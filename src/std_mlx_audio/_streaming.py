# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Windowed streaming session for the MLX ASR backends.

This plugin decodes every family through mlx-audio's batch ``model.generate``,
a call over a whole utterance (long audio is chunked internally, but each call
re-reads its input from the start). Some upstream models also have streaming
entry points (Nemotron's cache-aware ``stream_generate``); the plugin does not
use them today. This session synthesizes streaming output by a
**re-decode-the-window** strategy (the same honest approach
``std-faster-whisper`` uses for a batch engine), and the capabilities declared in
``_metadata.py`` match exactly what that strategy delivers.

Strategy
--------
1. **Intake.** A task owned by the session reads
   :meth:`TranscriptionSession.audio_chunks` as chunks arrive and appends them to
   a bounded inbox, so the session knows how much audio has arrived. The bound is
   ``max_window_s`` of audio (``_UNCAPPED_INBOX_S`` with no cap). While the inbox
   is at or over it, the intake reads no further chunk, so the library's bounded
   audio queue fills and pushes back on the client as it does without an intake.
   The inbox can therefore hold the bound plus one chunk (the chunk that crossed
   it); the float32 window and the byte copy a pass takes are separate buffers.
   These bound the session's own buffers, not the process's memory.
   Only whole samples leave the inbox; a byte of a sample split across chunks
   waits there for the rest.
2. **Pacing.** A decode pass is due when the inbox holds at least
   ``redecode_interval_s`` of audio, when it is full, when its oldest sample has
   waited ``redecode_interval_s`` (so audio is decoded even when the client stops
   sending), or when the input has ended; ``redecode_interval_s`` is the one
   setting for how often the window is decoded. There is no rest between passes.
   The pass lets the intake settle (yields until no chunk has arrived for a few
   milliseconds) and then takes everything in the inbox, so audio that arrived
   during a slow decode, or while another session used the loop, is taken in one
   step. Before each decode the session yields to the loop once.

   **Backlog** (with a cap). When a pass commits heads at the cap and the inbox
   has filled again meanwhile, the pass takes it again and keeps committing
   heads; it decodes the rest of the window (the ``partial``) only once the
   backlog is absorbed. While a session works through a backlog it therefore
   emits ``final`` events for the committed heads and no ``partial`` until it
   has caught up, and the window is not decoded again for each step.

   The session keeps up only while its sustained end-to-end service capacity
   exceeds the rate at which audio arrives. That capacity covers every decode
   (each pass decodes the whole window again, so a window near the cap costs a
   decode near the cap's length however little audio is new, and a cap commit
   adds the head's decode; during a backlog only heads are decoded), the
   settling, PCM conversion, event handling, and other work on the loop. When it
   falls short the session falls behind; the cap still bounds each decode and
   the inbox bound holds the client back. The decode runs inline on the
   event-loop thread — MLX arrays are thread-bound, so it cannot be offloaded to
   a worker thread; see :meth:`MlxAudioStreamingSession._decode`.
3. **Normal pass.** The whole window is decoded. Leading segments that end at
   least ``settle_margin_s`` before the end of the window become ``final``
   events and their audio is dropped; the rest of the window is one ``partial``.
4. **Commits ahead of the normal pass.** Two more kinds of commit happen before
   the normal pass decodes the window:

   * the **window cap**: the inbox moves into the window in portions of at most
     the cap, and while the window is at least ``max_window_s`` long, a bounded
     head is committed, again and again, until less than the cap remains. That
     also holds on the final pass and for a single huge chunk, so no decode
     covers more than ``max_window_s`` and the window never holds more than twice
     the cap. How the head is chosen depends on whether the backend can return a
     segment boundary inside a window of the cap's length
     (:func:`~std_mlx_audio.backends.has_inner_boundaries`, decided from the
     model family and its ``chunk_duration`` before any decode):

     - if it can (Whisper, Parakeet; Qwen3-ASR only with a ``chunk_duration``
       shorter than the cap), the first ``max_window_s`` of the window is decoded
       and committed up to its last settled segment (one that ends
       ``settle_margin_s`` before the prefix ends) or, failing that, its last
       inner boundary, provided the commit covers at least half the cap;
     - otherwise (Qwen3-ASR at its default ``chunk_duration``), or when no such
       boundary covers half the cap, the window is cut in the longest pause of
       the ten seconds before the cap (never before half the cap), and the head
       before the cut is decoded once on its own. With no pause there, the cut
       goes at the least loud point, so speech with no pause is still bounded;
   * a **pause** of at least ``commit_pause_s`` after speech (when set, and only
     for a backend that cannot return inner boundaries): the audio up to the
     middle of the pause is decoded once on its own and committed.

   Pauses come from a cheap energy profile of the window (frame RMS;
   :func:`_frame_rms` and :func:`_quiet_threshold`).
5. On ``end_audio`` the same cap commits run, then the rest is decoded and
   finalized, then ``done``. A byte left over from an incomplete last sample is
   dropped with a ``partial_sample_dropped`` diagnostic.

Positions (the dropped origin, the window, the decoded frontier) are integer
sample counts; they become seconds only inside an event, so the audio cursor
never moves backwards by a rounding step. ``audio_processed_until`` is the end of
the audio a successful decode has covered, never audio that is merely buffered.

Input that is queued all at once (a whole file through ``audio=``, or a client
pushing faster than real time) is a backlog from the start. With a cap it is
taken in bounded steps and decoded in bounded heads, and yields ``final`` events
without a ``partial`` until the backlog is absorbed; with ``max_window_s=None``
no cap heads are committed, and it still produces partials.

Honesty
-------
The model re-decodes the window each pass and may rewrite ANY not-yet-finalized
text, so every ``partial`` leaves ``stable_text`` out, which makes it ``""``
(``partial_stability=false``). We never emit ``supersede``
(``re_segments=false``). Segment ids are synthesized
deterministically and monotonically (``seg-0`` ...). Every commit is a seam: the
next window is decoded without the earlier text, so a word that straddles a cut
can be lost or doubled, and a cut in a mid-sentence pause can make the model end
the head with a period. Cutting in pauses makes word damage rare, not impossible.
This is a *pragmatic* streaming adapter for batch decoders, not a true
incremental recognizer — the README and findings doc say so plainly.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray
from standard_asr import RuntimeParams, TranscriptionEvent, TranscriptionSession
from standard_asr.contract.language import effective_language
from standard_asr.contract.results import Segment

from . import backends
from ._config import MlxAudioConfig, MlxAudioParams
from ._guidance import echoes_prompt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .engine import MlxAudioASR

_LOGGER = logging.getLogger(__name__)

#: MLX STT backends run at 16 kHz mono; wire frames are negotiated to this rate.
_SAMPLE_RATE = backends.SAMPLE_RATE
#: int16 <-> float32 scaling (canonical wire decode is /32768; spec AI R4).
_PCM_SCALE = 32768.0

#: Inbox bound when there is no cap, in seconds of audio.
_UNCAPPED_INBOX_S = 30.0
#: Pacing: before taking the inbox, yield to the loop in steps of this many
#: seconds until a step passes with no new chunk and no other work blocking the
#: loop (a step that took over twice this long was blocked) ...
_SETTLE_QUIET_S = 0.01
#: ... but for at most this many steps, so a steady stream cannot hold a decode
#: back.
_SETTLE_MAX_STEPS = 5

#: Energy profile: frame length in seconds (20 ms; a word gap spans several).
_FRAME_S = 0.02
#: Samples per energy frame.
_FRAME_SAMPLES = round(_FRAME_S * _SAMPLE_RATE)
#: Energy profile: frames averaged when looking for the least loud cut point in
#: speech with no pause (100 ms, as mlx-audio's own ``split_audio_into_chunks``
#: does), so a stop consonant's short closure inside a word does not win.
_SMOOTH_FRAMES = 5
#: Quiet threshold: the noise level is the window's 5th-percentile frame RMS (so
#: pauses must fill at least 5% of the window to be measured) and the speech
#: level its 99th-percentile frame RMS (so speech filling just over 1% of the
#: window still counts). A frame is quiet when it is no louder than the noise
#: level plus this fraction of the gap up to the speech level.
_QUIET_FRACTION = 0.1
#: Quiet threshold: a window whose speech level is less than this many times its
#: noise level (12 dB) is treated as holding no pause at all: no frame is quiet.
_MIN_SPEECH_TO_NOISE = 4.0
#: Quiet threshold: an absolute floor (-60 dBFS RMS).
_QUIET_FLOOR = 1e-3
#: Window cap: seconds before the cap searched for a pause to cut in. Long enough
#: to hold a sentence end in most dictation, so the cut need not split a sentence.
_CUT_SEARCH_S = 10.0
#: Window cap: the last seconds of the window are not searched, so the audio just
#: heard (possibly a word still being spoken) stays in the window.
_MIN_TAIL_S = 1.0


def _rebase(origin: float, t: float | None) -> float | None:
    """Shift a window-relative time to absolute, preserving an absent value.

    Segment timestamps are optional in the core schema; a segment without one
    keeps ``None`` rather than inventing a time.

    Args:
        origin: The sliding-window origin in absolute seconds.
        t: A window-relative time, or ``None``.

    Returns:
        The absolute time, or ``None``.
    """
    return None if t is None else origin + t


def _pcm_s16le_to_float32(data: bytes) -> NDArray[np.float32]:
    """Decode canonical 16-bit LE PCM bytes into a float32 mono waveform.

    A trailing odd byte (half a sample split across chunks) is dropped.

    Args:
        data: Raw ``pcm_s16le`` bytes (mono).

    Returns:
        A ``float32`` array in ``[-1, 1)``; empty if ``data`` is too short.
    """
    if len(data) < 2:
        return np.zeros(0, dtype=np.float32)
    usable = len(data) - (len(data) % 2)
    samples: NDArray[np.int16] = np.frombuffer(data[:usable], dtype="<i2")
    return np.array(samples, dtype=np.float32) / _PCM_SCALE


def _seconds(samples: int) -> float:
    """Convert a sample count to seconds."""
    return samples / _SAMPLE_RATE


def _frame_rms(audio: NDArray[np.float32]) -> NDArray[np.float64]:
    """Return the RMS of each ``_FRAME_S`` frame of ``audio``.

    A trailing partial frame (under 20 ms) is ignored.

    Args:
        audio: A float32 mono waveform at ``_SAMPLE_RATE``.

    Returns:
        One RMS value per whole frame; empty if ``audio`` is shorter than a frame.
    """
    count = audio.size // _FRAME_SAMPLES
    if count == 0:
        return np.zeros(0)
    power = np.square(audio[: count * _FRAME_SAMPLES], dtype=np.float64)
    starts = np.arange(0, count * _FRAME_SAMPLES, _FRAME_SAMPLES)
    return np.sqrt(np.add.reduceat(power, starts) / _FRAME_SAMPLES)


def _quiet_threshold(rms: NDArray[np.float64]) -> float:
    """Return the RMS at or below which a frame of this window counts as quiet.

    The threshold sits ``_QUIET_FRACTION`` of the way from the window's noise
    level (5th-percentile frame RMS) up to its speech level (99th percentile).
    Its limits, all of which fall back to "no frame is quiet" (no pause commit;
    a cap cut then goes to the least loud point) or to a poorer pause choice:

    * speech less than 12 dB (``_MIN_SPEECH_TO_NOISE``) above a steady noise
      floor, or a window of noise alone, shows no pause;
    * pauses filling less than about 5% of the window are not measured as the
      noise level, so the weakest stretches of speech take that role;
    * speech filling less than about 1% of the window is not seen as speech.

    Args:
        rms: The window's frame RMS profile (not empty).

    Returns:
        The quiet threshold; negative when no frame counts as quiet.
    """
    noise = float(np.percentile(rms, 5))
    speech = float(np.percentile(rms, 99))
    if speech < max(_QUIET_FLOOR, _MIN_SPEECH_TO_NOISE * noise):
        return -1.0
    return max(_QUIET_FLOOR, noise + _QUIET_FRACTION * (speech - noise))


def _quiet_runs(rms: NDArray[np.float64]) -> list[tuple[int, int]]:
    """Return the runs of quiet frames as ``(first, end)`` frame indexes.

    Args:
        rms: The window's frame RMS profile (not empty).

    Returns:
        Each maximal run of frames at or below :func:`_quiet_threshold`, in order;
        ``end`` is exclusive.
    """
    quiet = (rms <= _quiet_threshold(rms)).astype(np.int8)
    edges = np.diff(np.concatenate(([0], quiet, [0])))
    starts = np.flatnonzero(edges == 1).tolist()
    ends = np.flatnonzero(edges == -1).tolist()
    return list(zip(starts, ends, strict=True))


def _pause_cut(rms: NDArray[np.float64], min_pause_s: float) -> int | None:
    """Find the latest pause of at least ``min_pause_s`` that follows speech.

    A pause is a run of quiet frames (see :func:`_quiet_threshold`). A run at the
    very start of the window follows no speech in this window, so it does not
    count; that also keeps a cut from firing twice on the same pause, since the
    second half of a cut pause starts the next window. A window where no frame
    counts as quiet has no such run.

    Args:
        rms: The window's frame RMS profile.
        min_pause_s: The shortest pause that counts, in seconds.

    Returns:
        The window-relative sample at the middle of that pause, or ``None``.
    """
    if rms.size == 0:
        return None
    min_frames = math.ceil(min_pause_s / _FRAME_S - 1e-9)
    found = [(s + e) // 2 for s, e in _quiet_runs(rms) if s > 0 and e - s >= min_frames]
    return found[-1] * _FRAME_SAMPLES if found else None


def _cap_cut(rms: NDArray[np.float64], lo: int, hi: int) -> tuple[int, bool]:
    """Find the best cut frame of the window between frames ``lo`` and ``hi``.

    The best point is the middle of the longest pause (run of quiet frames)
    inside the range, because a longer pause is more likely to end a sentence
    than to sit between two words of one. With no quiet frame in the range, it
    is the least loud point, after averaging the energy over ``_SMOOTH_FRAMES``
    frames.

    Args:
        rms: The window's frame RMS profile.
        lo: Earliest allowed cut frame.
        hi: Latest allowed cut frame (exclusive); greater than ``lo``.

    Returns:
        The cut as a window-relative sample, and whether it lies in a pause.
    """
    clipped = [(min(e, hi) - max(s, lo), max(s, lo), min(e, hi)) for s, e in _quiet_runs(rms)]
    inside = [run for run in clipped if run[0] > 0]
    if inside:
        _, first, end = max(inside)  # the longest; the later one on a tie
        return (first + end) // 2 * _FRAME_SAMPLES, True
    half = _SMOOTH_FRAMES // 2
    padded = np.pad(np.square(rms), half, mode="edge")  # no false minimum at the edges
    energy = np.convolve(padded, np.ones(_SMOOTH_FRAMES) / _SMOOTH_FRAMES, mode="valid")
    return (lo + int(np.argmin(energy[lo:hi]))) * _FRAME_SAMPLES, False


def _clamp(segments: list[Segment], samples: int) -> list[Segment]:
    """Clamp segment and word times to the decoded audio's length.

    Segments and words are clamped independently: a word can run past the audio
    even when its segment does not.

    mlx-audio's Qwen3-ASR pads input shorter than one second and reports the
    padded length, so a short head can come back as ``[0, 1]``. Times past the
    audio that was actually decoded would overlap the next window.

    Args:
        segments: Segments with times relative to the decoded audio.
        samples: Length of the decoded audio in samples.

    Returns:
        The segments, with every time at most the audio's length.
    """
    limit = _seconds(samples)
    out: list[Segment] = []
    for seg in segments:
        update: dict[str, Any] = {}
        if seg.end is not None and seg.end > limit:
            assert seg.start is not None  # The schema forbids an end without a start.
            update.update(start=min(seg.start, limit), end=limit)
        if seg.words and any(w.end > limit for w in seg.words):
            update["words"] = [
                w.model_copy(update={"start": min(w.start, limit), "end": min(w.end, limit)})
                for w in seg.words
            ]
        out.append(seg.model_copy(update=update) if update else seg)
    return out


class MlxAudioStreamingSession(TranscriptionSession):
    """A windowed streaming session backed by the engine's bound backend.

    Args:
        engine: The owning engine (model-loaded by the time :meth:`_produce`
            runs).
        gated_params: Frozen, already-gated runtime parameters (spec RT R5).
        redecode_interval_s: Minimum seconds of new audio between two decodes of
            the window. A lower bound on spacing, not a fixed cadence: each
            decode takes all the audio that has arrived, so after a slow decode
            the next one covers more than the interval. Audio below one interval
            is decoded once its oldest sample has waited this long. This is the
            one setting for how often the window is decoded: there is no rest
            between passes.
        settle_margin_s: A segment is finalized once it ends at least this many
            seconds before the end of the decoded audio; its audio is then
            dropped from the window. Qwen3-ASR returns one segment per
            ``chunk_duration`` of audio, so with the default (1200 s) its segment
            never ends before a window shorter than that does: then this has no
            effect, except that ``0`` finalizes the whole window on every decode,
            and its commits come from the cap, pauses, and the end of the input.
        max_window_s: Cap on the window length in seconds; no decode covers more
            than this, and it also bounds the inbox (see the module docstring for
            how a head is chosen). ``None`` disables the cap: the window then
            grows until a settled segment or a pause commits it, without limit,
            and the inbox is bounded by ``_UNCAPPED_INBOX_S``.
        commit_pause_s: When set, a pause of at least this many seconds after
            speech commits the audio before it right away, without waiting for
            the cap: that audio is decoded once on its own and finalized. Applies
            only to a backend that cannot return a segment boundary inside a
            window (see :func:`~std_mlx_audio.backends.has_inner_boundaries`);
            for the others, settled segments already commit promptly. It keeps
            the window short and delivers a ``final`` soon after the speaker
            pauses, at the cost of more seams. ``None`` commits only at settled
            segments, the cap, and the end of the input.
        **session_kwargs: Forwarded to :class:`TranscriptionSession` (deadlines,
            buffer sizes, ``strict_lifecycle``).

    Raises:
        ValueError: If ``max_window_s`` is shorter than one 20 ms energy frame.
    """

    def __init__(
        self,
        engine: MlxAudioASR,
        gated_params: RuntimeParams,
        *,
        redecode_interval_s: float = 1.5,
        settle_margin_s: float = 2.0,
        max_window_s: float | None = 30.0,
        commit_pause_s: float | None = None,
        **session_kwargs: Any,
    ) -> None:
        super().__init__(**session_kwargs)
        if max_window_s is not None and round(max_window_s * _SAMPLE_RATE) < _FRAME_SAMPLES:
            raise ValueError(
                f"max_window_s={max_window_s} is shorter than one {_FRAME_S * 1000:.0f} ms "
                "frame; use a cap of at least a few seconds, or None for no cap."
            )
        self._engine = engine
        self._params = gated_params
        self._redecode_interval_s = redecode_interval_s
        self._settle_margin_s = settle_margin_s
        self._commit_pause_s = commit_pause_s
        # Positions are integer sample counts (seconds only inside events).
        self._interval_bytes = max(2, 2 * round(redecode_interval_s * _SAMPLE_RATE))
        self._cap = None if max_window_s is None else round(max_window_s * _SAMPLE_RATE)
        # The inbox bound in bytes: the cap's worth of audio, or a fixed amount.
        bound_samples = self._cap if self._cap is not None else _UNCAPPED_INBOX_S * _SAMPLE_RATE
        self._inbox_limit = 2 * round(bound_samples)
        # Whether a window of the cap's length can hold a segment boundary.
        self._has_boundaries = backends.has_inner_boundaries(
            type(engine).backend, self._mlx_params(), max_window_s
        )
        # The base TranscriptionSession reserves several private attribute names
        # (`_buffer`, `_audio_queue`, ...); we name our audio window `_window`
        # to avoid clobbering them.
        self._window: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        # Samples already committed and dropped off the FRONT of the window. Only
        # ever increases; window-relative times map to session time through it.
        self._origin = 0
        # End of the audio a successful decode has covered (absolute samples).
        # This, not the buffered audio, is what `audio_processed_until` reports.
        self._processed = 0
        self._resolved_language = self._resolve_language()
        # Count of segments already emitted as `final` (also their next id). A
        # finalized segment's id and text are immutable and its audio is dropped.
        self._finalized_count = 0
        # The text the application shows for the open `partial` (segment
        # `seg-{_finalized_count}`), or None when no partial is open. A commit that
        # decodes to no text still closes an open partial, and a decode that
        # withdraws its text clears it once, so stale text never lingers.
        self._partial_shown: str | None = None
        # Intake: audio read from `audio_chunks()` and not yet taken by a pass.
        self._inbox = bytearray()
        # Loop time at which the inbox's oldest complete sample arrived, or None
        # when it holds no complete sample.
        self._oldest_at: float | None = None
        self._input_ended = False
        self._arrived = asyncio.Event()
        self._space = asyncio.Event()
        self._intake: asyncio.Task[None] | None = None

    def _mlx_params(self) -> MlxAudioParams:
        """Return the request's MLX provider params (defaults when none were set)."""
        provider = self._params.provider_params
        return provider if isinstance(provider, MlxAudioParams) else MlxAudioParams()

    def _resolve_language(self) -> str | None:
        """Resolve the effective language to forward to the backend.

        Returns:
            A BCP-47 tag, or ``None`` for auto-detect (also ``None`` for a
            fixed-language model whose default is ``"auto"``).
        """
        config = cast(MlxAudioConfig, self._engine.config)
        caps = type(self._engine).declared_capabilities
        resolved = effective_language(
            self._params.language,
            config.default_language,
            has_language_axis=bool(type(self._engine).properties.selectable_languages),
            runtime_override_supported=bool(caps.supports("streaming.language.runtime_override")),
        )
        return None if (resolved is None or resolved == "auto") else resolved

    def _decode(self, audio: NDArray[np.float32], *, want_words: bool) -> list[Segment]:
        """Run a full backend decode over ``audio`` (blocking; main thread).

        IMPORTANT — MLX threading constraint: MLX arrays (incl. model weights)
        are bound to the Metal stream of the thread they were created on, and a
        stream cannot be used from another thread. The model is loaded on the
        event-loop thread, so generation MUST run on that same thread — offloading
        to ``asyncio.to_thread`` raises ``RuntimeError: There is no Stream(gpu, N)
        in current thread``. We therefore decode inline (this blocks the loop for
        the decode duration). See ``docs/STANDARD_ASR_FINDINGS.md``.

        Args:
            audio: The audio to decode, as a float32 array.
            want_words: Whether to request word-level timestamps.

        Returns:
            The decoded Standard ASR segments for this audio.
        """
        engine = self._engine
        backend = type(engine).backend
        config = cast(MlxAudioConfig, engine.config)
        gen_kwargs = backend.generate_kwargs(
            resolved_language=self._resolved_language,
            want_words=want_words,
            params=self._mlx_params(),
            config=config,
            prompt=self._params.prompt,
        )
        model = cast(Any, engine.model)
        source = backends.to_mlx_array(np.ascontiguousarray(audio, dtype=np.float32))
        native = model.generate(backends.adapt_audio_source(backend, source), **gen_kwargs)
        result = backend.to_result(
            native, duration=backends.waveform_duration(audio), want_words=want_words
        )
        # A window with no speech can come back as the prompt itself (see _guidance).
        if echoes_prompt(
            result.text, gen_kwargs.get("system_prompt", gen_kwargs.get("initial_prompt"))
        ):
            return []
        return list(result.segments or [])

    # ------------------------------------------------------------------ #
    # Intake and pacing
    # ------------------------------------------------------------------ #
    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        """Drive the windowed re-decode loop and yield streaming events.

        Yields:
            ``final`` events for committed segments, ``partial`` for the
            in-progress tail, ``progress`` heartbeats carrying the audio cursor,
            and a terminal ``done`` (or a non-recoverable ``error``).
        """
        self._engine.ensure_loaded(mode="streaming")
        want_words = backends.map_word_timestamps(self._params.word_timestamps)
        self._intake = asyncio.ensure_future(self._take_in())
        try:
            while True:
                await self._wait_until_due()
                if not self._input_ended:
                    await self._settle_intake()
                self._raise_intake_failure()
                final_pass = self._input_ended
                async for event in self._decode_pass(want_words=want_words, final_pass=final_pass):
                    yield event
                    if event.type == "error":  # pragma: no cover - the base stops at a terminal
                        return
                if final_pass:
                    break
            yield TranscriptionEvent.done(audio_processed_until=_seconds(self._processed))
        finally:
            await self._stop_intake()

    async def _close(self) -> None:
        """Stop the intake task on every exit path, then close the base."""
        await self._stop_intake()
        await super()._close()

    async def _take_in(self) -> None:
        """Read fed audio into the inbox as it arrives (the intake task).

        Stops reading while the inbox is full; a pass that takes the inbox makes
        room again (:attr:`_space`).
        """
        loop = asyncio.get_running_loop()
        try:
            async for chunk in self.audio_chunks():
                self._inbox.extend(chunk)
                if self._oldest_at is None and len(self._inbox) >= 2:
                    self._oldest_at = loop.time()
                self._arrived.set()
                while len(self._inbox) >= self._inbox_limit:
                    self._space.clear()
                    await self._space.wait()
        finally:
            self._input_ended = True
            self._arrived.set()

    async def _stop_intake(self) -> None:
        """Stop the intake task, wait for it, and retrieve how it ended.

        Retrieving the outcome keeps a failure that lost the race against
        shutdown from being reported as never retrieved; it does not replace the
        terminal event the session already has (the failure is logged).
        """
        task = self._intake
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.wait({task})
        if not task.cancelled() and task.exception() is not None:
            _LOGGER.debug(
                "Audio intake ended with %r while the session was ending.", task.exception()
            )

    def _raise_intake_failure(self) -> None:
        """Re-raise an exception the intake task ended with, in the producer.

        The base session turns it into the matching terminal ``error``.
        """
        task = self._intake
        if task is not None and task.done() and not task.cancelled():
            failure = task.exception()
            if failure is not None:
                raise failure

    def _inbox_full(self) -> bool:
        """Whether the inbox has reached its bound."""
        return len(self._inbox) >= self._inbox_limit

    async def _wait_until_due(self) -> None:
        """Wait until a decode pass is due.

        A pass is due once the inbox holds ``redecode_interval_s`` of audio, is
        full, or holds a complete sample that has waited ``redecode_interval_s``,
        or once the input has ended. The wait wakes on a chunk, the end of the
        input, or the age deadline, so buffered audio is decoded even when the
        client sends nothing more. An empty inbox, or one holding only part of a
        sample, never makes a pass due. There is no rest after a pass (see
        ``docs/DESIGN.md`` §4 for the rules that were tried and why none stayed).
        """
        loop = asyncio.get_running_loop()
        while not self._input_ended:
            size = len(self._inbox)
            if size >= self._interval_bytes or self._inbox_full():
                break
            timeout: float | None = None
            if self._oldest_at is not None:
                timeout = self._oldest_at + self._redecode_interval_s - loop.time()
                if timeout <= 0:
                    break
            self._arrived.clear()
            try:
                await asyncio.wait_for(self._arrived.wait(), timeout)
            except asyncio.TimeoutError:
                break

    async def _settle_intake(self) -> None:
        """Let chunks already on their way reach the inbox before taking it.

        After a slow decode (this session's, or another session's on the same
        loop), the transport and the intake task still hold audio that arrived
        meanwhile. Yield to the loop in ``_SETTLE_QUIET_S`` steps until one passes
        with no new chunk and without the loop being blocked by other work, for
        at most ``_SETTLE_MAX_STEPS`` steps, and not at all once the inbox is full.
        """
        loop = asyncio.get_running_loop()
        for _ in range(_SETTLE_MAX_STEPS):
            if self._input_ended or self._inbox_full():
                return
            size = len(self._inbox)
            started = loop.time()
            await asyncio.sleep(_SETTLE_QUIET_S)
            blocked = loop.time() - started > 2 * _SETTLE_QUIET_S
            if len(self._inbox) == size and not blocked:
                return

    def _take_inbox(self, *, final_pass: bool) -> bytes:
        """Take every whole sample out of the inbox and make room for the intake.

        A byte left over from a sample split across chunks stays for the next
        chunk; at the end of the input it is dropped, with a diagnostic.

        Args:
            final_pass: Whether the input has ended.

        Returns:
            The taken ``pcm_s16le`` bytes (an even number).
        """
        usable = len(self._inbox) - len(self._inbox) % 2
        taken = bytes(self._inbox[:usable])
        del self._inbox[:usable]
        self._oldest_at = None
        if final_pass and self._inbox:
            self._inbox.clear()
            _LOGGER.warning("Audio input ended in the middle of a sample; one byte was dropped.")
            self.emit_diagnostic(
                code="partial_sample_dropped",
                level="warning",
                message=(
                    "The audio input ended in the middle of a 16-bit sample; the "
                    "incomplete last byte was dropped."
                ),
            )
        self._space.set()
        return taken

    # ------------------------------------------------------------------ #
    # Decode passes
    # ------------------------------------------------------------------ #
    async def _decode_pass(
        self, *, want_words: bool, final_pass: bool
    ) -> AsyncIterator[TranscriptionEvent]:
        """Move the inbox into the window, commit what is due, decode the rest.

        The inbox moves in portions of at most the inbox bound, with the cap
        commits run after each portion, so the window never holds more than the
        cap plus one portion.

        **Backlog.** If those commits committed at least one head and the inbox
        has meanwhile filled again (at least one interval of audio, full, or the
        input has ended with audio left), the pass takes the inbox again and
        repeats, before any pause commit and before decoding the rest of the
        window. So while a session works through a backlog it emits ``final``
        events for the committed heads and no ``partial`` until it has caught up:
        the rest of the window is decoded only once the backlog is absorbed. A
        round that commits no head ends the repetition (it cannot have taken
        enough audio to reach the cap). A pass that commits no head at all
        behaves as without this rule.

        Args:
            want_words: Whether word-level timestamps were requested.
            final_pass: ``True`` after ``end_audio``: finalize everything.

        Yields:
            The pass's events in order; an ``error`` event ends the pass.
        """
        while True:
            taken = self._take_inbox(final_pass=final_pass)
            origin = self._origin
            for start in range(0, len(taken), self._inbox_limit):
                self._append_pcm(taken[start : start + self._inbox_limit])
                async for event in self._commit_over_cap(want_words=want_words):
                    yield event
                    if event.type == "error":  # pragma: no cover - base stops at a terminal
                        return
            del taken
            if self._origin == origin or not await self._backlog_waiting():
                break
        if not final_pass and self._commit_pause_s is not None and not self._has_boundaries:
            cut = _pause_cut(_frame_rms(self._window), self._commit_pause_s)
            if cut is not None and cut > 0:
                for event in await self._commit_head(cut, want_words=want_words):
                    yield event
                    if event.type == "error":  # pragma: no cover - base stops at a terminal
                        return
        for event in await self._decode_rest(want_words=want_words, final_pass=final_pass):
            yield event

    async def _backlog_waiting(self) -> bool:
        """Whether more audio is waiting after a round of head commits.

        Lets audio already on its way reach the inbox first (as before a pass).

        Returns:
            Whether the inbox holds at least one interval of audio, is full, or
            holds audio after the input has ended.
        """
        await self._settle_intake()
        size = len(self._inbox)
        if size >= self._interval_bytes or self._inbox_full():
            return True
        return self._input_ended and size >= 2

    async def _commit_over_cap(self, *, want_words: bool) -> AsyncIterator[TranscriptionEvent]:
        """Commit bounded heads until the window is under the cap.

        Args:
            want_words: Whether word-level timestamps were requested.

        Yields:
            ``final`` events (or one ``error``) for each committed head.
        """
        while self._cap is not None and self._window.size >= self._cap:
            if self._has_boundaries:
                events = await self._commit_settled_prefix(self._cap, want_words=want_words)
                if events is not None:
                    for event in events:
                        yield event
                    if events[-1].type == "error":  # pragma: no cover - base stops at a terminal
                        return
                    continue
            events = await self._commit_head(self._cap_cut_point(), want_words=want_words)
            for event in events:
                yield event
            if events[-1].type == "error":  # pragma: no cover - the base stops at a terminal
                return

    def _cap_cut_point(self) -> int:
        """Return where to cut an over-cap window at a pause.

        The best point (:func:`_cap_cut`) of the ``_CUT_SEARCH_S`` seconds before
        the cap, but not before half the cap and not in the last ``_MIN_TAIL_S``
        of the window. Only the audio up to the end of that range is analysed.
        When the range is empty (a cap of a second or two), the cut falls back to
        the cap itself, which can leave nothing in the window.

        Returns:
            The cut as a window-relative sample, between 1 and the cap.
        """
        cap = cast(int, self._cap)
        cap_frames = cap // _FRAME_SAMPLES
        half_frames = -(-((cap + 1) // 2) // _FRAME_SAMPLES)  # at least half the cap, rounded up
        lo = max(half_frames, cap_frames - round(_CUT_SEARCH_S / _FRAME_S))
        tail_frames = round(_MIN_TAIL_S / _FRAME_S)
        hi = min(cap_frames, self._window.size // _FRAME_SAMPLES - tail_frames)
        if hi - lo < 2:
            return cap
        cut, quiet = _cap_cut(_frame_rms(self._window[: hi * _FRAME_SAMPLES]), lo, hi)
        if not quiet:
            _LOGGER.debug(
                "No pause in %.1f-%.1f s of the window; cutting at its least loud point (%.2f s).",
                lo * _FRAME_S,
                hi * _FRAME_S,
                _seconds(cut),
            )
        return max(1, cut)

    async def _commit_settled_prefix(
        self, length: int, *, want_words: bool
    ) -> list[TranscriptionEvent] | None:
        """Decode the window's first ``length`` samples and commit at a boundary.

        Commits up to the last settled segment (one that ends ``settle_margin_s``
        before the prefix ends) or, if that covers less than half the prefix, up
        to the last segment boundary inside it, finalizing every segment up to
        and including the one that ends there. A full-cap decode must buy real
        progress, so a commit covering less than half the prefix is not made.

        Args:
            length: The prefix length in samples (the cap).
            want_words: Whether word-level timestamps were requested.

        Returns:
            The events (``final`` events, or one ``error``), or ``None`` if no
            boundary covers half the prefix (the caller then cuts at a pause).
        """
        segments = await self._safe_decode(self._window[:length], want_words=want_words)
        if isinstance(segments, TranscriptionEvent):
            return [segments]
        segments = _clamp(segments, length)
        self._processed = max(self._processed, self._origin + length)
        prefix_s = _seconds(length)
        # The boundary is chosen as a segment index, and the segments up to and
        # including it are finalized: a time converted to a sample count and back
        # can land just below the end it came from, so it must not decide which
        # segments the cut includes.
        last_settled = last_inner = -1
        for index, seg in enumerate(segments):
            # An unknown end cannot anchor a cut or be skipped by a later boundary.
            if seg.end is None:
                break
            if seg.end <= prefix_s - self._settle_margin_s:
                last_settled = index
            if seg.end < prefix_s - _FRAME_S:
                last_inner = index
        for index in (last_settled, last_inner):
            if index < 0:
                continue
            boundary = segments[index].end
            assert boundary is not None  # Only measured ends enter the boundary run.
            cut = round(boundary * _SAMPLE_RATE)
            if 2 * cut >= length:  # at least half the prefix
                # The same lifecycle rules as a head commit: a prefix with no
                # text still closes an open partial, before its audio goes.
                events = self._finalize_all(segments[: index + 1], cut)
                self._trim_window(cut)
                return events
        return None

    async def _commit_head(self, cut: int, *, want_words: bool) -> list[TranscriptionEvent]:
        """Decode the window's first ``cut`` samples on their own and commit them.

        Args:
            cut: Window-relative cut in samples (at least 1).
            want_words: Whether word-level timestamps were requested.

        Returns:
            ``final`` events for the head (one empty ``final`` closing the open
            partial if the head decodes to no text), one ``progress`` event if
            there is nothing to finalize, or an ``error`` event.
        """
        cut = max(1, min(cut, self._window.size))
        segments = await self._safe_decode(self._window[:cut], want_words=want_words)
        if isinstance(segments, TranscriptionEvent):
            return [segments]
        segments = _clamp(segments, cut)
        self._processed = max(self._processed, self._origin + cut)
        events = self._finalize_all(segments, cut)
        self._trim_window(cut)
        return events

    async def _decode_rest(self, *, want_words: bool, final_pass: bool) -> list[TranscriptionEvent]:
        """Decode the window (now under the cap) and build the pass's last events.

        Args:
            want_words: Whether word-level timestamps were requested.
            final_pass: ``True`` after ``end_audio`` (finalize every segment).

        Returns:
            The ordered events.
        """
        size = self._window.size
        if size == 0:
            # Nothing to decode; the final pass must still close an open partial.
            if final_pass:
                return self._finalize_all([], 0)
            return [TranscriptionEvent.progress(audio_processed_until=_seconds(self._processed))]
        segments = await self._safe_decode(self._window, want_words=want_words)
        if isinstance(segments, TranscriptionEvent):
            return [segments]
        segments = _clamp(segments, size)
        self._processed = max(self._processed, self._origin + size)
        if final_pass:
            return self._finalize_all(segments, size)
        return self._build_events(segments)

    async def _safe_decode(
        self, audio: NDArray[np.float32], *, want_words: bool
    ) -> list[Segment] | TranscriptionEvent:
        """Decode ``audio``, turning a backend failure into an ``error`` event.

        Before decoding, it yields once to the loop and then re-raises an
        intake failure, if the intake task has ended with one, so a failed input
        is noticed before each decode, also within a long pass.

        Args:
            audio: The audio to decode.
            want_words: Whether word-level timestamps were requested.

        Returns:
            The decoded segments, or a non-recoverable ``engine_error`` event.
        """
        # Yield once so other loop tasks (e.g. the audio pump) make progress
        # before we block the loop on the inline MLX decode (see _decode: MLX is
        # thread-bound, so we cannot offload to a worker thread).
        await asyncio.sleep(0)
        # A failed intake ends the session before any more decode work (the base
        # turns the exception into the terminal error).
        self._raise_intake_failure()
        try:
            return self._decode(audio, want_words=want_words)
        except Exception as exc:
            _LOGGER.exception("MLX streaming decode failed")
            return TranscriptionEvent.make_error(
                code="engine_error",
                recoverable=False,
                extra={"detail": f"{type(exc).__name__}: {exc}"},
            )

    # ------------------------------------------------------------------ #
    # Events and the window
    # ------------------------------------------------------------------ #
    def _append_pcm(self, data: bytes) -> None:
        """Append decoded PCM bytes to the running float32 window.

        Args:
            data: ``pcm_s16le`` mono bytes.
        """
        if not data:
            return
        samples = _pcm_s16le_to_float32(data)
        if samples.size:
            self._window = np.concatenate([self._window, samples])

    def _final_event(self, seg: Segment) -> TranscriptionEvent:
        """Build the ``final`` for the next segment id and count it as finalized.

        Args:
            seg: The segment, with window-relative timestamps.

        Returns:
            The ``final`` event, with absolute timestamps.
        """
        origin = _seconds(self._origin)
        event = TranscriptionEvent.final(
            segment_id=f"seg-{self._finalized_count}",
            text=seg.text,
            start=_rebase(origin, seg.start),
            end=_rebase(origin, seg.end),
            words=self._shift_words(seg.words, origin),
            audio_processed_until=_seconds(self._processed),
        )
        self._finalized_count += 1
        self._partial_shown = None
        return event

    def _finalize_all(self, segments: list[Segment], samples: int) -> list[TranscriptionEvent]:
        """Finalize every segment of a committed stretch of audio.

        Used for every commit that finalizes audio and drops it: a head commit, a
        boundary commit at the cap, and the final pass (also when the window is
        already empty). A segment with empty text (recognition output with
        nothing in it) is not content and produces no event. When the audio
        decodes to no text but a ``partial`` is still open, that partial is
        closed with an empty ``final`` (an application keeps showing an open
        partial's text otherwise). That empty ``final`` and the one empty
        ``partial`` that clears withdrawn text (:meth:`_build_events`) are the
        only empty content events the session emits.

        Args:
            segments: Segments decoded from the window's first ``samples``.
            samples: Length of the committed audio in samples.

        Returns:
            The ``final`` events, or one ``progress`` event if there is nothing
            to finalize.
        """
        events = [self._final_event(seg) for seg in segments if seg.text]
        if not events and self._partial_shown is not None:
            _LOGGER.debug(
                "Committed %.2f s of audio decoded to no text; closing the open partial.",
                _seconds(samples),
            )
            empty = Segment(start=0.0, end=_seconds(samples), text="")
            events.append(self._final_event(empty))
        return events or [
            TranscriptionEvent.progress(audio_processed_until=_seconds(self._processed))
        ]

    def _build_events(self, segments: list[Segment]) -> list[TranscriptionEvent]:
        """Turn a decoded segment list into final/partial/progress events.

        ``segments`` come from decoding the CURRENT window, so their timestamps are
        window-relative; we map them to absolute session time with the origin.
        Leading segments that end at least ``settle_margin_s`` before the end of
        the window become ``final`` events with stable, monotonic ids; the
        remaining tail is one ``partial``. A segment with empty text is not
        content and produces no event, and a tail with no text produces no
        ``partial``, with one exception: when the open partial still shows text,
        one ``partial`` with empty text clears it. When nothing is left to emit,
        a ``progress`` event carries the cursor. Finalized audio is then dropped
        from the front of the window (the slide). The ``partial`` carries no
        stable text (the model may still rewrite the tail).

        Args:
            segments: Standard ASR ``Segment`` objects from this decode (window-
                relative timestamps, clamped to the window).

        Returns:
            The ordered events for this pass.
        """
        origin = _seconds(self._origin)  # decode-time origin
        settle_before = _seconds(self._window.size) - self._settle_margin_s

        # Leading run of segments settled enough to finalize.
        settled = 0
        for seg in segments:
            # An unknown end stays in the partial tail until its audio is committed.
            if seg.end is not None and seg.end <= settle_before:
                settled += 1
            else:
                break

        events = [self._final_event(seg) for seg in segments[:settled] if seg.text]
        tail = segments[settled:]
        tail_text = "".join(s.text for s in tail)
        if tail_text:
            tail_words: list[Any] | None = None
            for s in tail:
                if s.words:
                    tail_words = (tail_words or []) + list(s.words)
            # The aggregate span is legal only in the measured / start-only /
            # unavailable shapes: when the first tail segment has no start, a
            # later segment's end must not be emitted alone (end-without-start
            # is unrepresentable), so the whole span degrades to unavailable.
            tail_start = _rebase(origin, tail[0].start)
            tail_end = _rebase(origin, tail[-1].end) if tail_start is not None else None
            events.append(
                TranscriptionEvent.partial(
                    segment_id=f"seg-{self._finalized_count}",
                    text=tail_text,
                    start=tail_start,
                    end=tail_end,
                    words=self._shift_words(tail_words, origin),
                    audio_processed_until=_seconds(self._processed),
                )
            )
            self._partial_shown = tail_text
        elif self._partial_shown:
            # The decode withdrew the text the open partial shows: clear it once
            # (a revision, so legitimately content). Later empty results find
            # nothing shown and produce `progress` only.
            events.append(
                TranscriptionEvent.partial(
                    segment_id=f"seg-{self._finalized_count}",
                    text="",
                    audio_processed_until=_seconds(self._processed),
                )
            )
            self._partial_shown = ""
        elif not events:
            events.append(
                TranscriptionEvent.progress(audio_processed_until=_seconds(self._processed))
            )

        # Slide the window: drop the audio of the segments we just finalized so the
        # next decode runs over a bounded tail. Done last, so the events above used
        # the pre-trim origin.
        if settled:
            last_settled_end = segments[settled - 1].end
            assert last_settled_end is not None  # Only measured ends can settle.
            self._trim_window(round(last_settled_end * _SAMPLE_RATE))
        return events

    def _trim_window(self, samples: int) -> None:
        """Drop ``samples`` of audio off the front of the window.

        Advances the origin by exactly the number of samples dropped, so the
        origin and the window always sum to the same absolute position.

        Args:
            samples: Window-relative samples to drop (clamped to the window).
        """
        samples = max(0, min(samples, self._window.size))
        self._window = self._window[samples:]
        self._origin += samples

    @staticmethod
    def _shift_words(words: list[Any] | None, offset: float) -> list[Any] | None:
        """Return ``words`` with start/end shifted by ``offset`` seconds (absolute).

        Returns the input unchanged when there is nothing to shift (no words, or a
        zero origin), so the non-sliding path stays allocation-free and identical.

        Args:
            words: The window-relative :class:`~standard_asr.contract.results.Word` list, or
                ``None``.
            offset: Seconds to add (the sliding-window origin).

        Returns:
            A new list of shifted words, or the original/``None``.
        """
        if not words or offset <= 0.0:
            return words
        return [
            w.model_copy(update={"start": w.start + offset, "end": w.end + offset}) for w in words
        ]


__all__ = ["MlxAudioStreamingSession"]
