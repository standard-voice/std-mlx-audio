# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Windowed streaming session for the MLX ASR backends.

None of the MLX STT backends here (Qwen3-ASR, Whisper, Parakeet) expose a true
low-latency *incremental* decoder in mlx-audio — each ``model.generate`` is a
batch call over a whole utterance (they chunk long audio internally, but still
re-read from the start). This session synthesizes streaming output by a
**re-decode-the-window** strategy (the same honest approach
``std-faster-whisper`` uses for a batch engine), and the capabilities declared in
``_metadata.py`` match exactly what that strategy delivers.

Strategy
--------
1. Consume fed PCM frames via :meth:`TranscriptionSession.audio_chunks` and
   append them to a **sliding** float32 window.
2. Whenever at least ``redecode_interval_s`` of *new* audio has arrived, re-run
   the **whole current window** through the engine's bound backend (inline on the
   event-loop thread — MLX arrays are thread-bound, so the decode cannot be
   offloaded to a worker thread; see :meth:`MlxAudioStreamingSession._decode`)
   and emit:
   * a ``final`` for every segment that ends comfortably before the decode
     frontier (``settle_margin_s`` behind the last decoded timestamp) and has not
     been finalized yet;
   * one ``partial`` carrying the tail (everything after the last finalized
     segment) as the single in-progress segment.
3. **Slide the window:** once a segment is finalized, its audio is dropped from
   the front of the window (it can never change again), so the next decode runs
   over a bounded tail instead of the whole utterance — O(n) per pass rather than
   O(n²) over a long session. The decoder's window-relative timestamps are mapped
   back to absolute session time through an ``_origin_s`` offset (the duration
   already dropped), so segment/word times and the audio cursor stay absolute and
   monotonic. ``max_window_s`` is a hard cap that force-finalizes leading segments
   if speech is so continuous it never settles on its own.
4. On ``end_audio`` do a last decode and finalize the tail, then ``done``.

Honesty
-------
The model re-decodes the (current) window each pass and may rewrite ANY not-yet-
finalized text, so ``stable_until=0`` on every ``partial`` (``word_stability=false``)
and a segment is finalized only once it is at least ``settle_margin_s`` behind the
frontier. We never emit ``supersede`` (``re_segments=false``). Segment ids are
synthesized deterministically and monotonically (``seg-0`` ...). This is a
*pragmatic* streaming adapter for batch decoders, not a true incremental
recognizer — the README and findings doc say so plainly.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray
from standard_asr import RuntimeParams, TranscriptionEvent, TranscriptionSession
from standard_asr.language import effective_language

from . import backends
from ._config import MlxAudioConfig, MlxAudioParams

if TYPE_CHECKING:  # pragma: no cover - typing only
    from standard_asr.results import Segment

    from .engine import MlxAudioASR

_LOGGER = logging.getLogger(__name__)

#: MLX STT backends run at 16 kHz mono; wire frames are negotiated to this rate.
_SAMPLE_RATE = backends.SAMPLE_RATE
#: int16 <-> float32 scaling (canonical wire decode is /32768; spec AI R4).
_PCM_SCALE = 32768.0


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


class MlxAudioStreamingSession(TranscriptionSession):
    """A windowed streaming session backed by the engine's bound backend.

    Args:
        engine: The owning engine (model-loaded by the time :meth:`_produce`
            runs).
        gated_params: Frozen, already-gated runtime parameters (spec RT R5).
        redecode_interval_s: Minimum seconds of new audio to accumulate before
            re-decoding the window (latency vs. compute trade-off).
        settle_margin_s: A segment is finalized only once it ends at least this
            many seconds behind the last decoded timestamp (conservative
            stability under re-decode). Once finalized, its audio is dropped from
            the window.
        max_window_s: Hard cap on the sliding window length; leading segments are
            force-finalized to keep the window (and so each decode's cost) bounded
            under long, sparsely-segmented speech. ``None`` disables the cap.
        **session_kwargs: Forwarded to :class:`TranscriptionSession` (deadlines,
            buffer sizes, ``strict_lifecycle``).
    """

    def __init__(
        self,
        engine: MlxAudioASR,
        gated_params: RuntimeParams,
        *,
        redecode_interval_s: float = 1.5,
        settle_margin_s: float = 2.0,
        max_window_s: float | None = 30.0,
        **session_kwargs: Any,
    ) -> None:
        super().__init__(**session_kwargs)
        self._engine = engine
        self._params = gated_params
        self._redecode_interval_s = redecode_interval_s
        self._settle_margin_s = settle_margin_s
        self._max_window_s = max_window_s
        # The base TranscriptionSession reserves several private attribute names
        # (`_buffer`, `_audio_queue`, ...); we name our audio window `_window`
        # to avoid clobbering them.
        self._window = np.zeros(0, dtype=np.float32)
        # Absolute seconds of audio already dropped off the FRONT of the window
        # (the sliding-window origin). Window-relative decode timestamps are mapped
        # to absolute session time by adding this; it only ever increases.
        self._origin_s = 0.0
        self._resolved_language = self._resolve_language()
        # Count of segments already emitted as `final` (also their next id). A
        # finalized segment's id and text are immutable and its audio is dropped.
        self._finalized_count = 0

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
            audio: The accumulated window as a float32 array.
            want_words: Whether to request word-level timestamps.

        Returns:
            The decoded Standard ASR segments for this window.
        """
        engine = self._engine
        backend = type(engine).backend
        config = cast(MlxAudioConfig, engine.config)
        mlx_params = (
            self._params.provider_params
            if isinstance(self._params.provider_params, MlxAudioParams)
            else MlxAudioParams()
        )
        gen_kwargs = backend.generate_kwargs(
            resolved_language=self._resolved_language,
            want_words=want_words,
            params=mlx_params,
            config=config,
        )
        model = cast(Any, engine.model)
        source = backends.to_mlx_array(np.ascontiguousarray(audio, dtype=np.float32))
        native = model.generate(backends.adapt_audio_source(backend, source), **gen_kwargs)
        result = backend.to_result(
            native, duration=backends.waveform_duration(audio), want_words=want_words
        )
        return list(result.segments or [])

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        """Drive the windowed re-decode loop and yield streaming events.

        Yields:
            ``final`` events for settled segments, ``partial`` for the in-progress
            tail, ``progress`` heartbeats carrying the audio cursor, and a
            terminal ``done``.
        """
        self._engine.ensure_loaded()
        want_words = backends.map_word_timestamps(self._params.word_timestamps)
        pending = bytearray()
        bytes_since_decode = 0
        bytes_per_interval = int(self._redecode_interval_s * _SAMPLE_RATE * 2)

        async for chunk in self.audio_chunks():
            pending.extend(chunk)
            bytes_since_decode += len(chunk)
            if bytes_since_decode < bytes_per_interval:
                continue
            self._append_pcm(bytes(pending))
            pending.clear()
            bytes_since_decode = 0
            for event in await self._redecode(want_words=want_words, final_pass=False):
                yield event

        # Flush any tail audio that did not reach a full interval.
        if pending:
            self._append_pcm(bytes(pending))
            pending.clear()

        # Final pass: decode whatever we have and finalize everything.
        for event in await self._redecode(want_words=want_words, final_pass=True):
            yield event
        yield TranscriptionEvent.done(audio_processed_until=self._cursor_seconds())

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

    def _window_seconds(self) -> float:
        """Return the duration of the CURRENT (post-trim) window in seconds."""
        return self._window.size / _SAMPLE_RATE

    def _cursor_seconds(self) -> float:
        """Return the absolute audio time processed so far (seconds).

        This is the sliding-window origin plus the current window length, so it
        stays monotonic across trims (a trim moves audio from the window into the
        origin without changing the sum).
        """
        return self._origin_s + self._window_seconds()

    async def _redecode(self, *, want_words: bool, final_pass: bool) -> list[TranscriptionEvent]:
        """Re-decode the window and build the events for this pass.

        Args:
            want_words: Whether word-level timestamps were requested.
            final_pass: ``True`` on the post-``end_audio`` flush (finalize all
                remaining segments regardless of the settle margin).

        Returns:
            The ordered events to yield for this pass.
        """
        cursor = self._cursor_seconds()
        if self._window.size == 0:
            return [TranscriptionEvent.progress(audio_processed_until=cursor)]
        # Yield once so other loop tasks (e.g. the audio pump) make progress
        # before we block the loop on the inline MLX decode (see _decode: MLX is
        # thread-bound, so we cannot offload to a worker thread).
        await asyncio.sleep(0)
        try:
            segments = self._decode(self._window, want_words=want_words)
        except Exception as exc:
            _LOGGER.exception("MLX streaming decode failed")
            return [
                TranscriptionEvent.make_error(
                    code="engine_error",
                    recoverable=False,
                    extra={"detail": f"{type(exc).__name__}: {exc}"},
                )
            ]
        return self._build_events(segments, cursor=cursor, final_pass=final_pass)

    def _build_events(
        self, segments: list[Segment], *, cursor: float, final_pass: bool
    ) -> list[TranscriptionEvent]:
        """Turn a decoded segment list into final/partial/progress events.

        ``segments`` come from decoding the CURRENT window, so their timestamps are
        window-relative; we map them to absolute session time with
        :attr:`_origin_s`. Segments that end at least ``settle_margin_s`` behind the
        frontier (or all of them on the final pass) become ``final`` events with
        stable, monotonic ids; the remaining tail is one ``partial``. Finalized
        audio is then dropped from the front of the window (the slide).
        ``stable_until`` is always 0 (the model may still rewrite the tail).

        Args:
            segments: Standard ASR ``Segment`` objects from this decode (window-
                relative timestamps).
            cursor: The absolute audio time processed so far (seconds).
            final_pass: Whether to finalize all remaining segments.

        Returns:
            The ordered events for this pass.
        """
        origin = self._origin_s  # decode-time origin; the segment times are relative to it
        window_len = self._window_seconds()
        settle_before = window_len - self._settle_margin_s  # window-relative frontier

        # Leading run of segments settled enough to finalize (all, on the final pass).
        if final_pass:
            settled = len(segments)
        else:
            settled = 0
            for seg in segments:
                if seg.end <= settle_before:
                    settled += 1
                else:
                    break
            # Sliding-window hard cap: if trimming to the last settled segment would
            # still leave a window longer than ``max_window_s``, force-finalize more
            # leading segments so the next decode (and memory) stays bounded under
            # long, sparsely-segmented speech.
            if self._max_window_s is not None:
                while (
                    settled < len(segments)
                    and ((window_len - segments[settled - 1].end) if settled else window_len)
                    > self._max_window_s
                ):
                    settled += 1

        events: list[TranscriptionEvent] = []
        for j in range(settled):
            seg = segments[j]
            events.append(
                TranscriptionEvent.final(
                    segment_id=f"seg-{self._finalized_count + j}",
                    text=seg.text,
                    stable_until=0,
                    start=origin + seg.start,
                    end=origin + seg.end,
                    words=self._shift_words(seg.words, origin),
                    audio_processed_until=cursor,
                )
            )
        self._finalized_count += settled

        if not final_pass and settled < len(segments):
            tail = segments[settled:]
            tail_text = "".join(s.text for s in tail)
            tail_words: list[Any] | None = None
            for s in tail:
                if s.words:
                    tail_words = (tail_words or []) + list(s.words)
            events.append(
                TranscriptionEvent.partial(
                    segment_id=f"seg-{self._finalized_count}",
                    text=tail_text,
                    stable_until=0,
                    start=origin + tail[0].start,
                    end=origin + tail[-1].end,
                    words=self._shift_words(tail_words, origin),
                    audio_processed_until=cursor,
                )
            )
        elif not events:
            events.append(TranscriptionEvent.progress(audio_processed_until=cursor))

        # Slide the window: drop the audio of the segments we just finalized so the
        # next decode runs over a bounded tail (origin advances to keep times
        # absolute). Done last, so the events above used the pre-trim origin.
        if settled and not final_pass:
            self._trim_window(segments[settled - 1].end)
        return events

    def _trim_window(self, cut_seconds: float) -> None:
        """Drop ``cut_seconds`` of audio off the front of the window.

        Advances :attr:`_origin_s` by exactly the duration actually dropped (from
        the sample count, not the requested seconds), so the origin and the window
        always sum to the same absolute cursor.

        Args:
            cut_seconds: Window-relative time to drop (the end of the last
                finalized segment). Clamped to the available window.
        """
        if cut_seconds <= 0.0 or self._window.size == 0:
            return
        cut_samples = max(0, min(round(cut_seconds * _SAMPLE_RATE), self._window.size))
        if cut_samples == 0:
            return
        self._window = self._window[cut_samples:]
        self._origin_s += cut_samples / _SAMPLE_RATE

    @staticmethod
    def _shift_words(words: list[Any] | None, offset: float) -> list[Any] | None:
        """Return ``words`` with start/end shifted by ``offset`` seconds (absolute).

        Returns the input unchanged when there is nothing to shift (no words, or a
        zero origin), so the non-sliding path stays allocation-free and identical.

        Args:
            words: The window-relative :class:`~standard_asr.results.Word` list, or
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
