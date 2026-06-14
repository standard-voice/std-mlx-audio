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
   accumulate them into one growing float32 buffer (the "window").
2. Whenever at least ``redecode_interval_s`` of *new* audio has arrived, re-run
   the **whole buffer** through the engine's bound backend (inline on the
   event-loop thread — MLX arrays are thread-bound, so the decode cannot be
   offloaded to a worker thread; see :meth:`MlxAudioStreamingSession._decode`)
   and emit:
   * a ``final`` for every segment that ends comfortably before the decode
     frontier (``settle_margin_s`` behind the last decoded timestamp) and has not
     been finalized yet;
   * one ``partial`` carrying the tail (everything after the last finalized
     segment) as the single in-progress segment.
3. On ``end_audio`` do a last full decode and finalize the tail, then ``done``.

Honesty
-------
The model re-decodes the entire window each pass and may rewrite ANY earlier
text, so ``stable_until=0`` on every ``partial`` (``word_stability=false``) and a
segment is finalized only once it is several seconds behind the frontier. We
never emit ``supersede`` (``re_segments=false``). Segment ids are synthesized
deterministically (``seg-0`` ...). This is a *pragmatic* streaming adapter for
batch decoders, not a true incremental recognizer — the README and findings doc
say so plainly.
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
            stability under re-decode).
        **session_kwargs: Forwarded to :class:`TranscriptionSession` (deadlines,
            buffer sizes, ``strict_lifecycle``).
    """

    def __init__(
        self,
        engine: MlxAudioASR,
        gated_params: RuntimeParams,
        *,
        redecode_interval_s: float = 5.0,
        settle_margin_s: float = 4.0,
        **session_kwargs: Any,
    ) -> None:
        super().__init__(**session_kwargs)
        self._engine = engine
        self._params = gated_params
        self._redecode_interval_s = redecode_interval_s
        self._settle_margin_s = settle_margin_s
        # The base TranscriptionSession reserves several private attribute names
        # (`_buffer`, `_audio_queue`, ...); we name our audio window `_window`
        # to avoid clobbering them.
        self._window = np.zeros(0, dtype=np.float32)
        self._resolved_language = self._resolve_language()
        # Index up to which we have already emitted a `final`. Finalized ids are
        # immutable.
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
        native = model.generate(source, **gen_kwargs)
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
        yield TranscriptionEvent.done(audio_processed_until=self._window_seconds())

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
        """Return the duration of the accumulated window in seconds."""
        return self._window.size / _SAMPLE_RATE

    async def _redecode(self, *, want_words: bool, final_pass: bool) -> list[TranscriptionEvent]:
        """Re-decode the window and build the events for this pass.

        Args:
            want_words: Whether word-level timestamps were requested.
            final_pass: ``True`` on the post-``end_audio`` flush (finalize all
                remaining segments regardless of the settle margin).

        Returns:
            The ordered events to yield for this pass.
        """
        cursor = self._window_seconds()
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

        Segments that end at least ``settle_margin_s`` behind ``cursor`` (or all
        of them on the final pass) become ``final`` events with stable ids; the
        remaining tail is one ``partial``. ``stable_until`` is always 0 (the
        model may rewrite the window).

        Args:
            segments: Standard ASR ``Segment`` objects from this decode.
            cursor: The audio time processed so far (seconds).
            final_pass: Whether to finalize all remaining segments.

        Returns:
            The ordered events for this pass.
        """
        events: list[TranscriptionEvent] = []
        settle_before = cursor - self._settle_margin_s
        settled = (
            len(segments) if final_pass else sum(1 for s in segments if s.end <= settle_before)
        )
        for idx in range(self._finalized_count, settled):
            seg = segments[idx]
            events.append(
                TranscriptionEvent.final(
                    segment_id=f"seg-{idx}",
                    text=seg.text,
                    stable_until=0,
                    start=seg.start,
                    end=seg.end,
                    words=seg.words,
                    audio_processed_until=cursor,
                )
            )
        self._finalized_count = max(self._finalized_count, settled)

        if not final_pass and settled < len(segments):
            tail = segments[settled:]
            tail_text = "".join(s.text for s in tail)
            tail_words: list[Any] | None = None
            for s in tail:
                if s.words:
                    tail_words = (tail_words or []) + list(s.words)
            events.append(
                TranscriptionEvent.partial(
                    segment_id=f"seg-{settled}",
                    text=tail_text,
                    stable_until=0,
                    start=tail[0].start,
                    end=tail[-1].end,
                    words=tail_words,
                    audio_processed_until=cursor,
                )
            )
        elif not events:
            events.append(TranscriptionEvent.progress(audio_processed_until=cursor))
        return events


__all__ = ["MlxAudioStreamingSession"]
