# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Streaming-path tests for the windowed MLX session.

Runs entirely against the injected fake loader (no weights, no MLX). The fake's
``output_fn`` returns segments based on how much audio has accumulated, so we can
simulate the window growing across re-decodes and assert the partial -> final
progression plus the spec event-sequence contract. Covers all three families
since the session dispatches to the bound backend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from standard_asr import TranscriptionEvent
from standard_asr.audio_format import AudioFormat
from standard_asr.capabilities import FinalityCap, ReconnectCap, StreamTimestampsCap
from standard_asr.compliance import check_event_sequence

from std_mlx_audio import ParakeetTdt06BV3, Qwen3Asr06B, WhisperTiny
from std_mlx_audio.engine import MlxAudioASR

from .conftest import (
    FakeAlignedResult,
    FakeAlignedSentence,
    FakeAlignedToken,
    FakeLoader,
    FakeSTTOutput,
    silent_pcm,
)

_FMT = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)


def _window_seconds(audio: Any) -> float:
    return len(audio) / 16000.0


# --------------------------------------------------------------------------- #
# Capability honesty (fail-closed) — per family
# --------------------------------------------------------------------------- #
def test_streaming_capabilities_are_conservative() -> None:
    for cls in (Qwen3Asr06B, WhisperTiny, ParakeetTdt06BV3):
        caps = cls.declared_capabilities
        assert caps.supports("streaming_input") is True
        assert caps.supports("streaming_output") is True
        assert caps.supports("streaming.emits_partials") is True
        # Windowed re-decode => no stability, no supersede, no reconnect.
        assert caps.supports("streaming.word_stability") is False
        assert caps.supports("streaming.re_segments") is False
        assert isinstance(caps.node_at("streaming.reconnect"), ReconnectCap)
        assert caps.node_at("streaming.reconnect").mode == "unsupported"  # type: ignore[union-attr]
        assert isinstance(caps.node_at("streaming.finality_level"), FinalityCap)
        assert caps.node_at("streaming.finality_level").mode == "final"  # type: ignore[union-attr]
        assert isinstance(caps.node_at("streaming.timestamps"), StreamTimestampsCap)
        assert caps.node_at("streaming.timestamps").mode == "post_align"  # type: ignore[union-attr]


def test_parakeet_streaming_no_language_override() -> None:
    # Fixed-language model: per-request override is fail-closed in streaming too.
    assert (
        ParakeetTdt06BV3.declared_capabilities.supports("streaming.language.runtime_override")
        is False
    )


# --------------------------------------------------------------------------- #
# Live windowed run — Qwen3-ASR (STTOutput chunk segments)
# --------------------------------------------------------------------------- #
async def _drive(engine: MlxAudioASR) -> tuple[list[TranscriptionEvent], Any]:
    """Feed ~12s of audio in 1s chunks; collect events + the reduced result."""
    events: list[TranscriptionEvent] = []
    chunk = silent_pcm(1.0)
    async with engine.start_transcription(audio_format=_FMT) as session:
        session.feed([chunk] * 12)
        async for event in session:
            events.append(event)
    return events, session.result()


async def test_qwen_streaming_emits_partials_then_finals(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        secs = _window_seconds(audio)
        segs = [{"text": "First chunk. ", "language": "English", "start": 0.0, "end": 2.0}]
        if secs >= 8:
            segs.append({"text": "Second chunk. ", "language": "English", "start": 2.0, "end": 5.0})
            segs.append({"text": "trailing tail", "language": "English", "start": 5.0, "end": secs})
        return FakeSTTOutput(text="".join(s["text"] for s in segs), segments=segs)

    fake_loader(output_fn=output_fn)
    events, result = await _drive(Qwen3Asr06B())

    types = [e.type for e in events]
    assert "partial" in types
    assert "final" in types
    assert types[-1] == "done"
    # Recorded stream obeys the segment/event-order contract.
    report = check_event_sequence([e for e in events if e.type != "progress"])
    assert report.passed, [i.message for i in report.issues]
    assert "First chunk." in result.text


# --------------------------------------------------------------------------- #
# Live windowed run — Parakeet (AlignedResult; different return type)
# --------------------------------------------------------------------------- #
async def test_parakeet_streaming_dispatches_to_backend(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeAlignedResult:
        secs = _window_seconds(audio)
        sents = [
            FakeAlignedSentence(
                text="First sentence. ",
                start=0.0,
                end=2.0,
                tokens=[FakeAlignedToken(text="First sentence. ", start=0.0, end=2.0)],
            )
        ]
        if secs >= 8:
            sents.append(
                FakeAlignedSentence(
                    text="tail",
                    start=5.0,
                    end=secs,
                    tokens=[FakeAlignedToken(text="tail", start=5.0, end=secs)],
                )
            )
        return FakeAlignedResult(text="".join(s.text for s in sents), sentences=sents)

    fake_loader(output_fn=output_fn)
    events, result = await _drive(ParakeetTdt06BV3())
    assert events[-1].type == "done"
    assert any(e.type == "final" for e in events)
    assert "First sentence." in result.text


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
async def test_streaming_silence_only_emits_progress_and_done(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # Empty transcript on every decode => only progress + done, no partial/final.
    fake_loader(output=FakeSTTOutput(text="", segments=[]))
    events, _ = await _drive(WhisperTiny())
    assert events[-1].type == "done"
    assert all(e.type in ("progress", "done") for e in events)


async def test_streaming_decode_error_surfaces_error_event(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="x"))
    loader.model.raise_on_generate = RuntimeError("decode failed")
    events: list[TranscriptionEvent] = []
    async with WhisperTiny().start_transcription(audio_format=_FMT) as session:
        session.feed([silent_pcm(6.0)])
        async for event in session:
            events.append(event)
    assert any(e.type == "error" for e in events)


async def test_streaming_tail_carries_words_when_requested(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # word_timestamps=word => the moving partial tail must carry word data too.
    from standard_asr import RuntimeParams

    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        secs = _window_seconds(audio)
        return FakeSTTOutput(
            text="hello world",
            language="en",
            segments=[
                {
                    "text": "hello world",
                    "start": 0.0,
                    "end": secs,
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.9},
                        {"word": "world", "start": 0.5, "end": 1.0, "probability": 0.9},
                    ],
                }
            ],
        )

    fake_loader(output_fn=output_fn)
    events: list[TranscriptionEvent] = []
    async with WhisperTiny().start_transcription(
        audio_format=_FMT, params=RuntimeParams(language="en", word_timestamps="word")
    ) as session:
        session.feed([silent_pcm(1.0)] * 7)
        async for event in session:
            events.append(event)
    partials = [e for e in events if e.type == "partial"]
    assert partials, "expected at least one partial"
    assert any(e.words for e in partials), "partial tail should carry words"


async def test_streaming_whole_input_path(fake_loader: Callable[..., FakeLoader]) -> None:
    # OpenAI-style: submit a whole waveform, stream the result.
    import numpy as np
    from standard_asr import RuntimeParams
    from standard_asr.audio_input import AudioArray

    fake_loader(
        output=FakeSTTOutput(
            text="whole input.",
            language=["English"],
            segments=[{"text": "whole input.", "language": "English", "start": 0.0, "end": 3.0}],
        )
    )
    audio = AudioArray(np.zeros(16000 * 3, dtype=np.float32), 16000)
    events: list[TranscriptionEvent] = []
    async with Qwen3Asr06B().start_transcription(
        audio=audio, params=RuntimeParams(language="en")
    ) as session:
        async for event in session:
            events.append(event)
    assert events[-1].type == "done"
    assert "whole input." in session.result().text
