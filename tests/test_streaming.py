# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Streaming-path tests for the windowed MLX session.

Runs entirely against the injected fake loader: no weights are loaded, though
``mlx_audio`` and MLX are imported (see ``conftest.py``). The fake's
``output_fn`` returns segments based on how much audio has accumulated, so we can
simulate the window growing across re-decodes and assert the partial -> final
progression plus the spec event-sequence contract. Covers all three families
since the session dispatches to the bound backend.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from standard_asr import TranscriptionEvent
from standard_asr.audio.format import AudioFormat
from standard_asr.compliance import check_event_sequence
from standard_asr.contract.capabilities import FinalityCap, ReconnectCap, StreamTimestampsCap

from std_mlx_audio import ParakeetTdt06BV3, Qwen3Asr06B, WhisperTiny
from std_mlx_audio.engine import MlxAudioASR

from .conftest import (
    FakeAlignedResult,
    FakeAlignedSentence,
    FakeAlignedToken,
    FakeLoader,
    FakeSTTOutput,
    VirtualTimeLoop,
    on_virtual_clock,
    run_virtual,
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
        assert caps.supports("streaming.partial_stability") is False
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
def _drive_live(
    engine: MlxAudioASR, chunks: list[bytes], *, chunk_s: float = 1.0, params: Any = None
) -> tuple[list[TranscriptionEvent], Any]:
    """Send ``chunks`` live (one per ``chunk_s``, on virtual time); collect events.

    Returns:
        The events and the reduced result.
    """

    async def main(loop: VirtualTimeLoop) -> tuple[list[TranscriptionEvent], Any]:
        events: list[TranscriptionEvent] = []
        session = on_virtual_clock(
            engine.start_transcription(audio_format=_FMT, params=params), loop
        )
        async with session:

            async def feeder() -> None:
                for i, chunk in enumerate(chunks):
                    await asyncio.sleep(max(0.0, (i + 1) * chunk_s - loop.time()))
                    await session.send_audio(chunk)
                await session.end_audio()

            task = asyncio.ensure_future(feeder())
            async for event in session:
                events.append(event)
            await task
        return events, session.result()

    return run_virtual(main)


def _drive(engine: MlxAudioASR) -> tuple[list[TranscriptionEvent], Any]:
    """Send ~12s of audio live in 1s chunks; collect events + the reduced result."""
    return _drive_live(engine, [silent_pcm(1.0)] * 12)


def test_qwen_streaming_emits_partials_then_finals(
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
    events, result = _drive(Qwen3Asr06B())

    types = [e.type for e in events]
    assert "partial" in types
    assert "final" in types
    assert types[-1] == "done"
    # Recorded stream obeys the segment/event-order contract and does not exceed
    # what the engine declares (progress events included: they carry the cursor).
    report = check_event_sequence(events, capabilities=Qwen3Asr06B.declared_capabilities)
    assert report.passed, [i.message for i in report.issues]
    assert "First chunk." in result.text


# --------------------------------------------------------------------------- #
# Live windowed run — Parakeet (AlignedResult; different return type)
# --------------------------------------------------------------------------- #
def test_parakeet_streaming_dispatches_to_backend(
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
    events, result = _drive(ParakeetTdt06BV3())
    assert events[-1].type == "done"
    assert any(e.type == "final" for e in events)
    assert "First sentence." in result.text
    report = check_event_sequence(events, capabilities=ParakeetTdt06BV3.declared_capabilities)
    assert report.passed, [i.message for i in report.issues]


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_streaming_silence_only_emits_progress_and_done(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # Empty transcript on every decode => only progress + done, no partial/final.
    fake_loader(output=FakeSTTOutput(text="", segments=[]))
    events, _ = _drive(WhisperTiny())
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


def test_streaming_tail_carries_words_when_requested(
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
    events, _ = _drive_live(
        WhisperTiny(),
        [silent_pcm(1.0)] * 7,
        params=RuntimeParams(language="en", word_timestamps="word"),
    )
    partials = [e for e in events if e.type == "partial"]
    assert partials, "expected at least one partial"
    assert any(e.words for e in partials), "partial tail should carry words"
    report = check_event_sequence(events, capabilities=WhisperTiny.declared_capabilities)
    assert report.passed, [i.message for i in report.issues]


async def test_streaming_whole_input_path(fake_loader: Callable[..., FakeLoader]) -> None:
    # OpenAI-style: submit a whole waveform, stream the result.
    import numpy as np
    from standard_asr import RuntimeParams
    from standard_asr.audio.input import AudioArray

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


# --------------------------------------------------------------------------- #
# Sliding window: knobs wire through; the window stays bounded; finalized
# timestamps stay absolute (not reset on each trim).
# --------------------------------------------------------------------------- #
def test_streaming_config_knobs_default_and_override() -> None:
    from std_mlx_audio._config import MlxAudioConfig

    cfg = MlxAudioConfig(engine="mlx-audio")
    # Defaults are the snappier (lower-latency) values.
    assert (cfg.redecode_interval_s, cfg.settle_margin_s, cfg.max_window_s) == (1.5, 2.0, 30.0)
    assert cfg.commit_pause_s is None

    # --set overrides (constructor kwargs) reach the constructed session.
    session = WhisperTiny(
        redecode_interval_s=0.7, settle_margin_s=1.0, max_window_s=4.0, commit_pause_s=0.6
    ).start_transcription(audio_format=_FMT)
    assert session._redecode_interval_s == 0.7  # type: ignore[attr-defined]
    assert session._settle_margin_s == 1.0  # type: ignore[attr-defined]
    assert session._cap == 4 * 16000  # type: ignore[attr-defined]  # the cap in samples
    assert session._inbox_limit == 2 * 4 * 16000  # type: ignore[attr-defined]  # in bytes
    assert session._commit_pause_s == 0.6  # type: ignore[attr-defined]


def test_streaming_knobs_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # The application tunes streaming through these variables; they must keep
    # their names.
    monkeypatch.setenv("STANDARD_ASR_MLX_AUDIO__REDECODE_INTERVAL_S", "0.3")
    monkeypatch.setenv("STANDARD_ASR_MLX_AUDIO__SETTLE_MARGIN_S", "0.4")
    monkeypatch.setenv("STANDARD_ASR_MLX_AUDIO__COMMIT_PAUSE_S", "0.8")
    session = Qwen3Asr06B().start_transcription(audio_format=_FMT)
    assert session._redecode_interval_s == 0.3  # type: ignore[attr-defined]
    assert session._settle_margin_s == 0.4  # type: ignore[attr-defined]
    assert session._commit_pause_s == 0.8  # type: ignore[attr-defined]


def test_streaming_sliding_window_bounds_and_keeps_absolute_time(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    """A decoder that tiles its (relative-timed) window into 1s segments.

    With a sliding window the session must (a) keep the decode input BOUNDED by
    ``max_window_s`` (the old whole-buffer strategy would re-decode the entire
    ~16s feed), and (b) re-base each decode's window-relative timestamps to
    absolute session time, so finalized ids are contiguous/monotonic and their
    ends climb far past the window size.
    """

    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        secs = _window_seconds(audio)
        n = max(1, int(secs))
        segs = [{"text": f"w{i} ", "start": float(i), "end": float(i + 1)} for i in range(n)]
        segs[-1]["end"] = max(segs[-1]["end"], secs)  # tail runs to the frontier
        return FakeSTTOutput(text="".join(s["text"] for s in segs), segments=segs)

    loader = fake_loader(output_fn=output_fn)
    engine = WhisperTiny(redecode_interval_s=1.0, settle_margin_s=1.0, max_window_s=4.0)
    events, _ = _drive_live(engine, [silent_pcm(1.0)] * 16)

    finals = [e for e in events if e.type == "final"]
    partials = [e for e in events if e.type == "partial"]

    # (a) Window BOUNDED: no decode ever sees more than ~max_window + one interval
    #     of audio (vs ~16s for the old grow-forever buffer).
    max_audio_s = max(len(c["audio"]) for c in loader.model.generate_calls) / 16000.0
    assert max_audio_s <= 6.0, f"window not bounded: {max_audio_s:.1f}s"

    # (b) Absolute, monotonic timestamps + contiguous ids (proves the origin re-base
    #     across trims; without it, ends would never exceed the ~4s window).
    assert finals, "expected finals"
    ids = [int(e.segment_id.split("-")[1]) for e in finals]
    assert ids == list(range(len(ids))), ids
    ends = [e.end for e in finals]
    assert ends == sorted(ends)
    assert max(ends) > 6.0, f"timestamps look window-relative, not absolute: {max(ends)}"

    # The audio cursor stays monotonic and reaches ~the fed duration.
    cursors = [e.audio_processed_until for e in events if e.audio_processed_until is not None]
    assert cursors == sorted(cursors)
    assert cursors[-1] >= 15.0

    # Not 5s-batched: many partials over the 16s feed.
    assert len(partials) >= 8
    assert events[-1].type == "done"


# --------------------------------------------------------------------------- #
# Optional segment timestamps (the core schema allows a segment without times)
# --------------------------------------------------------------------------- #
def test_build_events_segment_without_end_never_settles(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # A segment with no end time cannot be proven settled: it stays in the
    # partial tail (with honest None times) instead of being finalized on a
    # timestamp the decode never produced.
    import numpy as np
    from standard_asr import RuntimeParams
    from standard_asr.contract.results import Segment

    from std_mlx_audio._streaming import MlxAudioStreamingSession, _clamp

    fake_loader(output=FakeSTTOutput(text="x"))
    session = MlxAudioStreamingSession(
        WhisperTiny(), RuntimeParams(), settle_margin_s=0.0, max_window_s=None
    )
    session._window = np.zeros(16000 * 10, dtype=np.float32)
    segments = _clamp([Segment(text="no end yet", start=None, end=None)], 16000 * 10)
    events = session._build_events(segments)
    (partial,) = events
    assert partial.type == "partial"
    assert partial.start is None and partial.end is None
    assert session._finalized_count == 0
    (final,) = session._finalize_all(segments, 16000 * 10)
    assert final.type == "final"
    assert final.start is None and final.end is None


async def test_cap_boundary_stops_at_segment_without_end(
    fake_loader: Callable[..., FakeLoader],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unknown end cannot anchor a cut. The cap must fall back to decoding
    # a bounded head on its own, without committing text past the unknown end.
    import numpy as np
    from standard_asr import RuntimeParams
    from standard_asr.contract.results import Segment

    from std_mlx_audio._streaming import MlxAudioStreamingSession

    fake_loader(output=FakeSTTOutput(text="x"))
    session = MlxAudioStreamingSession(
        WhisperTiny(), RuntimeParams(), settle_margin_s=100.0, max_window_s=10.0
    )
    session._window = np.zeros(16000 * 10, dtype=np.float32)
    segments = [Segment(text="a", start=None, end=None), Segment(text="b", start=0.0, end=9.5)]
    monkeypatch.setattr(session, "_decode", lambda *_args, **_kwargs: segments)
    assert await session._commit_settled_prefix(16000 * 10, want_words=False) is None
    assert session._finalized_count == 0
    assert session._window.size == 16000 * 10
    (partial,) = session._build_events(segments)
    assert partial.start is None and partial.end is None


async def test_cap_commits_at_the_last_inner_boundary(
    fake_loader: Callable[..., FakeLoader],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no settled segment, the cap commits through the last inner boundary.
    import numpy as np
    from standard_asr import RuntimeParams
    from standard_asr.contract.results import Segment

    from std_mlx_audio._streaming import MlxAudioStreamingSession

    fake_loader(output=FakeSTTOutput(text="x"))
    session = MlxAudioStreamingSession(
        WhisperTiny(), RuntimeParams(), settle_margin_s=100.0, max_window_s=10.0
    )
    session._window = np.zeros(16000 * 10, dtype=np.float32)
    segments = [
        Segment(text="a", start=0.0, end=7.5),
        Segment(text="b", start=7.5, end=9.0),
        Segment(text="c", start=9.0, end=10.0),
    ]
    monkeypatch.setattr(session, "_decode", lambda *_args, **_kwargs: segments)
    events = await session._commit_settled_prefix(16000 * 10, want_words=False)
    assert events is not None
    # Trimming to 9.0 leaves the last second for the next decode.
    finals = [e for e in events if e.type == "final"]
    assert [e.text for e in finals] == ["a", "b"]
    assert session._finalized_count == 2
    assert session._window.size == 16000  # 1.0 s left after the trim
