# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend-adapter tests — the "one engine, many models" normalization core.

Each backend maps a DIFFERENT native ``generate`` call-shape + return type onto
the single constant Standard ASR result schema. These tests pin that mapping
directly (no engine, no MLX): language translation in ``generate_kwargs`` and
output normalization in ``to_result`` for Qwen3-ASR, Whisper, and Parakeet.
"""

from __future__ import annotations

import math

import pytest
from standard_asr.runtime_params import WordTimestampGranularity

from std_mlx_audio import MlxAudioConfig, MlxAudioParams
from std_mlx_audio.backends import (  # pyright: ignore[reportPrivateUsage]
    ParakeetBackend,
    Qwen3AsrBackend,
    WhisperBackend,
    _opt_float,
    _opt_unit,
    map_word_timestamps,
    waveform_duration,
)

from .conftest import (
    FakeAlignedResult,
    FakeAlignedSentence,
    FakeAlignedToken,
    FakeSTTOutput,
    float_array,
)

_CONFIG = MlxAudioConfig()
_PARAMS = MlxAudioParams()


# --------------------------------------------------------------------------- #
# Qwen3-ASR backend
# --------------------------------------------------------------------------- #
class TestQwen3AsrBackend:
    backend = Qwen3AsrBackend()

    def test_generate_kwargs_maps_language_to_english_name(self) -> None:
        kw = self.backend.generate_kwargs(
            resolved_language="zh-CN", want_words=False, params=_PARAMS, config=_CONFIG
        )
        assert kw["language"] == "Chinese"  # BCP-47 -> Qwen English name
        # Sampler knobs forwarded.
        assert kw["temperature"] == 0.0
        assert kw["max_tokens"] == 8192

    def test_generate_kwargs_omits_language_when_auto(self) -> None:
        kw = self.backend.generate_kwargs(
            resolved_language=None, want_words=False, params=_PARAMS, config=_CONFIG
        )
        assert "language" not in kw  # None => auto-detect upstream

    def test_generate_kwargs_drops_unsupported_language(self) -> None:
        # A tag Qwen does not support falls back to auto rather than a bad name.
        kw = self.backend.generate_kwargs(
            resolved_language="sw", want_words=False, params=_PARAMS, config=_CONFIG
        )
        assert "language" not in kw

    def test_generate_kwargs_forwards_system_prompt(self) -> None:
        params = MlxAudioParams(system_prompt="medical terms")
        kw = self.backend.generate_kwargs(
            resolved_language=None, want_words=False, params=params, config=_CONFIG
        )
        assert kw["system_prompt"] == "medical terms"

    def test_to_result_normalizes_sttoutput(self) -> None:
        native = FakeSTTOutput(
            text="Hello world.",
            segments=[{"text": "Hello world.", "language": "English", "start": 0.0, "end": 2.5}],
            language=["English"],
            generation_tokens=12,
            generation_tps=99.5,
        )
        result = self.backend.to_result(native, duration=2.5, want_words=False)
        assert result.text == "Hello world."
        assert result.detected_language == "en"  # list[name] -> dominant -> BCP-47
        assert result.duration == 2.5
        assert result.segments is not None and len(result.segments) == 1
        assert result.segments[0].start == 0.0 and result.segments[0].end == 2.5
        # Qwen3-ASR never emits word timing.
        assert result.words is None
        assert result.segments[0].words is None
        # Token accounting surfaces in extra (engine-specific channel).
        assert result.extra["generation_tokens"] == 12
        assert result.extra["generation_tps"] == 99.5

    def test_to_result_handles_empty(self) -> None:
        result = self.backend.to_result(FakeSTTOutput(text=""), duration=None, want_words=False)
        assert result.text == ""
        assert result.segments is None
        assert result.detected_language is None


# --------------------------------------------------------------------------- #
# Whisper backend
# --------------------------------------------------------------------------- #
class TestWhisperBackend:
    backend = WhisperBackend()

    def test_generate_kwargs_maps_language_to_iso_code(self) -> None:
        kw = self.backend.generate_kwargs(
            resolved_language="ja", want_words=True, params=_PARAMS, config=_CONFIG
        )
        assert kw["language"] == "ja"
        assert kw["word_timestamps"] is True
        assert kw["return_timestamps"] is True

    def test_generate_kwargs_omits_language_when_auto(self) -> None:
        kw = self.backend.generate_kwargs(
            resolved_language=None, want_words=False, params=_PARAMS, config=_CONFIG
        )
        assert "language" not in kw
        assert kw["word_timestamps"] is False

    def test_to_result_with_words(self) -> None:
        native = FakeSTTOutput(
            text="Hi there.",
            language="en",
            segments=[
                {
                    "text": "Hi there.",
                    "start": 0.0,
                    "end": 1.0,
                    "avg_logprob": -0.2,
                    "no_speech_prob": 0.01,
                    "words": [
                        {"word": "Hi", "start": 0.0, "end": 0.4, "probability": 0.99},
                        {"word": "there.", "start": 0.4, "end": 1.0, "probability": 0.95},
                    ],
                }
            ],
        )
        result = self.backend.to_result(native, duration=1.0, want_words=True)
        assert result.detected_language == "en"
        assert result.words is not None and len(result.words) == 2
        assert result.words[0].text == "Hi"
        assert result.words[0].probability == 0.99
        assert result.segments is not None
        assert result.segments[0].avg_logprob == -0.2
        assert result.segments[0].words is not None and len(result.segments[0].words) == 2

    def test_to_result_without_words_omits_word_data(self) -> None:
        # SEGMENT request (want_words=False) must NOT back-fill word data (TR.3).
        native = FakeSTTOutput(
            text="Hi.",
            language="en",
            segments=[
                {
                    "text": "Hi.",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [{"word": "Hi.", "start": 0.0, "end": 1.0, "probability": 0.9}],
                }
            ],
        )
        result = self.backend.to_result(native, duration=1.0, want_words=False)
        assert result.words is None
        assert result.segments is not None and result.segments[0].words is None


# --------------------------------------------------------------------------- #
# Parakeet backend (different return type: AlignedResult)
# --------------------------------------------------------------------------- #
class TestParakeetBackend:
    backend = ParakeetBackend()

    def test_generate_kwargs_is_empty(self) -> None:
        # Parakeet takes no language / sampler args.
        kw = self.backend.generate_kwargs(
            resolved_language="en", want_words=True, params=_PARAMS, config=_CONFIG
        )
        assert kw == {}

    def test_to_result_maps_sentences_and_tokens(self) -> None:
        native = FakeAlignedResult(
            text="Hello world.",
            sentences=[
                FakeAlignedSentence(
                    text="Hello world.",
                    start=0.0,
                    end=1.2,
                    tokens=[
                        FakeAlignedToken(text="Hello", start=0.0, end=0.6),
                        FakeAlignedToken(text=" world.", start=0.6, end=1.2),
                    ],
                )
            ],
        )
        result = self.backend.to_result(native, duration=1.2, want_words=True)
        assert result.text == "Hello world."
        # Parakeet is fixed-language: never reports a detected language.
        assert result.detected_language is None
        assert result.segments is not None and len(result.segments) == 1
        assert result.words is not None and len(result.words) == 2
        assert result.words[0].text == "Hello"
        # TDT path has no per-token probability.
        assert result.words[0].probability is None

    def test_to_result_without_words(self) -> None:
        native = FakeAlignedResult(
            text="Hi.",
            sentences=[
                FakeAlignedSentence(
                    text="Hi.",
                    start=0.0,
                    end=0.5,
                    tokens=[FakeAlignedToken(text="Hi.", start=0.0, end=0.5)],
                )
            ],
        )
        result = self.backend.to_result(native, duration=0.5, want_words=False)
        assert result.words is None
        assert result.segments is not None and result.segments[0].words is None


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def test_clamp_span_repairs_inverted_and_negative_timings() -> None:
    # A negative native span must not blow up Segment construction.
    native = FakeSTTOutput(
        text="x",
        language="en",
        segments=[{"text": "x", "start": -0.5, "end": -1.0}],
    )
    result = WhisperBackend().to_result(native, duration=1.0, want_words=False)
    assert result.segments is not None
    seg = result.segments[0]
    assert seg.start >= 0.0 and seg.end >= seg.start


def test_clamp_span_repairs_positive_inverted_span() -> None:
    # end < start with both positive: end is clamped up to start (the e<s branch).
    native = FakeSTTOutput(
        text="x", language="en", segments=[{"text": "x", "start": 5.0, "end": 3.0}]
    )
    result = WhisperBackend().to_result(native, duration=6.0, want_words=False)
    assert result.segments is not None
    seg = result.segments[0]
    assert seg.start == 5.0 and seg.end == 5.0


def test_waveform_duration() -> None:
    assert waveform_duration(float_array(2.0)) == pytest.approx(2.0)
    assert waveform_duration(float_array(0.0)) == 0.0


def test_map_word_timestamps() -> None:
    assert map_word_timestamps(WordTimestampGranularity.WORD) is True
    assert map_word_timestamps(WordTimestampGranularity.SEGMENT) is False
    assert map_word_timestamps(None) is False


def test_opt_float_filters_garbage_and_nonfinite() -> None:
    assert _opt_float(None) is None
    assert _opt_float("not a number") is None
    assert _opt_float(math.nan) is None
    assert _opt_float(math.inf) is None
    assert _opt_float(1.5) == 1.5


def test_opt_unit_clamps_to_zero_one() -> None:
    assert _opt_unit(1.5) == 1.0
    assert _opt_unit(-0.2) == 0.0
    assert _opt_unit(0.5) == 0.5
    assert _opt_unit(None) is None
