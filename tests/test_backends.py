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
    AlignedResultBackend,
    GenericSttBackend,
    Qwen3AsrBackend,
    SttFamilySpec,
    WhisperBackend,
    _opt_float,
    _opt_unit,
    adapt_audio_source,
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
# Aligned-output backend (different return type: AlignedResult) — Parakeet/Nemotron
# --------------------------------------------------------------------------- #
class TestAlignedResultBackend:
    backend = AlignedResultBackend(model_types=("parakeet",))

    def test_model_types_bound_per_instance(self) -> None:
        # One class, two families: the instance carries the family it adapts.
        assert AlignedResultBackend(model_types=("parakeet",)).model_types == ("parakeet",)
        assert AlignedResultBackend(model_types=("nemotron_asr",)).model_types == ("nemotron_asr",)
        assert self.backend.audio_as_list is False

    def test_generate_kwargs_is_empty(self) -> None:
        # Aligned-output models take no language / sampler args we drive.
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


# --------------------------------------------------------------------------- #
# Generic STTOutput backend (one data-driven adapter, many families)
# --------------------------------------------------------------------------- #
class TestGenericSttBackend:
    def test_model_types_and_audio_as_list_from_spec(self) -> None:
        backend = GenericSttBackend(SttFamilySpec(model_types=("voxtral",), audio_as_list=True))
        assert backend.model_types == ("voxtral",)
        assert backend.audio_as_list is True
        # Default spec: no list wrapping.
        assert GenericSttBackend(SttFamilySpec(model_types=("mms",))).audio_as_list is False

    def test_no_language_axis_passes_nothing(self) -> None:
        # A family with no language_kwarg (e.g. Moonshine) gets an empty surface.
        backend = GenericSttBackend(SttFamilySpec(model_types=("moonshine",)))
        kw = backend.generate_kwargs(
            resolved_language="en", want_words=False, params=_PARAMS, config=_CONFIG
        )
        assert kw == {}

    def test_iso_language_passed_through_as_subtag(self) -> None:
        # SenseVoice-style: language=<iso subtag> (region stripped).
        backend = GenericSttBackend(
            SttFamilySpec(model_types=("sensevoice",), language_kwarg="language")
        )
        kw = backend.generate_kwargs(
            resolved_language="zh-Hans-CN", want_words=False, params=_PARAMS, config=_CONFIG
        )
        assert kw == {"language": "zh"}

    def test_language_omitted_when_auto(self) -> None:
        backend = GenericSttBackend(
            SttFamilySpec(model_types=("sensevoice",), language_kwarg="language")
        )
        kw = backend.generate_kwargs(
            resolved_language=None, want_words=False, params=_PARAMS, config=_CONFIG
        )
        assert "language" not in kw

    def test_canary_source_and_default_target(self) -> None:
        # Canary: source_lang from the language axis; target defaults to source so a
        # non-English source TRANSCRIBES (not silently translates to English).
        backend = GenericSttBackend(
            SttFamilySpec(
                model_types=("canary",),
                language_kwarg="source_lang",
                translate_target_kwarg="target_lang",
            )
        )
        kw = backend.generate_kwargs(
            resolved_language="de-DE", want_words=False, params=_PARAMS, config=_CONFIG
        )
        assert kw == {"source_lang": "de", "target_lang": "de"}

    def test_canary_translation_target_override(self) -> None:
        backend = GenericSttBackend(
            SttFamilySpec(
                model_types=("canary",),
                language_kwarg="source_lang",
                translate_target_kwarg="target_lang",
            )
        )
        params = MlxAudioParams(target_language="en")
        kw = backend.generate_kwargs(
            resolved_language="de", want_words=False, params=params, config=_CONFIG
        )
        assert kw == {"source_lang": "de", "target_lang": "en"}

    def test_granite_translation_via_language_kwarg(self) -> None:
        # Granite: no spoken-language axis; target_language drives its `language` arg.
        backend = GenericSttBackend(
            SttFamilySpec(model_types=("granite_speech",), translate_target_kwarg="language")
        )
        no_target = backend.generate_kwargs(
            resolved_language=None, want_words=False, params=_PARAMS, config=_CONFIG
        )
        assert no_target == {}  # transcribe (no translation requested)
        with_target = backend.generate_kwargs(
            resolved_language=None,
            want_words=False,
            params=MlxAudioParams(target_language="fr-FR"),
            config=_CONFIG,
        )
        assert with_target == {"language": "fr"}

    def test_niche_knobs_forwarded_only_when_set(self) -> None:
        backend = GenericSttBackend(
            SttFamilySpec(
                model_types=("fun_asr_nano",),
                language_kwarg="language",
                forward=(("hotwords", "hotwords"), ("itn", "use_itn")),
            )
        )
        # Unset -> omitted (model defaults).
        assert (
            backend.generate_kwargs(
                resolved_language=None, want_words=False, params=_PARAMS, config=_CONFIG
            )
            == {}
        )
        # Set -> forwarded under the family's kwarg name (note itn<-use_itn rename).
        params = MlxAudioParams(hotwords=["MLX", "Qwen"], use_itn=False)
        kw = backend.generate_kwargs(
            resolved_language=None, want_words=False, params=params, config=_CONFIG
        )
        assert kw == {"hotwords": ["MLX", "Qwen"], "itn": False}

    def test_to_result_text_only_when_no_segment_timing(self) -> None:
        backend = GenericSttBackend(SttFamilySpec(model_types=("moonshine",)))
        native = FakeSTTOutput(
            text="hello.",
            segments=[{"text": "hello.", "start": 0.0, "end": 0.0}],  # placeholder timing
            generation_tokens=5,
        )
        result = backend.to_result(native, duration=1.0, want_words=False)
        assert result.text == "hello."
        assert result.segments is None  # honest: placeholder timing not surfaced
        assert result.words is None
        assert result.detected_language is None
        assert result.extra["generation_tokens"] == 5  # throughput stats still surfaced

    def test_to_result_segments_when_real_timing(self) -> None:
        backend = GenericSttBackend(
            SttFamilySpec(
                model_types=("cohere_asr",), language_kwarg="language", segment_timing=True
            )
        )
        native = FakeSTTOutput(
            text="a b",
            segments=[
                {"text": "a", "start": 0.0, "end": 0.5},
                {"text": " b", "start": 0.5, "end": 1.0},
            ],
        )
        result = backend.to_result(native, duration=1.0, want_words=False)
        assert result.segments is not None and len(result.segments) == 2
        assert result.segments[1].start == 0.5 and result.segments[1].end == 1.0
        assert result.segments[0].words is None  # never word timing on STTOutput families

    def test_to_result_detected_language_when_reported(self) -> None:
        backend = GenericSttBackend(
            SttFamilySpec(
                model_types=("sensevoice",),
                language_kwarg="language",
                reports_detected_language=True,
            )
        )
        native = FakeSTTOutput(text="你好", language="zh")
        result = backend.to_result(native, duration=1.0, want_words=False)
        assert result.detected_language == "zh"  # ISO reported -> BCP-47


def test_adapt_audio_source_wraps_only_for_list_families() -> None:
    list_backend = GenericSttBackend(SttFamilySpec(model_types=("voxtral",), audio_as_list=True))
    plain_backend = GenericSttBackend(SttFamilySpec(model_types=("mms",)))
    sentinel = object()
    assert adapt_audio_source(list_backend, sentinel) == [sentinel]
    assert adapt_audio_source(plain_backend, sentinel) is sentinel
