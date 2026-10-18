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
from standard_asr.contract.params import WordTimestampGranularity

from std_mlx_audio import MlxAudioConfig, MlxAudioParams
from std_mlx_audio.backends import (  # pyright: ignore[reportPrivateUsage]
    TURN_SEPARATOR,
    AlignedResultBackend,
    ChunkJoin,
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
                model_types=("cohere_asr",),
                language_kwarg="language",
                segment_timing=True,
                chunk_join=ChunkJoin(),
            )
        )
        native = FakeSTTOutput(
            text="a b",
            segments=[
                {"text": "a", "start": 0.0, "end": 0.5},
                {"text": "b", "start": 0.5, "end": 1.0},
            ],
        )
        result = backend.to_result(native, duration=1.0, want_words=False)
        assert result.segments is not None and len(result.segments) == 2
        assert result.segments[1].start == 0.5 and result.segments[1].end == 1.0
        assert result.segments[0].words is None  # never word timing on STTOutput families
        assert [s.text for s in result.segments] == ["a", " b"]
        assert result.text == "a b"

    def test_a_timed_spec_without_a_chunk_join_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="chunk_join"):
            SttFamilySpec(model_types=("glmasr",), segment_timing=True)

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


def test_single_segment_span_and_the_boundary_rule() -> None:
    from std_mlx_audio import CohereAsr, FunAsrNano, GlmAsrNano, MlxAudioParams
    from std_mlx_audio.backends import has_inner_boundaries

    params = MlxAudioParams()
    qwen = Qwen3AsrBackend()
    assert qwen.single_segment_span(params) == 1200.0
    assert qwen.single_segment_span(MlxAudioParams(chunk_duration=4.0)) == 4.0
    # mlx-audio's splitter advances at least one second and pads to a second.
    assert qwen.single_segment_span(MlxAudioParams(chunk_duration=0.1)) == 1.0
    assert WhisperBackend().single_segment_span(params) is None
    assert AlignedResultBackend(model_types=("parakeet",)).single_segment_span(params) is None
    assert FunAsrNano.backend.single_segment_span(params) == 1200.0
    assert GlmAsrNano.backend.single_segment_span(params) == 30.0
    # Cohere decodes without VAD by default: one segment up to its 35 s clip.
    assert CohereAsr.backend.single_segment_span(params) == 35.0
    # Content boundaries: always. Fixed chunks: only when a chunk is shorter than
    # the window; with no cap, never.
    assert has_inner_boundaries(WhisperBackend(), params, None) is True
    assert has_inner_boundaries(qwen, params, 30.0) is False
    assert has_inner_boundaries(qwen, MlxAudioParams(chunk_duration=4.0), 30.0) is True
    assert has_inner_boundaries(qwen, MlxAudioParams(chunk_duration=4.0), None) is False
    assert has_inner_boundaries(GlmAsrNano.backend, params, 30.0) is False
    assert has_inner_boundaries(CohereAsr.backend, params, 30.0) is False
    assert has_inner_boundaries(CohereAsr.backend, params, 40.0) is True
    # The review's two subsecond cases: 0.5 s with chunk 0.1, 0.8 s with chunk 0.3.
    assert has_inner_boundaries(qwen, MlxAudioParams(chunk_duration=0.1), 0.5) is False
    assert has_inner_boundaries(qwen, MlxAudioParams(chunk_duration=0.3), 0.8) is False


# --------------------------------------------------------------------------- #
# Segment text and whitespace (spec TR.1): text is the segment texts
# concatenated, and every character is mlx-audio's, plus the separator
# mlx-audio itself puts between two chunks.
# --------------------------------------------------------------------------- #
def _texts(result: object) -> list[str]:
    segments = getattr(result, "segments", None)
    assert segments is not None
    return [segment.text for segment in segments]


def _chunks(*texts: str) -> list[dict[str, object]]:
    return [{"text": text, "start": float(i), "end": float(i + 1)} for i, text in enumerate(texts)]


class TestSegmentTextWhitespace:
    def test_qwen_space_between_chunks_moves_onto_the_next_segment(self) -> None:
        # mlx-audio: full text = " ".join(chunk texts); segments carry each chunk alone.
        native = FakeSTTOutput(
            text="Hello world. How are you?",
            segments=_chunks("Hello world.", "How are you?"),
            language=["English", "English"],
        )
        result = Qwen3AsrBackend().to_result(native, duration=2.0, want_words=False)
        assert _texts(result) == ["Hello world.", " How are you?"]
        assert result.text == native.text

    def test_qwen_keeps_its_space_after_an_empty_chunk(self) -> None:
        # " ".join does not skip an empty chunk, so mlx-audio's text has two spaces.
        native = FakeSTTOutput(text=" ".join(["a", "", "b"]), segments=_chunks("a", "", "b"))
        result = Qwen3AsrBackend().to_result(native, duration=3.0, want_words=False)
        assert result.text == "a  b" == native.text

    def test_qwen_chunks_of_unknown_language_keep_mlx_audios_space(self) -> None:
        # No source names a language: the chunk dicts have no "language" key,
        # STTOutput.language is a list (_chunk_languages reads it only as a
        # string), and the request resolved none. The fallback is the family's
        # own separator, mlx-audio's space, here between two Chinese chunks.
        native = FakeSTTOutput(
            text="今天天气 很好。", segments=_chunks("今天天气", "很好。"), language=["Chinese"] * 2
        )
        result = Qwen3AsrBackend().to_result(
            native, duration=2.0, want_words=False, resolved_language=None
        )
        assert result.text == "今天天气 很好。" == native.text

    def test_qwen_text_is_never_stripped(self) -> None:
        native = FakeSTTOutput(text=" hi ", segments=_chunks(" hi "))
        result = Qwen3AsrBackend().to_result(native, duration=1.0, want_words=False)
        assert result.text == " hi "

    def test_qwen_after_earlier_text_in_an_unknown_language_starts_with_its_space(
        self,
    ) -> None:
        native = FakeSTTOutput(text="", segments=_chunks("", "Next."))
        result = Qwen3AsrBackend().to_result(
            native, duration=2.0, want_words=False, follows_text=True
        )
        # One space before the first text, not one per chunk.
        assert _texts(result) == ["", " Next."]

    @pytest.mark.parametrize(
        ("chunk_language", "expected"),
        [
            ("English", " Next."),
            ("Chinese", "Next."),
            ("Cantonese", "Next."),
            ("Japanese", "Next."),
            (None, " Next."),
        ],
    )
    def test_qwen_seam_separator_follows_the_chunks_language(
        self, chunk_language: str | None, expected: str
    ) -> None:
        segments = [{"text": "Next.", "start": 0.0, "end": 1.0, "language": chunk_language}]
        native = FakeSTTOutput(text="Next.", segments=segments, language=[chunk_language])
        result = Qwen3AsrBackend().to_result(
            native, duration=1.0, want_words=False, follows_text=True
        )
        assert _texts(result) == [expected]

    def test_chinese_chunks_join_with_nothing_at_a_seam_and_inside_a_decode(
        self,
    ) -> None:
        # mlx-audio puts " " between two Chinese chunks; the plugin puts nothing
        # there, inside a decode and at a seam alike.
        segments = [
            {"text": "今天", "start": 0.0, "end": 1.0, "language": "Chinese"},
            {"text": "很好。", "start": 1.0, "end": 2.0, "language": "Chinese"},
        ]
        native = FakeSTTOutput(text="今天 很好。", segments=segments, language=["Chinese"] * 2)
        result = Qwen3AsrBackend().to_result(
            native, duration=2.0, want_words=False, follows_text=True
        )
        assert _texts(result) == ["今天", "很好。"]

    def test_unknown_language_seams_use_the_family_separator(self) -> None:
        from std_mlx_audio import FunAsrNano, GlmAsrNano

        # GLM-ASR reports no language; an auto-detecting Fun-ASR reports None.
        glm = FakeSTTOutput(text="好。", segments=_chunks("好。"))
        fun = FakeSTTOutput(
            text="好。",
            segments=[{"text": "好。", "start": 0.0, "end": 1.0, "language": None}],
        )
        for backend, native in ((GlmAsrNano.backend, glm), (FunAsrNano.backend, fun)):
            result = backend.to_result(native, duration=1.0, want_words=False, follows_text=True)
            assert _texts(result) == [" 好。"]

    def test_requested_language_decides_a_chunk_that_reports_none(self) -> None:
        from std_mlx_audio import GlmAsrNano

        # GLM-ASR reports no language, so the request's language decides the
        # separator between two chunks, in batch and at a seam.
        native = FakeSTTOutput(text="今天 很好。", segments=_chunks("今天", "很好。"))
        backend = GlmAsrNano.backend
        for language, expected in (("zh-Hant", ["今天", "很好。"]), (None, ["今天", " 很好。"])):
            result = backend.to_result(
                native, duration=2.0, want_words=False, resolved_language=language
            )
            assert _texts(result) == expected
        result = backend.to_result(
            FakeSTTOutput(text="好。", segments=_chunks("好。")),
            duration=1.0,
            want_words=False,
            follows_text=True,
            resolved_language="zh",
        )
        assert _texts(result) == ["好。"]

    def test_cohere_seam_uses_the_decode_language(self) -> None:
        from std_mlx_audio import CohereAsr

        for language, expected in (("ja", "はい。"), ("en", " Yes.")):
            native = FakeSTTOutput(
                text=expected.strip(), segments=_chunks(expected.strip()), language=language
            )
            result = CohereAsr.backend.to_result(
                native, duration=1.0, want_words=False, follows_text=True
            )
            assert _texts(result) == [expected]

    def test_qwen_without_text_has_no_seam_space(self) -> None:
        native = FakeSTTOutput(text="", segments=_chunks(""))
        result = Qwen3AsrBackend().to_result(
            native, duration=1.0, want_words=False, follows_text=True
        )
        assert result.text == ""

    def test_whisper_segments_carry_their_own_leading_space(self) -> None:
        # mlx-audio decodes each segment's tokens; English segments start with " ".
        native = FakeSTTOutput(
            text=" Hello world. How are you?",
            language="en",
            segments=_chunks(" Hello world.", " How are you?"),
        )
        for follows_text in (False, True):
            result = WhisperBackend().to_result(
                native, duration=2.0, want_words=False, follows_text=follows_text
            )
            assert _texts(result) == [" Hello world.", " How are you?"]
            assert result.text == native.text  # leading space kept

    def test_whisper_chinese_segments_join_with_nothing(self) -> None:
        native = FakeSTTOutput(
            text="你好。再见。", language="zh", segments=_chunks("你好。", "再见。")
        )
        result = WhisperBackend().to_result(native, duration=2.0, want_words=False)
        assert result.text == "你好。再见。"

    def test_aligned_text_is_the_sentences_without_mlx_audios_strip(self) -> None:
        # mlx-audio: AlignedResult.text = "".join(sentence texts).strip(); a sentence
        # starts with the space its first token carries.
        sentences = [
            FakeAlignedSentence(
                text=" Hello world.",
                start=0.0,
                end=1.0,
                tokens=[
                    FakeAlignedToken(text=" Hello", start=0.0, end=0.5),
                    FakeAlignedToken(text=" world.", start=0.5, end=1.0),
                ],
            ),
            FakeAlignedSentence(
                text=" Bye.",
                start=1.0,
                end=1.5,
                tokens=[FakeAlignedToken(text=" Bye.", start=1.0, end=1.5)],
            ),
        ]
        native = FakeAlignedResult(text="Hello world. Bye.", sentences=sentences)
        result = AlignedResultBackend(model_types=("parakeet",)).to_result(
            native, duration=1.5, want_words=True, follows_text=True
        )
        assert _texts(result) == [" Hello world.", " Bye."]
        assert result.text == " Hello world. Bye."

    def test_fun_asr_leaves_empty_chunks_out_of_the_join(self) -> None:
        from std_mlx_audio import FunAsrNano

        native = FakeSTTOutput(text="a b", segments=_chunks("a", "", "b"))
        result = FunAsrNano.backend.to_result(native, duration=3.0, want_words=False)
        assert _texts(result) == ["a", "", " b"]
        assert result.text == native.text

    def test_glm_keeps_a_chunk_edge_space_mlx_audio_strips(self) -> None:
        # mlx-audio: (" ".join(all_texts)).strip(); the segments are the unstripped
        # chunk texts, and the plugin keeps their characters.
        from std_mlx_audio import GlmAsrNano

        native = FakeSTTOutput(text="a  b", segments=_chunks(" a", "", "b"))
        result = GlmAsrNano.backend.to_result(native, duration=3.0, want_words=False)
        assert _texts(result) == [" a", " ", " b"]
        assert result.text == " a  b"

    @pytest.mark.parametrize(
        ("language", "texts"),
        [
            ("en", ["Hello.", "", "Bye."]),
            ("zh", ["你好。", "", "再见。"]),
            ("ja", ["こんにちは。", "さようなら。"]),
        ],
    )
    def test_cohere_text_matches_mlx_audios_own_join(self, language: str, texts: list[str]) -> None:
        cohere = pytest.importorskip("mlx_audio.stt.models.cohere_asr.cohere_asr")
        from std_mlx_audio import CohereAsr

        expected = cohere.join_chunk_texts(texts, language=language)
        native = FakeSTTOutput(text=expected, segments=_chunks(*texts), language=language)
        result = CohereAsr.backend.to_result(native, duration=3.0, want_words=False)
        assert result.text == expected

    def test_cohere_no_space_languages_match_mlx_audio(self) -> None:
        cohere = pytest.importorskip("mlx_audio.stt.models.cohere_asr.cohere_asr")
        from std_mlx_audio.languages import COHERE_LANGUAGES, NO_SPACE_LANGUAGES

        # The plugin's no-space languages that Cohere supports must be exactly
        # mlx-audio's own list, so the Cohere batch join stays mlx-audio's.
        supported = NO_SPACE_LANGUAGES.intersection(COHERE_LANGUAGES)
        assert supported == frozenset(cohere.NO_SPACE_LANGS)

    def test_text_only_family_keeps_mlx_audios_text_verbatim(self) -> None:
        backend = GenericSttBackend(SttFamilySpec(model_types=("moonshine",)))
        result = backend.to_result(FakeSTTOutput(text=" hello. "), duration=1.0, want_words=False)
        assert result.text == " hello. "
        assert result.segments is None

    def test_vibevoice_turns_join_with_a_line_break_kept_verbatim(self) -> None:
        from std_mlx_audio import VibeVoiceAsr

        native = FakeSTTOutput(
            text='[{"Content": "..."}]',
            segments=[
                {"start": 0.0, "end": 1.0, "speaker_id": 0, "text": "你好。"},
                {"start": 1.0, "end": 2.0, "speaker_id": 1, "text": ""},
                {"start": 2.0, "end": 3.0, "speaker_id": 1, "text": "我很好。 "},
            ],
        )
        result = VibeVoiceAsr.backend.to_result(native, duration=3.0, want_words=False)
        assert TURN_SEPARATOR == "\n"
        assert result.text == "你好。\n我很好。 "
        assert result.segments is None

    def test_vibevoice_without_parsed_turns_keeps_the_raw_text(self) -> None:
        from std_mlx_audio import VibeVoiceAsr

        native = FakeSTTOutput(text="not json", segments=[])
        result = VibeVoiceAsr.backend.to_result(native, duration=1.0, want_words=False)
        assert result.text == "not json"

    def test_every_timed_preset_names_its_chunk_join(self) -> None:
        import std_mlx_audio

        for name in std_mlx_audio.__all__:
            cls = getattr(std_mlx_audio, name)
            backend = getattr(cls, "backend", None)
            if isinstance(backend, GenericSttBackend) and backend.spec.segment_timing:
                assert backend.spec.chunk_join is not None, name
