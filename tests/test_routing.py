# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Engine routing, family enforcement, and generic-family batch tests.

Covers the "many models" machinery added on top of the original three families:
the loaded-model/backend family check (the latent-bug fix), the ``model_type``
load override, Voxtral's list-audio adaptation, and end-to-end batch transcription
for the generic STTOutput / aligned families incl. their per-family decode knobs.
All against the injected fake loader — never a real model or download.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from standard_asr import RuntimeParams
from standard_asr.audio_input import AudioArray
from standard_asr.exceptions import DiscoveryError

from std_mlx_audio import (
    Canary1BV2,
    FireRedAsr2Aed,
    FunAsrNano,
    GraniteSpeech1B,
    MlxAudioParams,
    Mms1BAll,
    MoonshineTiny,
    NemotronAsrStreaming06B,
    SenseVoiceSmall,
    VibeVoiceAsr,
    VoxtralMini3B,
    WhisperTiny,
)
from std_mlx_audio.engine import _model_family  # pyright: ignore[reportPrivateUsage]

from .conftest import (
    FakeAlignedResult,
    FakeAlignedSentence,
    FakeAlignedToken,
    FakeLoader,
    FakeSTTOutput,
)

_RATE = 16000


def _array_input(seconds: float = 1.0) -> AudioArray:
    return AudioArray(np.zeros(int(seconds * _RATE), dtype=np.float32), _RATE)


def _model_of_family(family: str) -> object:
    """Build a stand-in model whose class ``__module__`` mimics mlx-audio's layout."""
    cls = type("Model", (), {})
    cls.__module__ = f"mlx_audio.stt.models.{family}.{family}"
    return cls()


# --------------------------------------------------------------------------- #
# Family detection + enforcement (the latent-bug fix)
# --------------------------------------------------------------------------- #
def test_model_family_extracts_segment_after_models() -> None:
    assert _model_family(_model_of_family("whisper")) == "whisper"
    assert _model_family(_model_of_family("mega_asr")) == "mega_asr"


def test_model_family_none_when_unintrospectable() -> None:
    assert _model_family(None) is None
    # A class whose module is not under mlx_audio.stt.models -> cannot determine.
    other = type("Thing", (), {})
    other.__module__ = "some.other.module"
    assert _model_family(other()) is None


def test_verify_model_family_passes_on_match() -> None:
    engine = WhisperTiny()
    engine._model = _model_of_family("whisper")  # pyright: ignore[reportPrivateUsage]
    engine._verify_model_family()  # pyright: ignore[reportPrivateUsage]  # no raise


def test_verify_model_family_raises_on_mismatch() -> None:
    # A model_path override resolving to a different family must fail loudly, not
    # silently run through the wrong adapter (the cardinal sin).
    engine = WhisperTiny()
    engine._model = _model_of_family("sensevoice")  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(DiscoveryError, match="sensevoice"):
        engine._verify_model_family()  # pyright: ignore[reportPrivateUsage]


def test_verify_model_family_skips_when_unintrospectable() -> None:
    engine = WhisperTiny()
    engine._model = object()  # pyright: ignore[reportPrivateUsage]  # no module family
    engine._verify_model_family()  # pyright: ignore[reportPrivateUsage]  # does not block


def test_mismatch_surfaces_through_load(fake_loader: Callable[..., FakeLoader]) -> None:
    # End to end: the loader returns a model of the wrong family -> DiscoveryError.
    # Give THIS fake model a one-off class whose module looks like another family,
    # without mutating the shared FakeMlxModel class (which would pollute other tests).
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    loader.model.__class__ = type(
        "Model", (type(loader.model),), {"__module__": "mlx_audio.stt.models.voxtral.voxtral"}
    )
    with pytest.raises(DiscoveryError, match="voxtral"):
        WhisperTiny().prepare()


# --------------------------------------------------------------------------- #
# model_type load override (MMS mislabels its config as wav2vec2)
# --------------------------------------------------------------------------- #
def test_mms_pins_model_type_on_load(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    Mms1BAll().prepare()
    assert loader.load_calls[0]["model_type"] == "mms"


def test_default_preset_does_not_pin_model_type(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    MoonshineTiny().prepare()
    assert "model_type" not in loader.load_calls[0]


# --------------------------------------------------------------------------- #
# Voxtral list-audio adaptation
# --------------------------------------------------------------------------- #
def test_voxtral_wraps_audio_in_list(fake_loader: Callable[..., FakeLoader]) -> None:
    import mlx.core as mx

    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    VoxtralMini3B().transcribe(_array_input(), RuntimeParams(language="en"))
    audio = loader.model.generate_calls[0]["audio"]
    assert isinstance(audio, list) and len(audio) == 1
    assert isinstance(audio[0], mx.array)
    assert loader.model.generate_calls[0]["language"] == "en"


# --------------------------------------------------------------------------- #
# Generic STTOutput families — end-to-end batch
# --------------------------------------------------------------------------- #
def test_moonshine_text_only(fake_loader: Callable[..., FakeLoader]) -> None:
    # No language axis: transcribe with no language must resolve cleanly, and a
    # placeholder-timing STTOutput yields text with NO segments (honest).
    loader = fake_loader(
        output=FakeSTTOutput(
            text="hello world", segments=[{"text": "hello world", "start": 0.0, "end": 0.0}]
        )
    )
    result = MoonshineTiny().transcribe(_array_input())
    assert result.text == "hello world"
    assert result.segments is None
    assert "language" not in loader.model.generate_calls[0]


def test_sensevoice_passes_language_and_reports_detected(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="你好", language="zh"))
    result = SenseVoiceSmall().transcribe(_array_input(), RuntimeParams(language="zh"))
    assert loader.model.generate_calls[0]["language"] == "zh"
    assert result.detected_language == "zh"


def test_sensevoice_use_itn_forwarded(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="hi", language="en"))
    SenseVoiceSmall().transcribe(
        _array_input(), RuntimeParams(language="en", provider_params=MlxAudioParams(use_itn=True))
    )
    assert loader.model.generate_calls[0]["use_itn"] is True


def test_fun_asr_hotwords_and_itn(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    FunAsrNano().transcribe(
        _array_input(),
        RuntimeParams(
            language="zh", provider_params=MlxAudioParams(hotwords=["MLX"], use_itn=False)
        ),
    )
    call = loader.model.generate_calls[0]
    assert call["language"] == "zh"
    assert call["hotwords"] == ["MLX"]
    assert call["itn"] is False  # provider use_itn -> family's `itn` kwarg


def test_canary_translation(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="hello"))
    Canary1BV2().transcribe(
        _array_input(),
        RuntimeParams(language="de", provider_params=MlxAudioParams(target_language="en")),
    )
    call = loader.model.generate_calls[0]
    assert call["source_lang"] == "de" and call["target_lang"] == "en"


def test_granite_translation_target(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="bonjour"))
    GraniteSpeech1B().transcribe(
        _array_input(), RuntimeParams(provider_params=MlxAudioParams(target_language="fr"))
    )
    assert loader.model.generate_calls[0]["language"] == "fr"


def test_fireredasr2_beam_size(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    FireRedAsr2Aed().transcribe(
        _array_input(), RuntimeParams(provider_params=MlxAudioParams(beam_size=5))
    )
    assert loader.model.generate_calls[0]["beam_size"] == 5


def test_vibevoice_context(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    VibeVoiceAsr().transcribe(
        _array_input(), RuntimeParams(provider_params=MlxAudioParams(context="medical terms"))
    )
    assert loader.model.generate_calls[0]["context"] == "medical terms"


def test_nemotron_aligned_output_with_words(fake_loader: Callable[..., FakeLoader]) -> None:
    fake_loader(
        output=FakeAlignedResult(
            text="hi there",
            sentences=[
                FakeAlignedSentence(
                    text="hi there",
                    start=0.0,
                    end=0.8,
                    tokens=[
                        FakeAlignedToken(text="hi", start=0.0, end=0.4),
                        FakeAlignedToken(text=" there", start=0.4, end=0.8),
                    ],
                )
            ],
        )
    )
    result = NemotronAsrStreaming06B().transcribe(
        _array_input(), RuntimeParams(word_timestamps="word")
    )
    assert result.text == "hi there"
    assert result.words is not None and len(result.words) == 2
    assert result.detected_language is None  # aligned families don't report language
