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
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from standard_asr import RuntimeParams
from standard_asr.audio_input import AudioArray
from standard_asr.exceptions import DiscoveryError

from std_mlx_audio import (
    Canary1BV2,
    CohereAsr,
    FireRedAsr2Aed,
    FunAsrNano,
    GraniteSpeech1B,
    MlxAudioParams,
    Mms1BAll,
    MoonshineTiny,
    NemotronAsrStreaming06B,
    Qwen2Audio7B,
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
# Voxtral path adaptation (wants_path)
# --------------------------------------------------------------------------- #
def test_voxtral_receives_path_for_array_input(fake_loader: Callable[..., FakeLoader]) -> None:
    # Voxtral's generate only works via its file-path branch (the array path needs
    # a `format` arg mlx-audio never passes); the engine materializes a temp WAV
    # from the negotiated array and hands generate the path string.
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    VoxtralMini3B().transcribe(_array_input(), RuntimeParams(language="en"))
    audio = loader.model.generate_calls[0]["audio"]
    assert isinstance(audio, str) and audio.endswith(".wav")
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


# --------------------------------------------------------------------------- #
# Qwen2-Audio decode prompt (audio-LLM -> bare transcript)
# --------------------------------------------------------------------------- #
def test_qwen2_audio_passes_default_transcription_prompt(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The audio-LLM's own default ("Please transcribe the speech.") yields
    # conversational output; the preset injects a strict transcription prompt so
    # generate returns a bare transcript.
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    Qwen2Audio7B().transcribe(_array_input(), RuntimeParams())
    assert "only the transcript" in loader.model.generate_calls[0]["prompt"].lower()


def test_qwen2_audio_system_prompt_overrides_default(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # A caller-supplied system_prompt takes precedence over the preset default.
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    Qwen2Audio7B().transcribe(
        _array_input(),
        RuntimeParams(provider_params=MlxAudioParams(system_prompt="Just the words.")),
    )
    assert loader.model.generate_calls[0]["prompt"] == "Just the words."


# --------------------------------------------------------------------------- #
# VibeVoice transcript rebuilt from parsed diarization segments (not raw JSON)
# --------------------------------------------------------------------------- #
def test_vibevoice_text_rebuilt_from_segments(fake_loader: Callable[..., FakeLoader]) -> None:
    # VibeVoice returns its transcript inside STTOutput.segments (parsed from a
    # diarization JSON) while STTOutput.text is the raw JSON string; the result
    # text must be the joined segment text, with no (undeclared) segments emitted.
    raw_json = '[{"start":0.0,"end":1.0,"text":"hello"},{"start":1.0,"end":2.0,"text":"world"}]'
    fake_loader(
        output=FakeSTTOutput(
            text=raw_json,
            segments=[
                {"start": 0.0, "end": 1.0, "speaker_id": 0, "text": "hello"},
                {"start": 1.0, "end": 2.0, "speaker_id": 1, "text": "world"},
            ],
        )
    )
    result = VibeVoiceAsr().transcribe(_array_input(), RuntimeParams())
    assert result.text == "hello world"  # rebuilt from segments, not the raw JSON
    assert result.segments is None  # declared text-only: segments are not emitted


# --------------------------------------------------------------------------- #
# Cohere-ASR loads from the repo's mlx-int8/ subfolder
# --------------------------------------------------------------------------- #
def test_cohere_loads_from_subfolder(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The repo stores its checkpoint under mlx-int8/; the loader snapshot-downloads
    # the repo and points load() at the subfolder (not the root, where there is no
    # config.json), dropping the now-meaningless revision for the local path.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *_a, **_k: str(tmp_path))
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    CohereAsr().prepare()
    assert loader.load_calls[0]["model_path"] == str(tmp_path / "mlx-int8")
    assert "revision" not in loader.load_calls[0]


def test_cohere_subfolder_download_failure_raises(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A snapshot-download failure surfaces as a clean DiscoveryError, not a raw OSError.
    import huggingface_hub

    def _boom(*_a: Any, **_k: Any) -> str:
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _boom)
    fake_loader(output=FakeSTTOutput(text="hi"))
    with pytest.raises(DiscoveryError, match="cohere-asr-mlx"):
        CohereAsr().prepare()
