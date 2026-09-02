# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Batch-path tests for the MLX engine (real adapter logic, fake model).

Exercises the engine ``transcribe`` template end to end against the injected
fake loader: lazy loading + download policy, language-axis resolution per family,
audio source mapping (array / path / bytes), the batch error contract, and the
init config / env-var fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from standard_asr import RuntimeParams
from standard_asr.audio.input import AudioArray
from standard_asr.contract.exceptions import (
    DiscoveryError,
    InvalidProviderParamError,
    TranscriptionError,
)
from standard_asr.contract.params import ProviderParams

from std_mlx_audio import (
    MlxAudioConfig,
    MlxAudioParams,
    ParakeetTdt06BV3,
    Qwen3Asr06B,
    WhisperTiny,
)

from .conftest import (
    FAKE_SNAPSHOT_DIR,
    FakeAlignedResult,
    FakeAlignedSentence,
    FakeAlignedToken,
    FakeLoader,
    FakeSnapshot,
    FakeSTTOutput,
)

_RATE = 16000


def _array_input(seconds: float = 1.0) -> AudioArray:
    n = int(seconds * _RATE)
    return AudioArray(np.zeros(n, dtype=np.float32), _RATE)


# --------------------------------------------------------------------------- #
# Lazy loading + download policy
# --------------------------------------------------------------------------- #
def test_init_is_pure_no_load(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="x"))
    engine = Qwen3Asr06B()
    # Constructing must not touch the loader (spec IC.9 — weights load lazily).
    assert loader.load_calls == []
    assert engine._model is None  # pyright: ignore[reportPrivateUsage]


def test_transcribe_loads_preset_repo(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="hi", language=["English"]))
    engine = Qwen3Asr06B()
    engine.transcribe(_array_input(), RuntimeParams(language="en"))
    # Resolution is plugin-side: the snapshot is downloaded by repo id and the
    # loader receives the resolved LOCAL directory, never the repo id.
    assert FakeSnapshot.last_download_kwargs["repo_id"] == "mlx-community/Qwen3-ASR-0.6B-4bit"
    assert loader.load_calls[0]["model_path"] == FAKE_SNAPSHOT_DIR


def test_model_path_override_wins(fake_loader: Callable[..., FakeLoader], tmp_path: Path) -> None:
    # The artifact guard inspects the path before loading, so the override must
    # be a real MLX checkpoint directory.
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"\x00")
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    engine = Qwen3Asr06B(model_path=str(tmp_path))
    engine.prepare()
    assert loader.load_calls[0]["model_path"] == str(tmp_path)
    assert FakeSnapshot.download_calls == 0


def test_download_disabled_forces_local_only(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Per download-policy.md, an explicit "0"/false/no disables downloads (an
    # UNSET variable defaults to enabled, so we must set it explicitly here).
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    FakeSnapshot.cached_path = FAKE_SNAPSHOT_DIR
    Qwen3Asr06B().prepare()
    # The plugin-side resolution stayed cache-only and the loader received the
    # cached local directory; no online call was made.
    assert FakeSnapshot.last_kwargs["local_files_only"] is True
    assert FakeSnapshot.download_calls == 0
    assert loader.load_calls[0]["model_path"] == FAKE_SNAPSHOT_DIR


def test_local_files_only_config_forces_local(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The config flag forces local-only even when downloads are globally allowed.
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "1")
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    FakeSnapshot.cached_path = FAKE_SNAPSHOT_DIR
    Qwen3Asr06B(local_files_only=True).prepare()
    assert FakeSnapshot.last_kwargs["local_files_only"] is True
    assert FakeSnapshot.download_calls == 0
    assert loader.load_calls[0]["model_path"] == FAKE_SNAPSHOT_DIR


def test_download_enabled_allows_network(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "1")
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    Qwen3Asr06B().prepare()
    assert FakeSnapshot.last_download_kwargs["local_files_only"] is False
    assert loader.load_calls[0]["model_path"] == FAKE_SNAPSHOT_DIR


def test_load_failure_with_ready_snapshot_is_engine_fault(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The snapshot is cached and complete; a loader failure is then an
    # engine-execution fault (batch R7 mapping), never an artifact error.
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    FakeSnapshot.cached_path = FAKE_SNAPSHOT_DIR
    loader.raise_on_load = RuntimeError("metal kaput")
    with pytest.raises(TranscriptionError, match="failed to load"):
        Qwen3Asr06B().prepare()


def test_missing_mlx_audio_raises_discovery_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate mlx-audio import failure inside the lazy loader.
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mlx_audio.stt" or name.startswith("mlx_audio"):
            raise ImportError("no mlx here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(DiscoveryError, match="mlx-audio is not installed"):
        Qwen3Asr06B().prepare()


# --------------------------------------------------------------------------- #
# Language-axis resolution
# --------------------------------------------------------------------------- #
def test_qwen_forwards_resolved_language(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="hi", language=["English"]))
    engine = Qwen3Asr06B()
    engine.transcribe(_array_input(), RuntimeParams(language="de"))
    assert loader.model.generate_calls[0]["language"] == "German"


def test_parakeet_ignores_language_override(fake_loader: Callable[..., FakeLoader]) -> None:
    # Parakeet declares runtime_override=False; a per-request language must be
    # rejected by the gate. (Honest fail-closed behaviour.)
    fake_loader(
        output=FakeAlignedResult(
            text="hi",
            sentences=[
                FakeAlignedSentence(
                    text="hi",
                    start=0.0,
                    end=0.5,
                    tokens=[FakeAlignedToken(text="hi", start=0.0, end=0.5)],
                )
            ],
        )
    )
    engine = ParakeetTdt06BV3()
    with pytest.raises(Exception):  # noqa: B017 - gated as unsupported feature
        engine.transcribe(_array_input(), RuntimeParams(language="fr"))


def test_parakeet_default_language_runs(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(
        output=FakeAlignedResult(
            text="hi there",
            sentences=[
                FakeAlignedSentence(
                    text="hi there",
                    start=0.0,
                    end=0.8,
                    tokens=[FakeAlignedToken(text="hi there", start=0.0, end=0.8)],
                )
            ],
        )
    )
    engine = ParakeetTdt06BV3()
    result = engine.transcribe(_array_input())  # no per-request language
    assert result.text == "hi there"
    # No language argument is ever passed to Parakeet.
    assert "language" not in loader.model.generate_calls[0]


# --------------------------------------------------------------------------- #
# Audio source mapping
# --------------------------------------------------------------------------- #
def test_array_passed_through_as_mlx_array(fake_loader: Callable[..., FakeLoader]) -> None:
    import mlx.core as mx

    loader = fake_loader(output=FakeSTTOutput(text="hi", language=["English"]))
    Qwen3Asr06B().transcribe(_array_input(2.0), RuntimeParams(language="en"))
    audio = loader.model.generate_calls[0]["audio"]
    # Converted to mx.array (the one decoded shape every MLX backend accepts;
    # numpy fails Parakeet's bfloat16 cast).
    assert isinstance(audio, mx.array)
    assert audio.shape[0] == 2 * _RATE


def test_path_input_passed_through(fake_loader: Callable[..., FakeLoader], tmp_path: Path) -> None:
    # A real WAV path negotiates to ENCODED_FILE (we accept it); the engine hands
    # the path straight to mlx-audio.
    import wave

    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_RATE)
        w.writeframes(np.zeros(_RATE, dtype="<i2").tobytes())
    loader = fake_loader(output=FakeSTTOutput(text="hi", language=["English"]))
    Qwen3Asr06B().transcribe(str(wav), RuntimeParams(language="en"))
    assert loader.model.generate_calls[0]["audio"] == str(wav)


# --------------------------------------------------------------------------- #
# Error contract + provider params
# --------------------------------------------------------------------------- #
def test_generate_failure_wrapped_as_transcription_error(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    loader.model.raise_on_generate = ValueError("boom")
    engine = Qwen3Asr06B()
    with pytest.raises(TranscriptionError) as exc:
        engine.transcribe(_array_input(), RuntimeParams(language="en"))
    assert isinstance(exc.value.__cause__, ValueError)  # native exception preserved


def test_priming_failure_downgrades_to_warning(
    fake_loader: Callable[..., FakeLoader], caplog: pytest.LogCaptureFixture
) -> None:
    # A family that rejects the 0.1 s priming silence must still load and
    # transcribe real audio; the engine downgrades the priming failure to a
    # warning instead of failing the load.
    loader = fake_loader(output=FakeSTTOutput(text="hi", language=["English"]))
    loader.model.raise_on_prime = RuntimeError("prime rejected")
    with caplog.at_level("WARNING", logger="std_mlx_audio.engine"):
        result = Qwen3Asr06B().transcribe(_array_input(), RuntimeParams(language="en"))
    assert result.text == "hi"
    assert loader.model.priming_call is not None
    assert any("Priming generate failed" in r.message for r in caplog.records)


def test_provider_params_accepted(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="hi", language=["English"]))
    params = RuntimeParams(
        language="en", provider_params=MlxAudioParams(temperature=0.3, top_p=0.9)
    )
    Qwen3Asr06B().transcribe(_array_input(), params)
    assert loader.model.generate_calls[0]["temperature"] == 0.3
    assert loader.model.generate_calls[0]["top_p"] == 0.9


def test_wrong_provider_params_rejected(fake_loader: Callable[..., FakeLoader]) -> None:
    class OtherParams(ProviderParams):
        foo: int = 1

    fake_loader(output=FakeSTTOutput(text="hi"))
    with pytest.raises(InvalidProviderParamError):
        Qwen3Asr06B().transcribe(_array_input(), RuntimeParams(provider_params=OtherParams()))


# --------------------------------------------------------------------------- #
# Config / env
# --------------------------------------------------------------------------- #
def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STANDARD_ASR_MLX_AUDIO__REVISION", "refs/pr/9")
    engine = WhisperTiny()
    cfg = engine.config
    assert isinstance(cfg, MlxAudioConfig)
    assert cfg.revision == "refs/pr/9"


def test_explicit_kwarg_beats_preset_default() -> None:
    # Parakeet defaults default_language to "en"; an explicit value wins.
    engine = ParakeetTdt06BV3(default_language="fr")
    assert engine.config.default_language == "fr"


def test_hf_token_is_secret() -> None:
    engine = Qwen3Asr06B(hf_token="sk-secret")
    dumped = repr(engine.config)
    assert "sk-secret" not in dumped  # SecretStr masks it
