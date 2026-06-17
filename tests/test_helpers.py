# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the small pure helpers (language mapping, PCM, edge paths)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from std_mlx_audio import languages
from std_mlx_audio._streaming import (  # pyright: ignore[reportPrivateUsage]
    _pcm_s16le_to_float32,
)
from std_mlx_audio.backends import WhisperBackend
from std_mlx_audio.engine import _disable_tqdm_monitor_thread  # pyright: ignore[reportPrivateUsage]

from .conftest import FakeLoader, FakeSTTOutput


# --------------------------------------------------------------------------- #
# Language mapping
# --------------------------------------------------------------------------- #
def test_to_qwen_name_and_whisper_code_roundtrip() -> None:
    assert languages.to_qwen_name("ja") == "Japanese"
    assert languages.to_whisper_code("ja") == "ja"
    assert languages.to_qwen_name("zh-Hans-CN") == "Chinese"  # primary subtag


def test_unsupported_tag_maps_to_none() -> None:
    assert languages.to_qwen_name("sw") is None
    assert languages.to_whisper_code("sw") is None


def test_to_iso_subtag_strips_region_and_script() -> None:
    # ISO-taking families get the bare primary subtag (no Qwen/Whisper table).
    assert languages.to_iso_subtag("en-US") == "en"
    assert languages.to_iso_subtag("zh-Hans-CN") == "zh"
    assert languages.to_iso_subtag("DE") == "de"
    # Unlike the Whisper/Qwen mappers, an out-of-table tag is still returned (the
    # standard layer has already gated it against the family's selectable set).
    assert languages.to_iso_subtag("sw") == "sw"


def test_from_backend_language_code_and_name() -> None:
    assert languages.from_backend_language("en") == "en"  # ISO code path
    assert languages.from_backend_language("German") == "de"  # English-name path
    assert languages.from_backend_language("Chinese,English") == "zh"  # first wins


def test_from_backend_language_empty_and_unknown() -> None:
    assert languages.from_backend_language(None) is None
    assert languages.from_backend_language("") is None
    assert languages.from_backend_language("   ") is None
    assert languages.from_backend_language("Klingon") is None


def test_whisper_backend_drops_unsupported_language() -> None:
    # to_whisper_code returns None -> no language kwarg (auto-detect).
    from std_mlx_audio import MlxAudioConfig, MlxAudioParams

    kw = WhisperBackend().generate_kwargs(
        resolved_language="sw",
        want_words=False,
        params=MlxAudioParams(),
        config=MlxAudioConfig(),
    )
    assert "language" not in kw


# --------------------------------------------------------------------------- #
# PCM decode
# --------------------------------------------------------------------------- #
def test_pcm_decode_empty_and_odd_byte() -> None:
    assert _pcm_s16le_to_float32(b"").size == 0
    assert _pcm_s16le_to_float32(b"\x01").size == 0  # < 2 bytes
    # Odd trailing byte dropped (one whole sample decoded).
    decoded = _pcm_s16le_to_float32(b"\x00\x40\x07")
    assert decoded.size == 1
    assert decoded[0] == pytest.approx(0.5, abs=1e-3)


def test_pcm_decode_roundtrip() -> None:
    samples = np.array([0, 16384, -16384], dtype="<i2")
    decoded = _pcm_s16le_to_float32(samples.tobytes())
    assert decoded.size == 3
    assert decoded[1] == pytest.approx(0.5, abs=1e-3)


# --------------------------------------------------------------------------- #
# tqdm monitor suppression (best-effort, never raises)
# --------------------------------------------------------------------------- #
def test_disable_tqdm_monitor_is_idempotent_and_safe() -> None:
    _disable_tqdm_monitor_thread()
    _disable_tqdm_monitor_thread()  # idempotent
    import tqdm

    assert tqdm.tqdm.monitor_interval == 0


def test_disable_tqdm_monitor_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def boom(name: str, *args: object, **kwargs: object) -> object:
        if name == "tqdm":
            raise ImportError("no tqdm")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", boom)
    _disable_tqdm_monitor_thread()  # must not raise


# --------------------------------------------------------------------------- #
# Engine revision passthrough + bytes input
# --------------------------------------------------------------------------- #
def test_revision_forwarded_to_loader(fake_loader: Callable[..., FakeLoader]) -> None:
    from std_mlx_audio import Qwen3Asr06B

    loader = fake_loader(output=FakeSTTOutput(text="hi"))
    Qwen3Asr06B(revision="refs/pr/1").prepare()
    assert loader.load_calls[0]["revision"] == "refs/pr/1"


def test_source_for_rejects_empty_prepared() -> None:
    # Defensive guard: negotiation never delivers an empty shape, but the engine
    # raises a portable error rather than silently mis-transcribing if it does.
    from standard_asr.audio_conversion import PreparedAudio
    from standard_asr.audio_input import InputKind
    from standard_asr.exceptions import TranscriptionError

    from std_mlx_audio import Qwen3Asr06B

    empty = PreparedAudio(kind=InputKind.ARRAY)  # all payloads None
    with pytest.raises(TranscriptionError, match="no array, path, or bytes"):
        Qwen3Asr06B()._source_for(empty)  # pyright: ignore[reportPrivateUsage]


def test_source_for_wants_path_returns_path_unchanged() -> None:
    # A wants_path family (Voxtral-Mini) is handed a file path directly.
    from standard_asr.audio_conversion import PreparedAudio
    from standard_asr.audio_input import InputKind

    from std_mlx_audio import VoxtralMini3B

    src, dur = VoxtralMini3B()._source_for(  # pyright: ignore[reportPrivateUsage]
        PreparedAudio(kind=InputKind.ENCODED_FILE, path="/tmp/x.wav")
    )
    assert src == "/tmp/x.wav" and dur is None


def test_source_for_wants_path_materializes_bytes() -> None:
    import os

    from standard_asr.audio_conversion import PreparedAudio
    from standard_asr.audio_input import InputKind

    from std_mlx_audio import VoxtralMini3B

    src, dur = VoxtralMini3B()._source_for(  # pyright: ignore[reportPrivateUsage]
        PreparedAudio(kind=InputKind.ENCODED_BYTES, data=b"RIFFfake")
    )
    assert isinstance(src, str) and os.path.exists(src) and dur is None
    os.unlink(src)


def test_source_for_wants_path_materializes_array_to_wav() -> None:
    # An array negotiated for a wants_path family becomes a 16 kHz mono temp WAV.
    import os
    import wave

    import numpy as np
    from standard_asr.audio_conversion import PreparedAudio
    from standard_asr.audio_input import InputKind

    from std_mlx_audio import VoxtralMini3B

    arr = np.zeros(16000, dtype=np.float32)
    src, dur = VoxtralMini3B()._source_for(  # pyright: ignore[reportPrivateUsage]
        PreparedAudio(kind=InputKind.ARRAY, array=arr, sample_rate=16000)
    )
    assert isinstance(src, str) and src.endswith(".wav")
    with wave.open(src, "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getnframes() == 16000
    os.unlink(src)
    assert dur == 1.0


def test_source_for_wants_path_rejects_empty_prepared() -> None:
    from standard_asr.audio_conversion import PreparedAudio
    from standard_asr.audio_input import InputKind
    from standard_asr.exceptions import TranscriptionError

    from std_mlx_audio import VoxtralMini3B

    with pytest.raises(TranscriptionError, match="no array, path, or bytes"):
        VoxtralMini3B()._source_for(  # pyright: ignore[reportPrivateUsage]
            PreparedAudio(kind=InputKind.ARRAY)
        )


def test_prepared_to_pcm_requires_array() -> None:
    from standard_asr.audio_conversion import PreparedAudio
    from standard_asr.audio_input import InputKind
    from standard_asr.exceptions import TranscriptionError

    from std_mlx_audio.engine import _prepared_to_pcm  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(TranscriptionError, match="not delivered as an array"):
        _prepared_to_pcm(PreparedAudio(kind=InputKind.ENCODED_FILE, path="/x.wav"))


def test_append_pcm_empty_and_subsample_are_noops() -> None:
    from standard_asr import RuntimeParams

    from std_mlx_audio import WhisperTiny
    from std_mlx_audio._streaming import (  # pyright: ignore[reportPrivateUsage]
        MlxAudioStreamingSession,
    )

    session = MlxAudioStreamingSession(WhisperTiny(), RuntimeParams())
    before = session._window.size  # pyright: ignore[reportPrivateUsage]
    session._append_pcm(b"")  # empty -> early return
    # A single odd byte decodes to zero samples -> the size-guard false branch.
    session._append_pcm(b"\x01")  # pyright: ignore[reportPrivateUsage]
    assert session._window.size == before  # pyright: ignore[reportPrivateUsage]


def test_bytes_input_written_to_tempfile(fake_loader: Callable[..., FakeLoader]) -> None:
    # An AudioBytes input (encoded) negotiates to ENCODED_BYTES; the engine
    # materializes a temp file for mlx-audio (which has no bytes entry point).
    import io
    import wave
    from pathlib import Path

    from standard_asr import RuntimeParams
    from standard_asr.audio_input import AudioBytes

    from std_mlx_audio import Qwen3Asr06B

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    loader = fake_loader(output=FakeSTTOutput(text="hi", language=["English"]))
    Qwen3Asr06B().transcribe(AudioBytes(buf.getvalue()), RuntimeParams(language="en"))
    audio_arg = loader.model.generate_calls[0]["audio"]
    # The engine passes a path string to a temp .wav it wrote.
    assert isinstance(audio_arg, str)
    assert Path(audio_arg).exists()
    Path(audio_arg).unlink(missing_ok=True)
