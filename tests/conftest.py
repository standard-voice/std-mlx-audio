# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared fakes for the std-mlx-audio test suite.

CRITICAL: these tests NEVER load a real MLX model or download weights, and they
do not require Apple Silicon. The engine imports ``mlx_audio.stt.load`` lazily
inside ``_ensure_model_loaded``; we monkeypatch that symbol to return a fake
model whose ``generate`` yields a controllable native output of the right SHAPE
for each backend family (``STTOutput``-like for Qwen3-ASR / Whisper,
``AlignedResult``-like for Parakeet). This exercises the real adapter logic
(language mapping, output normalization, streaming windowing) against fakes.
Real-inference verification is a separate, opt-in script
(``scripts/verify_inference.py``), not part of this suite.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Native output fakes (one per backend return shape)
# --------------------------------------------------------------------------- #
@dataclass
class FakeSTTOutput:
    """Stand-in for mlx-audio's ``STTOutput`` (Qwen3-ASR / Whisper return)."""

    text: str
    segments: list[dict[str, Any]] | None = None
    language: Any = None
    generation_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    generation_tps: float = 0.0


@dataclass
class FakeAlignedToken:
    """Stand-in for Parakeet's ``AlignedToken``."""

    text: str
    start: float
    end: float
    duration: float = 0.0


@dataclass
class FakeAlignedSentence:
    """Stand-in for Parakeet's ``AlignedSentence``."""

    text: str
    start: float
    end: float
    tokens: list[FakeAlignedToken] = field(default_factory=list)


@dataclass
class FakeAlignedResult:
    """Stand-in for Parakeet's ``AlignedResult``."""

    text: str
    sentences: list[FakeAlignedSentence] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Fake model + loader
# --------------------------------------------------------------------------- #
class FakeMlxModel:
    """A fake mlx-audio model recording the kwargs passed to ``generate``.

    ``output`` is the canned native return; ``output_fn`` (if set) computes the
    return from ``(audio, kwargs)`` so streaming tests can vary segments by how
    much audio has accumulated.
    """

    def __init__(self, *, output: Any = None, output_fn: Callable[..., Any] | None = None) -> None:
        self.output = output
        self.output_fn = output_fn
        self.generate_calls: list[dict[str, Any]] = []
        self.priming_call: dict[str, Any] | None = None
        self.raise_on_generate: BaseException | None = None
        self.raise_on_prime: BaseException | None = None

    def generate(self, audio: Any, **kwargs: Any) -> Any:
        # The engine primes the generation thread with one silence generate
        # right after load (MlxAudioASR._prime_generation_thread, MLX stream
        # thread affinity), so the FIRST call on a freshly loaded model is
        # always the prime. Keep it out of generate_calls — tests assert on
        # real transcribe/streaming calls — and keep it fully inert (no
        # output_fn, no raise): the engine discards its result either way.
        if self.priming_call is None:
            self.priming_call = {"audio": audio, **kwargs}
            if self.raise_on_prime is not None:
                raise self.raise_on_prime
            return self.output
        self.generate_calls.append({"audio": audio, **kwargs})
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        if self.output_fn is not None:
            return self.output_fn(audio, kwargs)
        return self.output


class FakeLoader:
    """Records ``load`` calls and returns a preset fake model."""

    def __init__(self, model: FakeMlxModel) -> None:
        self.model = model
        self.load_calls: list[dict[str, Any]] = []
        self.raise_on_load: BaseException | None = None

    def __call__(self, model_path: str, **kwargs: Any) -> FakeMlxModel:
        self.load_calls.append({"model_path": model_path, **kwargs})
        if self.raise_on_load is not None:
            raise self.raise_on_load
        return self.model


#: The path the fake snapshot resolution returns for an online "download" when
#: a test has not staged a concrete cache directory.
FAKE_SNAPSHOT_DIR = "/fake/snapshots/deadbeef"


class FakeSnapshot:
    """Controls the fake ``huggingface_hub.snapshot_download``.

    ``cached_path`` is the local hit a ``local_files_only=True`` call returns
    (``None`` = not cached: the call raises, like the real helper). An online
    call records itself, optionally raises, and can flip the cache to
    ``download_target`` (simulating a completed acquisition).
    """

    cached_path: str | None = None
    download_target: str | None = None
    raise_on_download: BaseException | None = None
    last_kwargs: dict[str, Any] = {}
    #: Kwargs of the last ONLINE call only (a later cache-only status query
    #: overwrites ``last_kwargs``, so acquisition tests read this one).
    last_download_kwargs: dict[str, Any] = {}
    download_calls: int = 0

    @classmethod
    def reset(cls) -> None:
        cls.cached_path = None
        cls.download_target = None
        cls.raise_on_download = None
        cls.last_kwargs = {}
        cls.last_download_kwargs = {}
        cls.download_calls = 0


def _fake_snapshot_download(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    token: str | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """Mirror the ``snapshot_download`` contract against ``FakeSnapshot``."""
    FakeSnapshot.last_kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "token": token,
        "allow_patterns": allow_patterns,
    }
    if local_files_only:
        if FakeSnapshot.cached_path is None:
            raise FileNotFoundError("snapshot is not cached locally")
        return FakeSnapshot.cached_path
    FakeSnapshot.last_download_kwargs = dict(FakeSnapshot.last_kwargs)
    FakeSnapshot.download_calls += 1
    if FakeSnapshot.raise_on_download is not None:
        raise FakeSnapshot.raise_on_download
    if FakeSnapshot.download_target is not None:
        FakeSnapshot.cached_path = FakeSnapshot.download_target
    return FakeSnapshot.cached_path or FAKE_SNAPSHOT_DIR


def install_fake_loader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output: Any = None,
    output_fn: Callable[..., Any] | None = None,
) -> FakeLoader:
    """Patch ``mlx_audio.stt.load`` (as imported by the engine) with a fake.

    The engine does ``from mlx_audio.stt import load`` *inside*
    ``_ensure_model_loaded``, so we patch the attribute on the real
    ``mlx_audio.stt`` module; the lazy import then resolves to our fake. The
    plugin-side snapshot resolution (``huggingface_hub.snapshot_download``) is
    faked alongside it, backed by :class:`FakeSnapshot`, so no test ever
    touches the network.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        output: The canned native ``generate`` return.
        output_fn: Optional ``(audio, kwargs) -> native`` override.

    Returns:
        The installed :class:`FakeLoader` (inspect ``load_calls`` / ``model``).
    """
    import huggingface_hub
    import mlx_audio.stt as stt

    FakeSnapshot.reset()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)
    model = FakeMlxModel(output=output, output_fn=output_fn)
    loader = FakeLoader(model)
    monkeypatch.setattr(stt, "load", loader)
    return loader


@pytest.fixture
def fake_loader(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeLoader]:
    """Return an installer so each test sets the native output it wants.

    Usage::

        def test_x(fake_loader):
            loader = fake_loader(output=FakeSTTOutput(text="hi"))
            ...
    """

    def _install(*, output: Any = None, output_fn: Callable[..., Any] | None = None) -> FakeLoader:
        return install_fake_loader(monkeypatch, output=output, output_fn=output_fn)

    return _install


def silent_pcm(seconds: float, sample_rate: int = 16000) -> bytes:
    """Return ``seconds`` of silent 16-bit LE PCM mono bytes."""
    return np.zeros(int(seconds * sample_rate), dtype="<i2").tobytes()


def float_array(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    """Return ``seconds`` of a non-silent float32 mono waveform."""
    n = int(seconds * sample_rate)
    t = np.linspace(0.0, seconds, n, endpoint=False, dtype=np.float32)
    return (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
