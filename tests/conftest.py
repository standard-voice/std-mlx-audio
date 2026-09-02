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
from pathlib import Path
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


def _make_fake_snapshot_dir() -> str:
    """Create one REAL checkpoint-shaped snapshot directory for the fakes.

    The plugin's status path verifies completeness on disk (config.json plus a
    weights file), so the default fake resolution must point at a directory
    that actually has that shape.
    """
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="std-mlx-fake-")) / "snapshots" / ("deadbeef" * 5)
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"\x00" * 8)
    # Every preset-declared closure (both axes) must be satisfied by this
    # shared default (the per-family files coexist harmlessly): the Whisper
    # and Qwen3 processor files, the transformers tokenizer files, the tekken
    # tokenizer, the SenseVoice bpe + normalization stats, the Canary and
    # Cohere SentencePiece model, the FireRed dict/cmvn, and the MMS vocab.
    # It doubles as the warm companion-tokenizer cache (tokenizer.json).
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "merges.txt",
        "tekken.json",
        "chn_jpn_yue_eng_ko_spectok.bpe.model",
        "am.mvn",
        "tokenizer.model",
        "dict.txt",
        "cmvn.json",
        "vocab.json",
    ):
        (root / name).write_text("{}")
    return str(root)


#: The path the fake snapshot resolution returns for an online "download" when
#: a test has not staged a concrete cache directory. A real checkpoint-shaped
#: directory (see ``_make_fake_snapshot_dir``).
FAKE_SNAPSHOT_DIR = _make_fake_snapshot_dir()

#: The one companion tokenizer repo the plugin declares (VibeVoice); calls for
#: it route through the FakeSnapshot companion seam.
_COMPANION_TOKENIZER_REPO = "Qwen/Qwen2.5-7B"


class FakeSnapshot:
    """Controls the fake ``huggingface_hub.snapshot_download``.

    ``cached_path`` is the local hit a ``local_files_only=True`` call returns
    (``None`` = not cached: the call raises, like the real helper). An online
    call records itself, optionally raises, and can flip the cache to
    ``download_target`` (simulating a completed acquisition).

    The VibeVoice companion tokenizer repo resolves through its own seam
    (``companion_cached_path`` and the companion bookkeeping): its default is
    the warm shared directory so the rest of the suite never trips over a
    cold companion cache, and the VibeVoice tests set it explicitly.
    """

    cached_path: str | None = None
    download_target: str | None = None
    #: Raised by a cache-only resolution (models an unreadable cache).
    raise_on_resolve: BaseException | None = None
    raise_on_download: BaseException | None = None
    #: Optional side effect run by an online call (models the transfer
    #: materializing files, e.g. completing a partial snapshot).
    on_download: Any = None
    last_kwargs: dict[str, Any] = {}
    #: Kwargs of the last ONLINE call only (a later cache-only status query
    #: overwrites ``last_kwargs``, so acquisition tests read this one).
    last_download_kwargs: dict[str, Any] = {}
    download_calls: int = 0
    #: The cached hit for the companion tokenizer repo (``None`` = cold).
    companion_cached_path: str | None = None
    #: Raised by a cache-only companion resolution (models an unreadable
    #: cache -- distinct from the not-in-cache miss the cold default
    #: raises).
    raise_on_companion_resolve: BaseException | None = None
    #: The canned answer of the fake ``offline_tokenizer_loads`` probe
    #: (the real probe runs transformers; its own unit tests exercise it
    #: against real directories).
    offline_tokenizer_result: bool = True
    #: Arguments of every fake offline-probe call.
    offline_probe_calls: list[tuple[str, str]] = []
    #: Kwargs of the last companion call (cache-only or online).
    companion_last_kwargs: dict[str, Any] = {}
    #: Kwargs of the last ONLINE companion call (a later cache-only status
    #: probe overwrites ``companion_last_kwargs``).
    companion_last_download_kwargs: dict[str, Any] = {}
    companion_download_calls: int = 0
    raise_on_companion_download: BaseException | None = None

    @classmethod
    def reset(cls) -> None:
        cls.cached_path = None
        cls.download_target = None
        cls.raise_on_resolve = None
        cls.raise_on_download = None
        cls.on_download = None
        cls.last_kwargs = {}
        cls.last_download_kwargs = {}
        cls.download_calls = 0
        cls.companion_cached_path = FAKE_SNAPSHOT_DIR
        cls.companion_last_kwargs = {}
        cls.companion_last_download_kwargs = {}
        cls.companion_download_calls = 0
        cls.raise_on_companion_download = None
        cls.raise_on_companion_resolve = None
        cls.offline_tokenizer_result = True
        cls.offline_probe_calls = []


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
    if repo_id == _COMPANION_TOKENIZER_REPO:
        FakeSnapshot.companion_last_kwargs = {
            "repo_id": repo_id,
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "token": token,
            "allow_patterns": allow_patterns,
        }
        if local_files_only:
            if FakeSnapshot.raise_on_companion_resolve is not None:
                raise FakeSnapshot.raise_on_companion_resolve
            if FakeSnapshot.companion_cached_path is None:
                from huggingface_hub.errors import LocalEntryNotFoundError

                raise LocalEntryNotFoundError("companion tokenizer is not cached")
            return FakeSnapshot.companion_cached_path
        FakeSnapshot.companion_last_download_kwargs = dict(FakeSnapshot.companion_last_kwargs)
        FakeSnapshot.companion_download_calls += 1
        if FakeSnapshot.raise_on_companion_download is not None:
            raise FakeSnapshot.raise_on_companion_download
        if FakeSnapshot.companion_cached_path is None:
            FakeSnapshot.companion_cached_path = FAKE_SNAPSHOT_DIR
        return FakeSnapshot.companion_cached_path
    FakeSnapshot.last_kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "token": token,
        "allow_patterns": allow_patterns,
    }
    if local_files_only:
        if FakeSnapshot.raise_on_resolve is not None:
            raise FakeSnapshot.raise_on_resolve
        if FakeSnapshot.cached_path is None:
            # Mirror the real helper: not-in-cache surfaces as the documented
            # LocalEntryNotFoundError, the one failure the plugin may read as
            # reliable evidence of a missing snapshot.
            from huggingface_hub.errors import LocalEntryNotFoundError

            raise LocalEntryNotFoundError("snapshot is not cached locally")
        return FakeSnapshot.cached_path
    FakeSnapshot.last_download_kwargs = dict(FakeSnapshot.last_kwargs)
    FakeSnapshot.download_calls += 1
    if FakeSnapshot.raise_on_download is not None:
        raise FakeSnapshot.raise_on_download
    if FakeSnapshot.on_download is not None:
        FakeSnapshot.on_download()
    if FakeSnapshot.download_target is not None:
        FakeSnapshot.cached_path = FakeSnapshot.download_target
    return FakeSnapshot.cached_path or FAKE_SNAPSHOT_DIR


class FakeHfApi:
    """Controls the fake ``huggingface_hub.HfApi`` source metadata queries.

    A refresh re-resolves the mutable revision through ``model_info``;
    ``remote_sha`` is the commit the fake source answers with (``None``
    models a source that names no commit), and ``raise_on_model_info``
    models an unreachable or rejecting source.
    """

    remote_sha: str | None = None
    raise_on_model_info: BaseException | None = None
    model_info_calls: list[dict[str, Any]] = []
    last_token: str | None = None

    def __init__(self, token: str | None = None) -> None:
        FakeHfApi.last_token = token

    def model_info(self, repo_id: str, *, revision: str | None = None) -> Any:
        FakeHfApi.model_info_calls.append({"repo_id": repo_id, "revision": revision})
        if FakeHfApi.raise_on_model_info is not None:
            raise FakeHfApi.raise_on_model_info
        import types

        return types.SimpleNamespace(sha=FakeHfApi.remote_sha)

    @classmethod
    def reset(cls) -> None:
        cls.remote_sha = None
        cls.raise_on_model_info = None
        cls.model_info_calls = []
        cls.last_token = None


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

    from std_mlx_audio import engine as engine_module

    FakeSnapshot.reset()
    FakeHfApi.reset()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)

    def _fake_offline_probe(checkpoint_dir: str, companion_repo: str) -> bool:
        # The real probe runs transformers against the checkpoint and the
        # default cache; its own unit tests exercise that against real
        # directories. Here the canned answer keeps engine-flow tests
        # hermetic.
        FakeSnapshot.offline_probe_calls.append((checkpoint_dir, companion_repo))
        return FakeSnapshot.offline_tokenizer_result

    monkeypatch.setattr(engine_module, "offline_tokenizer_loads", _fake_offline_probe)
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
