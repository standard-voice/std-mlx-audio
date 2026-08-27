# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Artifact-lifecycle status and acquisition (protocol 1.1).

Every test runs against the injected fakes; no weights are downloaded. The
matrix covers both configured shapes (Hub preset, operator ``model_path``),
the download-policy interactions, the subfolder layout, refresh, and the
error translations.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from standard_asr import (
    ARTIFACT_MISSING,
    ARTIFACT_READY,
    ARTIFACT_UNKNOWN,
    ARTIFACTS_READY,
    ARTIFACTS_UNAVAILABLE,
)
from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactUnavailableError,
)

from std_mlx_audio import CohereAsr, Mms1BAll, WhisperTiny

from .conftest import FAKE_SNAPSHOT_DIR, FakeLoader, FakeSnapshot

PINNED = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _downloads_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each test starts from the documented default (unset = allowed) and opts
    # into the disabled state explicitly.
    monkeypatch.delenv("STANDARD_ASR_ALLOW_DOWNLOAD", raising=False)
    monkeypatch.delenv("STANDARD_ASR_MODEL_DIR", raising=False)


def _snapshot_dir(tmp_path: Path, sha: str = PINNED) -> Path:
    snapshot = tmp_path / "snapshots" / sha
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 64)
    return snapshot


def _mlx_dir(tmp_path: Path) -> Path:
    root = tmp_path / "local-mlx"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"\x00" * 16)
    return root


# --------------------------------------------------------------------------- #
# Static declaration
# --------------------------------------------------------------------------- #
def test_declared_metadata_upper_bounds() -> None:
    artifacts = WhisperTiny.declared_metadata.artifacts
    assert artifacts.acquisition_applicable is True
    assert artifacts.supports_explicit_acquisition is True
    assert artifacts.may_acquire_during_inference is True


def test_protocol_version_is_1_1() -> None:
    assert WhisperTiny.properties.protocol_version == "1.1.0"


# --------------------------------------------------------------------------- #
# Status: Hub preset
# --------------------------------------------------------------------------- #
def test_hub_cold_cache_reports_missing_and_acquirable(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    fake_loader()
    report = WhisperTiny().artifact_status()
    assert report.applicable is True
    assert report.readiness == ARTIFACTS_UNAVAILABLE
    (requirement,) = report.requirements
    assert requirement.state == ARTIFACT_MISSING
    assert requirement.required_for_inference is True
    assert requirement.can_acquire_now is True
    assert requirement.acquisition_blocker is None
    assert requirement.may_acquire_during_inference is True
    assert requirement.source_is_mutable is True


def test_hub_cold_cache_with_downloads_disabled(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_loader()
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    (requirement,) = WhisperTiny().artifact_status().requirements
    assert requirement.state == ARTIFACT_MISSING
    assert requirement.can_acquire_now is False
    assert requirement.acquisition_blocker == "downloads_disabled"
    assert requirement.may_acquire_during_inference is False


def test_hub_ready_cache_reports_location_size_and_commit(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    fake_loader()
    snapshot = _snapshot_dir(tmp_path)
    FakeSnapshot.cached_path = str(snapshot)
    report = WhisperTiny().artifact_status()
    assert report.readiness == ARTIFACTS_READY
    (requirement,) = report.requirements
    assert requirement.state == ARTIFACT_READY
    assert requirement.location == snapshot
    assert requirement.size_bytes == 66
    # The commit is read off the Hugging Face snapshots/<sha> layout.
    assert requirement.artifact_version == PINNED


def test_pinned_revision_is_immutable(fake_loader: Callable[..., FakeLoader]) -> None:
    fake_loader()
    (requirement,) = WhisperTiny(revision=PINNED).artifact_status().requirements
    assert requirement.source_is_mutable is False
    assert requirement.artifact_version == PINNED


def test_subfolder_present_is_ready_at_the_subfolder(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    fake_loader()
    snapshot = _snapshot_dir(tmp_path)
    (snapshot / "mlx-int8").mkdir()
    (snapshot / "mlx-int8" / "config.json").write_text("{}")
    (snapshot / "mlx-int8" / "model.safetensors").write_bytes(b"\x00" * 8)
    FakeSnapshot.cached_path = str(snapshot)
    (requirement,) = CohereAsr().artifact_status().requirements
    assert requirement.state == ARTIFACT_READY
    assert requirement.location == snapshot / "mlx-int8"
    # The commit is still read off the snapshots/<sha> layout, one level up.
    assert requirement.artifact_version == PINNED


def test_subfolder_absent_is_incomplete_and_acquirable(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    fake_loader()
    FakeSnapshot.cached_path = str(_snapshot_dir(tmp_path))
    (requirement,) = CohereAsr().artifact_status().requirements
    # The snapshot resolved but the preset's checkpoint subfolder is not in
    # it: this is a detectable interrupted acquisition, and reporting it as
    # incomplete keeps it repairable through pull (an online resolution
    # fetches only the files that are absent).
    assert requirement.state == "incomplete"
    assert requirement.can_acquire_now is True


def test_mms_is_one_filtered_snapshot_requirement(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The current plugin has no language-adapter selector, so MMS deliberately
    # reports ONE snapshot requirement rather than per-language adapters.
    fake_loader()
    report = Mms1BAll().artifact_status()
    (requirement,) = report.requirements
    assert requirement.artifact_id == "mlx-snapshot"


# --------------------------------------------------------------------------- #
# Status: operator model_path
# --------------------------------------------------------------------------- #
def test_model_path_missing_needs_provide_artifacts(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    fake_loader()
    report = WhisperTiny(model_path=str(tmp_path / "absent")).artifact_status()
    assert report.readiness == ARTIFACTS_UNAVAILABLE
    (requirement,) = report.requirements
    assert requirement.state == ARTIFACT_MISSING
    assert requirement.acquisition_blocker == "action_required"
    (action,) = requirement.required_actions
    assert action.kind == "provide_artifacts"


def test_model_path_ready_directory(fake_loader: Callable[..., FakeLoader], tmp_path: Path) -> None:
    fake_loader()
    local = _mlx_dir(tmp_path)
    (requirement,) = WhisperTiny(model_path=str(local)).artifact_status().requirements
    assert requirement.state == ARTIFACT_READY
    assert requirement.location == local
    assert requirement.size_bytes == 18


def test_model_path_without_checkpoint_shape_is_incomplete_with_guidance(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # The most common operator mistake (pointing one level too high) is
    # provably not a complete checkpoint -- the check already ran and
    # answered, so the state is incomplete, with a concrete next step.
    fake_loader()
    (requirement,) = WhisperTiny(model_path=str(tmp_path)).artifact_status().requirements
    assert requirement.state == "incomplete"
    assert requirement.acquisition_blocker == "action_required"
    (action,) = requirement.required_actions
    assert action.kind == "provide_artifacts"
    assert "config.json" in action.message


def test_model_path_pointing_at_a_file_is_incomplete(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    fake_loader()
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"\x00")
    (requirement,) = WhisperTiny(model_path=str(target)).artifact_status().requirements
    assert requirement.state == "incomplete"
    (action,) = requirement.required_actions
    assert "DIRECTORY" in action.message


def test_prepare_incomplete_model_path_is_unavailable(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # The guard translates the already-answered incompleteness instead of a
    # strict=False load producing fluent garbage or an opaque native failure.
    fake_loader()
    with pytest.raises(ArtifactUnavailableError) as exc_info:
        WhisperTiny(model_path=str(tmp_path)).prepare()
    assert exc_info.value.reason == "incomplete"


def test_sharded_checkpoint_missing_a_shard_is_incomplete(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # A partial shard set would load with strict=False into random weights --
    # fluent garbage. The safetensors index is the completeness authority.
    import json

    fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    index = {
        "weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
        }
    }
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(index))
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"\x00")
    FakeSnapshot.cached_path = str(snapshot)
    (requirement,) = WhisperTiny().artifact_status().requirements
    assert requirement.state == "incomplete"
    # The second shard arrives; the same snapshot is now provably complete.
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"\x00")
    (requirement,) = WhisperTiny().artifact_status().requirements
    assert requirement.state == ARTIFACT_READY


def test_mms_requires_the_base_weights_beside_adapters(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # config.json plus a single language adapter must NOT report ready: the
    # base encoder is required_snapshot_files on the preset.
    fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "adapter.fra.safetensors").write_bytes(b"\x00")
    FakeSnapshot.cached_path = str(snapshot)
    (requirement,) = Mms1BAll().artifact_status().requirements
    assert requirement.state == "incomplete"
    (snapshot / "model.safetensors").write_bytes(b"\x00")
    (requirement,) = Mms1BAll().artifact_status().requirements
    assert requirement.state == ARTIFACT_READY


def test_loader_receives_name_parts_for_model_path_too(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # The routing fix is symmetric: an operator checkpoint for a preset whose
    # repo config omits model_type must not infer its family from an
    # arbitrary directory name.
    from std_mlx_audio import ParakeetTdt06BV3

    loader = fake_loader()
    local = tmp_path / "ckpt"
    local.mkdir()
    (local / "config.json").write_text("{}")
    (local / "model.safetensors").write_bytes(b"\x00")
    ParakeetTdt06BV3(model_path=str(local)).prepare()
    parts = loader.load_calls[0]["model_name_parts"]
    assert parts and parts[0] == "parakeet"


# --------------------------------------------------------------------------- #
# Explicit acquisition and refresh
# --------------------------------------------------------------------------- #
def test_pull_acquires_and_reports_ready(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    fake_loader()
    FakeSnapshot.download_target = str(_snapshot_dir(tmp_path))
    phases: list[str] = []
    report = WhisperTiny(hf_token="hf_abc", download_root=str(tmp_path)).acquire_artifacts(
        progress=lambda event: phases.append(event.phase)
    )
    assert report.readiness == ARTIFACTS_READY
    assert FakeSnapshot.download_calls == 1
    assert FakeSnapshot.last_download_kwargs["repo_id"] == WhisperTiny.hf_repo
    assert FakeSnapshot.last_download_kwargs["local_files_only"] is False
    # download_root and the secret token reach the resolution (they were
    # discarded or never forwarded before this rollout).
    assert FakeSnapshot.last_download_kwargs["cache_dir"] == str(tmp_path)
    assert FakeSnapshot.last_download_kwargs["token"] == "hf_abc"
    # The filter is the upstream loader's own allow-patterns list.
    assert "*.safetensors" in (FakeSnapshot.last_download_kwargs["allow_patterns"] or [])
    assert phases[0] == "resolving"
    assert "transferring" in phases
    assert phases[-1] == "finalizing"
    # Artifact-only: no model load, no priming generate.
    assert FakeSnapshot.last_kwargs["local_files_only"] is True  # final status query


def test_pull_never_loads_or_primes(fake_loader: Callable[..., FakeLoader], tmp_path: Path) -> None:
    loader = fake_loader()
    FakeSnapshot.download_target = str(_snapshot_dir(tmp_path))
    WhisperTiny().acquire_artifacts()
    assert loader.load_calls == []
    assert loader.model.priming_call is None


def _gated_error() -> Exception:
    import httpx
    from huggingface_hub.errors import GatedRepoError

    response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/x"))
    return GatedRepoError("gated", response=response)


def _unauthorized_error() -> Exception:
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    response = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co/x"))
    return HfHubHTTPError("401 unauthorized", response=response)


def test_pull_gated_repo_reports_request_access(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The REAL upstream error class: name-based matching would silently stop
    # working on a rename or subclass.
    fake_loader()
    FakeSnapshot.raise_on_download = _gated_error()
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        WhisperTiny().acquire_artifacts()
    assert exc_info.value.reason == "action_required"
    (action,) = exc_info.value.required_actions
    assert action.kind == "request_access"


def test_pull_unauthenticated_repo_reports_authenticate(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    fake_loader()
    FakeSnapshot.raise_on_download = _unauthorized_error()
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        WhisperTiny().acquire_artifacts()
    assert exc_info.value.reason == "action_required"
    (action,) = exc_info.value.required_actions
    assert action.kind == "authenticate"


def test_pull_native_failure_is_failed_with_cause(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    fake_loader()
    FakeSnapshot.raise_on_download = RuntimeError("dns exploded")
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        WhisperTiny().acquire_artifacts()
    assert exc_info.value.reason == "failed"
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_pull_blocked_by_download_policy(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_loader()
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        WhisperTiny().acquire_artifacts()
    assert exc_info.value.reason == "downloads_disabled"
    assert FakeSnapshot.download_calls == 0


def test_refresh_re_resolves_a_ready_mutable_source(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    fake_loader()
    FakeSnapshot.cached_path = str(_snapshot_dir(tmp_path))
    engine = WhisperTiny()
    assert engine.acquire_artifacts().readiness == ARTIFACTS_READY
    assert FakeSnapshot.download_calls == 0  # plain pull: ready is a no-op
    engine.acquire_artifacts(refresh=True)
    assert FakeSnapshot.download_calls == 1  # refresh re-resolves the branch


def test_refresh_skips_a_pinned_revision(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    fake_loader()
    FakeSnapshot.cached_path = str(_snapshot_dir(tmp_path))
    report = WhisperTiny(revision=PINNED).acquire_artifacts(refresh=True)
    assert report.readiness == ARTIFACTS_READY
    assert FakeSnapshot.download_calls == 0


def test_pull_on_missing_model_path_raises_action_required(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    fake_loader()
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        WhisperTiny(model_path=str(tmp_path / "absent")).acquire_artifacts()
    assert exc_info.value.reason == "action_required"


# --------------------------------------------------------------------------- #
# Implicit-path translation (the loading guard)
# --------------------------------------------------------------------------- #
def test_prepare_cold_cache_downloads_disabled_is_unavailable(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_loader()
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    with pytest.raises(ArtifactUnavailableError) as exc_info:
        WhisperTiny().prepare()
    assert exc_info.value.reason == "downloads_disabled"
    assert exc_info.value.report.readiness == ARTIFACTS_UNAVAILABLE


def test_prepare_missing_model_path_is_unavailable(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    fake_loader()
    with pytest.raises(ArtifactUnavailableError) as exc_info:
        WhisperTiny(model_path=str(tmp_path / "absent")).prepare()
    assert exc_info.value.reason == "missing"


def test_prepare_cold_cache_resolution_failure_is_acquisition_error(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    fake_loader()
    FakeSnapshot.raise_on_download = OSError("dns exploded")
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        WhisperTiny().prepare()
    assert exc_info.value.reason == "failed"


def test_tree_size_returns_none_on_walk_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from std_mlx_audio import _artifacts

    def _raise(self: Path, pattern: str) -> object:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "rglob", _raise)
    assert _artifacts._tree_size_bytes(tmp_path) is None


def test_acquire_hook_ignores_a_target_set_without_the_hub_id(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    from standard_asr.contract.artifacts import ArtifactContext

    fake_loader()
    WhisperTiny()._acquire_artifacts(ArtifactContext(mode="batch"), (), False, None)
    assert FakeSnapshot.download_calls == 0


def test_status_unknown_when_resolution_stack_unavailable(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    from std_mlx_audio import _artifacts

    fake_loader()

    def _boom(*_a: object, **_k: object) -> object:
        raise ImportError("no huggingface_hub")

    monkeypatch.setattr(_artifacts, "snapshot", _boom)
    (requirement,) = WhisperTiny().artifact_status().requirements
    assert requirement.state == ARTIFACT_UNKNOWN


# --------------------------------------------------------------------------- #
# Round-1 review regressions
# --------------------------------------------------------------------------- #
def test_loader_receives_repo_derived_model_name_parts(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # Upstream infers the model family from the LAST path component when
    # config.json omits model_type (NeMo-format repos): a local snapshot path
    # would feed it the commit hash, so the engine must hand it the
    # repo-derived name parts explicitly or Parakeet-lineage presets brick.
    from std_mlx_audio import ParakeetTdt06BV3

    loader = fake_loader()
    FakeSnapshot.cached_path = FAKE_SNAPSHOT_DIR
    ParakeetTdt06BV3().prepare()
    parts = loader.load_calls[0]["model_name_parts"]
    assert parts and parts[0] == "parakeet"


def test_partial_snapshot_reports_incomplete_and_pull_repairs(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # Upstream resolution returns an existing snapshot directory WITHOUT
    # verifying its contents; readiness must come from the completeness check,
    # and pull must be able to repair the interrupted download.
    fake_loader()
    partial = tmp_path / "snapshots" / PINNED
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}")  # weights never arrived
    FakeSnapshot.cached_path = str(partial)
    report = WhisperTiny().artifact_status()
    assert report.readiness == ARTIFACTS_UNAVAILABLE
    (requirement,) = report.requirements
    assert requirement.state == "incomplete"
    assert requirement.can_acquire_now is True

    def _complete_download() -> None:
        (partial / "model.safetensors").write_bytes(b"\x00" * 8)

    FakeSnapshot.on_download = _complete_download
    repaired = WhisperTiny().acquire_artifacts()
    assert repaired.readiness == ARTIFACTS_READY
    assert FakeSnapshot.download_calls == 1


def test_refresh_respects_engine_local_files_only(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # local_files_only is the engine's own offline policy; the core template
    # only sees the global toggle, so the hook must self-gate the refresh.
    fake_loader()
    FakeSnapshot.cached_path = str(_snapshot_dir(tmp_path))
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        WhisperTiny(local_files_only=True).acquire_artifacts(refresh=True)
    assert exc_info.value.reason == "downloads_disabled"
    assert FakeSnapshot.download_calls == 0


def test_unreadable_cache_reports_unknown_not_missing(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # A permission failure is not evidence of absence: claiming missing would
    # tell an offline operator to enable downloads for a permissions bug.
    fake_loader()
    FakeSnapshot.raise_on_resolve = PermissionError("refs unreadable")
    (requirement,) = WhisperTiny().artifact_status().requirements
    assert requirement.state == ARTIFACT_UNKNOWN
    assert requirement.acquisition_blocker is None


def test_warm_cache_with_branch_revision_reports_resolved_commit(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # The resolved commit wins over the configured mutable reference, so a
    # refresh that moves a branch is observable through artifact_version.
    fake_loader()
    FakeSnapshot.cached_path = str(_snapshot_dir(tmp_path))
    (requirement,) = WhisperTiny(revision="main").artifact_status().requirements
    assert requirement.source_is_mutable is True
    assert requirement.artifact_version == PINNED


def test_first_use_gated_repo_reports_request_access(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The reason comes from the discovered blocker, not from which code path
    # noticed it: the IMPLICIT first-use resolution carries the same action.
    fake_loader()
    FakeSnapshot.raise_on_download = _gated_error()
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        WhisperTiny().prepare()
    assert exc_info.value.reason == "action_required"
    (action,) = exc_info.value.required_actions
    assert action.kind == "request_access"


def test_transcribe_propagates_artifact_error_unwrapped(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The R7 exemption end to end: an availability failure inside the batch
    # pipeline reaches the caller as the artifact error, never wrapped into
    # TranscriptionError.
    import numpy as np

    fake_loader()
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    with pytest.raises(ArtifactUnavailableError):
        WhisperTiny().transcribe((np.zeros(16000, dtype=np.float32), 16000))


def test_streaming_session_emits_artifact_unavailable_terminal(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The streaming mapping end to end: the session's producer translates the
    # availability failure into the dedicated terminal code, not engine_error.
    from standard_asr import SyncSession
    from standard_asr.audio.format import AudioFormat

    fake_loader()
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    engine = WhisperTiny()
    session = engine.start_transcription(
        audio_format=AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)
    )
    with SyncSession(session) as sync:
        events = list(sync)
    (error,) = [event for event in events if event.type == "error"]
    assert error.code == "artifact_unavailable"
    assert error.recoverable is False


def test_error_translation_degrades_without_hub_errors_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from std_mlx_audio._artifacts import _is_local_entry_not_found, raise_for_gated_source

    real_import = builtins.__import__

    def _import(name: str, *a: object, **k: object) -> object:
        if name.startswith("huggingface_hub"):
            raise ImportError("no huggingface_hub")
        return real_import(name, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _import)
    assert _is_local_entry_not_found(FileNotFoundError("x")) is False
    assert raise_for_gated_source(RuntimeError("x"), "repo") is None
