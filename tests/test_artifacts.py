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

from std_mlx_audio import (
    Canary1BV2,
    CohereAsr,
    FireRedAsr2Aed,
    Mms1BAll,
    Qwen3Asr06B,
    VibeVoiceAsr,
    WhisperTiny,
)

from .conftest import FAKE_SNAPSHOT_DIR, FakeHfApi, FakeLoader, FakeSnapshot

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
    # The Whisper presets declare the feature-extractor config as snapshot
    # closure; the tokenizer file is a harmless extra.
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "preprocessor_config.json").write_text("{}")
    return snapshot


def _mlx_dir(tmp_path: Path) -> Path:
    root = tmp_path / "local-mlx"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"\x00" * 16)
    (root / "tokenizer.json").write_text("{}")
    (root / "preprocessor_config.json").write_text("{}")
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
    assert requirement.size_bytes == 70
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
    (snapshot / "mlx-int8" / "tokenizer.model").write_text("{}")
    (snapshot / "mlx-int8" / "tokenizer_config.json").write_text("{}")
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
    assert requirement.size_bytes == 22


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
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "preprocessor_config.json").write_text("{}")
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
    (snapshot / "vocab.json").write_text("{}")
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
    FakeHfApi.remote_sha = PINNED
    engine = WhisperTiny(hf_token="hf_abc")
    assert engine.acquire_artifacts().readiness == ARTIFACTS_READY
    assert FakeSnapshot.download_calls == 0  # plain pull: ready is a no-op
    assert FakeHfApi.model_info_calls == []  # plain pull never queries the source
    engine.acquire_artifacts(refresh=True)
    assert FakeSnapshot.download_calls == 1  # refresh re-resolves the branch
    # The re-resolution evidence: one metadata query against the preset's
    # repo, with the engine's credentials.
    (call,) = FakeHfApi.model_info_calls
    assert call == {"repo_id": "openai/whisper-tiny", "revision": None}
    assert FakeHfApi.last_token == "hf_abc"


def test_refresh_skips_a_pinned_revision(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    fake_loader()
    FakeSnapshot.cached_path = str(_snapshot_dir(tmp_path))
    report = WhisperTiny(revision=PINNED).acquire_artifacts(refresh=True)
    assert report.readiness == ARTIFACTS_READY
    assert FakeSnapshot.download_calls == 0
    assert FakeHfApi.model_info_calls == []  # a pinned commit needs no query


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
        (partial / "tokenizer.json").write_text("{}")
        (partial / "preprocessor_config.json").write_text("{}")

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
    assert FakeHfApi.model_info_calls == []  # gated before the source query


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


# --------------------------------------------------------------------------- #
# Round-3 review regressions: shard closures and refresh evidence
# --------------------------------------------------------------------------- #
def test_partial_shard_set_without_index_is_incomplete(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # An interrupted download can hold shard files while the index has not
    # arrived; each shard names its own closure (-NNNNN-of-NNNNN), and the
    # strict=False loader would load the fragment into fluent garbage.
    fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "preprocessor_config.json").write_text("{}")
    (snapshot / "model-00001-of-00003.safetensors").write_bytes(b"\x00" * 8)
    FakeSnapshot.cached_path = str(snapshot)
    (requirement,) = WhisperTiny().artifact_status().requirements
    assert requirement.state == "incomplete"

    (snapshot / "model-00002-of-00003.safetensors").write_bytes(b"\x00" * 8)
    (snapshot / "model-00003-of-00003.safetensors").write_bytes(b"\x00" * 8)
    (requirement,) = WhisperTiny().artifact_status().requirements
    assert requirement.state == ARTIFACT_READY


def test_whisper_missing_processor_config_is_incomplete(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # The Whisper backend needs the WhisperProcessor beside the weights:
    # without it the loader only WARNS and the first generate fails
    # ("Processor not found"). The single point of failure is the
    # feature-extractor config (the OpenAI repos ship both tokenizer
    # layouts), and the snapshot axis gates only the Hub snapshot: a
    # model_path may use any layout the flexible loader accepts, so it
    # stays ready without the file (round-9 review).
    fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 8)
    (snapshot / "tokenizer.json").write_text("{}")
    FakeSnapshot.cached_path = str(snapshot)
    (requirement,) = WhisperTiny().artifact_status().requirements
    assert requirement.state == "incomplete"

    local = tmp_path / "local"
    local.mkdir()
    (local / "config.json").write_text("{}")
    (local / "model.safetensors").write_bytes(b"\x00" * 8)
    (requirement,) = WhisperTiny(model_path=str(local)).artifact_status().requirements
    assert requirement.state == ARTIFACT_READY

    (snapshot / "preprocessor_config.json").write_text("{}")
    (requirement,) = WhisperTiny().artifact_status().requirements
    assert requirement.state == ARTIFACT_READY


def test_unreadable_shard_index_is_incomplete(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # An index that cannot be parsed proves nothing; it must never count as
    # a complete checkpoint.
    fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "preprocessor_config.json").write_text("{}")
    (snapshot / "model.safetensors.index.json").write_text("{not json")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 8)
    FakeSnapshot.cached_path = str(snapshot)
    (requirement,) = WhisperTiny().artifact_status().requirements
    assert requirement.state == "incomplete"


def test_model_path_honors_preset_required_files(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # A model_path still loads through the preset's family: an MMS directory
    # with an adapter but no base weights would reach the strict=False loader
    # as a fragment, so the preset's required files apply to it too.
    fake_loader()
    root = tmp_path / "mms"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "adapter.eng.safetensors").write_bytes(b"\x00" * 8)
    (requirement,) = Mms1BAll(model_path=str(root)).artifact_status().requirements
    assert requirement.state == "incomplete"

    (root / "model.safetensors").write_bytes(b"\x00" * 8)
    (root / "vocab.json").write_text("{}")
    (requirement,) = Mms1BAll(model_path=str(root)).artifact_status().requirements
    assert requirement.state == ARTIFACT_READY


def test_implicit_load_rechecks_completeness_after_resolution(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # Resolution success is not completeness: with the remote unreachable the
    # hub client silently falls back to the local cache, so an allowed
    # implicit acquisition can return the same incomplete directory status
    # just reported. The loader must never receive that fragment.
    loader = fake_loader()
    partial = tmp_path / "snapshots" / PINNED
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}")  # weights never arrived
    FakeSnapshot.cached_path = str(partial)
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        WhisperTiny().prepare()
    assert exc_info.value.reason == "failed"
    assert "complete" in str(exc_info.value)
    assert loader.load_calls == []  # the fragment never reached the loader


def test_refresh_fails_when_the_source_is_unreachable(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # The downloader silently falls back to the local cache when the remote
    # is unreachable, so refresh must fail without re-resolution evidence
    # instead of reporting the stale cache as fresh (spec AR.4).
    fake_loader()
    FakeSnapshot.cached_path = str(_snapshot_dir(tmp_path))
    FakeHfApi.raise_on_model_info = ConnectionError("network is down")
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        WhisperTiny().acquire_artifacts(refresh=True)
    assert exc_info.value.reason == "failed"
    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert FakeSnapshot.download_calls == 0  # fail fast, before any transfer


def test_refresh_detects_a_stale_cache_fallback(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # The source re-resolved to a new commit, but the download resolved the
    # old snapshot (the mid-transfer fallback): success here would claim
    # freshness the source never confirmed.
    fake_loader()
    FakeSnapshot.cached_path = str(_snapshot_dir(tmp_path))
    FakeHfApi.remote_sha = "f" * 40
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        WhisperTiny().acquire_artifacts(refresh=True)
    assert exc_info.value.reason == "failed"
    assert "fell back" in str(exc_info.value)


def test_refresh_gated_source_query_reports_request_access(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # An access rejection during the metadata query carries the same
    # discovered action as one during the transfer.
    import httpx
    from huggingface_hub.errors import GatedRepoError

    fake_loader()
    FakeSnapshot.cached_path = str(_snapshot_dir(tmp_path))
    response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/x"))
    FakeHfApi.raise_on_model_info = GatedRepoError("gated", response=response)
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        WhisperTiny().acquire_artifacts(refresh=True)
    assert exc_info.value.reason == "action_required"
    (action,) = exc_info.value.required_actions
    assert action.kind == "request_access"


def test_refresh_fails_when_the_source_names_no_commit(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    fake_loader()
    FakeSnapshot.cached_path = str(_snapshot_dir(tmp_path))
    FakeHfApi.remote_sha = None
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        WhisperTiny().acquire_artifacts(refresh=True)
    assert exc_info.value.reason == "failed"
    assert "commit" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Round-8 review regressions: per-family inference closures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("preset_name", "family_files"),
    [
        # Decode falls back to numeric token ids without the bpe model, and
        # feature normalization is silently skipped without am.mvn.
        ("SenseVoiceSmall", ("chn_jpn_yue_eng_ko_spectok.bpe.model", "am.mvn")),
        # Missing dict.txt yields a silently EMPTY transcript; missing
        # cmvn.json silently skips normalization.
        ("FireRedAsr2Aed", ("dict.txt", "cmvn.json")),
        # The upstream hook swallows the tokenizer load failure and decode
        # falls back to per-id characters.
        ("MoonshineTiny", ("tokenizer.json",)),
        # The tokenizer is loaded optionally and its absence raises only at
        # the first generate.
        ("Canary1BV2", ("tokenizer.model",)),
        # The hook's AutoTokenizer + WhisperFeatureExtractor calls each
        # break without their single-point files (per-file ablation).
        (
            "Qwen3Asr06B",
            ("merges.txt", "preprocessor_config.json", "tokenizer_config.json", "vocab.json"),
        ),
        # AutoTokenizer breaks without the fast serialization (no slow
        # fallback in this repo).
        ("GlmAsrNano", ("tokenizer.json",)),
        # The realtime tokenizer reads tekken.json by fixed name and raises
        # at load without it.
        ("VoxtralRealtime4B", ("tekken.json",)),
    ],
)
def test_family_inference_closure_gates_ready(
    fake_loader: Callable[..., FakeLoader],
    tmp_path: Path,
    preset_name: str,
    family_files: tuple[str, ...],
) -> None:
    # A snapshot holding only config + weights must NOT report ready when a
    # preset-declared closure file is absent: silent-corruption files
    # (checkpoint axis) and repo-layout single points of failure (snapshot
    # axis) both gate the Hub snapshot (rounds 8-9; every listed file is
    # verified against the installed loader AND the preset's repo).
    import std_mlx_audio

    preset = getattr(std_mlx_audio, preset_name)
    fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 8)
    FakeSnapshot.cached_path = str(snapshot)
    (requirement,) = preset().artifact_status().requirements
    assert requirement.state == "incomplete"

    for name in family_files:
        (snapshot / name).write_text("{}")
    (requirement,) = preset().artifact_status().requirements
    assert requirement.state == ARTIFACT_READY


def test_snapshot_filter_includes_the_mvn_gap(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # SenseVoice's am.mvn is load-bearing (normalization stats, no config
    # fallback) but the upstream default allow patterns never fetch it; the
    # plugin's snapshot filter must close that gap or the required file
    # could never arrive.
    fake_loader()
    with pytest.raises(ArtifactUnavailableError):
        # Any resolution records the kwargs; the cold cache then raises.
        from std_mlx_audio import SenseVoiceSmall

        SenseVoiceSmall(local_files_only=True).prepare()
    patterns = FakeSnapshot.last_kwargs["allow_patterns"]
    assert "*.mvn" in patterns
    assert "*.model" in patterns  # the upstream defaults are still present


# --------------------------------------------------------------------------- #
# Round-9 review regressions: the two closure axes and the companion tokenizer
# --------------------------------------------------------------------------- #
def test_canary_model_path_accepts_the_loader_alternatives(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # The Canary hook reads tokenizer.model, tokens.txt, OR a
    # config-embedded tokenizer. Requiring the Hub repo's exact filename
    # rejected valid local checkpoints (round-9 review), so the snapshot
    # axis must not gate a model_path.
    loader = fake_loader()
    root = tmp_path / "canary"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"\x00" * 8)
    (root / "tokens.txt").write_text("a 1\n")
    engine = Canary1BV2(model_path=str(root))
    (requirement,) = engine.artifact_status().requirements
    assert requirement.state == ARTIFACT_READY
    engine.prepare()
    assert loader.load_calls[0]["model_path"] == str(root)


def test_firered_dead_spm_file_is_not_required(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # The loader assigns train_bpe1000.model to a field no decode path ever
    # reads (dead upstream code): requiring it rejected working checkpoints
    # for a file that changes nothing (round-9 review).
    fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    for name in ("config.json", "dict.txt", "cmvn.json"):
        (snapshot / name).write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 8)
    FakeSnapshot.cached_path = str(snapshot)
    (requirement,) = FireRedAsr2Aed().artifact_status().requirements
    assert requirement.state == ARTIFACT_READY


def test_cohere_missing_tokenizer_config_gates_a_model_path(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # Without tokenizer_config.json the upstream tokenizer silently falls
    # back to an EMPTY additional_special_tokens list, so the 200-plus
    # language, task, and speaker tags leak into the transcript unstripped
    # (round-9 review): the checkpoint axis gates a model_path too.
    fake_loader()
    local = tmp_path / "cohere"
    local.mkdir()
    (local / "config.json").write_text("{}")
    (local / "model.safetensors").write_bytes(b"\x00" * 8)
    (local / "tokenizer.model").write_text("{}")
    (requirement,) = CohereAsr(model_path=str(local)).artifact_status().requirements
    assert requirement.state == "incomplete"

    (local / "tokenizer_config.json").write_text("{}")
    (requirement,) = CohereAsr(model_path=str(local)).artifact_status().requirements
    assert requirement.state == ARTIFACT_READY


def test_loud_family_fragment_reports_incomplete_and_plain_pull_repairs(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # Round-9 review: with an empty closure, an interrupted snapshot of a
    # family whose loader fails loudly still reported READY, the load
    # failure was misclassified as an engine fault, and plain pull was a
    # no-op (only refresh could repair). With the ablation-verified closure
    # declared, status is honest and plain pull completes the fragment.
    fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 8)
    for name in ("merges.txt", "preprocessor_config.json", "tokenizer_config.json"):
        (snapshot / name).write_text("{}")
    # vocab.json never arrived.
    FakeSnapshot.cached_path = str(snapshot)
    engine = Qwen3Asr06B()
    (requirement,) = engine.artifact_status().requirements
    assert requirement.state == "incomplete"
    assert requirement.can_acquire_now is True

    FakeSnapshot.on_download = lambda: (snapshot / "vocab.json").write_text("{}")
    report = engine.acquire_artifacts()
    assert report.readiness == ARTIFACTS_READY


def test_vibevoice_cold_companion_degrades_status_and_pull_fetches_it(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # The checkpoint repo ships no tokenizer files; the upstream hook
    # Hub-fetches Qwen/Qwen2.5-7B at load time. A complete snapshot without
    # the cached companion is NOT offline-ready, and pull must acquire the
    # companion into the DEFAULT cache -- the upstream from_pretrained call
    # reads only there (round-9 review).
    fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 8)
    FakeSnapshot.cached_path = str(snapshot)
    FakeSnapshot.companion_cached_path = None
    engine = VibeVoiceAsr()
    (requirement,) = engine.artifact_status().requirements
    assert requirement.state == "incomplete"
    assert requirement.can_acquire_now is True

    report = engine.acquire_artifacts()
    assert report.readiness == ARTIFACTS_READY
    assert FakeSnapshot.companion_download_calls == 1
    assert FakeSnapshot.companion_last_download_kwargs["cache_dir"] is None
    assert FakeSnapshot.companion_last_download_kwargs["token"] is None
    patterns = FakeSnapshot.companion_last_download_kwargs["allow_patterns"] or []
    assert "tokenizer.json" in patterns
    assert "tokenizer_config.json" in patterns
    # Loading by REPO ID resolves AutoConfig first, so the closure must
    # carry config.json (round-10 review, verified against the Hub).
    assert "config.json" in patterns


def test_vibevoice_companion_needs_the_model_config(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # A companion cache holding the tokenizer but not config.json is NOT
    # loadable by repo id: transformers resolves AutoConfig first, so an
    # offline load fails and an online load silently fetches the file
    # (round-10 review, verified end to end against the Hub). The probe
    # must not report such a cache as warm.
    fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 8)
    FakeSnapshot.cached_path = str(snapshot)
    companion = tmp_path / "companion"
    companion.mkdir()
    (companion / "tokenizer.json").write_text("{}")
    FakeSnapshot.companion_cached_path = str(companion)
    (requirement,) = VibeVoiceAsr().artifact_status().requirements
    assert requirement.state == "incomplete"

    (companion / "config.json").write_text("{}")
    (requirement,) = VibeVoiceAsr().artifact_status().requirements
    assert requirement.state == ARTIFACT_READY


def test_vibevoice_pull_translates_a_gated_companion(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # A companion rejection during pull carries the discovered action, the
    # same way the main snapshot's gated translation does.
    fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 8)
    FakeSnapshot.cached_path = str(snapshot)
    FakeSnapshot.companion_cached_path = None
    FakeSnapshot.raise_on_companion_download = _gated_error()
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        VibeVoiceAsr().acquire_artifacts()
    assert exc_info.value.reason == "action_required"

    # A plain network failure propagates for the template to wrap.
    FakeSnapshot.raise_on_companion_download = OSError("offline")
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        VibeVoiceAsr().acquire_artifacts()
    assert exc_info.value.reason == "failed"
    assert isinstance(exc_info.value.__cause__, OSError)


def test_vibevoice_cold_companion_with_downloads_disabled_refuses_load(
    fake_loader: Callable[..., FakeLoader],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Loading would let the upstream hook fetch the tokenizer PAST the
    # no-download policy (spec AR.9); the engine must refuse loudly first.
    loader = fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 8)
    FakeSnapshot.cached_path = str(snapshot)
    FakeSnapshot.companion_cached_path = None
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    engine = VibeVoiceAsr()
    (requirement,) = engine.artifact_status().requirements
    assert requirement.state == "incomplete"
    assert requirement.acquisition_blocker == "downloads_disabled"
    with pytest.raises(ArtifactUnavailableError) as exc_info:
        engine.prepare()
    assert exc_info.value.reason == "downloads_disabled"
    assert loader.load_calls == []


def test_vibevoice_model_path_companion_policy(
    fake_loader: Callable[..., FakeLoader],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A model_path checkpoint is complete on disk, but the load still needs
    # the companion tokenizer: with downloads disabled and a cold cache the
    # engine refuses (naming the tokenizer, not the checkpoint); with
    # downloads allowed the requirement is incomplete AND acquirable -- the
    # companion is the one part the plugin can acquire for an operator
    # path, so pull fetches exactly it (round-10 review), and a direct
    # load pre-fetches it through the same classified path.
    loader = fake_loader()
    local = tmp_path / "vibevoice"
    local.mkdir()
    (local / "config.json").write_text("{}")
    (local / "model.safetensors").write_bytes(b"\x00" * 8)
    FakeSnapshot.companion_cached_path = None
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    engine = VibeVoiceAsr(model_path=str(local))
    (requirement,) = engine.artifact_status().requirements
    assert requirement.state == "incomplete"
    assert requirement.acquisition_blocker == "downloads_disabled"
    with pytest.raises(ArtifactUnavailableError) as exc_info:
        engine.prepare()
    assert "tokenizer" in str(exc_info.value)
    assert loader.load_calls == []

    monkeypatch.delenv("STANDARD_ASR_ALLOW_DOWNLOAD")
    engine = VibeVoiceAsr(model_path=str(local))
    (requirement,) = engine.artifact_status().requirements
    assert requirement.state == "incomplete"
    assert requirement.can_acquire_now is True
    assert requirement.may_acquire_during_inference is True
    phases: list[str] = []
    report = engine.acquire_artifacts(progress=lambda event: phases.append(event.phase))
    assert report.readiness == ARTIFACTS_READY
    assert "transferring" in phases
    assert FakeSnapshot.companion_download_calls == 1
    assert FakeSnapshot.download_calls == 0  # the checkpoint is not the plugin's
    # Ready is not "inference cannot acquire": the upstream loader still
    # addresses the companion by unpinned repo id with no offline flag, so
    # while downloads are permitted a load may revalidate and fetch it
    # (round-11 review; mirrors the Hub branch's policy narrowing).
    (requirement,) = report.requirements
    assert requirement.state == ARTIFACT_READY
    assert requirement.can_acquire_now is False
    assert requirement.may_acquire_during_inference is True

    engine = VibeVoiceAsr(model_path=str(local))
    engine.prepare()  # the warm companion is not re-fetched
    assert FakeSnapshot.companion_download_calls == 1
    assert loader.load_calls[0]["model_path"] == str(local)

    # With downloads disabled and the companion warm, the requirement is
    # ready and the effective may_acquire_during_inference narrows to
    # False, like every other policy-narrowed requirement.
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    (requirement,) = VibeVoiceAsr(model_path=str(local)).artifact_status().requirements
    assert requirement.state == ARTIFACT_READY
    assert requirement.may_acquire_during_inference is False
    monkeypatch.delenv("STANDARD_ASR_ALLOW_DOWNLOAD")

    # A gated companion rejection during the model_path pull carries the
    # discovered action, like every other gated translation; a plain
    # network failure propagates for the template to wrap.
    FakeSnapshot.companion_cached_path = None
    FakeSnapshot.raise_on_companion_download = _gated_error()
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        VibeVoiceAsr(model_path=str(local)).acquire_artifacts()
    assert exc_info.value.reason == "action_required"
    FakeSnapshot.raise_on_companion_download = OSError("offline")
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        VibeVoiceAsr(model_path=str(local)).acquire_artifacts()
    assert exc_info.value.reason == "failed"
    assert isinstance(exc_info.value.__cause__, OSError)


def test_vibevoice_bundled_tokenizer_needs_no_companion(
    fake_loader: Callable[..., FakeLoader],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The upstream hook tries the checkpoint directory FIRST and falls
    # back to the companion repo only when that fails, so a checkpoint
    # bundling a loadable tokenizer is self-contained: no companion
    # requirement, no fetch, and no acquisition possible during inference
    # (round-12 review; the accepted layouts are ablation-proven against
    # the installed transformers).
    loader = fake_loader()
    local = tmp_path / "vibevoice"
    local.mkdir()
    (local / "config.json").write_text("{}")
    (local / "model.safetensors").write_bytes(b"\x00" * 8)
    (local / "tokenizer.json").write_text("{}")
    FakeSnapshot.companion_cached_path = None
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    engine = VibeVoiceAsr(model_path=str(local))
    (requirement,) = engine.artifact_status().requirements
    assert requirement.state == ARTIFACT_READY
    engine.prepare()
    assert FakeSnapshot.companion_download_calls == 0
    assert loader.load_calls[0]["model_path"] == str(local)

    monkeypatch.delenv("STANDARD_ASR_ALLOW_DOWNLOAD")
    (requirement,) = VibeVoiceAsr(model_path=str(local)).artifact_status().requirements
    assert requirement.state == ARTIFACT_READY
    assert requirement.may_acquire_during_inference is False

    # A bare vocab + merges pair without the tokenizer config provably
    # FAILS to load beside the checkpoint's own config, so it does not
    # count as bundled and the companion is still required.
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "config.json").write_text("{}")
    (bare / "model.safetensors").write_bytes(b"\x00" * 8)
    (bare / "vocab.json").write_text("{}")
    (bare / "merges.txt").write_text("")
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    (requirement,) = VibeVoiceAsr(model_path=str(bare)).artifact_status().requirements
    assert requirement.state == "incomplete"
    assert requirement.acquisition_blocker == "downloads_disabled"


def test_vibevoice_bundled_hub_snapshot_needs_no_companion(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # The same self-containment applies to a Hub snapshot: if the repo
    # ever ships tokenizer files, the load stays local and a cold
    # companion cache must not degrade readiness.
    fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 8)
    (snapshot / "tokenizer.json").write_text("{}")
    FakeSnapshot.cached_path = str(snapshot)
    FakeSnapshot.companion_cached_path = None
    (requirement,) = VibeVoiceAsr().artifact_status().requirements
    assert requirement.state == ARTIFACT_READY


def test_acquire_companion_applies_the_download_policy_at_entry(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Spec AR.3 applies the download toggle before the transfer; the
    # preflight that made the local target runnable is a separate earlier
    # query, so the hook entry re-applies the policy like acquire() does
    # (round-12 review).
    from std_mlx_audio._artifacts import acquire_companion

    fake_loader()
    monkeypatch.setenv("STANDARD_ASR_ALLOW_DOWNLOAD", "0")
    config = VibeVoiceAsr(model_path="/nonexistent").config
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        acquire_companion(config, "Qwen/Qwen2.5-7B", None)
    assert exc_info.value.reason == "downloads_disabled"
    assert FakeSnapshot.companion_download_calls == 0


def test_vibevoice_implicit_load_prefetches_the_companion(
    fake_loader: Callable[..., FakeLoader], tmp_path: Path
) -> None:
    # On the allowed implicit path the plugin fetches the companion itself,
    # so the transfer is classified (and a failure surfaces as a failed
    # acquisition) instead of happening inside the upstream hook.
    loader = fake_loader()
    snapshot = tmp_path / "snapshots" / PINNED
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\x00" * 8)
    FakeSnapshot.cached_path = str(snapshot)
    FakeSnapshot.companion_cached_path = None
    VibeVoiceAsr().prepare()
    assert FakeSnapshot.companion_download_calls == 1
    assert loader.load_calls  # the load followed the fetch

    FakeSnapshot.reset()
    FakeSnapshot.cached_path = str(snapshot)
    FakeSnapshot.companion_cached_path = None
    FakeSnapshot.raise_on_companion_download = OSError("offline")
    with pytest.raises(ArtifactAcquisitionError) as exc_info:
        VibeVoiceAsr().prepare()
    assert exc_info.value.reason == "failed"
    assert "companion" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, OSError)


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
