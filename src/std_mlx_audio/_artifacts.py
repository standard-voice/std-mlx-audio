# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Inference-artifact status and acquisition for the mlx-audio engine.

The engine's one logical requirement is the preset's MLX snapshot (weights,
config, tokenizer -- filtered by the upstream allow patterns). Resolution is
plugin-side: the engine snapshot-downloads through ``huggingface_hub`` itself
and hands ``mlx_audio.stt.load`` a LOCAL directory, because the upstream
loader accepts no cache root, token, or offline flag (it silently swallows
unknown kwargs). Doing it here is what makes ``download_root``, ``hf_token``,
and ``local_files_only`` real controls instead of declared-but-inert config.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from standard_asr.contract.artifacts import (
    ARTIFACT_ACTION_AUTHENTICATE,
    ARTIFACT_ACTION_PROVIDE_ARTIFACTS,
    ARTIFACT_ACTION_REQUEST_ACCESS,
    ARTIFACT_BLOCKER_ACTION_REQUIRED,
    ARTIFACT_BLOCKER_DOWNLOADS_DISABLED,
    ARTIFACT_INCOMPLETE,
    ARTIFACT_MISSING,
    ARTIFACT_PROGRESS_TRANSFERRING,
    ARTIFACT_READY,
    ARTIFACT_UNKNOWN,
    ArtifactAction,
    ArtifactProgress,
    ArtifactProgressCallback,
    ArtifactReport,
    ArtifactRequirement,
)
from standard_asr.contract.exceptions import ArtifactAcquisitionError
from standard_asr.runtime.downloads import allow_downloads, resolve_download_root

if TYPE_CHECKING:
    from ._config import MlxAudioConfig

#: The engine's single logical requirement id for a Hub-sourced snapshot.
HUB_ARTIFACT_ID = "mlx-snapshot"
#: The requirement id when an operator-provided ``model_path`` overrides it.
LOCAL_ARTIFACT_ID = "mlx-local-path"

#: A full commit hash pins the snapshot; anything else is a mutable reference.
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _is_local_entry_not_found(exc: BaseException) -> bool:
    """Return whether a resolution failure is the documented cache miss.

    Args:
        exc: The failure raised by a cache-only resolution.

    Returns:
        ``True`` only for the upstream ``LocalEntryNotFoundError`` class.
    """
    try:
        from huggingface_hub.errors import (  # pyright: ignore[reportMissingModuleSource]
            LocalEntryNotFoundError,
        )
    except Exception:
        return False
    return isinstance(exc, LocalEntryNotFoundError)


def _looks_like_checkpoint(root: Path, required_files: tuple[str, ...] = ()) -> bool:
    """Return whether a directory is a provably complete MLX checkpoint.

    Resolution success only proves the snapshot directory exists; upstream
    documents that it cannot verify the files inside, so completeness is this
    plugin's own check. A sharded checkpoint is complete only when EVERY file
    in its safetensors index is present: the upstream loader globs whatever
    shards exist and loads them with ``strict=False``, so a missing shard
    would silently leave parameters at random init -- fluent garbage, the
    cardinal sin.

    Args:
        root: Candidate checkpoint directory.
        required_files: Extra preset-declared files that must exist (for
            example the MMS base weights beside its per-language adapters).

    Returns:
        ``True`` when the config, the index closure, and every required file
        are present.
    """
    if not (root / "config.json").is_file():
        return False
    for name in required_files:
        if not (root / name).is_file():
            return False
    index = root / "model.safetensors.index.json"
    if index.is_file():
        import json

        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
            shards = {str(value) for value in weight_map.values()}
        except Exception:
            # An unreadable index cannot prove completeness.
            return False
        return bool(shards) and all((root / shard).is_file() for shard in shards)
    # Upstream load_weights consumes safetensors and npz only.
    return any(root.glob("*.safetensors")) or any(root.glob("*.npz"))


def normalized_model_path(model_path: str) -> Path:
    """Return the operator ``model_path`` in its one canonical absolute form.

    Status inspection and the loader MUST agree on this form: expanding only
    on the status side would report a tilde path as ready while the loader
    receives the raw string.

    Args:
        model_path: The configured local checkpoint path.

    Returns:
        The expanded, resolved path.
    """
    return Path(model_path).expanduser().resolve()


def _revision_is_pinned(revision: str | None) -> bool:
    """Return whether a configured revision is an immutable commit hash.

    Args:
        revision: Configured Hugging Face revision, or ``None``.

    Returns:
        ``True`` only for a full 40-hex commit hash.
    """
    return revision is not None and _COMMIT_SHA.fullmatch(revision) is not None


def _tree_size_bytes(root: Path) -> int | None:
    """Sum regular-file sizes under a resolved snapshot root.

    Args:
        root: Resolved artifact directory.

    Returns:
        The logical size in bytes, or ``None`` when the walk fails.
    """
    try:
        return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
    except OSError:
        return None


def _allow_patterns() -> list[str]:
    """Return the upstream loader's snapshot filter, imported so it never drifts.

    Returns:
        The mlx-audio ``DEFAULT_ALLOW_PATTERNS`` list.
    """
    from mlx_audio.utils import (  # pyright: ignore[reportMissingImports]
        DEFAULT_ALLOW_PATTERNS,
    )

    return DEFAULT_ALLOW_PATTERNS


def snapshot(config: MlxAudioConfig, hf_repo: str, *, local_files_only: bool) -> Path:
    """Resolve or acquire the preset's filtered snapshot.

    Args:
        config: Resolved engine configuration.
        hf_repo: The preset's Hugging Face repo id.
        local_files_only: ``True`` for a cache-only resolution (no network).

    Returns:
        The snapshot directory.

    Raises:
        Exception: Whatever ``huggingface_hub`` raises when the snapshot cannot
            be resolved under the requested policy.
    """
    import huggingface_hub  # pyright: ignore[reportMissingModuleSource]

    download_root = resolve_download_root(config.download_root, has_library_default=True)
    token = config.hf_token.get_secret_value() if config.hf_token is not None else None
    # snapshot_download's overloads carry Unknown generics (user_agent /
    # tqdm_class), so pyright strict flags the call as partially unknown; the
    # return is the snapshot path string.
    resolved: str = huggingface_hub.snapshot_download(  # pyright: ignore[reportUnknownMemberType]
        hf_repo,
        revision=config.revision,
        cache_dir=None if download_root is None else str(download_root),
        local_files_only=local_files_only,
        token=token,
        allow_patterns=_allow_patterns(),
    )
    # A relative download_root yields a relative snapshot path; the report's
    # location field requires (and callers deserve) the absolute form.
    return Path(resolved).expanduser().resolve()


def _local_path_requirement(config: MlxAudioConfig) -> ArtifactRequirement:
    """Build the externally provided requirement for a ``model_path`` config.

    Args:
        config: Resolved engine configuration with ``model_path`` set.

    Returns:
        The single logical requirement for the operator-provided directory.
    """
    assert config.model_path is not None
    path = normalized_model_path(config.model_path)
    if not path.exists():
        state = ARTIFACT_MISSING
        message: str | None = (
            f"Provide an MLX checkpoint directory at {path} (the configured "
            "model_path), or unset model_path to use the preset's Hub repo."
        )
    elif path.is_file():
        state = ARTIFACT_INCOMPLETE
        message = (
            f"The configured model_path {path} is a file; point it at the "
            "MLX checkpoint DIRECTORY containing config.json and its weights."
        )
    elif _looks_like_checkpoint(path):
        state = ARTIFACT_READY
        message = None
    else:
        # The directory exists but is provably not a complete checkpoint (the
        # loader requires config.json, and an incomplete shard set would load
        # with strict=False into fluent garbage). This check already ran and
        # answered, so the state is incomplete, not unknown.
        state = ARTIFACT_INCOMPLETE
        message = (
            f"The configured model_path {path} has no config.json plus "
            "complete weights; point it at the MLX checkpoint directory "
            "itself."
        )

    return ArtifactRequirement(
        artifact_id=LOCAL_ARTIFACT_ID,
        label="Operator-provided MLX checkpoint directory",
        state=state,
        required_for_inference=True,
        can_acquire_now=False,
        may_acquire_during_inference=False,
        source_is_mutable=False,
        acquisition_blocker=None if state == ARTIFACT_READY else ARTIFACT_BLOCKER_ACTION_REQUIRED,
        required_actions=()
        if message is None
        else (ArtifactAction(kind=ARTIFACT_ACTION_PROVIDE_ARTIFACTS, message=message),),
        location=path if path.exists() else None,
        size_bytes=_tree_size_bytes(path) if state == ARTIFACT_READY else None,
    )


def _hub_requirement(
    config: MlxAudioConfig,
    hf_repo: str,
    subfolder: str | None,
    required_files: tuple[str, ...] = (),
) -> ArtifactRequirement:
    """Build the Hub-preset requirement from a cache-only resolution.

    Args:
        config: Resolved engine configuration.
        hf_repo: The preset's Hugging Face repo id.
        subfolder: Optional checkpoint subfolder inside the snapshot.

    Returns:
        The single logical requirement for the preset's snapshot.
    """
    downloads_blocked = config.local_files_only or not allow_downloads()
    location: Path | None = None
    try:
        resolved = snapshot(config, hf_repo, local_files_only=True)
    except Exception as exc:
        # Only the documented not-in-cache failure is reliable evidence of a
        # missing snapshot; anything else (the resolution stack unavailable,
        # an unreadable cache, a permission failure) is not evidence of
        # absence -- unknown never means ready, and never claims missing.
        state = ARTIFACT_MISSING if _is_local_entry_not_found(exc) else ARTIFACT_UNKNOWN
    else:
        # Resolution success only proves the snapshot directory exists;
        # completeness (and the preset's subfolder) is this plugin's check.
        # An interrupted download is repairable: a later online resolution
        # fetches only the files that are absent.
        location = resolved / subfolder if subfolder is not None else resolved
        if location.is_dir() and _looks_like_checkpoint(location, required_files):
            state = ARTIFACT_READY
        else:
            state = ARTIFACT_INCOMPLETE
            if not location.is_dir():
                # The preset's subfolder is not in the snapshot yet; report
                # the content that DOES exist (the snapshot root).
                location = resolved

    # Prefer the resolved commit over the configured mutable reference: two
    # engines resolving the same snapshot must report the same version, and a
    # refresh that moves a branch must be observable through this field.
    artifact_version = config.revision
    if location is not None:
        if location.parent.name == "snapshots":
            artifact_version = location.name
        elif location.parent.parent.name == "snapshots":
            # A subfolder location sits one level below the commit directory.
            artifact_version = location.parent.name

    if state == ARTIFACT_READY:
        can_acquire_now = False
        blocker = None
    elif downloads_blocked:
        can_acquire_now = False
        blocker = ARTIFACT_BLOCKER_DOWNLOADS_DISABLED
    else:
        can_acquire_now = True
        blocker = None

    label = f"mlx-audio snapshot {hf_repo}"
    if subfolder is not None:
        label += f" ({subfolder})"
    return ArtifactRequirement(
        artifact_id=HUB_ARTIFACT_ID,
        label=label,
        state=state,
        required_for_inference=True,
        can_acquire_now=can_acquire_now,
        may_acquire_during_inference=not downloads_blocked,
        source_is_mutable=not _revision_is_pinned(config.revision),
        acquisition_blocker=blocker,
        location=location,
        # The present logical size is reported for incomplete content too --
        # exactly what an operator wants to see for an interrupted download.
        size_bytes=_tree_size_bytes(location) if location is not None else None,
        artifact_version=artifact_version,
    )


def status_requirement(
    config: MlxAudioConfig,
    hf_repo: str,
    subfolder: str | None,
    required_files: tuple[str, ...] = (),
) -> ArtifactRequirement:
    """Report the engine's one logical requirement for the resolved config.

    Args:
        config: Resolved engine configuration.
        hf_repo: The preset's Hugging Face repo id.
        subfolder: Optional checkpoint subfolder inside the snapshot.
        required_files: Preset-declared files that must exist for readiness.

    Returns:
        The requirement for either the operator path or the Hub preset.
    """
    if config.model_path is not None:
        return _local_path_requirement(config)
    return _hub_requirement(config, hf_repo, subfolder, required_files)


def raise_for_gated_source(
    exc: BaseException,
    hf_repo: str,
    report: ArtifactReport | None = None,
) -> None:
    """Translate a gated or unauthenticated Hub rejection into actions.

    Shared by the explicit acquisition hook and the implicit first-use path:
    the reason comes from the discovered blocker, not from which code path
    noticed it.

    Args:
        exc: The native failure raised by the Hub client.
        hf_repo: The preset's Hugging Face repo id, for the message.
        report: The preflight report to attach, when the caller has one.

    Returns:
        None when the failure is not an access problem.

    Raises:
        ArtifactAcquisitionError: With ``reason="action_required"`` and the
            discovered action when the source is gated or rejects the
            credentials.
    """
    try:
        from huggingface_hub.errors import (  # pyright: ignore[reportMissingModuleSource]
            GatedRepoError,
            HfHubHTTPError,
        )
    except Exception:
        return
    if isinstance(exc, GatedRepoError):
        raise ArtifactAcquisitionError(
            f"The {hf_repo} repository is gated.",
            reason="action_required",
            report=report,
            required_actions=(
                ArtifactAction(
                    kind=ARTIFACT_ACTION_REQUEST_ACCESS,
                    message=(
                        "Accept the model terms or request access on its "
                        "Hugging Face page, then configure hf_token."
                    ),
                ),
            ),
            hint="Set the hf_token config field after access is granted.",
        ) from exc
    response = getattr(exc, "response", None)
    if isinstance(exc, HfHubHTTPError) and getattr(response, "status_code", None) == 401:
        raise ArtifactAcquisitionError(
            f"The {hf_repo} repository rejected the request as unauthenticated.",
            reason="action_required",
            report=report,
            required_actions=(
                ArtifactAction(
                    kind=ARTIFACT_ACTION_AUTHENTICATE,
                    message="Configure a valid hf_token for this repository.",
                ),
            ),
            hint="Set the hf_token config field.",
        ) from exc


def acquire(
    config: MlxAudioConfig,
    hf_repo: str,
    progress: ArtifactProgressCallback | None,
) -> None:
    """Materialize the preset's filtered snapshot without loading a model.

    ``snapshot_download`` re-resolves a mutable revision on every online call,
    so one code path serves both plain acquisition and an explicit refresh.
    ``model.generate`` is never called here.

    Args:
        config: Resolved engine configuration.
        hf_repo: The preset's Hugging Face repo id.
        progress: Optional serialized progress observer.

    Returns:
        None.

    Raises:
        ArtifactAcquisitionError: With ``reason="action_required"`` when the
            source reports a gated repository; every other native failure
            propagates for the template to wrap as ``reason="failed"``.
    """
    if config.local_files_only or not allow_downloads():
        raise ArtifactAcquisitionError(
            "Acquisition needs a network transfer, and downloads are disabled for this engine.",
            reason="downloads_disabled",
            hint=(
                "Unset local_files_only and set STANDARD_ASR_ALLOW_DOWNLOAD=1 "
                "to permit the transfer."
            ),
        )
    if progress is not None:
        # huggingface_hub reports per-file progress on its own terminal bars,
        # not through a callback seam; emit one honest indeterminate transfer
        # event instead of fabricating totals.
        progress(
            ArtifactProgress(phase=ARTIFACT_PROGRESS_TRANSFERRING, artifact_id=HUB_ARTIFACT_ID)
        )
    try:
        snapshot(config, hf_repo, local_files_only=False)
    except Exception as exc:
        raise_for_gated_source(exc, hf_repo)
        raise


def resolve_for_load(config: MlxAudioConfig, hf_repo: str, *, ready: bool) -> Path:
    """Return the local snapshot directory the loader should be pointed at.

    Args:
        config: Resolved engine configuration.
        hf_repo: The preset's Hugging Face repo id.
        ready: Whether status already proved a complete cached snapshot.

    Returns:
        The local snapshot root.

    Raises:
        Exception: Whatever the resolution raises; the caller classifies it as
            a failed implicit acquisition when the cache was not ready.
    """
    local_only = ready or config.local_files_only or not allow_downloads()
    return snapshot(config, hf_repo, local_files_only=local_only)
