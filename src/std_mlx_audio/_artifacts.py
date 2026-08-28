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
    ArtifactAcquisitionBlocker,
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

#: A sharded safetensors file names its own closure
#: (``model-00002-of-00005.safetensors``): every sibling the name implies
#: must be present before the fragment counts as weights.
_SHARD_NAME = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.safetensors$")

#: The files fetched for a preset's companion tokenizer repo (VibeVoice): the
#: fast-tokenizer serialization, the slow-path pair transformers falls back
#: to, the config files naming the class and special tokens, and the model
#: config -- loading by REPO ID resolves AutoConfig from config.json even
#: when the tokenizer files alone would satisfy a local-directory load
#: (verified end to end against the Hub, round-10 review: without it the
#: offline load fails, and an online load silently fetches it).
_COMPANION_TOKENIZER_PATTERNS = (
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def companion_tokenizer_cached(companion_repo: str) -> bool:
    """Return whether the companion tokenizer can load from the local cache.

    The probe mirrors the upstream fallback exactly: mlx-audio's VibeVoice
    hook calls ``AutoTokenizer.from_pretrained("<companion>")`` with no
    ``cache_dir`` and no token, so this reads the DEFAULT Hugging Face cache
    -- honoring ``download_root`` here would report ready while the upstream
    call still sees a cold default cache.

    Args:
        companion_repo: The Hub repo id the upstream loader fetches its
            tokenizer from.

    Returns:
        ``True`` when the cached snapshot holds the model config (the
        repo-id load resolves AutoConfig first) plus a loadable tokenizer
        (the fast serialization, or the slow-path vocab + merges pair).
        Any resolution failure counts as not proven cached: the load gate
        must not wave through a fetch it cannot rule out.
    """
    import huggingface_hub  # pyright: ignore[reportMissingModuleSource]

    try:
        resolved: str = huggingface_hub.snapshot_download(  # pyright: ignore[reportUnknownMemberType]
            companion_repo,
            local_files_only=True,
            allow_patterns=list(_COMPANION_TOKENIZER_PATTERNS),
        )
    except Exception:
        return False
    root = Path(resolved)
    if not (root / "config.json").is_file():
        return False
    return (root / "tokenizer.json").is_file() or (
        (root / "vocab.json").is_file() and (root / "merges.txt").is_file()
    )


def bundled_companion_tokenizer(root: Path) -> bool:
    """Return whether a checkpoint directory bundles a loadable tokenizer.

    The upstream VibeVoice hook tries ``AutoTokenizer.from_pretrained``
    on the checkpoint directory FIRST and falls back to the companion Hub
    repo only when that fails, so a checkpoint that bundles its tokenizer
    never needs the companion. The accepted layouts are the ones proven
    to load beside the checkpoint's own (foreign-model-type) config.json
    against the installed transformers (round-12 review, per-layout
    ablation): the fast serialization alone, or the tokenizer config with
    the slow vocab + merges pair. A bare vocab + merges pair without the
    tokenizer config provably fails and does NOT count.

    Args:
        root: The checkpoint directory.

    Returns:
        ``True`` when the local tokenizer load provably succeeds.
    """
    return (root / "tokenizer.json").is_file() or (
        (root / "tokenizer_config.json").is_file()
        and (root / "vocab.json").is_file()
        and (root / "merges.txt").is_file()
    )


def fetch_companion_tokenizer(companion_repo: str) -> None:
    """Materialize the companion tokenizer into the DEFAULT Hub cache.

    The transfer must land where the upstream loader's own
    ``AutoTokenizer.from_pretrained`` call (no ``cache_dir``, no token) will
    look for it, so this never honors ``download_root``.

    Args:
        companion_repo: The Hub repo id to fetch the tokenizer files from.

    Raises:
        Exception: Whatever ``huggingface_hub`` raises when the transfer
            fails.
    """
    import huggingface_hub  # pyright: ignore[reportMissingModuleSource]

    huggingface_hub.snapshot_download(  # pyright: ignore[reportUnknownMemberType]
        companion_repo,
        local_files_only=False,
        allow_patterns=list(_COMPANION_TOKENIZER_PATTERNS),
    )


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


def checkpoint_complete(root: Path, required_files: tuple[str, ...] = ()) -> bool:
    """Return whether a directory is a provably complete MLX checkpoint.

    Resolution success only proves the snapshot directory exists; upstream
    documents that it cannot verify the files inside, so completeness is this
    plugin's own check. A sharded checkpoint is complete only when EVERY
    shard is present -- named by the safetensors index when it exists, or by
    the shards' own ``-NNNNN-of-NNNNN`` names when the index has not arrived
    yet: the upstream loader globs whatever shards exist and loads them with
    ``strict=False``, so a missing shard would silently leave parameters at
    random init -- fluent garbage, the cardinal sin.

    The weights closure is checked structurally; non-weight inference files
    (tokenizers, normalization stats) vary per model family, so they enter
    through ``required_files`` from the presets' two declared axes (round-9
    review). ``required_checkpoint_files`` are files the installed loader
    reads by one fixed name and whose absence silently corrupts output
    (SenseVoice, MMS, FireRed, Moonshine, Cohere); they gate every
    checkpoint, an operator ``model_path`` included. ``required_snapshot_files``
    are files whose absence provably breaks inference for the preset's OWN
    repo layout; they gate only the Hub snapshot (acquisition completeness),
    because the flexible upstream loaders accept alternative local layouts
    (Canary reads ``tokenizer.model``, ``tokens.txt``, or a config-embedded
    tokenizer; transformers falls back from ``tokenizer.json`` to
    vocab + merges) that an exact-name check would wrongly reject. Every
    declared file is verified as a single point of failure against the
    installed loader (per-file ablation or the loader's code); a fragment
    that dodges both lists (a dual-layout repo missing every alternative at
    once) still fails LOUDLY at load or at the first request, and
    ``pull --refresh`` re-fetches whatever files its snapshot lacks.

    Args:
        root: Candidate checkpoint directory.
        required_files: Preset-declared files that must exist -- the
            checkpoint axis for a ``model_path``, the union of both axes
            for a Hub snapshot.

    Returns:
        ``True`` when the config, the shard closure, and every required file
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
    # No index (yet): an interrupted download can hold shard files while the
    # index is still absent, so honor the closure each shard name declares.
    matches = [
        match
        for match in (_SHARD_NAME.fullmatch(entry.name) for entry in root.glob("*.safetensors"))
        if match is not None
    ]
    groups = {(match["prefix"], int(match["total"])) for match in matches}
    for prefix, total in groups:
        for ordinal in range(1, total + 1):
            if not (root / f"{prefix}-{ordinal:05d}-of-{total:05d}.safetensors").is_file():
                return False
    if groups:
        return True
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
    """Return the snapshot filter: the upstream defaults plus known gaps.

    The base list is imported so it never drifts. ``*.mvn`` is the one known
    gap (round-8 review): SenseVoice's ``am.mvn`` carries its feature
    normalization stats and its config has no fallback, yet the upstream
    defaults never fetch the file -- every default-pattern snapshot loads
    with normalization silently skipped. The extra pattern matches nothing
    in the other preset repos.

    Returns:
        The mlx-audio ``DEFAULT_ALLOW_PATTERNS`` list plus ``*.mvn``.
    """
    from mlx_audio.utils import (  # pyright: ignore[reportMissingImports]
        DEFAULT_ALLOW_PATTERNS,
    )

    return [*DEFAULT_ALLOW_PATTERNS, "*.mvn"]


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


def _local_path_requirement(
    config: MlxAudioConfig,
    required_files: tuple[str, ...] = (),
    companion_repo: str | None = None,
) -> ArtifactRequirement:
    """Build the externally provided requirement for a ``model_path`` config.

    The preset's checkpoint-axis ``required_files`` apply here too: a
    ``model_path`` still loads through the preset's model family, so a
    directory lacking a file the loader reads by one fixed name (the MMS
    base weights, SenseVoice's normalization stats) would silently corrupt
    output. The snapshot-axis files do NOT apply: the flexible upstream
    loaders accept alternative local layouts that an exact-name check would
    wrongly reject (round-9 review).

    Args:
        config: Resolved engine configuration with ``model_path`` set.
        required_files: Preset-declared checkpoint-axis files.
        companion_repo: Optional Hub repo the upstream loader fetches its
            tokenizer from at load time (VibeVoice).

    Returns:
        The single logical requirement for the operator-provided directory.
    """
    assert config.model_path is not None
    path = normalized_model_path(config.model_path)
    blocker: ArtifactAcquisitionBlocker | None = ARTIFACT_BLOCKER_ACTION_REQUIRED
    can_acquire_now = False
    may_acquire_during_inference = False
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
    elif not checkpoint_complete(path, required_files):
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
    elif (
        companion_repo is not None
        and not bundled_companion_tokenizer(path)
        and not companion_tokenizer_cached(companion_repo)
    ):
        # The checkpoint itself is complete, but the family's tokenizer
        # lives in a separate Hub repo the upstream loader fetches at load
        # time, and neither the checkpoint (a bundled tokenizer makes the
        # companion moot, round-12 review) nor the default cache holds it:
        # inference is not offline-ready (same substance as the Hub
        # branch, round-10 review). With downloads allowed the companion
        # is the one part the plugin CAN acquire for an operator path --
        # pull fetches it, and a load would fetch it implicitly -- so the
        # requirement is acquirable; with downloads disabled the transfer
        # is forbidden and the load must refuse. This is the ONLY local
        # shape whose blocker is not action_required (the engine's guard
        # keys on it).
        state = ARTIFACT_INCOMPLETE
        message = None
        if config.local_files_only or not allow_downloads():
            blocker = ARTIFACT_BLOCKER_DOWNLOADS_DISABLED
        else:
            blocker = None
            can_acquire_now = True
            may_acquire_during_inference = True
    else:
        state = ARTIFACT_READY
        message = None
        if (
            companion_repo is not None
            and not bundled_companion_tokenizer(path)
            and not (config.local_files_only or not allow_downloads())
        ):
            # Even with the companion warm, the upstream loader addresses
            # it by UNPINNED Hub repo id with no offline flag, so a load
            # may still revalidate against the source and fetch an updated
            # file while downloads are permitted: the effective
            # may_acquire_during_inference is True, mirroring the Hub
            # branch's policy narrowing (round-11 review). A bundled
            # tokenizer keeps the load entirely local, so the field stays
            # False then.
            may_acquire_during_inference = True

    return ArtifactRequirement(
        artifact_id=LOCAL_ARTIFACT_ID,
        label="Operator-provided MLX checkpoint directory",
        state=state,
        required_for_inference=True,
        can_acquire_now=can_acquire_now,
        may_acquire_during_inference=may_acquire_during_inference,
        source_is_mutable=False,
        acquisition_blocker=None if state == ARTIFACT_READY else blocker,
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
    companion_repo: str | None = None,
) -> ArtifactRequirement:
    """Build the Hub-preset requirement from a cache-only resolution.

    Args:
        config: Resolved engine configuration.
        hf_repo: The preset's Hugging Face repo id.
        subfolder: Optional checkpoint subfolder inside the snapshot.
        required_files: Preset-declared files that must exist for readiness
            (the union of the checkpoint and snapshot axes).
        companion_repo: Optional Hub repo the upstream loader fetches its
            tokenizer from at load time (VibeVoice).

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
        if location.is_dir() and checkpoint_complete(location, required_files):
            state = ARTIFACT_READY
        else:
            state = ARTIFACT_INCOMPLETE
            if not location.is_dir():
                # The preset's subfolder is not in the snapshot yet; report
                # the content that DOES exist (the snapshot root).
                location = resolved
    if (
        state == ARTIFACT_READY
        and companion_repo is not None
        and location is not None
        and not bundled_companion_tokenizer(location)
        and not companion_tokenizer_cached(companion_repo)
    ):
        # The snapshot is complete, but the family's tokenizer lives in a
        # separate Hub repo the upstream loader fetches at load time, and
        # neither the snapshot (a bundled tokenizer would keep the load
        # local) nor the default cache holds it: inference is not
        # offline-ready, and pull must still acquire the companion.
        state = ARTIFACT_INCOMPLETE

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
    checkpoint_files: tuple[str, ...] = (),
    snapshot_files: tuple[str, ...] = (),
    companion_repo: str | None = None,
) -> ArtifactRequirement:
    """Report the engine's one logical requirement for the resolved config.

    The two file axes gate different questions (round-9 review): the
    checkpoint axis (silent-corruption files, one fixed name each) applies
    to every checkpoint, while the snapshot axis (files whose absence
    provably breaks inference for the preset's own repo layout) applies
    only to the Hub snapshot -- an operator ``model_path`` may use any
    alternative layout the flexible upstream loaders accept.

    Args:
        config: Resolved engine configuration.
        hf_repo: The preset's Hugging Face repo id.
        subfolder: Optional checkpoint subfolder inside the snapshot.
        checkpoint_files: Preset-declared files every checkpoint needs.
        snapshot_files: Preset-declared files only the Hub snapshot needs.
        companion_repo: Optional Hub repo the upstream loader fetches its
            tokenizer from at load time (VibeVoice).

    Returns:
        The requirement for either the operator path or the Hub preset.
    """
    if config.model_path is not None:
        return _local_path_requirement(config, checkpoint_files, companion_repo)
    return _hub_requirement(
        config,
        hf_repo,
        subfolder,
        checkpoint_files + snapshot_files,
        companion_repo,
    )


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


def _remote_commit(config: MlxAudioConfig, hf_repo: str) -> str:
    """Re-resolve the preset's revision against the source and return it.

    The positive evidence spec AR.4 requires for a refresh: the native
    downloader silently falls back to the local cache when the remote is
    unreachable, so download success can never prove that the mutable
    reference was re-resolved -- this metadata query can.

    Args:
        config: Resolved engine configuration.
        hf_repo: The preset's Hugging Face repo id.

    Returns:
        The commit hash the source currently resolves the revision to.

    Raises:
        ArtifactAcquisitionError: If the source answered without naming a
            commit.
        Exception: Whatever the metadata query raises when the source is
            unreachable or rejects the request.
    """
    import huggingface_hub  # pyright: ignore[reportMissingModuleSource]

    token = config.hf_token.get_secret_value() if config.hf_token is not None else None
    info = huggingface_hub.HfApi(token=token).model_info(hf_repo, revision=config.revision)
    sha = info.sha
    if sha is None:
        raise ArtifactAcquisitionError(
            f"The source metadata for {hf_repo} did not name a commit, so "
            "the refresh cannot be verified.",
            reason="failed",
        )
    return sha


def acquire(
    config: MlxAudioConfig,
    hf_repo: str,
    progress: ArtifactProgressCallback | None,
    *,
    refresh: bool = False,
    companion_repo: str | None = None,
) -> None:
    """Materialize the preset's filtered snapshot without loading a model.

    An incomplete snapshot is completed by the same online call (upstream
    fetches only the files that are absent). A refresh cannot trust that
    call alone: ``snapshot_download`` silently falls back to the local cache
    when the remote is unreachable, so a refresh first re-resolves the
    mutable revision against the source and then verifies that the resolved
    snapshot is that resolution (spec AR.4). A companion tokenizer is
    fetched into the default cache after the snapshot; it is a best-effort
    convenience fetch of the upstream loader's own source, so the refresh
    evidence covers the snapshot only. ``model.generate`` is never called
    here.

    Args:
        config: Resolved engine configuration.
        hf_repo: The preset's Hugging Face repo id.
        progress: Optional serialized progress observer.
        refresh: Whether the mutable revision must be re-resolved.
        companion_repo: Optional Hub repo the upstream loader fetches its
            tokenizer from at load time (VibeVoice).

    Returns:
        None.

    Raises:
        ArtifactAcquisitionError: With ``reason="action_required"`` when the
            source reports a gated repository; with ``reason="failed"`` when
            a refresh cannot prove the source was re-resolved; every other
            native failure propagates for the template to wrap as
            ``reason="failed"``.
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
    expected_commit: str | None = None
    if refresh and not _revision_is_pinned(config.revision):
        try:
            expected_commit = _remote_commit(config, hf_repo)
        except ArtifactAcquisitionError:
            raise
        except Exception as exc:
            raise_for_gated_source(exc, hf_repo)
            raise ArtifactAcquisitionError(
                f"Refresh needs to re-resolve the {hf_repo} source reference, "
                f"and the source metadata query failed ({type(exc).__name__}).",
                reason="failed",
                hint="Make the Hugging Face endpoint reachable, then retry.",
            ) from exc
    if progress is not None:
        # huggingface_hub reports per-file progress on its own terminal bars,
        # not through a callback seam; emit one honest indeterminate transfer
        # event instead of fabricating totals.
        progress(
            ArtifactProgress(phase=ARTIFACT_PROGRESS_TRANSFERRING, artifact_id=HUB_ARTIFACT_ID)
        )
    try:
        resolved = snapshot(config, hf_repo, local_files_only=False)
    except Exception as exc:
        raise_for_gated_source(exc, hf_repo)
        raise
    # Outside the snapshots/<commit> cache layout the resolution carries no
    # comparable commit; the layout is guaranteed here because the helper
    # always resolves through a Hub cache root.
    if (
        expected_commit is not None
        and resolved.parent.name == "snapshots"
        and resolved.name != expected_commit
    ):
        raise ArtifactAcquisitionError(
            f"Refresh resolved a {hf_repo} snapshot that is not the "
            "source's current commit: the source reference moved during the "
            "refresh, or the downloader fell back to the local cache without "
            "reaching the source.",
            reason="failed",
            hint="Retry while the source is reachable.",
        )
    if companion_repo is not None and not bundled_companion_tokenizer(resolved):
        # A snapshot that bundles its own loadable tokenizer is
        # self-contained: fetching the companion would be wasted transfer,
        # and a companion outage must not fail a pull whose target content
        # is independently usable (round-13 review).
        try:
            fetch_companion_tokenizer(companion_repo)
        except Exception as exc:
            raise_for_gated_source(exc, companion_repo)
            raise


def acquire_companion(
    config: MlxAudioConfig,
    companion_repo: str,
    progress: ArtifactProgressCallback | None,
) -> None:
    """Fetch only the companion tokenizer (a ``model_path`` pull target).

    An operator-provided checkpoint is not the plugin's to acquire, but its
    companion tokenizer is: this is the acquisition hook's path for a
    ``model_path`` requirement whose only deficiency is the cold companion
    cache. The download policy is applied again at this entry, like
    :func:`acquire`'s (spec AR.3 applies the toggle before the transfer;
    the preflight that made the target runnable is a separate earlier
    query).

    Args:
        config: Resolved engine configuration.
        companion_repo: The Hub repo id to fetch the tokenizer files from.
        progress: Optional serialized progress observer.

    Raises:
        ArtifactAcquisitionError: With ``reason="downloads_disabled"`` when
            the policy forbids the transfer; with
            ``reason="action_required"`` when the source is gated or
            rejects the credentials; every other native failure propagates
            for the template to wrap as ``reason="failed"``.
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
        progress(
            ArtifactProgress(phase=ARTIFACT_PROGRESS_TRANSFERRING, artifact_id=LOCAL_ARTIFACT_ID)
        )
    try:
        fetch_companion_tokenizer(companion_repo)
    except Exception as exc:
        raise_for_gated_source(exc, companion_repo)
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
