# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The Standard ASR engine class for the MLX ASR backend.

A thin, typed adapter over the upstream ``mlx-audio`` package (``mlx_audio.stt``)
that makes EVERY MLX STT model family usable by any Standard ASR application. A
single engine (engine_id ``mlx-audio``) exposes ~20 models — Qwen3-ASR (the
headliner), Whisper, Parakeet, Nemotron, SenseVoice, Voxtral, Canary, GLM-ASR,
Granite Speech, Fun-ASR, VibeVoice, Moonshine, MMS, FireRedASR2, Qwen2-Audio —
by binding each entry-point preset to a
``(hf_repo, ModelBackend, properties, capabilities)`` tuple.

It subclasses :class:`EngineBase` and:

* binds each preset's per-family :class:`~std_mlx_audio._metadata.MlxAudioProperties`
  and fail-closed ``DeclaredCapabilities`` (built via the ``stt_properties`` /
  ``stt_capabilities`` factories, or the original per-family subclasses);
* keeps ``__init__`` pure and loads weights lazily in
  :meth:`_ensure_model_loaded` (spec IC.9), then verifies the loaded model's
  family matches the bound backend (:meth:`_verify_model_family`) so a stray
  ``model_path`` fails loudly instead of mis-transcribing;
* implements :meth:`_transcribe` (batch) by dispatching to the bound
  :class:`~std_mlx_audio.backends.ModelBackend`, and :meth:`_start_transcription`
  (windowed streaming, see :mod:`std_mlx_audio._streaming`).

The family heterogeneity (different native ``generate`` signatures and return
types) lives entirely in the bound backend; this class is family-agnostic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray
from standard_asr import (
    RuntimeParams,
    TranscriptionResult,
    TranscriptionSession,
)
from standard_asr.audio.format import AudioFormat
from standard_asr.contract.artifacts import (
    ARTIFACT_BLOCKER_DOWNLOADS_DISABLED,
    ARTIFACT_INCOMPLETE,
    ARTIFACT_MISSING,
    ARTIFACT_READY,
    ArtifactContext,
    ArtifactProgressCallback,
    ArtifactReport,
    ArtifactRequirement,
)
from standard_asr.contract.capabilities import DeclaredCapabilities
from standard_asr.contract.exceptions import (
    ArtifactAcquisitionError,
    ArtifactUnavailableError,
    DiscoveryError,
    TranscriptionError,
)
from standard_asr.contract.language import effective_language
from standard_asr.contract.params import ProviderParams
from standard_asr.engine import (
    ArtifactDeclaration,
    BaseConfig,
    BaseProperties,
    DeclaredEngineMetadata,
    Diagnostic,
    EngineBase,
    Mode,
    PreparedAudio,
)
from standard_asr.runtime.downloads import allow_downloads

from . import backends
from ._artifacts import (
    HUB_ARTIFACT_ID,
    LOCAL_ARTIFACT_ID,
    acquire,
    acquire_companion,
    bundled_companion_tokenizer,
    checkpoint_complete,
    companion_tokenizer_cached,
    fetch_companion_tokenizer,
    normalized_model_path,
    offline_tokenizer_loads,
    raise_for_gated_source,
    resolve_for_load,
    status_requirement,
)
from ._config import MlxAudioConfig, MlxAudioParams
from ._metadata import (
    _PARAKEET_CAPABILITIES,
    _QWEN_CAPABILITIES,
    _WHISPER_CAPABILITIES,
    _WORD_TS_NONE,
    _WORD_TS_SEGMENT,
    _WORD_TS_WORD,
    ParakeetTdt06BV3Properties,
    Qwen3Asr06BProperties,
    Qwen3Asr17BProperties,
    WhisperLargeV3TurboProperties,
    WhisperTinyProperties,
    stt_capabilities,
    stt_properties,
)
from .backends import (
    AlignedResultBackend,
    GenericSttBackend,
    ModelBackend,
    Qwen3AsrBackend,
    SttFamilySpec,
    WhisperBackend,
    map_word_timestamps,
    waveform_duration,
)
from .languages import (
    CANARY_LANGUAGES,
    COHERE_LANGUAGES,
    FUN_ASR_DETECTABLE_LANGUAGES,
    FUN_ASR_LANGUAGES,
    SENSEVOICE_DETECTABLE_LANGUAGES,
    SENSEVOICE_LANGUAGES,
    VOXTRAL_LANGUAGES,
)

_LOGGER = logging.getLogger(__name__)
_SAMPLE_RATE = backends.SAMPLE_RATE


class MlxAudioASR(EngineBase):
    """Standard ASR adapter for an MLX STT model (abstract base for the presets).

    Each concrete preset is a subclass that overrides:

    * :attr:`hf_repo` — the Hugging Face MLX repo id to load;
    * :attr:`backend` — the :class:`~std_mlx_audio.backends.ModelBackend` that
      knows this family's ``generate`` call-shape and output mapping;
    * :attr:`properties` and :attr:`declared_capabilities` — the family's static
      identity and honest capabilities.

    Model selection is by preset (spec IC.7), never an init ``model`` field; a
    local ``model_path`` config override (spec IC.7 weights/path) still wins when
    set.

    Args:
        **kwargs: Configuration overrides for :class:`MlxAudioConfig`.
    """

    #: The Hugging Face MLX repo id this preset loads. Overridden per preset; a
    #: local ``model_path`` config override wins when set.
    hf_repo: ClassVar[str] = ""
    #: The backend adapter for this preset's model family. Overridden per preset.
    backend: ClassVar[ModelBackend]
    #: Per-preset config defaults applied when the caller does not specify them
    #: (spec IC.6 / LANG R1): a fixed-language preset whose ``selectable_languages``
    #: omits the ``"auto"`` directive (Parakeet) MUST default ``default_language``
    #: to a concrete member of its set, not the engine-wide ``"auto"`` default —
    #: otherwise LANG R1 fails. Explicit ``kwargs`` always win over these.
    default_config_overrides: ClassVar[dict[str, Any]] = {}
    #: Optional ``model_type`` to force on the loader (spec IC.7). A few HF repos
    #: ship a ``config.json`` whose ``model_type`` does not match the mlx-audio
    #: family that can actually run them — e.g. MMS reports ``"wav2vec2"`` but is
    #: served by the ``mms`` family — so the preset pins the correct family rather
    #: than letting auto-detection mis-route. ``None`` (default) auto-detects.
    load_model_type: ClassVar[str | None] = None
    #: Optional repo SUBFOLDER holding the actual checkpoint when a repo stores its
    #: weights + config below the root rather than at it. mlx-audio's ``load``
    #: resolves ``config.json`` at the repo root and fails (``Config not found``)
    #: when the files live in a subfolder (e.g. Cohere-ASR ships everything under
    #: ``mlx-int8/``). When set, the loader snapshot-downloads the repo and points
    #: ``load`` at ``<snapshot>/<subfolder>``. ``None`` (default) loads from root.
    hf_subfolder: ClassVar[str | None] = None
    #: Files the installed family loader reads by ONE fixed name and whose
    #: absence silently corrupts output (numeric ids, empty text, character
    #: soup, unstripped special tokens). They gate EVERY checkpoint of the
    #: family -- an operator ``model_path`` included -- beyond the generic
    #: checkpoint shape (config.json + weights / a complete shard index).
    #: Each entry is verified against the installed loader's code.
    required_checkpoint_files: ClassVar[tuple[str, ...]] = ()
    #: Files whose absence provably breaks inference for THIS preset's own
    #: repo layout (verified by per-file loader ablation against the
    #: installed stack). They gate only the Hub snapshot -- acquisition
    #: completeness -- never an operator ``model_path``: the flexible
    #: upstream loaders accept alternative local layouts (Canary reads
    #: tokenizer.model, tokens.txt, or a config-embedded tokenizer;
    #: transformers falls back from tokenizer.json to vocab + merges) that
    #: an exact-name check would wrongly reject (round-9 review).
    required_snapshot_files: ClassVar[tuple[str, ...]] = ()
    #: Optional Hub repo the upstream loader fetches its tokenizer from at
    #: LOAD time (VibeVoice ships no tokenizer files and its hook falls back
    #: to a Hub fetch that bypasses ``local_files_only``). When set, status
    #: requires the companion tokenizer in the DEFAULT Hugging Face cache,
    #: ``pull`` acquires it, and a load under a no-download policy refuses
    #: rather than letting the upstream fetch violate the policy silently.
    companion_tokenizer_repo: ClassVar[str | None] = None

    #: Static upper bounds shared by every preset in this plugin-owned family:
    #: each Hub preset supports explicit snapshot acquisition (plugin-side
    #: ``snapshot_download``) and can also acquire on first use inside
    #: ``_ensure_model_loaded``. A ``model_path`` instance narrows both to an
    #: externally provided requirement dynamically.
    declared_metadata: ClassVar[DeclaredEngineMetadata] = DeclaredEngineMetadata(
        artifacts=ArtifactDeclaration(
            acquisition_applicable=True,
            supports_explicit_acquisition=True,
            may_acquire_during_inference=True,
        )
    )
    provider_params_type: ClassVar[type[ProviderParams] | None] = MlxAudioParams
    config_type: ClassVar[type[BaseConfig[str]] | None] = MlxAudioConfig

    def __init__(self, **kwargs: Any) -> None:
        """Capture configuration (pure; weights load lazily, spec IC.9).

        Config is built via ``from_env``: unset fields fall back to
        ``STANDARD_ASR_MLX_AUDIO__*`` environment variables (spec IC.4; double
        underscore between engine and field segments), explicit ``kwargs`` win,
        and the HF token is wrapped in ``SecretStr`` by construction.
        Per-preset :attr:`default_config_overrides` seed defaults the caller did
        not provide (e.g. a fixed-language preset's ``default_language``).

        Args:
            **kwargs: Configuration overrides.
        """
        merged = {**type(self).default_config_overrides, **kwargs}
        self.config = MlxAudioConfig.from_env("mlx-audio", **merged)
        self._model: object | None = None

    # ------------------------------------------------------------------ #
    # Lazy model loading
    # ------------------------------------------------------------------ #
    @property
    def model(self) -> object:
        """The loaded MLX model (loads it on first access).

        Returns:
            The underlying mlx-audio model instance (an ``nn.Module``).

        Raises:
            DiscoveryError: If mlx-audio is not installed.
            ArtifactUnavailableError: If required artifacts cannot resolve
                under the current policy.
            ArtifactAcquisitionError: If an allowed first-use acquisition
                fails.
            TranscriptionError: If a locally available model fails to load.
        """
        self._ensure_model_loaded()
        assert self._model is not None  # _ensure_model_loaded raises otherwise
        return self._model

    def ensure_loaded(self, *, mode: Mode = "batch") -> None:
        """Public alias for the lazy loader (used by the streaming session).

        Args:
            mode: The inference mode whose path is loading; artifact errors
                raised here carry a report resolved for this mode.

        Raises:
            DiscoveryError: If mlx-audio is not installed.
            ArtifactUnavailableError: If required artifacts cannot resolve
                under the current policy.
            ArtifactAcquisitionError: If an allowed first-use acquisition
                fails.
            TranscriptionError: If a locally available model fails to load.
        """
        self._ensure_model_loaded(mode=mode)

    def _ensure_model_loaded(self, *, mode: Mode = "batch") -> None:
        """Load the MLX model lazily via ``mlx_audio.stt.load``.

        The artifact guard runs first, then the snapshot is resolved
        PLUGIN-SIDE (``huggingface_hub.snapshot_download`` with the resolved
        cache root, token, revision, and offline flag) and the loader receives
        a local directory. The upstream loader accepts none of those controls
        (it silently swallows unknown kwargs), so plugin-side resolution is
        what makes ``download_root``, ``hf_token``, and ``local_files_only``
        honored rather than declared-but-inert.

        Raises:
            DiscoveryError: If mlx-audio is not installed.
            ArtifactUnavailableError: If required artifacts are missing and
                cannot be acquired under the current policy.
            ArtifactAcquisitionError: If an allowed first-use acquisition
                fails.
            TranscriptionError: If a locally available model fails to load.
        """
        if self._model is not None:
            return
        # huggingface_hub (pulled in by mlx-audio) uses tqdm for download bars,
        # which spawns a persistent `tqdm_monitor` DAEMON thread on first use.
        # That thread is harmless at interpreter exit, but the standard
        # sync-bridge compliance check flags ANY leaked background thread, so an
        # unsuppressed monitor makes a fully-correct engine fail compliance.
        # Disabling the monitor (cosmetic only) is the adapter's responsibility —
        # we own the lifecycle of what loading the model spawns. See
        # docs/STANDARD_ASR_FINDINGS.md.
        _disable_tqdm_monitor_thread()
        try:
            from mlx_audio.stt import load  # pyright: ignore[reportMissingImports]
        except Exception as exc:
            raise DiscoveryError(
                "mlx-audio is not installed (or MLX is unavailable on this "
                "platform — MLX requires Apple Silicon with Metal). Install "
                "'std-mlx-audio' with its dependencies on a supported Mac "
                "(pip install std-mlx-audio)."
            ) from exc

        config = cast(MlxAudioConfig, self.config)
        local_only = config.local_files_only or not allow_downloads()
        # The artifact guard: the same cheap inspection artifact_status()
        # reports. It decides whether a failure below is a failed implicit
        # acquisition or an engine-execution fault. The report on a guard
        # error is built from this same requirement (no second inspection, no
        # TOCTOU window) with the caller's mode.
        requirement = status_requirement(
            config,
            type(self).hf_repo,
            type(self).hf_subfolder,
            type(self).required_checkpoint_files,
            type(self).required_snapshot_files,
            type(self).companion_tokenizer_repo,
        )
        report = ArtifactReport.from_requirements(
            mode=mode, applicable=True, requirements=(requirement,)
        )
        if requirement.state != ARTIFACT_READY:
            if (
                config.model_path is not None
                and requirement.acquisition_blocker == ARTIFACT_BLOCKER_DOWNLOADS_DISABLED
            ):
                # The checkpoint directory itself is complete; what is
                # missing is the family's companion tokenizer, which the
                # upstream loader would fetch from the Hub PAST the
                # no-download policy. Refuse loudly instead.
                raise ArtifactUnavailableError(
                    f"The model_path checkpoint is complete, but its tokenizer "
                    f"comes from the {type(self).companion_tokenizer_repo} Hub "
                    "repo, the local cache does not hold it, and downloads are "
                    "disabled.",
                    reason="downloads_disabled",
                    report=report,
                    hint=(
                        "Enable downloads for one load to cache the tokenizer, "
                        "or warm the Hugging Face cache on a connected machine."
                    ),
                )
            if (
                config.model_path is not None
                and requirement.state in (ARTIFACT_MISSING, ARTIFACT_INCOMPLETE)
                and requirement.acquisition_blocker is not None
            ):
                # The status check already ran and answered (the path is
                # absent, a file, or provably not a complete checkpoint); the
                # loader would only turn that knowledge into an opaque native
                # failure -- or fluent garbage via a strict=False load. The
                # one local incomplete WITHOUT a blocker is the acquirable
                # cold companion tokenizer (downloads allowed): the load
                # proceeds and pre-fetches it below.
                raise ArtifactUnavailableError(
                    f"The configured model_path {config.model_path!r} is not a "
                    f"usable MLX checkpoint (state: {requirement.state}).",
                    reason=cast("Any", requirement.state),
                    report=report,
                    hint="Provide the complete MLX checkpoint directory or unset model_path.",
                )
            if config.model_path is None and local_only:
                raise ArtifactUnavailableError(
                    f"The {type(self).hf_repo} snapshot is not fully cached "
                    f"(state: {requirement.state}) and downloads are disabled.",
                    reason="downloads_disabled",
                    report=report,
                    hint=(
                        "Enable downloads (unset local_files_only and set "
                        "STANDARD_ASR_ALLOW_DOWNLOAD=1), then run "
                        "'standard-asr pull'."
                    ),
                )
            # A model_path in state unknown is deliberately attempted: unknown
            # is not evidence of unavailability (AR.2), and the loader is the
            # authoritative check for an unrecognized local layout.
        # Model selection is by preset (spec IC.7); a local model_path wins.
        # The path uses the same canonical form status inspected.
        load_kwargs: dict[str, Any] = {}
        if config.model_path is not None:
            model_source: str = str(normalized_model_path(config.model_path))
        else:
            try:
                snapshot_root = resolve_for_load(
                    config,
                    type(self).hf_repo,
                    ready=requirement.state == ARTIFACT_READY,
                )
            except Exception as exc:
                # The resolution was the allowed implicit acquisition path. An
                # access rejection carries its discovered action (the reason
                # comes from the blocker, not from which code path noticed).
                raise_for_gated_source(exc, type(self).hf_repo, report)
                raise ArtifactAcquisitionError(
                    f"First-use acquisition of the {type(self).hf_repo} "
                    f"snapshot failed: {type(exc).__name__}.",
                    reason="failed",
                    report=report,
                    hint="Run 'standard-asr pull' to acquire it explicitly.",
                ) from exc
            # Some repos keep the checkpoint in a subfolder (e.g. Cohere-ASR
            # under ``mlx-int8/``); mlx-audio's ``load`` only resolves
            # config.json at the path root, so point it at the subfolder.
            subfolder = type(self).hf_subfolder
            checkpoint_dir = snapshot_root / subfolder if subfolder else snapshot_root
            # Resolution success is not completeness: when the remote is
            # unreachable, the hub client silently falls back to whatever the
            # local cache holds, so the allowed implicit acquisition above can
            # return the same incomplete directory status just reported.
            # Re-verify with the status check's own rule before the
            # strict=False loader turns a fragment into fluent garbage.
            hub_required = type(self).required_checkpoint_files + type(self).required_snapshot_files
            if not (checkpoint_dir.is_dir() and checkpoint_complete(checkpoint_dir, hub_required)):
                raise ArtifactAcquisitionError(
                    f"The {type(self).hf_repo} snapshot resolved without a "
                    "complete checkpoint; the source may be unreachable and "
                    "the local cache incomplete.",
                    reason="failed",
                    report=report,
                    hint="Run 'standard-asr pull' while the source is reachable.",
                )
            model_source = str(checkpoint_dir)
        # Upstream infers the model family from the LAST path component when
        # config.json omits model_type (NeMo-format repos): a local path --
        # snapshot or operator-provided -- would feed it the commit hash or an
        # arbitrary directory name, so hand it the repo-derived name parts
        # explicitly on both branches (the preset knows its family by
        # construction; _verify_model_family stays the loud backstop for a
        # mismatched checkpoint).
        from mlx_audio.utils import (  # pyright: ignore[reportMissingImports]
            get_model_name_parts,
        )

        load_kwargs["model_name_parts"] = get_model_name_parts(type(self).hf_repo)
        # A few repos mislabel their config.json model_type (e.g. MMS says
        # "wav2vec2"); the preset pins the family so the loader does not mis-route.
        if type(self).load_model_type is not None:
            load_kwargs["model_type"] = type(self).load_model_type
        companion = type(self).companion_tokenizer_repo
        if companion is not None and local_only:
            # File-presence heuristics let this load through (status is
            # limited to cheap inspection, AR.2), but presence cannot
            # prove a malformed tokenizer file will load -- and the
            # upstream hook's fallback on ANY local failure fetches from
            # the Hub with no offline flag, past the no-download policy
            # (round-14 review). Prove the load cannot need a transfer
            # before the hook runs; this also owns the companion-unknown
            # status shape on the no-download path.
            if not offline_tokenizer_loads(model_source, companion):
                raise ArtifactUnavailableError(
                    "The tokenizer for this model cannot load without a "
                    f"network transfer: neither the checkpoint at "
                    f"{model_source!r} nor the local cache of {companion} "
                    "holds a loadable tokenizer, and downloads are disabled.",
                    reason="incomplete",
                    report=report,
                    hint=(
                        "Enable downloads for one load (or run "
                        "'standard-asr pull') to cache the tokenizer, then "
                        "retry offline."
                    ),
                )
        elif (
            companion is not None
            and not bundled_companion_tokenizer(Path(model_source))
            and companion_tokenizer_cached(companion) is not True
        ):
            # A bundled tokenizer keeps the load entirely local (the
            # upstream hook tries the checkpoint directory first); without
            # one, this branch runs with downloads allowed (the guard
            # above refused the no-download cold-cache case, and the
            # probe branch owns every no-download load): fetch the
            # companion through the plugin's own path so a transfer
            # failure classifies as a failed implicit acquisition, not an
            # opaque loader error from the upstream hook's uncontrolled
            # fallback fetch. An unproven cache state is fetched too --
            # the transfer resolves it or fails loudly.
            try:
                fetch_companion_tokenizer(companion)
            except Exception as exc:
                raise_for_gated_source(exc, companion, report)
                raise ArtifactAcquisitionError(
                    f"First-use acquisition of the {companion} companion "
                    f"tokenizer failed: {type(exc).__name__}.",
                    reason="failed",
                    report=report,
                    hint="Run 'standard-asr pull' while the source is reachable.",
                ) from exc
        try:
            self._model = load(model_source, **load_kwargs)
        except Exception as exc:
            raise TranscriptionError(
                f"mlx-audio failed to load the model at {model_source!r}: "
                f"{type(exc).__name__}. Ensure the checkpoint is a valid MLX "
                "bundle and the platform can run MLX; if this model worked "
                "before, run 'standard-asr pull --refresh' to re-fetch any "
                "files its snapshot lacks."
            ) from exc
        self._verify_model_family()
        self._prime_generation_thread()

    def _prime_generation_thread(self) -> None:
        """Run one tiny generate on the loading thread (MLX thread affinity).

        mlx-audio's generate path materializes its Metal stream state lazily,
        the first time ``model.generate`` runs. If that first run happens on a
        different thread than the load, MLX aborts the whole process with an
        uncaught C++ ``std::runtime_error`` ("There is no Stream(gpu, N) in
        current thread") — exactly the shape of a host app that warms the
        engine on its main thread (``ensure_loaded`` / the ``model`` property)
        and then streams from a session pump thread. Priming with 0.1 s of
        silence ON THIS THREAD pins the stream state here; afterwards,
        generates from any thread are stable (both orders verified
        empirically; present on at least mlx 0.31-0.32 / mlx-audio 0.4.4-0.4.5,
        so this is not version-gated). It also pre-compiles the Metal kernels,
        making the load a real warmup.

        A priming failure downgrades to a log warning: a family that rejects
        near-empty audio but works on real input must still load; the warning
        preserves the trace for the (rare) cross-thread crash window that
        remains.
        """
        backend = type(self).backend
        config = cast(MlxAudioConfig, self.config)
        gen_kwargs = backend.generate_kwargs(
            resolved_language=None,
            want_words=False,
            params=MlxAudioParams(),
            config=config,
        )
        silence: NDArray[np.float32] = np.zeros(_SAMPLE_RATE // 10, dtype=np.float32)
        source: Any
        if getattr(backend, "wants_path", False):
            source = self._array_to_wav_tempfile(silence)
        else:
            source = backends.to_mlx_array(silence)
        model = cast(Any, self._model)
        try:
            model.generate(backends.adapt_audio_source(backend, source), **gen_kwargs)
        except Exception:
            _LOGGER.warning(
                "Priming generate failed for %s; the first real generate should "
                "run on the thread that loaded the model, or MLX may abort the "
                "process (stream thread affinity).",
                type(self).__name__,
                exc_info=True,
            )

    def _verify_model_family(self) -> None:
        """Assert the loaded model's family matches this preset's backend (spec IC.7).

        mlx-audio's ``load`` auto-detects the model family from the checkpoint's
        ``config.json``; this preset, however, binds ONE backend that knows
        exactly one family's ``generate`` call-shape and output schema. If a
        ``model_path`` (or ``revision``) override resolves to a DIFFERENT family,
        running it through the wrong adapter would silently produce a wrong
        transcript or crash — the cardinal sin — so we fail loudly here instead.

        Raises:
            DiscoveryError: If the loaded model's family is not one this preset's
                backend declares in ``model_types``.
        """
        family = _model_family(self._model)
        if family is None:
            # Could not introspect the model's module path (unexpected layout);
            # do not block a possibly-valid load on a check we cannot make.
            return
        allowed = type(self).backend.model_types
        if family not in allowed:
            raise DiscoveryError(
                f"The loaded MLX model is a {family!r} model, which the "
                f"{type(self).properties.model_id!r} preset cannot run (its "
                f"backend handles {allowed}). This usually means a 'model_path' "
                "override points at a different model family; use the matching "
                "preset, or point 'model_path' at a compatible checkpoint."
            )

    def prepare(self) -> None:
        """Warm up the MLX model without transcribing (spec IC.11).

        Idempotent and synchronous. The warm-up loads the model and runs the
        thread-affinity priming; a cold cache acquires the snapshot on this
        path too (under the same download policy and artifact errors as
        inference). Explicit artifact-only acquisition, which never loads or
        primes, is :meth:`acquire_artifacts`.

        Raises:
            ArtifactUnavailableError: If required artifacts are missing and
                cannot be acquired under the current policy.
            ArtifactAcquisitionError: If an allowed acquisition attempt fails.
            TranscriptionError: If a locally available model fails to load.
        """
        self._ensure_model_loaded()

    # ------------------------------------------------------------------ #
    # Inference-artifact lifecycle (protocol 1.1)
    # ------------------------------------------------------------------ #
    def _artifact_requirements(
        self,
        context: ArtifactContext,
    ) -> tuple[bool, tuple[ArtifactRequirement, ...], tuple[Diagnostic, ...]]:
        """Report the snapshot requirement for the resolved config.

        Batch and streaming share the same snapshot, so the closure does not
        depend on the request context. The MMS preset deliberately reports ONE
        filtered-snapshot requirement (base plus every safetensors adapter):
        the current plugin has no language-adapter selector, so claiming
        request-dependent adapter requirements would describe an architecture
        it does not have.

        Args:
            context: Resolved, best-effort-gated request context.

        Returns:
            One requirement: the Hub snapshot or the operator-provided path.
        """
        config = cast(MlxAudioConfig, self.config)
        requirement = status_requirement(
            config,
            type(self).hf_repo,
            type(self).hf_subfolder,
            type(self).required_checkpoint_files,
            type(self).required_snapshot_files,
            type(self).companion_tokenizer_repo,
        )
        return True, (requirement,), ()

    def _acquire_artifacts(
        self,
        context: ArtifactContext,
        requirements: tuple[ArtifactRequirement, ...],
        refresh: bool,
        progress: ArtifactProgressCallback | None,
    ) -> None:
        """Acquire the preset's filtered snapshot without loading or priming.

        A ``model_path`` requirement is externally provided and reaches this
        hook only when its single acquirable part -- the companion tokenizer
        of a preset that declares one -- is what is missing; then the hook
        fetches exactly that. A refresh carries its own re-resolution
        evidence: ``snapshot_download`` silently falls back to the local
        cache when the remote is unreachable, so ``acquire`` verifies the
        source resolution itself (spec AR.4). ``pull`` never runs the
        priming inference that :meth:`prepare` keeps.

        Args:
            context: Resolved artifact context.
            requirements: Runnable acquisition and refresh targets.
            refresh: Whether mutable targets must be re-resolved.
            progress: Serialized progress observer, if requested.

        Returns:
            None.
        """
        config = cast(MlxAudioConfig, self.config)
        companion = type(self).companion_tokenizer_repo
        if any(item.artifact_id == HUB_ARTIFACT_ID for item in requirements):
            acquire(
                config,
                type(self).hf_repo,
                progress,
                refresh=refresh,
                companion_repo=companion,
            )
        elif companion is not None and any(
            item.artifact_id == LOCAL_ARTIFACT_ID for item in requirements
        ):
            acquire_companion(config, companion, progress)

    # ------------------------------------------------------------------ #
    # Batch
    # ------------------------------------------------------------------ #
    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        """Transcribe negotiated audio by dispatching to the bound backend.

        Resolves the language axis, builds the family-specific ``generate``
        kwargs via the backend, runs ``model.generate`` (blocking MLX call), and
        maps the native return value onto a constant-schema result via the same
        backend.

        Args:
            prepared: Engine-ready audio (an array, a file path, or in-memory
                bytes — one of the declared ``accepted_input`` shapes).
            params: Gated runtime parameters.

        Returns:
            A Standard ASR transcription result.

        Raises:
            TranscriptionError: If the MLX backend raises during inference. The
                batch error contract (spec RT R7) requires an engine-execution
                failure to surface as a portable ``TranscriptionError`` with the
                native exception preserved as ``__cause__``.
        """
        self._ensure_model_loaded()
        backend = type(self).backend
        config = cast(MlxAudioConfig, self.config)

        resolved = effective_language(
            params.language,
            config.default_language,
            has_language_axis=self._has_language_axis(),
            runtime_override_supported=self._runtime_override_supported(),
        )
        resolved_language = None if (resolved is None or resolved == "auto") else resolved
        want_words = map_word_timestamps(params.word_timestamps)
        mlx_params = (
            params.provider_params
            if isinstance(params.provider_params, MlxAudioParams)
            else MlxAudioParams()
        )

        source, duration = self._source_for(prepared)
        gen_kwargs = backend.generate_kwargs(
            resolved_language=resolved_language,
            want_words=want_words,
            params=mlx_params,
            config=config,
        )
        model = cast(Any, self._model)
        try:
            native = model.generate(backends.adapt_audio_source(backend, source), **gen_kwargs)
        except Exception as exc:
            raise TranscriptionError(f"MLX transcription failed: {type(exc).__name__}.") from exc
        return backend.to_result(native, duration=duration, want_words=want_words)

    def _source_for(self, prepared: PreparedAudio) -> tuple[Any, float | None]:
        """Map negotiated audio onto the source mlx-audio accepts (+ duration).

        mlx-audio ``model.generate`` accepts a path (it decodes/resamples
        internally) or a decoded waveform. We pass the negotiated float32 array
        through as an ``mx.array`` (the one decoded shape every backend accepts —
        see ``backends.to_mlx_array``), else the path; for in-memory bytes we
        materialize a temp file lazily (mlx-audio has no bytes entry point).

        Args:
            prepared: The negotiated audio (array / bytes / path).

        Returns:
            A ``(source, duration_seconds_or_None)`` pair. Duration is known only
            for the array path (from the sample count); for path/bytes the
            backend leaves duration ``None`` (mlx-audio does not return it).
        """
        # A few families' generate only works via their file-path branch (their
        # array path is broken upstream — e.g. Voxtral-Mini); hand them a path,
        # materializing a temp WAV from a negotiated array when needed.
        if getattr(type(self).backend, "wants_path", False):
            if prepared.path is not None:
                return prepared.path, None
            if prepared.data is not None:
                return self._bytes_to_tempfile(prepared.data), None
            if prepared.array is not None:
                arr_p: NDArray[np.float32] = np.ascontiguousarray(prepared.array, dtype=np.float32)
                return self._array_to_wav_tempfile(arr_p), waveform_duration(arr_p)
            raise TranscriptionError("Negotiated audio carried no array, path, or bytes payload.")
        if prepared.array is not None:
            # We declare accepted_sample_rates=[16000]; the standard layer
            # negotiates to it. Assert defensively — an off-rate array silently
            # produces wrong timings/text.
            assert prepared.sample_rate == _SAMPLE_RATE, (
                f"MLX STT requires 16 kHz audio; got {prepared.sample_rate} Hz "
                "(audio negotiation should have resampled to 16000)."
            )
            arr: NDArray[np.float32] = np.ascontiguousarray(prepared.array, dtype=np.float32)
            return backends.to_mlx_array(arr), waveform_duration(arr)
        if prepared.path is not None:
            return prepared.path, None
        if prepared.data is not None:
            return self._bytes_to_tempfile(prepared.data), None
        # Defensive: negotiation always delivers one of our accepted shapes.
        raise TranscriptionError("Negotiated audio carried no array, path, or bytes payload.")

    @staticmethod
    def _array_to_wav_tempfile(audio: NDArray[np.float32]) -> str:
        """Write a 16 kHz mono float32 waveform to a temp WAV and return its path.

        For ``wants_path`` families whose ``generate`` only accepts a file path;
        the array is already at the negotiated 16 kHz (accepted_sample_rates).

        Args:
            audio: A contiguous float32 mono waveform at :data:`_SAMPLE_RATE`.

        Returns:
            The temp ``.wav`` path (left on disk for the loader; the OS temp dir
            is reclaimed by the platform — we do not hold a handle).
        """
        import tempfile
        import wave

        # numpy's stubs leave the dtype of arithmetic results partially unknown
        # under pyright strict, so do the int16 PCM math through an Any buffer and
        # assert the final wire type as bytes for the wave writer.
        buf: Any = audio
        pcm16: Any = (buf.clip(-1.0, 1.0) * 32767.0).round().astype(np.int16)
        frames: bytes = pcm16.tobytes()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            with wave.open(handle, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(_SAMPLE_RATE)
                wav.writeframes(frames)
            return handle.name

    @staticmethod
    def _bytes_to_tempfile(data: bytes) -> str:
        """Write encoded audio bytes to a temp file mlx-audio can open.

        Args:
            data: Encoded audio bytes (a canonical WAV, per the standard layer's
                array->bytes negotiation, or a passed-through encoded upload).

        Returns:
            The temp file path (left on disk for mlx-audio to read; the OS temp
            dir is reclaimed by the platform — we do not hold a handle).
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(data)
            return handle.name

    def _has_language_axis(self) -> bool:
        """Return whether this preset exposes a selectable language axis.

        Returns:
            ``True`` when ``selectable_languages`` is non-empty (Qwen3-ASR /
            Whisper); ``False`` would mean no axis. Parakeet still lists its
            fixed languages, so it has an axis — but no runtime override (see
            :meth:`_runtime_override_supported`).
        """
        return bool(type(self).properties.selectable_languages)

    def _runtime_override_supported(self) -> bool:
        """Return whether per-request language override is supported (from caps).

        Reads the declared batch ``language.runtime_override`` flag, so a
        fixed-language model (Parakeet) correctly resolves to its default rather
        than honoring a per-request override.

        Returns:
            ``True`` iff the batch capabilities allow a runtime language override.
        """
        return bool(type(self).declared_capabilities.supports("batch.language.runtime_override"))

    # ------------------------------------------------------------------ #
    # Streaming (windowed)
    # ------------------------------------------------------------------ #
    def _start_transcription(
        self,
        *,
        gated_params: RuntimeParams,
        audio_format: AudioFormat | None,
        prepared_audio: PreparedAudio | None,
    ) -> TranscriptionSession:
        """Open a windowed streaming session (spec ST; see ``_streaming.py``).

        The base ``start_transcription`` template has already enforced the
        ``audio_format`` / ``audio`` exclusivity, validated the language config,
        run the fail-closed wire-format check, and gated + frozen the params
        (spec RT R5). We just build the session.

        For the whole-input path (``audio=...``) the base hands us a negotiated
        ``prepared_audio``; we pre-load it into the session's buffer so the
        OpenAI-style "submit a file, stream the result" pattern works too.

        Args:
            gated_params: Frozen, gated runtime parameters.
            audio_format: The incremental wire format, or ``None``.
            prepared_audio: The negotiated whole input, or ``None``.

        Returns:
            A windowed streaming session for this engine.
        """
        # Imported here (not at module top) so the streaming module's import of
        # this class does not cycle at import time.
        from ._streaming import MlxAudioStreamingSession

        config = cast(MlxAudioConfig, self.config)
        session = MlxAudioStreamingSession(
            self,
            gated_params,
            redecode_interval_s=config.redecode_interval_s,
            settle_margin_s=config.settle_margin_s,
            max_window_s=config.max_window_s,
        )
        if prepared_audio is not None:
            session.feed(_prepared_to_pcm(prepared_audio))
        return session


def _model_family(model: object | None) -> str | None:
    """Return the mlx-audio family name of a loaded model, or ``None``.

    mlx-audio instantiates each family's ``Model`` class from
    ``mlx_audio.stt.models.<family>.<module>``, so the family is the path segment
    immediately after ``models`` in the model class's ``__module__`` (e.g.
    ``mlx_audio.stt.models.whisper.whisper`` -> ``"whisper"``;
    ``...models.mega_asr.mega_asr`` -> ``"mega_asr"``). Used by
    :meth:`MlxAudioASR._verify_model_family` to fail loudly on a family/backend
    mismatch.

    Args:
        model: A loaded mlx-audio model instance, or ``None``.

    Returns:
        The family name, or ``None`` if it cannot be determined.
    """
    if model is None:
        return None
    parts = (type(model).__module__ or "").split(".")
    try:
        index = parts.index("models")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None


def _disable_tqdm_monitor_thread() -> None:
    """Disable tqdm's auto-spawned monitor daemon thread (idempotent, best-effort).

    Setting ``tqdm.monitor_interval = 0`` before any tqdm instance is created
    prevents the persistent ``tqdm_monitor`` thread that would otherwise leak past
    a session and trip the sync-bridge compliance check. Progress bars still
    render; only the background stall-detector thread is suppressed. If tqdm is
    absent or already started its monitor, this is a no-op (it only ever
    *disables*).
    """
    try:
        import tqdm

        if tqdm.tqdm.monitor_interval != 0:
            tqdm.tqdm.monitor_interval = 0
    except Exception:
        pass


def _prepared_to_pcm(prepared: PreparedAudio) -> bytes:
    """Convert negotiated whole-input audio into canonical pcm_s16le bytes.

    Used only on the streaming whole-input path. The negotiated audio is a 16 kHz
    mono float32 array (we declare ``accepted_input`` includes ``array`` and the
    standard layer resamples to our native rate), so we quantize to int16 LE.

    Args:
        prepared: The negotiated whole-input audio.

    Returns:
        Canonical 16-bit LE PCM bytes.

    Raises:
        TranscriptionError: If the whole input did not arrive as an array.
    """
    if prepared.array is None:
        raise TranscriptionError(
            "Streaming whole-input audio was not delivered as an array; "
            "expected a negotiated 16 kHz float32 array."
        )
    arr: NDArray[np.float32] = np.nan_to_num(
        prepared.array, nan=0.0, posinf=1.0, neginf=-1.0
    ).astype(np.float32)
    clipped: NDArray[np.float32] = arr.clip(-1.0, 1.0)
    quantized: NDArray[np.int16] = np.round(clipped * 32767.0).astype("<i2")
    return quantized.tobytes()


# --------------------------------------------------------------------------- #
# Presets. Each MLX model is its own entry point (spec IC.7) so discovery can
# enumerate every available model. A preset overrides only hf_repo, backend,
# properties, and declared_capabilities; the config, params, and the
# transcribe/stream pipeline are inherited unchanged. This is "one engine, many
# models" — three DIFFERENT backend families under one engine_id.
# --------------------------------------------------------------------------- #
#: The Qwen3-ASR repos ship a GPT2-style tokenizer (no tokenizer.json), so all
#: four files are single points of failure for the hook's AutoTokenizer +
#: WhisperFeatureExtractor calls (per-file ablation, round-9 review).
_QWEN3_SNAPSHOT_FILES = (
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
)


class Qwen3Asr06B(MlxAudioASR):
    """``mlx-audio/qwen3-asr-0.6b`` — small Qwen3-ASR (the headliner)."""

    hf_repo: ClassVar[str] = "mlx-community/Qwen3-ASR-0.6B-4bit"
    required_snapshot_files: ClassVar[tuple[str, ...]] = _QWEN3_SNAPSHOT_FILES
    backend: ClassVar[ModelBackend] = Qwen3AsrBackend()
    properties: ClassVar[BaseProperties] = Qwen3Asr06BProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _QWEN_CAPABILITIES


class Qwen3Asr17B(MlxAudioASR):
    """``mlx-audio/qwen3-asr-1.7b`` — larger, more accurate Qwen3-ASR."""

    hf_repo: ClassVar[str] = "mlx-community/Qwen3-ASR-1.7B-8bit"
    required_snapshot_files: ClassVar[tuple[str, ...]] = _QWEN3_SNAPSHOT_FILES
    backend: ClassVar[ModelBackend] = Qwen3AsrBackend()
    properties: ClassVar[BaseProperties] = Qwen3Asr17BProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _QWEN_CAPABILITIES


class ParakeetTdt06BV3(MlxAudioASR):
    """``mlx-audio/parakeet-tdt-0.6b-v3`` — NVIDIA Parakeet TDT (word timing).

    Parakeet has no runtime language selection and no ``"auto"`` directive in its
    ``selectable_languages``, so it defaults ``default_language`` to ``"en"`` (a
    member of its set) to satisfy LANG R1; the value is inert at inference (the
    model ignores language), but the standard layer requires a valid default for
    any engine exposing a language axis.
    """

    hf_repo: ClassVar[str] = "mlx-community/parakeet-tdt-0.6b-v3"
    backend: ClassVar[ModelBackend] = AlignedResultBackend(model_types=("parakeet",))
    properties: ClassVar[BaseProperties] = ParakeetTdt06BV3Properties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _PARAKEET_CAPABILITIES
    default_config_overrides: ClassVar[dict[str, Any]] = {"default_language": "en"}


class WhisperLargeV3Turbo(MlxAudioASR):
    """``mlx-audio/whisper-large-v3-turbo`` — fast multilingual Whisper.

    Points at the **OpenAI** repo, not ``mlx-community/whisper-large-v3-turbo``:
    mlx-audio's Whisper backend requires a ``WhisperProcessor`` (tokenizer +
    feature extractor) loaded from the repo, and the mlx-community Whisper repos
    ship only ``config.json`` + ``weights.safetensors`` (no processor files), so
    they fail at first transcription with "Processor not found". The OpenAI repos
    ship the full processor and mlx-audio loads/quantizes them on first use. See
    docs/STANDARD_ASR_FINDINGS.md.
    """

    hf_repo: ClassVar[str] = "openai/whisper-large-v3-turbo"
    # Without it the WhisperProcessor load only WARNS and the first generate
    # fails ("Processor not found"). The single point of failure is the
    # feature-extractor config: the OpenAI repos ship BOTH tokenizer layouts
    # (tokenizer.json and vocab + merges), so no single tokenizer file is
    # load-bearing (per-file ablation, round-9 review).
    required_snapshot_files: ClassVar[tuple[str, ...]] = ("preprocessor_config.json",)
    backend: ClassVar[ModelBackend] = WhisperBackend()
    properties: ClassVar[BaseProperties] = WhisperLargeV3TurboProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _WHISPER_CAPABILITIES


class WhisperTiny(MlxAudioASR):
    """``mlx-audio/whisper-tiny`` — smallest Whisper (smoke/tests).

    Points at the OpenAI repo (ships the required ``WhisperProcessor``); the
    mlx-community Whisper repos omit the processor and fail at load. See
    :class:`WhisperLargeV3Turbo`.
    """

    hf_repo: ClassVar[str] = "openai/whisper-tiny"
    # See WhisperLargeV3Turbo: the feature-extractor config is the single
    # point of failure for the processor load.
    required_snapshot_files: ClassVar[tuple[str, ...]] = ("preprocessor_config.json",)
    backend: ClassVar[ModelBackend] = WhisperBackend()
    properties: ClassVar[BaseProperties] = WhisperTinyProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _WHISPER_CAPABILITIES


# --------------------------------------------------------------------------- #
# Aligned-output preset (token timing): NVIDIA Nemotron ASR.
# Shares the AlignedResult shape with Parakeet, so it reuses AlignedResultBackend
# and declares word+segment timestamps; its language keys are model-specific, so
# language.runtime_override stays False (honest) — see backends.AlignedResultBackend.
# --------------------------------------------------------------------------- #
class NemotronAsrStreaming06B(MlxAudioASR):
    """``mlx-audio/nemotron-asr-streaming-0.6b`` — NVIDIA Nemotron ASR (word timing)."""

    hf_repo: ClassVar[str] = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
    backend: ClassVar[ModelBackend] = AlignedResultBackend(model_types=("nemotron_asr",))
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="nemotron-asr-streaming-0.6b",
        description=(
            "NVIDIA Nemotron 3.5 ASR streaming 0.6B (MLX); English ASR with "
            "precise word/segment timestamps."
        ),
        selectable=[],
        detectable=[],
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_WORD, runtime_override=False, streaming=True
    )


# --------------------------------------------------------------------------- #
# Generic STTOutput presets. Each binds a GenericSttBackend(SttFamilySpec(...))
# that declares the family's language axis, timing honesty, input quirks, and
# decode knobs. Capabilities mirror the spec: segment timestamps + streaming ONLY
# where the model emits real per-chunk timing; language.runtime_override ONLY
# where it accepts a language; otherwise text-only + batch-only (honest).
# --------------------------------------------------------------------------- #
class SenseVoiceSmall(MlxAudioASR):
    """``mlx-audio/sensevoice-small`` — FunAudioLLM SenseVoice (language ID + ITN)."""

    hf_repo: ClassVar[str] = "mlx-community/SenseVoiceSmall"
    # Both are SILENT when absent (round-8 review): without the bpe model
    # the decoder emits numeric token ids as the transcript, and without
    # am.mvn the feature normalization is silently skipped (the config
    # carries no cmvn fallback). The loader reads both by fixed name, so
    # they gate every checkpoint, model_path included. am.mvn needs the
    # plugin's extended snapshot allow patterns -- the upstream defaults
    # never fetch it.
    required_checkpoint_files: ClassVar[tuple[str, ...]] = (
        "am.mvn",
        "chn_jpn_yue_eng_ko_spectok.bpe.model",
    )
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(
            model_types=("sensevoice",),
            language_kwarg="language",
            reports_detected_language=True,
            forward=(("use_itn", "use_itn"),),
        )
    )
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="sensevoice-small",
        description=(
            "FunAudioLLM SenseVoice-Small (MLX); multilingual (zh/en/yue/ja/ko) "
            "ASR with language detection and optional ITN. Batch only."
        ),
        selectable=SENSEVOICE_LANGUAGES,
        detectable=SENSEVOICE_DETECTABLE_LANGUAGES,
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_NONE, runtime_override=True, streaming=False
    )


class CohereAsr(MlxAudioASR):
    """``mlx-audio/cohere-asr`` — Cohere ASR (14 languages, VAD segment timing).

    Caveat: the only public MLX checkpoint stores its weights in a repo
    *subfolder* (``mlx-int8/``), which ``mlx_audio.stt.load`` may not resolve from
    the repo root — point ``model_path`` at a local copy of that subfolder if a
    plain load fails. See VERIFICATION.md.
    """

    hf_repo: ClassVar[str] = "appautomaton/cohere-asr-mlx"
    # The repo stores its checkpoint (config.json, weights, tokenizer) under the
    # ``mlx-int8/`` subfolder, not the repo root, so point the loader there.
    hf_subfolder: ClassVar[str | None] = "mlx-int8"
    # SILENT when absent (round-9 review): the upstream tokenizer falls back
    # to hardcoded token DEFAULTS when tokenizer_config.json is missing, so
    # additional_special_tokens becomes empty and the 200-plus language,
    # task, and speaker tags leak into the transcript unstripped.
    required_checkpoint_files: ClassVar[tuple[str, ...]] = ("tokenizer_config.json",)
    # LOUD when absent (sentencepiece raises at load), so it gates only the
    # snapshot; the exact filename is this repo's layout.
    required_snapshot_files: ClassVar[tuple[str, ...]] = ("tokenizer.model",)
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(
            model_types=("cohere_asr",),
            language_kwarg="language",
            segment_timing=True,
        )
    )
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="cohere-asr",
        description=(
            "Cohere ASR (MLX, int8); 14-language ASR with VAD-based segment "
            "timing. Public checkpoint stores weights in a subfolder (see docs)."
        ),
        selectable=COHERE_LANGUAGES,
        detectable=[],
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_SEGMENT, runtime_override=True, streaming=True
    )
    default_config_overrides: ClassVar[dict[str, Any]] = {"default_language": "en"}


class FunAsrNano(MlxAudioASR):
    """``mlx-audio/fun-asr-nano`` — Fun-ASR-Nano (hotwords + ITN, per-chunk timing)."""

    hf_repo: ClassVar[str] = "mlx-community/Fun-ASR-Nano-2512"
    # No declared closure files: the hook loads its Qwen tokenizer from the
    # config-named ``Qwen3-0.6B/`` subdirectory, which ships BOTH tokenizer
    # layouts, so no single file is a point of failure (per-file ablation,
    # round-9 review); a snapshot missing the whole subdirectory fails
    # loudly at load and 'pull --refresh' repairs it.
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(
            model_types=("fun_asr_nano",),
            language_kwarg="language",
            segment_timing=True,
            forward=(("hotwords", "hotwords"), ("itn", "use_itn")),
        )
    )
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="fun-asr-nano",
        description=(
            "Fun-ASR-Nano (MLX); Qwen3-based ASR for Chinese/English/Japanese "
            "with hotword biasing, ITN, and per-chunk segment timing."
        ),
        selectable=FUN_ASR_LANGUAGES,
        detectable=FUN_ASR_DETECTABLE_LANGUAGES,
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_SEGMENT, runtime_override=True, streaming=True
    )


class VoxtralMini3B(MlxAudioASR):
    """``mlx-audio/voxtral-mini-3b`` — Mistral Voxtral-Mini 3B (multilingual).

    Voxtral's ``generate`` only works via its file-path branch: handed a decoded
    array it routes through a transformers processor whose
    ``apply_transcription_request`` needs a ``format`` argument mlx-audio never
    passes (``len(None)`` -> ``TypeError``). So this preset declares ``wants_path``
    and the engine hands it a path (materializing a temp WAV from a negotiated
    array/bytes when needed).

    KNOWN LIMITATION (upstream): even on the path branch, transformers'
    ``VoxtralProcessor.apply_transcription_request`` pulls in the full Mistral
    inference stack — ``librosa`` (``load_audio_as``), then ``mistral_common``
    (``TranscriptionRequest``), then ``mistral_common[audio]``, and further missing
    pieces — which mlx-audio does not declare and which does not converge cleanly.
    We deliberately do NOT add those heavyweight deps for one batch-only model, so
    this preset currently raises a dependency ``ImportError`` at transcription on a
    base install. **Use ``voxtral-realtime-4b`` for Voxtral on this stack** (it runs
    out of the box). ``wants_path`` is kept as the correct input shape so that, if
    the Mistral stack is installed, the path branch is the one that can work.
    """

    hf_repo: ClassVar[str] = "mlx-community/Voxtral-Mini-3B-2507-bf16"
    # Both are single points of failure for the hook's AutoProcessor call:
    # the feature-extractor config and the tekken tokenizer serialization
    # (per-file ablation, round-9 review).
    required_snapshot_files: ClassVar[tuple[str, ...]] = (
        "preprocessor_config.json",
        "tekken.json",
    )
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(
            model_types=("voxtral",),
            language_kwarg="language",
            wants_path=True,
        )
    )
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="voxtral-mini-3b",
        description=(
            "Mistral Voxtral-Mini 3B (MLX, bf16); multilingual ASR. Large (~9 GB); batch only."
        ),
        selectable=VOXTRAL_LANGUAGES,
        detectable=[],
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_NONE, runtime_override=True, streaming=False
    )
    default_config_overrides: ClassVar[dict[str, Any]] = {"default_language": "en"}


class Canary1BV2(MlxAudioASR):
    """``mlx-audio/canary-1b-v2`` — NVIDIA Canary (25 EU languages + translation).

    Set the ``target_language`` provider param to translate the transcript into a
    different language (Canary's source/target axes); otherwise it transcribes in
    the selected source language.

    Repo note: points at the ``CogniSoftOrg`` bf16 conversion, NOT the smaller
    ``TechHara`` q4/q8 repos. mlx-audio's Canary backend needs a SentencePiece
    tokenizer (a ``tokenizer.model`` file, an embedded base64 in config.json, or a
    ``tokens.txt`` paired with one) to ``decode()``; the TechHara repos ship only a
    ``tokens.json`` vocab list (no SentencePiece source), so loading them yields a
    tokenizer-less model that raises ``RuntimeError: Tokenizer not loaded`` at
    generate(). The CogniSoftOrg repo ships ``tokenizer.model`` and declares
    ``model_type=canary``, so it loads and decodes correctly.
    """

    hf_repo: ClassVar[str] = "CogniSoftOrg/canary-1b-v2-mlx-bf16"
    # Loaded optionally by the upstream hook; absence surfaces only at the
    # first generate, so it is part of the SNAPSHOT closure. It must not
    # gate a model_path: the hook also accepts a tokens.txt vocabulary or a
    # config-embedded base64 tokenizer, so a local checkpoint using either
    # alternative is valid (round-9 review).
    required_snapshot_files: ClassVar[tuple[str, ...]] = ("tokenizer.model",)
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(
            model_types=("canary",),
            language_kwarg="source_lang",
            translate_target_kwarg="target_lang",
        )
    )
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="canary-1b-v2",
        description=(
            "NVIDIA Canary 1B v2 (MLX, bf16); 25-language European ASR with "
            "speech translation (set target_language). ~3.2 GB. Batch only."
        ),
        selectable=CANARY_LANGUAGES,
        detectable=[],
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_NONE, runtime_override=True, streaming=False
    )
    default_config_overrides: ClassVar[dict[str, Any]] = {"default_language": "en"}


class Qwen2Audio7B(MlxAudioASR):
    """``mlx-audio/qwen2-audio-7b`` — Qwen2-Audio 7B Instruct (audio-LLM ASR)."""

    hf_repo: ClassVar[str] = "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
    # The single point of failure for the hook's AutoProcessor call; the
    # repo ships both tokenizer layouts, so no tokenizer file qualifies
    # (per-file ablation, round-9 review).
    required_snapshot_files: ClassVar[tuple[str, ...]] = ("preprocessor_config.json",)
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(
            model_types=("qwen2_audio",),
            # Qwen2-Audio is an instruction audio-LLM; its default prompt
            # ("Please transcribe the speech.") yields conversational output like
            # "The speech is in English, with the transcription being: '…'". A
            # strict prompt steers it to emit the bare transcript.
            default_prompt=(
                "Transcribe the spoken audio into text verbatim. "
                "Output only the transcript, with no preamble, labels, or quotation marks."
            ),
        )
    )
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="qwen2-audio-7b",
        description="Qwen2-Audio 7B Instruct (MLX, 4-bit); audio-LLM transcription. Batch only.",
        selectable=[],
        detectable=[],
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_NONE, runtime_override=False, streaming=False
    )


class GlmAsrNano(MlxAudioASR):
    """``mlx-audio/glm-asr-nano`` — GLM-ASR-Nano (per-chunk segment timing)."""

    hf_repo: ClassVar[str] = "mlx-community/GLM-ASR-Nano-2512-4bit"
    # The single point of failure for the hook's AutoTokenizer call: the
    # repo ships no slow-layout fallback (per-file ablation, round-9 review).
    required_snapshot_files: ClassVar[tuple[str, ...]] = ("tokenizer.json",)
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(model_types=("glmasr",), segment_timing=True)
    )
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="glm-asr-nano",
        description="GLM-ASR-Nano (MLX, 4-bit); compact ASR with per-chunk segment timing.",
        selectable=[],
        detectable=[],
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_SEGMENT, runtime_override=False, streaming=True
    )


class GraniteSpeech1B(MlxAudioASR):
    """``mlx-audio/granite-speech-1b`` — IBM Granite Speech (ASR + translation).

    Set the ``target_language`` provider param to translate the transcript
    (Granite's prompt-driven ``Translate the speech to <lang>`` mode); otherwise
    it transcribes in the spoken language.
    """

    hf_repo: ClassVar[str] = "mlx-community/granite-4.0-1b-speech-5bit"
    # No declared closure files: the repo ships BOTH tokenizer layouts
    # (tokenizer.json and vocab + merges), so no single file is a point of
    # failure for the hook's AutoTokenizer call (per-file ablation,
    # round-9 review).
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(model_types=("granite_speech",), translate_target_kwarg="language")
    )
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="granite-speech-1b",
        description=(
            "IBM Granite 4.0 1B Speech (MLX, 5-bit); ASR with prompt-driven "
            "speech translation (set target_language). Batch only."
        ),
        selectable=[],
        detectable=[],
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_NONE, runtime_override=False, streaming=False
    )


class GraniteSpeechNar2B(MlxAudioASR):
    """``mlx-audio/granite-speech-nar-2b`` — IBM Granite Speech NAR (fast, non-AR).

    Its ``generate`` -> ``_load_waveform`` accepts an ``mx.array`` directly (assumed
    16 kHz) but, given a file path, reads it with ``soundfile`` and REFUSES to
    resample (``ValueError: audio must be 16000 Hz``). So this preset takes a
    decoded array only (``array_only``): the standard layer resamples to 16 kHz and
    the engine passes the array straight through — no path, no soundfile.
    """

    hf_repo: ClassVar[str] = "mlx-community/granite-speech-4.1-2b-nar-mlx"
    # The single point of failure for the hook's AutoTokenizer call: unlike
    # the 1B repo, this one ships no slow-layout fallback (per-file
    # ablation, round-9 review).
    required_snapshot_files: ClassVar[tuple[str, ...]] = ("tokenizer.json",)
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(model_types=("granite_speech_nar",))
    )
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="granite-speech-nar-2b",
        description=(
            "IBM Granite 4.1 2B Speech NAR (MLX); fast non-autoregressive ASR. Batch only."
        ),
        selectable=[],
        detectable=[],
        array_only=True,
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_NONE, runtime_override=False, streaming=False
    )


class VibeVoiceAsr(MlxAudioASR):
    """``mlx-audio/vibevoice-asr`` — Microsoft VibeVoice-ASR (context-biased).

    The checkpoint repo ships NO tokenizer files; the upstream hook falls
    back to fetching the ``Qwen/Qwen2.5-7B`` tokenizer from the Hub at load
    time, past the engine's ``local_files_only``. The preset declares that
    repo as its :attr:`companion_tokenizer_repo`, so status reports the
    snapshot incomplete until the tokenizer is in the DEFAULT Hugging Face
    cache (``download_root`` cannot redirect the upstream call), ``pull``
    acquires it, and a load under a no-download policy refuses instead of
    letting the upstream fetch run. A checkpoint that bundles its own
    loadable tokenizer (the hook tries the checkpoint directory first) is
    self-contained and needs no companion. Residual caveat: with the
    tokenizer cached, downloads disabled, and the network reachable, the
    upstream call may still revalidate against the Hub and fetch an
    updated file if the source repo moved. See
    docs/STANDARD_ASR_FINDINGS.md.
    """

    hf_repo: ClassVar[str] = "mlx-community/VibeVoice-ASR-4bit"
    companion_tokenizer_repo: ClassVar[str | None] = "Qwen/Qwen2.5-7B"
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(
            model_types=("vibevoice_asr",),
            # VibeVoice returns a diarization JSON: STTOutput.text is the raw JSON
            # string while STTOutput.segments are the parsed {start,end,text} dicts.
            # Rebuild the transcript from the parsed segments rather than surfacing
            # raw JSON. We do NOT declare segment timing (the model stays honestly
            # text-only/batch-only), so segments are used for text only, not emitted.
            text_from_segments=True,
            forward=(("context", "context"),),
        )
    )
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="vibevoice-asr",
        description=(
            "Microsoft VibeVoice-ASR 8B (MLX, 4-bit); context-biased ASR "
            "(set the context provider param). Batch only."
        ),
        selectable=[],
        detectable=[],
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_NONE, runtime_override=False, streaming=False
    )


class MoonshineTiny(MlxAudioASR):
    """``mlx-audio/moonshine-tiny`` — UsefulSensors Moonshine tiny (English, ~27M)."""

    hf_repo: ClassVar[str] = "UsefulSensors/moonshine-tiny"
    # SILENT when absent (round-8 review): the upstream hook swallows the
    # tokenizer load failure and decode falls back to per-id characters, so
    # the file gates every checkpoint, model_path included. The repo ships
    # no slow-layout fallback (per-file ablation, round-9 review).
    required_checkpoint_files: ClassVar[tuple[str, ...]] = ("tokenizer.json",)
    backend: ClassVar[ModelBackend] = GenericSttBackend(SttFamilySpec(model_types=("moonshine",)))
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="moonshine-tiny",
        description="UsefulSensors Moonshine tiny (MLX); tiny fast English ASR. Batch only.",
        selectable=[],
        detectable=[],
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_NONE, runtime_override=False, streaming=False
    )


class Mms1BAll(MlxAudioASR):
    """``mlx-audio/mms-1b-all`` — Meta MMS-1B-all (multilingual CTC ASR).

    The upstream repo's ``config.json`` reports ``model_type="wav2vec2"`` even
    though the ``mms`` family runs it, so the preset pins ``load_model_type`` to
    ``"mms"`` to route it correctly. Heavy (~29 GB: base + language adapters).
    """

    hf_repo: ClassVar[str] = "facebook/mms-1b-all"
    #: The base encoder weights sit BESIDE 1198 per-language adapter files
    #: (a bare weights glob would report ready with only an adapter
    #: present), and vocab.json decodes the CTC ids -- without it the
    #: transcript is silently the numeric ids (round-8 review). Both are
    #: read by fixed name, so they gate every checkpoint, model_path
    #: included.
    required_checkpoint_files: ClassVar[tuple[str, ...]] = (
        "model.safetensors",
        "vocab.json",
    )
    backend: ClassVar[ModelBackend] = GenericSttBackend(SttFamilySpec(model_types=("mms",)))
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="mms-1b-all",
        description=(
            "Meta MMS-1B-all (MLX); multilingual CTC ASR. Large download "
            "(~29 GB); config model_type pinned to 'mms'. Batch only."
        ),
        selectable=[],
        detectable=[],
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_NONE, runtime_override=False, streaming=False
    )
    load_model_type: ClassVar[str | None] = "mms"


class FireRedAsr2Aed(MlxAudioASR):
    """``mlx-audio/fireredasr2-aed`` — FireRedASR2-AED (Chinese/English, beam search)."""

    hf_repo: ClassVar[str] = "mlx-community/FireRedASR2-AED-mlx"
    # Both are loaded optionally upstream (round-8 review): without
    # dict.txt the transcript is silently EMPTY, and without cmvn.json the
    # feature normalization is silently skipped. The repo's
    # train_bpe1000.model is NOT declared: the loader assigns it to a field
    # no decode path ever reads (dead upstream code, round-9 review), so
    # requiring it would reject a working checkpoint for a file that
    # changes nothing.
    required_checkpoint_files: ClassVar[tuple[str, ...]] = (
        "cmvn.json",
        "dict.txt",
    )
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(model_types=("fireredasr2",), forward=(("beam_size", "beam_size"),))
    )
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="fireredasr2-aed",
        description=(
            "FireRedASR2-AED (MLX); Chinese/English ASR with beam search "
            "(set beam_size). Batch only."
        ),
        selectable=[],
        detectable=[],
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_NONE, runtime_override=False, streaming=False
    )


class VoxtralRealtime4B(MlxAudioASR):
    """``mlx-audio/voxtral-realtime-4b`` — Mistral Voxtral-Mini 4B Realtime (English).

    Exposed through the windowed batch re-decode path; mlx-audio's native
    low-latency streaming session for this model is not yet wired in, so this
    preset declares batch only (honest) rather than incremental streaming.
    """

    hf_repo: ClassVar[str] = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
    # The family reads its tekken tokenizer by fixed name and raises
    # FileNotFoundError at load when it is absent (loud, so snapshot-only);
    # the repo ships no other tokenizer layout (round-9 review).
    required_snapshot_files: ClassVar[tuple[str, ...]] = ("tekken.json",)
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(model_types=("voxtral_realtime",))
    )
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="voxtral-realtime-4b",
        description=(
            "Mistral Voxtral-Mini 4B Realtime (MLX, 4-bit); English ASR. Batch "
            "only (native streaming not yet wired)."
        ),
        selectable=[],
        detectable=[],
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_NONE, runtime_override=False, streaming=False
    )


__all__ = [
    "Canary1BV2",
    "CohereAsr",
    "FireRedAsr2Aed",
    "FunAsrNano",
    "GlmAsrNano",
    "GraniteSpeech1B",
    "GraniteSpeechNar2B",
    "MlxAudioASR",
    "Mms1BAll",
    "MoonshineTiny",
    "NemotronAsrStreaming06B",
    "ParakeetTdt06BV3",
    "Qwen2Audio7B",
    "Qwen3Asr06B",
    "Qwen3Asr17B",
    "SenseVoiceSmall",
    "VibeVoiceAsr",
    "VoxtralMini3B",
    "VoxtralRealtime4B",
    "WhisperLargeV3Turbo",
    "WhisperTiny",
]
