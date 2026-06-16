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
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray
from standard_asr import (
    RuntimeParams,
    TranscriptionResult,
    TranscriptionSession,
)
from standard_asr.audio_format import AudioFormat
from standard_asr.capabilities import DeclaredCapabilities
from standard_asr.engine import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    PreparedAudio,
)
from standard_asr.exceptions import DiscoveryError, TranscriptionError
from standard_asr.language import effective_language
from standard_asr.runtime import allow_downloads, resolve_download_root
from standard_asr.runtime_params import ProviderParams

from . import backends
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
            DiscoveryError: If mlx-audio is missing or weights cannot load.
        """
        self._ensure_model_loaded()
        assert self._model is not None  # _ensure_model_loaded raises otherwise
        return self._model

    def ensure_loaded(self) -> None:
        """Public alias for the lazy loader (used by the streaming session)."""
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Load the MLX model lazily via ``mlx_audio.stt.load``.

        Honors the download policy (spec IC.9): when downloads are disabled
        (the ``local_files_only`` config flag, or ``STANDARD_ASR_ALLOW_DOWNLOAD``
        set to a disable value) we pass ``local_files_only=True`` so the loader
        uses only cached weights and fails loudly instead of reaching out to the
        network. (An UNSET toggle defaults to downloads-enabled per the policy.)

        Raises:
            DiscoveryError: If mlx-audio is missing, the platform is not
                Apple-Silicon/Metal, or weights cannot be loaded.
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
        # mlx-audio resolves repos via the HF hub cache (it HAS a library
        # default), so forward the None passthrough for the cache root unchanged
        # — forcing a directory would break offline loads of hub-cached models.
        resolve_download_root(config.download_root, has_library_default=True)
        # Model selection is by preset (spec IC.7); a local model_path wins.
        model_source = config.model_path or type(self).hf_repo
        load_kwargs: dict[str, Any] = {"local_files_only": local_only}
        if config.revision is not None:
            load_kwargs["revision"] = config.revision
        # A few repos mislabel their config.json model_type (e.g. MMS says
        # "wav2vec2"); the preset pins the family so the loader does not mis-route.
        if type(self).load_model_type is not None:
            load_kwargs["model_type"] = type(self).load_model_type
        try:
            self._model = load(model_source, **load_kwargs)
        except Exception as exc:
            raise DiscoveryError(
                f"Failed to load MLX model {model_source!r}. If downloads are "
                "disabled, set STANDARD_ASR_ALLOW_DOWNLOAD=1 or pre-download the "
                "model; ensure you are on Apple Silicon with Metal."
            ) from exc
        self._verify_model_family()

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
        """Preload model weights without transcribing (spec IC.11).

        Idempotent and synchronous; self-checks the download policy via
        :meth:`_ensure_model_loaded`.

        Raises:
            DiscoveryError: If weights cannot be loaded.
        """
        self._ensure_model_loaded()

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

        session = MlxAudioStreamingSession(self, gated_params)
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
class Qwen3Asr06B(MlxAudioASR):
    """``mlx-audio/qwen3-asr-0.6b`` — small Qwen3-ASR (the headliner)."""

    hf_repo: ClassVar[str] = "mlx-community/Qwen3-ASR-0.6B-4bit"
    backend: ClassVar[ModelBackend] = Qwen3AsrBackend()
    properties: ClassVar[BaseProperties] = Qwen3Asr06BProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _QWEN_CAPABILITIES


class Qwen3Asr17B(MlxAudioASR):
    """``mlx-audio/qwen3-asr-1.7b`` — larger, more accurate Qwen3-ASR."""

    hf_repo: ClassVar[str] = "mlx-community/Qwen3-ASR-1.7B-8bit"
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

    Voxtral's ``generate`` requires a decoded ``list[mx.array]``, so this preset
    accepts an array only (the standard layer decodes a file/bytes first) and the
    engine wraps it in a list (``audio_as_list``).
    """

    hf_repo: ClassVar[str] = "mlx-community/Voxtral-Mini-3B-2507-bf16"
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(
            model_types=("voxtral",),
            language_kwarg="language",
            audio_as_list=True,
        )
    )
    properties: ClassVar[BaseProperties] = stt_properties(
        model_name="voxtral-mini-3b",
        description=(
            "Mistral Voxtral-Mini 3B (MLX, bf16); multilingual ASR. Large "
            "(~9 GB); decoded-array input only; batch only."
        ),
        selectable=VOXTRAL_LANGUAGES,
        detectable=[],
        array_only=True,
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
    """

    hf_repo: ClassVar[str] = "TechHara/canary-1b-v2-mlx-q4"
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
            "NVIDIA Canary 1B v2 (MLX, q4); 25-language European ASR with "
            "speech translation (set target_language). Batch only."
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
    backend: ClassVar[ModelBackend] = GenericSttBackend(SttFamilySpec(model_types=("qwen2_audio",)))
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
    """``mlx-audio/granite-speech-nar-2b`` — IBM Granite Speech NAR (fast, non-AR)."""

    hf_repo: ClassVar[str] = "mlx-community/granite-speech-4.1-2b-nar-mlx"
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
    )
    declared_capabilities: ClassVar[DeclaredCapabilities] = stt_capabilities(
        word_timestamps=_WORD_TS_NONE, runtime_override=False, streaming=False
    )


class VibeVoiceAsr(MlxAudioASR):
    """``mlx-audio/vibevoice-asr`` — Microsoft VibeVoice-ASR (context-biased)."""

    hf_repo: ClassVar[str] = "mlx-community/VibeVoice-ASR-4bit"
    backend: ClassVar[ModelBackend] = GenericSttBackend(
        SttFamilySpec(model_types=("vibevoice_asr",), forward=(("context", "context"),))
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
