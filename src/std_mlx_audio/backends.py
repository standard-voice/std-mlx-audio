# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-family backend adapters: native mlx-audio output -> Standard ASR result.

This is the core of the "one engine, many models" design. mlx-audio exposes
several STT *architectures* behind a single ``load()`` loader, but their
``model.generate(...)`` calls take **different arguments** and return **different
shapes**:

* **Qwen3-ASR** (``qwen3_asr``) — an audio-conditioned LLM. ``generate(audio, *,
  language=<English name>, temperature, top_p, top_k, repetition_penalty,
  max_tokens, system_prompt, chunk_duration, ...)`` returns an ``STTOutput`` with
  ``segments=[{text, language, start, end}]`` (chunk-level, no word timing) and
  ``language`` as a *list* of names.
* **Whisper** (``whisper``) — ``generate(audio, *, language=<iso code>, task,
  temperature, initial_prompt, word_timestamps, return_timestamps, ...)`` returns
  an ``STTOutput`` whose ``segments`` carry per-word ``words`` (when
  ``word_timestamps=True``) and ``language`` as a single code.
* **Parakeet** (``parakeet``) — ``generate(audio, ...)`` (no language arg)
  returns an ``AlignedResult`` with ``.text`` and ``.sentences[].tokens[]``
  (token-level timing) — a wholly different return type.

Each :class:`ModelBackend` owns exactly one family's call-shape and
output-normalization, so the engine class stays family-agnostic: it loads the
model, negotiates audio, resolves the language axis, then hands off to the bound
backend. Adding a fourth family (e.g. Voxtral) is a new ``ModelBackend`` plus an
entry point — no change to the engine, config, or pipeline.

All adapters are pure and side-effect-free given a model + waveform, so they are
unit-testable against tiny fakes without MLX or downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from standard_asr.results import Segment, TranscriptionResult, Word
from standard_asr.runtime_params import WordTimestampGranularity

from . import languages
from ._config import MlxAudioParams
from .languages import from_backend_language

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from numpy.typing import NDArray

    from ._config import MlxAudioConfig

#: All MLX STT backends here consume 16 kHz mono audio (the standard layer
#: resamples to our native rate before us).
SAMPLE_RATE = 16000


def _as_list(value: Any) -> list[Any]:
    """Coerce a possibly-``None`` native field into a concrete ``list[Any]``.

    The mlx-audio return objects are typed ``Any`` here (we deliberately do not
    depend on their internal classes), so ``getattr(native, "segments", None)``
    is ``Any``; this narrows the value to a definite ``list[Any]`` (empty when
    ``None`` or not a list) so the downstream conversion helpers stay fully typed
    under pyright strict.

    Args:
        value: A native list-like field, or ``None``.

    Returns:
        A ``list[Any]`` (empty if ``value`` is ``None`` / not a list).
    """
    if isinstance(value, list):
        return cast("list[Any]", value)
    return []


class ModelBackend(Protocol):
    """One model family's call-shape + output normalization.

    Implementations translate a resolved BCP-47 language into the family's native
    surface, build the family-specific ``generate`` kwargs, and map the native
    return value onto a constant-schema :class:`TranscriptionResult`.
    """

    #: mlx-audio ``model_type`` values this backend handles. The engine asserts
    #: the *actually loaded* model's family is one of these before transcribing
    #: (see ``MlxAudioASR._verify_model_family``), so a ``model_path`` override
    #: pointing at a different family fails loudly instead of being run through
    #: the wrong adapter (which would silently produce a wrong transcript).
    model_types: tuple[str, ...]

    #: Whether this family's ``generate`` requires the audio as a ``list`` of
    #: arrays rather than a bare ``mx.array`` (only Voxtral does). The engine and
    #: the streaming session wrap the negotiated array accordingly.
    audio_as_list: bool

    def generate_kwargs(
        self,
        *,
        resolved_language: str | None,
        want_words: bool,
        params: MlxAudioParams,
        config: MlxAudioConfig,
    ) -> dict[str, Any]:
        """Build the keyword arguments for this family's ``model.generate``.

        Args:
            resolved_language: The effective BCP-47 language, or ``None`` for
                auto-detect.
            want_words: Whether word-level timestamps were requested.
            params: The engine-specific decoding knobs.
            config: The engine init config.

        Returns:
            Keyword arguments for ``model.generate(audio, **kwargs)``.
        """
        ...

    def to_result(
        self, native: Any, *, duration: float | None, want_words: bool
    ) -> TranscriptionResult:
        """Map this family's native ``generate`` return onto a Standard result.

        Args:
            native: The value returned by ``model.generate``.
            duration: Audio duration in seconds, if known (from the waveform).
            want_words: Whether word-level timestamps were requested (controls
                the null semantics of ``words``: ``None`` = not requested).

        Returns:
            A constant-schema :class:`TranscriptionResult`.
        """
        ...


# --------------------------------------------------------------------------- #
# Qwen3-ASR
# --------------------------------------------------------------------------- #
class Qwen3AsrBackend:
    """Backend for the Qwen3-ASR family (audio-conditioned LLM).

    Qwen3-ASR is the headliner. It validates ``language`` against English names
    and emits chunk-level segments without word timing, so it declares only the
    ``"segment"`` word-timestamp granularity (it can always serve per-chunk
    start/end, but never per-word). The full MLX sampler is exposed via
    :class:`MlxAudioParams`.
    """

    model_types: tuple[str, ...] = ("qwen3_asr", "mega_asr")
    audio_as_list: bool = False

    def generate_kwargs(
        self,
        *,
        resolved_language: str | None,
        want_words: bool,
        params: MlxAudioParams,
        config: MlxAudioConfig,
    ) -> dict[str, Any]:
        """Build Qwen3-ASR ``generate`` kwargs (English-name language + sampler).

        Args:
            resolved_language: Effective BCP-47 language, or ``None`` for auto.
            want_words: Ignored (Qwen3-ASR has no word-timestamp mode).
            params: Engine-specific decoding knobs.
            config: Engine init config (unused here; kept for protocol symmetry).

        Returns:
            Keyword arguments for ``Qwen3ASR.generate``.
        """
        del want_words, config
        kwargs: dict[str, Any] = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "repetition_penalty": params.repetition_penalty,
            "repetition_context_size": params.repetition_context_size,
            "max_tokens": params.max_tokens,
            "chunk_duration": params.chunk_duration,
        }
        # Auto-detect when no language resolved; otherwise translate to the
        # English NAME surface Qwen validates against. An unmappable tag falls
        # back to auto rather than sending an invalid name.
        if resolved_language is not None:
            name = languages.to_qwen_name(resolved_language)
            if name is not None:
                kwargs["language"] = name
        if params.system_prompt is not None:
            kwargs["system_prompt"] = params.system_prompt
        return kwargs

    def to_result(
        self, native: Any, *, duration: float | None, want_words: bool
    ) -> TranscriptionResult:
        """Map a Qwen3-ASR ``STTOutput`` onto a Standard result.

        ``STTOutput.segments`` are chunk dicts ``{text, language, start, end}``;
        ``STTOutput.language`` is a list of names. ``words`` is always ``None``
        (Qwen3-ASR emits no word timing — declaring ``want_words`` cannot change
        that, and the standard layer would have rejected a ``"word"`` request as
        incompatible before we got here).

        Args:
            native: The ``STTOutput`` from ``Qwen3ASR.generate``.
            duration: Audio duration in seconds, if known.
            want_words: Ignored (no word timing available).

        Returns:
            A constant-schema :class:`TranscriptionResult`.
        """
        del want_words
        text: str = (getattr(native, "text", "") or "").strip()
        segments = _segments_from_dicts(_as_list(getattr(native, "segments", None)))
        detected = _detected_from_qwen_language(getattr(native, "language", None))
        return TranscriptionResult(
            text=text,
            detected_language=detected,
            duration=duration,
            segments=segments or None,
            words=None,
            extra=_sttoutput_extra(native),
        )


# --------------------------------------------------------------------------- #
# Whisper
# --------------------------------------------------------------------------- #
class WhisperBackend:
    """Backend for the Whisper family (encoder-decoder, segment + word timing).

    Whisper takes ISO language codes, supports ``initial_prompt`` (the portable
    ``prompt`` maps to it), and can emit per-word timestamps, so it declares both
    ``"word"`` and ``"segment"`` granularities.
    """

    model_types: tuple[str, ...] = ("whisper",)
    audio_as_list: bool = False

    def generate_kwargs(
        self,
        *,
        resolved_language: str | None,
        want_words: bool,
        params: MlxAudioParams,
        config: MlxAudioConfig,
    ) -> dict[str, Any]:
        """Build Whisper ``generate`` kwargs (ISO code language + timestamps).

        Args:
            resolved_language: Effective BCP-47 language, or ``None`` for auto.
            want_words: Whether to request word-level timestamps.
            params: Engine-specific decoding knobs.
            config: Engine init config (unused; kept for protocol symmetry).

        Returns:
            Keyword arguments for ``Whisper.generate``.
        """
        del config
        kwargs: dict[str, Any] = {
            # Whisper takes a temperature (or a fallback tuple); forward the
            # single sampler temperature as the first step.
            "temperature": params.temperature,
            "word_timestamps": want_words,
            "return_timestamps": True,
        }
        if resolved_language is not None:
            code = languages.to_whisper_code(resolved_language)
            if code is not None:
                kwargs["language"] = code
        return kwargs

    def to_result(
        self, native: Any, *, duration: float | None, want_words: bool
    ) -> TranscriptionResult:
        """Map a Whisper ``STTOutput`` onto a Standard result.

        Whisper segments are dicts ``{text, start, end, avg_logprob,
        no_speech_prob, words?}``; ``words`` (per segment) are dicts ``{word,
        start, end, probability}``. We back-fill word-level data only when
        ``want_words`` (spec TR.3 null semantics: ``words=None`` means "not
        requested").

        Args:
            native: The ``STTOutput`` from ``Whisper.generate``.
            duration: Audio duration in seconds, if known.
            want_words: Whether word timestamps were requested.

        Returns:
            A constant-schema :class:`TranscriptionResult`.
        """
        text: str = (getattr(native, "text", "") or "").strip()
        segments, words = _segments_from_whisper(
            _as_list(getattr(native, "segments", None)), want_words=want_words
        )
        detected = languages.from_backend_language(getattr(native, "language", None))
        return TranscriptionResult(
            text=text,
            detected_language=detected,
            duration=duration,
            segments=segments or None,
            words=words if want_words else None,
        )


# --------------------------------------------------------------------------- #
# Parakeet
# --------------------------------------------------------------------------- #
class AlignedResultBackend:
    """Backend for the NeMo-style aligned-output families (Parakeet, Nemotron).

    NVIDIA Parakeet (TDT/CTC) and Nemotron ASR (RNN-T) return an
    ``AlignedResult`` (``.text`` + ``.sentences[].tokens[]`` with token-level
    timing) — not an ``STTOutput``, a wholly different shape from the generative
    backends. We map sentences to segments and tokens to words; because these
    models *always* produce token timing, they serve both ``"word"`` and
    ``"segment"`` granularities at no extra cost.

    Neither family exposes a language axis we can safely drive: Parakeet is
    fixed-language, and Nemotron selects language via *model-specific prompt
    keys* (not a portable BCP-47 surface — passing an unmatched key silently
    falls back to auto), so this backend passes no language and lets the model
    auto-handle it. The honest capability for both is therefore
    ``language.runtime_override=False`` (declared in ``_metadata.py``).

    Args:
        model_types: The mlx-audio ``model_type`` value(s) this instance adapts
            (e.g. ``("parakeet",)`` or ``("nemotron_asr",)``).
    """

    audio_as_list: bool = False

    def __init__(self, *, model_types: tuple[str, ...]) -> None:
        """Bind the aligned-output adapter to one family's ``model_type``(s).

        Args:
            model_types: The mlx-audio ``model_type`` value(s) this instance
                adapts (used by the engine's load-time family check).
        """
        self.model_types = model_types

    def generate_kwargs(
        self,
        *,
        resolved_language: str | None,
        want_words: bool,
        params: MlxAudioParams,
        config: MlxAudioConfig,
    ) -> dict[str, Any]:
        """Build the aligned-output ``generate`` kwargs (none).

        These models take no language argument we drive and no sampler; mlx-audio
        defaults to whole-file decoding (best for accuracy on our target clips),
        so we pass nothing and let the model decide.

        Args:
            resolved_language: Ignored (no driveable language axis).
            want_words: Ignored (always produces token timing).
            params: Ignored (no exposed knobs).
            config: Ignored.

        Returns:
            An empty kwargs dict.
        """
        del resolved_language, want_words, params, config
        return {}

    def to_result(
        self, native: Any, *, duration: float | None, want_words: bool
    ) -> TranscriptionResult:
        """Map an ``AlignedResult`` onto a Standard result.

        Each ``AlignedSentence`` becomes a :class:`Segment`; each
        ``AlignedToken`` becomes a :class:`Word` (with no probability — the TDT /
        RNN-T paths do not emit per-token confidence in mlx-audio). Word data is
        attached only when ``want_words`` (spec TR.3).

        Args:
            native: The ``AlignedResult`` from the model's ``generate``.
            duration: Audio duration in seconds, if known.
            want_words: Whether word timestamps were requested.

        Returns:
            A constant-schema :class:`TranscriptionResult`.

        Note:
            These models do not report a detected language; ``detected_language``
            is left ``None`` (honesty over guessing).
        """
        text: str = (getattr(native, "text", "") or "").strip()
        segments, words = _segments_from_aligned(
            _as_list(getattr(native, "sentences", None)), want_words=want_words
        )
        return TranscriptionResult(
            text=text,
            detected_language=None,
            duration=duration,
            segments=segments or None,
            words=words if want_words else None,
        )


# --------------------------------------------------------------------------- #
# Generic STTOutput families (one parameterized adapter, many models)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SttFamilySpec:
    """Declarative adapter spec for an ``STTOutput``-returning model family.

    The ~14 non-Whisper / non-Qwen mlx-audio STT families all return the same
    ``STTOutput`` shape (``.text`` + optional ``.segments`` dicts + optional
    ``.language``); they differ only in (a) how the language axis maps onto
    ``generate`` kwargs, (b) whether their segments carry *real* timing, (c)
    whether they report a detected language, (d) one input quirk (Voxtral wants a
    list), and (e) a few per-family decode knobs. Capturing those differences as
    data lets a single :class:`GenericSttBackend` serve all of them, so the risky
    kwargs/normalization logic lives (and is tested) in exactly one place.

    Attributes:
        model_types: mlx-audio ``model_type`` value(s) this spec adapts (the
            engine's load-time family check uses these).
        language_kwarg: The ``generate`` keyword that selects the *spoken*
            language (``"language"`` or Canary's ``"source_lang"``), or ``None``
            for a family with no driveable spoken-language axis. The resolved
            BCP-47 tag is passed through as its ISO primary subtag.
        translate_target_kwarg: The ``generate`` keyword that selects a
            *translation target* (Canary ``"target_lang"`` / Granite
            ``"language"``), or ``None`` if the family cannot translate. Driven by
            the ``target_language`` provider param.
        segment_timing: Whether ``STTOutput.segments`` carry real per-chunk
            start/end times (Cohere/Fun-ASR/GLM) — only then do we emit segments
            and declare the ``"segment"`` granularity. Families with placeholder
            (``0/0``), wall-clock, or absent timing set this ``False`` and return
            text only (honesty over fabricated spans).
        reports_detected_language: Whether ``STTOutput.language`` is a genuinely
            *detected* language to surface as ``detected_language`` (SenseVoice);
            families that merely echo the requested language set this ``False``.
        audio_as_list: Whether ``generate`` needs the audio as ``list[mx.array]``
            (Voxtral) rather than a bare array.
        forward: ``(generate_kwarg, provider_field)`` pairs to forward from
            :class:`MlxAudioParams` when the provider value is not ``None`` (e.g.
            ``("hotwords", "hotwords")``). Only knobs the family's ``generate``
            actually accepts are listed (Voxtral has no ``**kwargs``, so passing
            an unknown one would raise — the allow-list prevents that).
    """

    model_types: tuple[str, ...]
    language_kwarg: str | None = None
    translate_target_kwarg: str | None = None
    segment_timing: bool = False
    reports_detected_language: bool = False
    audio_as_list: bool = False
    forward: tuple[tuple[str, str], ...] = ()


class GenericSttBackend:
    """Adapter for any ``STTOutput`` family, parameterized by a :class:`SttFamilySpec`.

    One implementation covers every generative / encoder-decoder / CTC STT family
    mlx-audio exposes besides Whisper and Qwen3-ASR (which keep bespoke backends
    for word timestamps and English-name language respectively). The behavior is
    entirely data-driven by ``spec`` so each family's quirks are declared once,
    in one auditable table, rather than duplicated across a dozen classes.

    Args:
        spec: The family's declarative adapter spec.
    """

    def __init__(self, spec: SttFamilySpec) -> None:
        """Bind the generic adapter to one family's spec.

        Args:
            spec: The family's declarative adapter spec.
        """
        self.spec = spec
        # Mirror the spec's routing/input flags as plain attributes (not
        # properties) so the instance satisfies the ModelBackend protocol's
        # mutable-attribute members under pyright strict.
        self.model_types: tuple[str, ...] = spec.model_types
        self.audio_as_list: bool = spec.audio_as_list

    def generate_kwargs(
        self,
        *,
        resolved_language: str | None,
        want_words: bool,
        params: MlxAudioParams,
        config: MlxAudioConfig,
    ) -> dict[str, Any]:
        """Build the family's ``generate`` kwargs from the spec + provider params.

        Maps the resolved spoken language onto the spec's ``language_kwarg`` (as
        an ISO subtag), wires up speech-translation when a ``target_language`` is
        set, and forwards only the per-family decode knobs the family accepts.

        Args:
            resolved_language: Effective BCP-47 language, or ``None`` for auto.
            want_words: Ignored (no STTOutput family emits word timing).
            params: Engine-specific decoding knobs.
            config: Engine init config (unused; kept for protocol symmetry).

        Returns:
            Keyword arguments for the family's ``model.generate``.
        """
        del want_words, config
        spec = self.spec
        kwargs: dict[str, Any] = {}
        if spec.language_kwarg is not None and resolved_language is not None:
            kwargs[spec.language_kwarg] = languages.to_iso_subtag(resolved_language)
        if spec.translate_target_kwarg is not None:
            if params.target_language is not None:
                kwargs[spec.translate_target_kwarg] = languages.to_iso_subtag(
                    params.target_language
                )
            elif spec.translate_target_kwarg == "target_lang" and "source_lang" in kwargs:
                # Canary defaults target_lang="en"; without an explicit target we
                # must pin it to the source so a non-English source TRANSCRIBES
                # rather than silently translating to English.
                kwargs["target_lang"] = kwargs["source_lang"]
        for gen_kwarg, field in spec.forward:
            value = getattr(params, field, None)
            if value is not None:
                kwargs[gen_kwarg] = value
        return kwargs

    def to_result(
        self, native: Any, *, duration: float | None, want_words: bool
    ) -> TranscriptionResult:
        """Map the family's ``STTOutput`` onto a Standard result.

        Emits segments only when the family produces real per-chunk timing
        (``spec.segment_timing``); otherwise returns text only — never a
        fabricated span. Surfaces a detected language only when the family truly
        detects one (``spec.reports_detected_language``). No STTOutput family
        emits word timing, so ``words`` is always ``None``.

        Args:
            native: The ``STTOutput`` from the family's ``generate``.
            duration: Audio duration in seconds, if known.
            want_words: Ignored (no word timing available).

        Returns:
            A constant-schema :class:`TranscriptionResult`.
        """
        del want_words
        spec = self.spec
        text: str = (getattr(native, "text", "") or "").strip()
        segments = (
            _segments_from_dicts(_as_list(getattr(native, "segments", None)))
            if spec.segment_timing
            else None
        )
        detected = (
            from_backend_language(getattr(native, "language", None))
            if spec.reports_detected_language
            else None
        )
        return TranscriptionResult(
            text=text,
            detected_language=detected,
            duration=duration,
            segments=segments or None,
            words=None,
            extra=_sttoutput_extra(native),
        )


# --------------------------------------------------------------------------- #
# Shared conversion helpers (pure)
# --------------------------------------------------------------------------- #
def waveform_duration(audio: NDArray[np.float32]) -> float:
    """Return the duration of a 16 kHz mono waveform in seconds.

    Args:
        audio: A float32 mono waveform at :data:`SAMPLE_RATE`.

    Returns:
        Duration in seconds (``0.0`` for an empty array).
    """
    return float(audio.shape[0]) / SAMPLE_RATE if audio.size else 0.0


def to_mlx_array(audio: NDArray[np.float32]) -> Any:
    """Convert a numpy float32 waveform into an ``mx.array`` for mlx-audio.

    Why this is required, not optional: the Parakeet backend's ``generate`` does
    ``audio.astype(mx.bfloat16)`` on a non-string input, which raises
    ``TypeError: Cannot interpret 'mlx.core.bfloat16' as a data type`` on a numpy
    array (numpy cannot interpret an MLX dtype). Qwen3-ASR and Whisper accept both
    numpy and ``mx.array``, so passing an ``mx.array`` is the one shape every
    backend handles. The array must be created on the thread that owns the model's
    Metal stream (MLX arrays are thread-bound), which is the calling thread here.

    Args:
        audio: A contiguous float32 mono waveform at :data:`SAMPLE_RATE`.

    Returns:
        An ``mx.array`` (float32) wrapping the waveform.
    """
    import mlx.core as mx  # local import: MLX is Apple-Silicon-only

    return mx.array(audio)


def adapt_audio_source(backend: ModelBackend, source: Any) -> Any:
    """Adapt the negotiated audio source to the backend's ``generate`` call shape.

    Most mlx-audio families take the audio as-is (a bare ``mx.array`` or a file
    path); Voxtral's ``generate`` requires a ``list[mx.array]``
    (``backend.audio_as_list``), so a single source is wrapped in a one-element
    list for it. Centralized so the batch path and the streaming session adapt
    identically.

    Args:
        backend: The active backend (its ``audio_as_list`` flag decides).
        source: The negotiated audio source (an ``mx.array`` or a path).

    Returns:
        The source, wrapped in a one-element list iff the backend requires it.
    """
    return [source] if backend.audio_as_list else source


def _clamp_span(start: Any, end: Any) -> tuple[float, float]:
    """Coerce a (start, end) pair to a valid non-negative, non-inverted span.

    The Standard ``Segment`` / ``Word`` models reject negative or inverted spans
    (spec TR.2). Backend timings are generally clean, but chunk-offset arithmetic
    and floating error can occasionally produce ``end`` a hair below ``start`` or
    a tiny negative ``start``; we clamp rather than let construction raise and
    drop the whole transcript.

    Args:
        start: Native start time (seconds).
        end: Native end time (seconds).

    Returns:
        A ``(start, end)`` pair with ``0 <= start <= end``.
    """
    s = max(0.0, float(start))
    e = max(0.0, float(end))
    if e < s:
        e = s
    return s, e


def _segments_from_dicts(raw: list[dict[str, Any]]) -> list[Segment]:
    """Convert chunk-level dict segments (Qwen3-ASR) into ``Segment`` objects.

    Args:
        raw: A list of ``{text, start, end, ...}`` dicts.

    Returns:
        A list of :class:`Segment` (word-less).
    """
    out: list[Segment] = []
    for item in raw:
        start, end = _clamp_span(item.get("start", 0.0), item.get("end", 0.0))
        out.append(Segment(start=start, end=end, text=str(item.get("text", "")), words=None))
    return out


def _segments_from_whisper(
    raw: list[dict[str, Any]], *, want_words: bool
) -> tuple[list[Segment], list[Word]]:
    """Convert Whisper dict segments into ``Segment`` / flattened ``Word`` lists.

    Args:
        raw: Whisper segment dicts (each may carry a ``words`` list).
        want_words: Whether to materialize word-level data.

    Returns:
        A ``(segments, flattened_words)`` pair (``flattened_words`` empty when
        ``want_words`` is False).
    """
    segments: list[Segment] = []
    flat_words: list[Word] = []
    for seg in raw:
        seg_words: list[Word] | None = None
        if want_words:
            seg_words = _words_from_whisper(seg.get("words") or [])
            flat_words.extend(seg_words)
        start, end = _clamp_span(seg.get("start", 0.0), seg.get("end", 0.0))
        segments.append(
            Segment(
                start=start,
                end=end,
                text=str(seg.get("text", "")),
                words=seg_words,
                avg_logprob=_opt_float(seg.get("avg_logprob")),
                no_speech_prob=_opt_float(seg.get("no_speech_prob")),
                compression_ratio=_opt_float(seg.get("compression_ratio")),
            )
        )
    return segments, flat_words


def _words_from_whisper(raw: list[dict[str, Any]]) -> list[Word]:
    """Convert Whisper word dicts ``{word, start, end, probability}`` to ``Word``.

    Args:
        raw: Whisper per-word dicts.

    Returns:
        A list of :class:`Word`.
    """
    words: list[Word] = []
    for w in raw:
        start, end = _clamp_span(w.get("start", 0.0), w.get("end", 0.0))
        words.append(
            Word(
                start=start,
                end=end,
                text=str(w.get("word", "")),
                probability=_opt_unit(w.get("probability")),
            )
        )
    return words


def _segments_from_aligned(
    sentences: list[Any], *, want_words: bool
) -> tuple[list[Segment], list[Word]]:
    """Convert ``AlignedSentence`` objects (Parakeet/Nemotron) to ``Segment`` / ``Word``.

    Args:
        sentences: A list of ``AlignedSentence`` (each with ``.text``, ``.start``,
            ``.end``, ``.tokens``).
        want_words: Whether to materialize token-level word data.

    Returns:
        A ``(segments, flattened_words)`` pair.
    """
    segments: list[Segment] = []
    flat_words: list[Word] = []
    for sent in sentences:
        seg_words: list[Word] | None = None
        if want_words:
            seg_words = _words_from_aligned(getattr(sent, "tokens", None) or [])
            flat_words.extend(seg_words)
        start, end = _clamp_span(getattr(sent, "start", 0.0), getattr(sent, "end", 0.0))
        segments.append(
            Segment(
                start=start,
                end=end,
                text=str(getattr(sent, "text", "")),
                words=seg_words,
            )
        )
    return segments, flat_words


def _words_from_aligned(tokens: list[Any]) -> list[Word]:
    """Convert ``AlignedToken`` objects (Parakeet/Nemotron) to ``Word`` (no probability).

    Args:
        tokens: A list of ``AlignedToken`` (each with ``.text``, ``.start``,
            ``.end``).

    Returns:
        A list of :class:`Word`.
    """
    words: list[Word] = []
    for tok in tokens:
        start, end = _clamp_span(getattr(tok, "start", 0.0), getattr(tok, "end", 0.0))
        words.append(Word(start=start, end=end, text=str(getattr(tok, "text", ""))))
    return words


def _detected_from_qwen_language(value: Any) -> str | None:
    """Map Qwen3-ASR's ``language`` (str or list of names) to a BCP-47 tag.

    Args:
        value: The ``STTOutput.language`` — a list of names, a single name, or
            ``None``.

    Returns:
        The dominant detected language as BCP-47, or ``None``.
    """
    dominant: Any = value
    if isinstance(value, list):
        items = _as_list(value)
        dominant = items[0] if items else None
    return languages.from_backend_language(None if dominant is None else str(dominant))


def _opt_float(value: Any) -> float | None:
    """Coerce a value to ``float`` or ``None`` (drops non-finite/garbage).

    Args:
        value: Any value or ``None``.

    Returns:
        A finite float, or ``None``.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # Segment/Word reject NaN/Inf via allow_inf_nan=False; pre-filter so an odd
    # backend metric does not blow up the whole result.
    return result if result == result and result not in (float("inf"), float("-inf")) else None


def _opt_unit(value: Any) -> float | None:
    """Coerce a value to a ``[0, 1]`` probability or ``None``.

    Args:
        value: Any value or ``None``.

    Returns:
        A float clamped to ``[0, 1]``, or ``None``.
    """
    f = _opt_float(value)
    if f is None:
        return None
    return min(1.0, max(0.0, f))


def _sttoutput_extra(native: Any) -> dict[str, Any]:
    """Build the engine-specific ``extra`` from any mlx-audio ``STTOutput``.

    Surfaces the token-accounting / throughput stats mlx-audio reports on its
    ``STTOutput`` (shared across Qwen3-ASR and the generic generative families;
    spec TR.1: engine-specific values belong in ``result.extra``, never
    ``metadata``). Fields a given family does not populate are simply absent.

    Args:
        native: The ``STTOutput``.

    Returns:
        A small JSON-friendly mapping (empty if nothing useful is present).
    """
    extra: dict[str, Any] = {}
    for name in ("generation_tokens", "prompt_tokens", "total_tokens"):
        value = getattr(native, name, None)
        if value:
            extra[name] = int(value)
    tps = getattr(native, "generation_tps", None)
    if tps:
        extra["generation_tps"] = round(float(tps), 2)
    return extra


def map_word_timestamps(granularity: WordTimestampGranularity | None) -> bool:
    """Return whether a granularity requires word-level decoding.

    Only ``WORD`` flips a backend's word-timestamp pass on; ``SEGMENT`` (and
    ``None``) are served by the always-present segment spans and MUST NOT
    back-fill word data (spec TR.3).

    Args:
        granularity: The requested granularity, or ``None``.

    Returns:
        ``True`` iff word-level timestamps were requested.
    """
    return granularity == WordTimestampGranularity.WORD


__all__ = [
    "SAMPLE_RATE",
    "AlignedResultBackend",
    "GenericSttBackend",
    "ModelBackend",
    "Qwen3AsrBackend",
    "SttFamilySpec",
    "WhisperBackend",
    "adapt_audio_source",
    "map_word_timestamps",
    "to_mlx_array",
    "waveform_duration",
]
