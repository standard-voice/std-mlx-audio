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

from typing import TYPE_CHECKING, Any, Protocol, cast

from standard_asr.results import Segment, TranscriptionResult, Word
from standard_asr.runtime_params import WordTimestampGranularity

from . import languages
from ._config import MlxAudioParams

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

    #: mlx-audio ``model_type`` values this backend handles (for routing /
    #: sanity assertions). Informational; routing is by entry-point preset.
    model_types: tuple[str, ...]

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
            extra=_qwen_extra(native),
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
class ParakeetBackend:
    """Backend for the Parakeet family (NVIDIA TDT/CTC; token-aligned output).

    Parakeet takes no language argument (fixed-language model) and returns an
    ``AlignedResult`` (``.text`` + ``.sentences[].tokens[]`` with token-level
    timing) — not an ``STTOutput``. We map its sentences to segments and its
    tokens to words; because it *always* produces token timing, it can serve both
    ``"word"`` and ``"segment"`` granularities at no extra cost.
    """

    model_types: tuple[str, ...] = ("parakeet", "nemo")

    def generate_kwargs(
        self,
        *,
        resolved_language: str | None,
        want_words: bool,
        params: MlxAudioParams,
        config: MlxAudioConfig,
    ) -> dict[str, Any]:
        """Build Parakeet ``generate`` kwargs (no language arg).

        Parakeet has no language selection and no sampler; only ``chunk_duration``
        is meaningful for very long files, but mlx-audio defaults to whole-file
        decoding (``chunk_duration=None``), which is best for accuracy on our
        target clips, so we pass nothing and let the model decide.

        Args:
            resolved_language: Ignored (fixed-language model).
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
        """Map a Parakeet ``AlignedResult`` onto a Standard result.

        Each ``AlignedSentence`` becomes a :class:`Segment`; each
        ``AlignedToken`` becomes a :class:`Word` (with no probability — TDT does
        not emit per-token confidence in the mlx-audio path). Word data is
        attached only when ``want_words`` (spec TR.3).

        Args:
            native: The ``AlignedResult`` from ``Parakeet.generate``.
            duration: Audio duration in seconds, if known.
            want_words: Whether word timestamps were requested.

        Returns:
            A constant-schema :class:`TranscriptionResult`.

        Note:
            Parakeet does not report a detected language; ``detected_language``
            is left ``None`` (the model is fixed-language; honesty over guessing).
        """
        text: str = (getattr(native, "text", "") or "").strip()
        segments, words = _segments_from_parakeet(
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


def _segments_from_parakeet(
    sentences: list[Any], *, want_words: bool
) -> tuple[list[Segment], list[Word]]:
    """Convert Parakeet ``AlignedSentence`` objects to ``Segment`` / ``Word``.

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
            seg_words = _words_from_parakeet(getattr(sent, "tokens", None) or [])
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


def _words_from_parakeet(tokens: list[Any]) -> list[Word]:
    """Convert Parakeet ``AlignedToken`` objects to ``Word`` (no probability).

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


def _qwen_extra(native: Any) -> dict[str, Any]:
    """Build the engine-specific ``extra`` from a Qwen3-ASR ``STTOutput``.

    Surfaces the token-accounting / throughput stats mlx-audio reports (spec
    TR.1: engine-specific values belong in ``result.extra``, never ``metadata``).

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
    "ModelBackend",
    "ParakeetBackend",
    "Qwen3AsrBackend",
    "WhisperBackend",
    "map_word_timestamps",
    "to_mlx_array",
    "waveform_duration",
]
