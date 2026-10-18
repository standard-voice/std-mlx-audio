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
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from standard_asr.contract.params import WordTimestampGranularity
from standard_asr.contract.results import Segment, TranscriptionResult, Word

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

#: What the plugin puts between two speaker turns of a ``text_from_segments``
#: family (VibeVoice-ASR). mlx-audio returns the turns only as parsed JSON
#: items, with no transcript that joins them, so the separator is the plugin's
#: choice. A line break reads right in every language: a space would put a
#: space after a Chinese full stop, and no separator would run two English
#: sentences together.
TURN_SEPARATOR = "\n"


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


@dataclass(frozen=True)
class ChunkJoin:
    """How mlx-audio joins a chunked family's chunk texts into its full text.

    Qwen3-ASR, GLM-ASR, Fun-ASR, and Cohere-ASR split long input into chunks,
    return one segment per chunk carrying that chunk's text alone, and build
    ``STTOutput.text`` by joining the chunk texts with a separator. The
    separator belongs to the transcript, so the plugin puts it at the start of
    the segment that follows it, and the segment texts concatenated are the
    transcript. Each family's values copy that family's own join in mlx-audio;
    the family's ``ChunkJoin`` names the source line.

    One rule overrides mlx-audio's join, in batch and at a streaming seam
    alike (see :meth:`separator_before`): no separator goes before a chunk in
    a language of :data:`~std_mlx_audio.languages.NO_SPACE_LANGUAGES`.
    mlx-audio's Cohere-ASR join applies that rule; its Qwen3-ASR, GLM-ASR, and
    Fun-ASR joins put a space between two Chinese chunks, which inserts a
    space into the transcript. For those families the batch text therefore
    differs from ``STTOutput.text`` at a chunk boundary in such a language. A
    streaming seam needs the same decision because mlx-audio never sees the
    text on both sides of it.

    Attributes:
        separator: What mlx-audio puts between two chunk texts.
        skip_empty: Whether mlx-audio leaves a chunk with no text out of the
            join (no separator before or after it). ``False`` means a plain
            ``separator.join(texts)``, which puts a separator after an empty
            chunk too.
    """

    separator: str = " "
    skip_empty: bool = False

    def separator_before(self, chunk_language: str | None) -> str:
        """Return the separator that goes before a chunk's text.

        It is used between two chunks of one decode and at a streaming seam,
        before the first text of a later window.

        Args:
            chunk_language: The chunk's language as an ISO-639 primary subtag
                (see :func:`_chunk_languages`), or ``None`` when neither the
                chunk, the decode, nor the request names one.

        Returns:
            ``""`` if the chunk's language is one of
            :data:`~std_mlx_audio.languages.NO_SPACE_LANGUAGES`; otherwise
            :attr:`separator`, which is also the answer when the chunk's
            language is unknown.
        """
        if chunk_language in languages.NO_SPACE_LANGUAGES:
            return ""
        return self.separator

    def texts(
        self,
        texts: list[str],
        *,
        chunk_languages: list[str | None],
        follows_text: bool,
    ) -> list[str]:
        """Return the chunk texts, each with the separator that goes before it.

        Args:
            texts: The chunk texts, in order, exactly as mlx-audio returned them.
            chunk_languages: Each chunk's language as an ISO-639 primary
                subtag, or ``None`` where it is unknown, one per entry of
                ``texts``.
            follows_text: Whether text precedes the first chunk (a streaming
                session's later window). The first chunk that has text then
                starts with :meth:`separator_before`, and nothing is added
                before it.

        Returns:
            The texts with separators added. Without ``follows_text``,
            ``"".join(result)`` is mlx-audio's join of ``texts``, except that a
            chunk in a language of
            :data:`~std_mlx_audio.languages.NO_SPACE_LANGUAGES` gets no
            separator before it.
        """
        out: list[str] = []
        seam = follows_text  # the seam separator is due before the first text
        owed = False  # a separator is due before the next text
        for index, text in enumerate(texts):
            if seam and text:
                prefix = self.separator_before(chunk_languages[index])
                seam = False
            elif (owed and text) or (index > 0 and not self.skip_empty and not seam):
                # A plain join puts the separator before every chunk after the
                # first; a join that skips empty chunks, only before text.
                prefix = self.separator_before(chunk_languages[index])
            else:
                prefix = ""
            out.append(prefix + text)
            owed = owed or bool(text)
        return out


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
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Build the keyword arguments for this family's ``model.generate``.

        Args:
            resolved_language: The effective BCP-47 language, or ``None`` for
                auto-detect.
            want_words: Whether word-level timestamps were requested.
            params: The engine-specific decoding knobs.
            config: The engine init config.
            prompt: The portable guidance ``prompt``, already gated by the
                standard layer: only a model that declares
                ``<mode>.guidance.prompt`` is ever given one, and a backend
                that declares it MUST pass it to the model.

        Returns:
            Keyword arguments for ``model.generate(audio, **kwargs)``.
        """
        ...

    def to_result(
        self,
        native: Any,
        *,
        duration: float | None,
        want_words: bool,
        follows_text: bool = False,
        resolved_language: str | None = None,
    ) -> TranscriptionResult:
        """Map this family's native ``generate`` return onto a Standard result.

        Every segment text carries the characters it adds to the transcript,
        including the space that joins it to the segment before it, and
        ``text`` is the segment texts concatenated (spec TR.1). Nothing is
        stripped: the characters are the ones mlx-audio returned, plus, for a
        family that splits its input into chunks, the separator mlx-audio
        itself puts between two chunks (see :class:`ChunkJoin`).

        Args:
            native: The value returned by ``model.generate``.
            duration: Audio duration in seconds, if known (from the waveform).
            want_words: Whether word-level timestamps were requested (controls
                the null semantics of ``words``: ``None`` = not requested).
            follows_text: Whether this decode continues a transcript that
                already has text: a streaming session's later window. A chunked
                family then starts the first segment that has text with
                :meth:`ChunkJoin.separator_before`: none for a language written
                without spaces, the family's chunk separator otherwise. A
                family whose segment texts carry their own leading space
                (Whisper, Parakeet, Nemotron) ignores it.
            resolved_language: The request's effective BCP-47 language, or
                ``None`` for auto-detect. A chunked family uses it for a chunk
                whose language the decode does not report (see
                :class:`ChunkJoin`); other families ignore it.

        Returns:
            A constant-schema :class:`TranscriptionResult`.
        """
        ...

    def single_segment_span(self, params: MlxAudioParams) -> float | None:
        """Return the longest audio this family returns as one spanning segment.

        Some families split their output only at fixed internal chunks (Qwen3-ASR
        at ``chunk_duration``): audio up to that length always comes back as one
        segment from its start to its end, with no boundary inside. Others split
        at what they hear (sentences, pauses), whatever the length. The streaming
        session uses this to decide, before any decode, whether a window can
        contain a segment boundary (:func:`has_inner_boundaries`).

        Args:
            params: The engine-specific decoding knobs (some set the chunk).

        Returns:
            That length in seconds, or ``None`` for a family that splits at
            content boundaries.
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
    #: mlx-audio builds the full text as ``" ".join(all_texts)`` over every
    #: chunk, empty ones included (``Qwen3ASRModel.generate`` in
    #: ``mlx_audio/stt/models/qwen3_asr/qwen3_asr.py``, mlx-audio 0.4.4 line
    #: 1191), while each chunk's segment carries the chunk text alone. MegaASR
    #: delegates to the same ``generate`` (``mega_asr.py``, ``Model.generate``).
    chunk_join: ClassVar[ChunkJoin] = ChunkJoin(separator=" ")

    def single_segment_span(self, params: MlxAudioParams) -> float | None:
        """Return the effective chunk: mlx-audio returns one segment per chunk.

        mlx-audio's splitter (``split_audio_into_chunks`` in
        ``mlx_audio/stt/models/qwen3_asr/qwen3_asr.py``) advances at least one
        second per chunk and pads shorter input to a second, so a chunk is never
        shorter than one second, whatever ``chunk_duration`` says.

        Args:
            params: The engine-specific decoding knobs.

        Returns:
            ``max(chunk_duration, 1.0)`` in seconds.
        """
        return max(params.chunk_duration, 1.0)

    def generate_kwargs(
        self,
        *,
        resolved_language: str | None,
        want_words: bool,
        params: MlxAudioParams,
        config: MlxAudioConfig,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Build Qwen3-ASR ``generate`` kwargs (English-name language + sampler).

        Args:
            resolved_language: Effective BCP-47 language, or ``None`` for auto.
            want_words: Ignored (Qwen3-ASR has no word-timestamp mode).
            params: Engine-specific decoding knobs.
            config: Engine init config (unused here; kept for protocol symmetry).
            prompt: The portable guidance prompt, used as Qwen3-ASR's context.

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
        # Qwen3-ASR's context is the system turn of its chat template, which is
        # where the official package puts `context` (spec §5.3: Qwen3 `context`
        # is the portable `prompt`). The Qwen-specific `system_prompt` provider
        # param, when given, is the more specific request and wins.
        if params.system_prompt is not None:
            kwargs["system_prompt"] = params.system_prompt
        elif prompt is not None:
            kwargs["system_prompt"] = prompt
        return kwargs

    def to_result(
        self,
        native: Any,
        *,
        duration: float | None,
        want_words: bool,
        follows_text: bool = False,
        resolved_language: str | None = None,
    ) -> TranscriptionResult:
        """Map a Qwen3-ASR ``STTOutput`` onto a Standard result.

        ``STTOutput.segments`` are chunk dicts ``{text, language, start, end}``;
        ``STTOutput.language`` is a list of names. Each segment after the first
        gets the space mlx-audio puts between two chunks (:attr:`chunk_join`),
        except a chunk in Chinese, Cantonese, or Japanese, which gets none, so
        ``text`` is the segment texts concatenated. ``words`` is always ``None`` (Qwen3-ASR emits no word timing;
        declaring ``want_words`` cannot change that, and the standard layer
        would have rejected a ``"word"`` request as incompatible before we got
        here).

        Args:
            native: The ``STTOutput`` from ``Qwen3ASR.generate``.
            duration: Audio duration in seconds, if known.
            want_words: Ignored (no word timing available).
            follows_text: Whether the transcript already has text before this
                decode (see :meth:`ModelBackend.to_result`).
            resolved_language: The request's effective language (see
                :meth:`ModelBackend.to_result`).

        Returns:
            A constant-schema :class:`TranscriptionResult`.
        """
        del want_words
        segments = _segments_from_dicts(
            _as_list(getattr(native, "segments", None)),
            join=self.chunk_join,
            language=getattr(native, "language", None),
            follows_text=follows_text,
            resolved_language=resolved_language,
        )
        detected = _detected_from_qwen_language(getattr(native, "language", None))
        return TranscriptionResult(
            text=_text_of(segments, native),
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

    def single_segment_span(self, params: MlxAudioParams) -> float | None:
        """Return ``None``: Whisper splits segments at what it hears.

        Args:
            params: Unused.

        Returns:
            ``None``.
        """
        del params
        return None

    def generate_kwargs(
        self,
        *,
        resolved_language: str | None,
        want_words: bool,
        params: MlxAudioParams,
        config: MlxAudioConfig,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Build Whisper ``generate`` kwargs (ISO code language + timestamps).

        Args:
            resolved_language: Effective BCP-47 language, or ``None`` for auto.
            want_words: Whether to request word-level timestamps.
            params: Engine-specific decoding knobs.
            config: Engine init config (unused; kept for protocol symmetry).
            prompt: The portable guidance prompt, passed as ``initial_prompt``.

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
        # Whisper declares guidance.prompt; its native slot is `initial_prompt`.
        if prompt is not None:
            kwargs["initial_prompt"] = prompt
        return kwargs

    def to_result(
        self,
        native: Any,
        *,
        duration: float | None,
        want_words: bool,
        follows_text: bool = False,
        resolved_language: str | None = None,
    ) -> TranscriptionResult:
        """Map a Whisper ``STTOutput`` onto a Standard result.

        Whisper segments are dicts ``{text, start, end, avg_logprob,
        no_speech_prob, words?}``; ``words`` (per segment) are dicts ``{word,
        start, end, probability}``. We back-fill word-level data only when
        ``want_words`` (spec TR.3 null semantics: ``words=None`` means "not
        requested").

        Each segment text is mlx-audio's decode of that segment's tokens, which
        starts with the space that joins it to the text before it (for a
        language written with spaces). mlx-audio's full text is the decode of
        all those tokens at once (``Whisper.generate`` in
        ``mlx_audio/stt/models/whisper/whisper.py``, mlx-audio 0.4.4 lines
        1005 and 1299), so ``text`` is the segment texts concatenated, leading
        space included, and nothing is added at a streaming seam.

        Args:
            native: The ``STTOutput`` from ``Whisper.generate``.
            duration: Audio duration in seconds, if known.
            want_words: Whether word timestamps were requested.
            follows_text: Ignored: the segment texts carry their own spaces.
            resolved_language: Ignored, for the same reason.

        Returns:
            A constant-schema :class:`TranscriptionResult`.
        """
        del follows_text, resolved_language
        segments, words = _segments_from_whisper(
            _as_list(getattr(native, "segments", None)), want_words=want_words
        )
        detected = languages.from_backend_language(getattr(native, "language", None))
        return TranscriptionResult(
            text=_text_of(segments, native),
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

    def single_segment_span(self, params: MlxAudioParams) -> float | None:
        """Return ``None``: these families split into sentences.

        Args:
            params: Unused.

        Returns:
            ``None``.
        """
        del params
        return None

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
        prompt: str | None = None,
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
            prompt: Ignored (these models declare no guidance).

        Returns:
            An empty kwargs dict.
        """
        del resolved_language, want_words, params, config, prompt
        return {}

    def to_result(
        self,
        native: Any,
        *,
        duration: float | None,
        want_words: bool,
        follows_text: bool = False,
        resolved_language: str | None = None,
    ) -> TranscriptionResult:
        """Map an ``AlignedResult`` onto a Standard result.

        Each ``AlignedSentence`` becomes a :class:`Segment`; each
        ``AlignedToken`` becomes a :class:`Word` (with no probability — the TDT /
        RNN-T paths do not emit per-token confidence in mlx-audio). Word data is
        attached only when ``want_words`` (spec TR.3).

        A sentence's text is its tokens' texts concatenated, and a token that
        starts a word starts with a space (the tokenizer turns ``▁`` into
        ``" "``). mlx-audio builds ``AlignedResult.text`` as the sentence texts
        concatenated and then strips it (``sentences_to_result`` and
        ``AlignedResult.__post_init__`` in ``mlx_audio/stt/models/nemo/
        alignment.py``, mlx-audio 0.4.4 lines 74 and 37). ``text`` here is the
        sentence texts concatenated without that strip, so it usually starts
        with a space, as Whisper's does, and nothing is added at a streaming
        seam.

        Args:
            native: The ``AlignedResult`` from the model's ``generate``.
            duration: Audio duration in seconds, if known.
            want_words: Whether word timestamps were requested.
            follows_text: Ignored: the sentence texts carry their own spaces.
            resolved_language: Ignored, for the same reason.

        Returns:
            A constant-schema :class:`TranscriptionResult`.

        Note:
            These models do not report a detected language; ``detected_language``
            is left ``None`` (honesty over guessing).
        """
        del follows_text
        segments, words = _segments_from_aligned(
            _as_list(getattr(native, "sentences", None)), want_words=want_words
        )
        return TranscriptionResult(
            text=_text_of(segments, native),
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
        audio_as_list: Whether ``generate`` needs the audio wrapped in a one-element
            ``list`` rather than a bare array.
        wants_path: Whether ``generate`` must be handed a **file path** rather than
            a decoded array. Voxtral-Mini's ``generate`` routes an array through a
            transformers processor whose ``apply_transcription_request`` then needs
            a ``format`` argument mlx-audio never passes (``len(None)`` ->
            ``TypeError``); only its string-path branch (``load_audio_as``) avoids
            that. The engine materializes a temp WAV from a negotiated array/bytes
            so such a family always receives a path.
        default_prompt: A decode prompt passed as the ``prompt`` generate kwarg
            when the family's ``generate`` accepts one and no provider prompt is
            set. Used to steer an audio-LLM (Qwen2-Audio) to emit a bare transcript
            instead of conversational commentary.
        text_from_segments: Whether the family returns its transcript only inside
            ``STTOutput.segments`` (parsed structured/diarization JSON) while
            ``STTOutput.text`` carries the raw JSON. When set, the result text is
            rebuilt from the segments' texts (VibeVoice-ASR; see
            :data:`TURN_SEPARATOR`).
        chunk_join: How mlx-audio joins the family's chunk texts into its full
            text. Required with ``segment_timing`` (each segment is one chunk),
            and unused without it.
        single_segment_span_s: For a family with ``segment_timing``, the
            internal chunk length up to which mlx-audio returns one segment
            spanning the whole input (Fun-ASR 1200 s, GLM-ASR 30 s, Cohere 35 s:
            mlx-audio's defaults, which the plugin does not change), or ``None``
            for a family that splits at content boundaries.
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
    wants_path: bool = False
    default_prompt: str | None = None
    text_from_segments: bool = False
    chunk_join: ChunkJoin | None = None
    single_segment_span_s: float | None = None
    forward: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Reject a timed family without a chunk join.

        Raises:
            ValueError: If ``segment_timing`` is set and ``chunk_join`` is not:
                without it the segment texts would lack the separators that
                mlx-audio's full text has between chunks.
        """
        if self.segment_timing and self.chunk_join is None:
            raise ValueError(
                f"SttFamilySpec for {self.model_types} sets segment_timing but no "
                "chunk_join: name how mlx-audio joins this family's chunk texts."
            )


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
        self.wants_path: bool = spec.wants_path

    def single_segment_span(self, params: MlxAudioParams) -> float | None:
        """Return the spec's ``single_segment_span_s``.

        Args:
            params: Unused (the generic families' chunk length is not a knob).

        Returns:
            The span in seconds, or ``None`` for content boundaries.
        """
        del params
        return self.spec.single_segment_span_s

    def generate_kwargs(
        self,
        *,
        resolved_language: str | None,
        want_words: bool,
        params: MlxAudioParams,
        config: MlxAudioConfig,
        prompt: str | None = None,
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
            prompt: Ignored (no STTOutput family here declares guidance).

        Returns:
            Keyword arguments for the family's ``model.generate``.
        """
        del want_words, config, prompt
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
        # Steer an audio-LLM family toward a bare transcript. Only set when the
        # spec declares it (the family's generate accepts `prompt`); a provider
        # `system_prompt`, when given, takes precedence so users can override.
        if spec.default_prompt is not None and "prompt" not in kwargs:
            kwargs["prompt"] = params.system_prompt or spec.default_prompt
        return kwargs

    def to_result(
        self,
        native: Any,
        *,
        duration: float | None,
        want_words: bool,
        follows_text: bool = False,
        resolved_language: str | None = None,
    ) -> TranscriptionResult:
        """Map the family's ``STTOutput`` onto a Standard result.

        Emits segments only when the family produces real per-chunk timing
        (``spec.segment_timing``); otherwise returns text only — never a
        fabricated span. With segments, each segment after the first gets the
        separator mlx-audio puts between two chunks (``spec.chunk_join``), and
        ``text`` is the segment texts concatenated. Without segments, ``text``
        is ``STTOutput.text`` exactly as mlx-audio returned it, or, for a
        ``text_from_segments`` family, the parsed segment texts joined with
        :data:`TURN_SEPARATOR`. Surfaces a detected language only when the
        family truly detects one (``spec.reports_detected_language``). No
        STTOutput family emits word timing, so ``words`` is always ``None``.

        Args:
            native: The ``STTOutput`` from the family's ``generate``.
            duration: Audio duration in seconds, if known.
            want_words: Ignored (no word timing available).
            follows_text: Whether the transcript already has text before this
                decode (see :meth:`ModelBackend.to_result`).
            resolved_language: The request's effective language (see
                :meth:`ModelBackend.to_result`).

        Returns:
            A constant-schema :class:`TranscriptionResult`.
        """
        del want_words
        spec = self.spec
        raw_segments = _as_list(getattr(native, "segments", None))
        # Emitting segments is governed independently by ``segment_timing`` (only
        # families with real per-chunk timing declare it); a text-only family that
        # merely parses its transcript out of JSON does not gain segment spans.
        segments: list[Segment] | None = None
        if spec.segment_timing:
            assert spec.chunk_join is not None  # SttFamilySpec.__post_init__ requires it
            segments = _segments_from_dicts(
                raw_segments,
                join=spec.chunk_join,
                language=getattr(native, "language", None),
                follows_text=follows_text,
                resolved_language=resolved_language,
            )
        if segments:
            text = "".join(segment.text for segment in segments)
        elif spec.text_from_segments:
            # The family emits its transcript only inside structured segments
            # (VibeVoice: ``STTOutput.text`` is the raw diarization JSON, while
            # ``segments`` are the parsed dicts, one per speaker turn). Rebuild
            # the transcript from the turn texts, kept exactly as parsed; fall
            # back to the raw text only if parsing yielded nothing.
            turns = [str(d.get("text", "")) for d in raw_segments]
            text = TURN_SEPARATOR.join(turn for turn in turns if turn) or _native_text(native)
        else:
            text = _native_text(native)
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
def has_inner_boundaries(
    backend: ModelBackend, params: MlxAudioParams, window_s: float | None
) -> bool:
    """Whether decoding ``window_s`` of audio can return a boundary inside it.

    This is the one rule the streaming session uses for a backend's segment
    boundaries: it is decided from the family and its configuration, never from
    one decode's output (a short window of Whisper can come back as one segment
    and still have boundaries in a longer one).

    Args:
        backend: The bound backend.
        params: The engine-specific decoding knobs.
        window_s: The longest window the session decodes, or ``None`` (no cap).

    Returns:
        ``True`` for a family that splits at content boundaries; for a family that
        splits only at fixed internal chunks, ``True`` only when that chunk is
        shorter than ``window_s`` (with no cap, ``False``: such a family is
        treated as having no boundaries, and pauses drive its commits).
    """
    span = backend.single_segment_span(params)
    if span is None:
        return True
    return window_s is not None and span < window_s


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


def _native_text(native: Any) -> str:
    """Return ``native.text`` exactly as mlx-audio returned it (``""`` if absent).

    Args:
        native: The native ``generate`` return value.

    Returns:
        The text, unstripped.
    """
    return str(getattr(native, "text", "") or "")


def _text_of(segments: list[Segment], native: Any) -> str:
    """Return the result ``text`` for a family that emits segments.

    With segments, the text is their texts concatenated (spec TR.1). Without
    any, the result carries ``segments=None`` and the text stands alone, so it
    is mlx-audio's own text.

    Args:
        segments: The converted segments (may be empty).
        native: The native ``generate`` return value.

    Returns:
        The transcript text.
    """
    if segments:
        return "".join(segment.text for segment in segments)
    return _native_text(native)


def _chunk_languages(
    raw: list[dict[str, Any]], language: Any, resolved_language: str | None
) -> list[str | None]:
    """Return each chunk's language as an ISO-639 primary subtag, or ``None``.

    The first source that names a language wins: the chunk dict's own
    ``language`` (Qwen3-ASR, Fun-ASR), then the decode's ``STTOutput.language``
    when it is a string (Cohere-ASR), then the language the request asked for
    (GLM-ASR reports none). mlx-audio reports names such as ``"Chinese"`` or
    ISO codes; :func:`~std_mlx_audio.languages.from_backend_language` maps
    both.

    Args:
        raw: The chunk dicts, one per chunk.
        language: The ``STTOutput.language`` of the decode.
        resolved_language: The request's effective BCP-47 language, or
            ``None`` for auto-detect.

    Returns:
        One subtag or ``None`` per chunk.
    """
    requested = languages.to_iso_subtag(resolved_language) if resolved_language else None
    decode = languages.from_backend_language(language) if isinstance(language, str) else None
    out: list[str | None] = []
    for item in raw:
        own = item.get("language")
        tag = languages.from_backend_language(own) if isinstance(own, str) else None
        out.append(tag or decode or requested)
    return out


def _segments_from_dicts(
    raw: list[dict[str, Any]],
    *,
    join: ChunkJoin,
    language: Any,
    follows_text: bool,
    resolved_language: str | None = None,
) -> list[Segment]:
    """Convert chunk-level dict segments into ``Segment`` objects.

    Args:
        raw: A list of ``{text, start, end, ...}`` dicts, one per chunk.
        join: How mlx-audio joins the chunk texts.
        language: The ``STTOutput.language`` of the decode.
        follows_text: Whether text precedes the first chunk (see
            :meth:`ChunkJoin.texts`).
        resolved_language: The request's effective BCP-47 language, or
            ``None`` (see :func:`_chunk_languages`).

    Returns:
        A list of :class:`Segment` (word-less), each text carrying the
        separator mlx-audio puts before that chunk.
    """
    texts = join.texts(
        [str(item.get("text", "")) for item in raw],
        chunk_languages=_chunk_languages(raw, language, resolved_language),
        follows_text=follows_text,
    )
    out: list[Segment] = []
    for item, text in zip(raw, texts, strict=True):
        start, end = _clamp_span(item.get("start", 0.0), item.get("end", 0.0))
        out.append(Segment(start=start, end=end, text=text, words=None))
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
    spec TR.1: engine-specific values belong in ``result.extra``). Fields a
    given family does not populate are simply absent.

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
    "TURN_SEPARATOR",
    "AlignedResultBackend",
    "ChunkJoin",
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
