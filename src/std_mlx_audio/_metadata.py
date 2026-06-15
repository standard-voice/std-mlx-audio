# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Static Properties and declared Capabilities for the MLX ASR models.

Everything here is read at the *class* level without instantiating the engine
(``standard-asr show``, the registry, REST ``GET /v1/capabilities``), so
it MUST be honest and self-contained. Capabilities are declared **fail-closed**:
we declare only what each MLX backend genuinely delivers.

This is the "one engine, many models" surface: a single engine_id
(``mlx-audio``) carries several model families with *different* capabilities —
Qwen3-ASR (segment timing only, no per-word; full LLM sampler), Whisper (word +
segment timing, prompt guidance), and Parakeet (word + segment timing, but
NO runtime language selection). Each family is its own ``Properties`` +
``DeclaredCapabilities`` pair; the engine class binds one pair per preset.
"""

from __future__ import annotations

from typing import Literal

from standard_asr.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
    FinalityCap,
    FlagCap,
    GuidanceCaps,
    LanguageCaps,
    PromptCap,
    PromptConstraints,
    ReconnectCap,
    StreamingCapabilities,
    StreamingGuidanceCaps,
    StreamTimestampsCap,
    WordTimestampsCap,
)
from standard_asr.engine import BaseProperties, InputKind, SampleRateRange

from .languages import (
    PARAKEET_V3_LANGUAGES,
    QWEN_DETECTABLE_LANGUAGES,
    QWEN_SELECTABLE_LANGUAGES,
    WHISPER_DETECTABLE_LANGUAGES,
    WHISPER_SELECTABLE_LANGUAGES,
)

#: All MLX STT backends run at 16 kHz mono; the standard layer resamples to it.
_SAMPLE_RATE = 16000

#: Shared accepted input shapes. mlx-audio's loaders accept a decoded waveform
#: (mx.array / np.ndarray) OR a file path; we declare ARRAY (zero-copy from the
#: standard layer's negotiated float32 waveform) plus ENCODED_FILE /
#: ENCODED_BYTES so an app holding a file or in-memory upload avoids a needless
#: decode round-trip on its side (we feed the path/bytes straight through).
_ACCEPTED_INPUT: set[InputKind] = {
    InputKind.ARRAY,
    InputKind.ENCODED_FILE,
    InputKind.ENCODED_BYTES,
}


class MlxAudioProperties(BaseProperties):
    """Base static metadata shared by every MLX ASR preset.

    Per-family subclasses override ``model_name`` (MUST equal the entry-point
    key's model component so ``properties.model_id`` matches — compliance
    enforced), the language inventories, and the description.

    I/O boundaries:

    * ``accepted_input`` — array + encoded file/bytes (see ``_ACCEPTED_INPUT``).
    * ``native_sample_rate = 16000``; ``accepted_sample_rates = [16000]``. MLX STT
      models run at 16 kHz; the standard layer resamples anything else before us.
    * ``wire_encodings = ["pcm_s16le"]`` — streaming wire frames are canonical
      16-bit PCM, so an undeclared encoding is fail-closed-rejected rather than
      mis-framed (spec AI 3.2).
    """

    engine_id: str = "mlx-audio"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = _ACCEPTED_INPUT
    native_sample_rate: int = _SAMPLE_RATE
    accepted_sample_rates: list[int] | SampleRateRange | Literal["any"] = [_SAMPLE_RATE]
    wire_encodings: list[str] | None = ["pcm_s16le"]


# --------------------------------------------------------------------------- #
# Qwen3-ASR family
# --------------------------------------------------------------------------- #
# Qwen3-ASR emits chunk-level segment timing on every run but NO per-word
# timing, so it declares "segment" only (declaring "word" would be dishonest;
# omitting "segment" would falsely reject the cheapest always-satisfiable
# request — spec TR.3). It exposes a full LLM sampler via provider params, but
# the portable `prompt` maps to a Qwen chat slot we expose as the
# Qwen-specific `system_prompt` provider param instead (a free-text decode prompt
# has no portable Whisper-`initial_prompt`-equivalent here), so batch `guidance`
# is left unsupported (fail-closed) rather than mis-mapped.
_QWEN_WORD_TS = WordTimestampsCap(supported=True, granularities=["segment"])

_QWEN_CAPABILITIES = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        word_timestamps=_QWEN_WORD_TS,
    ),
    # Windowed streaming (re-decode strategy; Qwen3-ASR has no native streaming).
    # Honest consequences of re-decoding the whole window each pass:
    #   * emits_partials = True, word_stability = False (any earlier text may be
    #     rewritten -> stable_until=0), re_segments = False (never `supersede`),
    #   * reconnect unsupported (local in-process model), finality = final
    #     (a settled sentence won't change, but we make no post-processing
    #     immutability promise so not `closed`),
    #   * timestamps = post_align (mapped from per-window decode, not native
    #     frame-aligned streaming timestamps).
    streaming=StreamingCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        word_timestamps=_QWEN_WORD_TS,
        emits_partials=FlagCap(supported=True),
        re_segments=FlagCap(supported=False),
        word_stability=FlagCap(supported=False),
        reconnect=ReconnectCap(mode="unsupported"),
        finality_level=FinalityCap(mode="final"),
        timestamps=StreamTimestampsCap(mode="post_align"),
    ),
    streaming_input=FlagCap(supported=True),
    streaming_output=FlagCap(supported=True),
    self_resamples=FlagCap(supported=False),
)


class Qwen3Asr06BProperties(MlxAudioProperties):
    """``mlx-audio/qwen3-asr-0.6b`` — small Qwen3-ASR (fast; the headliner)."""

    model_name: str = "qwen3-asr-0.6b"
    selectable_languages: list[str] = QWEN_SELECTABLE_LANGUAGES
    detectable_languages: list[str] = QWEN_DETECTABLE_LANGUAGES
    description: str | None = (
        "Qwen3-ASR 0.6B (4-bit MLX), 30-language multilingual; fast Apple-Silicon "
        "inference. The headliner; smallest Qwen3-ASR for quick verification."
    )


class Qwen3Asr17BProperties(MlxAudioProperties):
    """``mlx-audio/qwen3-asr-1.7b`` — larger, more accurate Qwen3-ASR."""

    model_name: str = "qwen3-asr-1.7b"
    selectable_languages: list[str] = QWEN_SELECTABLE_LANGUAGES
    detectable_languages: list[str] = QWEN_DETECTABLE_LANGUAGES
    description: str | None = (
        "Qwen3-ASR 1.7B (8-bit MLX), 30-language multilingual; higher accuracy "
        "than 0.6B. Production Qwen3-ASR preset."
    )


# --------------------------------------------------------------------------- #
# Whisper family
# --------------------------------------------------------------------------- #
# Whisper emits per-segment start/end on every run and per-word timing when
# asked, so it declares both "word" and "segment". It honors a free-text
# `initial_prompt`, so batch guidance.prompt is supported (conservative token
# cap; Whisper truncates an over-long prompt silently otherwise).
_WHISPER_WORD_TS = WordTimestampsCap(supported=True, granularities=["word", "segment"])
_WHISPER_GUIDANCE = GuidanceCaps(
    prompt=PromptCap(supported=True, constraints=PromptConstraints(max_tokens=200)),
)

_WHISPER_CAPABILITIES = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        word_timestamps=_WHISPER_WORD_TS,
        guidance=_WHISPER_GUIDANCE,
    ),
    streaming=StreamingCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        word_timestamps=_WHISPER_WORD_TS,
        guidance=StreamingGuidanceCaps(prompt=_WHISPER_GUIDANCE.prompt),
        emits_partials=FlagCap(supported=True),
        re_segments=FlagCap(supported=False),
        word_stability=FlagCap(supported=False),
        reconnect=ReconnectCap(mode="unsupported"),
        finality_level=FinalityCap(mode="final"),
        timestamps=StreamTimestampsCap(mode="post_align"),
    ),
    streaming_input=FlagCap(supported=True),
    streaming_output=FlagCap(supported=True),
    self_resamples=FlagCap(supported=False),
)


class WhisperLargeV3TurboProperties(MlxAudioProperties):
    """``mlx-audio/whisper-large-v3-turbo`` — fast, near-large-v3 Whisper."""

    model_name: str = "whisper-large-v3-turbo"
    selectable_languages: list[str] = WHISPER_SELECTABLE_LANGUAGES
    detectable_languages: list[str] = WHISPER_DETECTABLE_LANGUAGES
    description: str | None = (
        "OpenAI Whisper large-v3-turbo (MLX), multilingual; word timestamps and "
        "prompt guidance. Fast production Whisper preset."
    )


class WhisperTinyProperties(MlxAudioProperties):
    """``mlx-audio/whisper-tiny`` — smallest Whisper (fast tests/smoke)."""

    model_name: str = "whisper-tiny"
    selectable_languages: list[str] = WHISPER_SELECTABLE_LANGUAGES
    detectable_languages: list[str] = WHISPER_DETECTABLE_LANGUAGES
    description: str | None = (
        "OpenAI Whisper tiny (MLX), multilingual; smallest/fastest preset for smoke runs and tests."
    )


# --------------------------------------------------------------------------- #
# Parakeet family
# --------------------------------------------------------------------------- #
# Parakeet ALWAYS produces token-level alignment, so it declares both "word" and
# "segment". Critically it has NO language argument (fixed-language model), so
# language.runtime_override is fail-closed FALSE — the standard layer will reject
# a per-request `language` override as unsupported rather than silently ignoring
# it. This is the honest declaration for a model whose language is not selectable.
_PARAKEET_WORD_TS = WordTimestampsCap(supported=True, granularities=["word", "segment"])

_PARAKEET_CAPABILITIES = DeclaredCapabilities(
    batch=BatchCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=False)),
        word_timestamps=_PARAKEET_WORD_TS,
    ),
    streaming=StreamingCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=False)),
        word_timestamps=_PARAKEET_WORD_TS,
        emits_partials=FlagCap(supported=True),
        re_segments=FlagCap(supported=False),
        word_stability=FlagCap(supported=False),
        reconnect=ReconnectCap(mode="unsupported"),
        finality_level=FinalityCap(mode="final"),
        timestamps=StreamTimestampsCap(mode="post_align"),
    ),
    streaming_input=FlagCap(supported=True),
    streaming_output=FlagCap(supported=True),
    self_resamples=FlagCap(supported=False),
)


class ParakeetTdt06BV3Properties(MlxAudioProperties):
    """``mlx-audio/parakeet-tdt-0.6b-v3`` — NVIDIA Parakeet TDT (25 EU languages).

    Parakeet's language is fixed by the model (not user-selectable), so
    ``selectable_languages`` lists the supported set WITHOUT an ``"auto"``
    directive — and the capabilities declare ``language.runtime_override=False``.
    """

    model_name: str = "parakeet-tdt-0.6b-v3"
    selectable_languages: list[str] = list(PARAKEET_V3_LANGUAGES)
    detectable_languages: list[str] = list(PARAKEET_V3_LANGUAGES)
    description: str | None = (
        "NVIDIA Parakeet TDT 0.6B v3 (MLX), 25 European languages; precise "
        "word/sentence timestamps. Word-timestamp specialist (weights CC-BY-4.0)."
    )


__all__ = [
    "_PARAKEET_CAPABILITIES",
    "_QWEN_CAPABILITIES",
    "_WHISPER_CAPABILITIES",
    "MlxAudioProperties",
    "ParakeetTdt06BV3Properties",
    "Qwen3Asr06BProperties",
    "Qwen3Asr17BProperties",
    "WhisperLargeV3TurboProperties",
    "WhisperTinyProperties",
]
