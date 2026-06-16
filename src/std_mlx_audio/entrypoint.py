# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Entry-point factory functions for the std-mlx-audio plugin.

One factory per model preset (spec IC.7: model selection = entry-point preset,
never an init ``model`` field), so ``standard-asr list`` / the registry /
a settings UI can enumerate every available model under the single ``mlx-audio``
engine. Each factory's return annotation is the **concrete** preset class (NOT
the ``StandardASR`` protocol) so the registry can resolve the class — and read
its class-level ``properties`` / ``declared_capabilities`` /
``provider_params_type`` — WITHOUT instantiating the engine
(``ModelRegistry.engine_class``; adapting_engine.md "Publish").

This is the "one engine, many models" surface in full: the single ``mlx-audio``
engine exposes every mlx-audio STT family (Qwen3-ASR, Whisper, Parakeet, plus
SenseVoice, Voxtral, Canary, Nemotron, Moonshine, GLM-ASR, Granite Speech,
Fun-ASR, VibeVoice, FireRedASR2, Qwen2-Audio, MMS, Cohere ASR) as its own preset.
"""

from __future__ import annotations

from typing import Any

from .engine import (
    Canary1BV2,
    CohereAsr,
    FireRedAsr2Aed,
    FunAsrNano,
    GlmAsrNano,
    GraniteSpeech1B,
    GraniteSpeechNar2B,
    Mms1BAll,
    MoonshineTiny,
    NemotronAsrStreaming06B,
    ParakeetTdt06BV3,
    Qwen2Audio7B,
    Qwen3Asr06B,
    Qwen3Asr17B,
    SenseVoiceSmall,
    VibeVoiceAsr,
    VoxtralMini3B,
    VoxtralRealtime4B,
    WhisperLargeV3Turbo,
    WhisperTiny,
)


def create_qwen3_asr_0_6b(**kwargs: Any) -> Qwen3Asr06B:
    """Return the ``mlx-audio/qwen3-asr-0.6b`` preset (the headliner; smallest Qwen3-ASR).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`Qwen3Asr06B`.

    Returns:
        A configured Qwen3-ASR 0.6B engine.
    """
    return Qwen3Asr06B(**kwargs)


def create_qwen3_asr_1_7b(**kwargs: Any) -> Qwen3Asr17B:
    """Return the ``mlx-audio/qwen3-asr-1.7b`` preset (larger, more accurate Qwen3-ASR).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`Qwen3Asr17B`.

    Returns:
        A configured Qwen3-ASR 1.7B engine.
    """
    return Qwen3Asr17B(**kwargs)


def create_parakeet_tdt_0_6b_v3(**kwargs: Any) -> ParakeetTdt06BV3:
    """Return the ``mlx-audio/parakeet-tdt-0.6b-v3`` preset (word-timestamp specialist).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`ParakeetTdt06BV3`.

    Returns:
        A configured Parakeet TDT 0.6B v3 engine.
    """
    return ParakeetTdt06BV3(**kwargs)


def create_whisper_large_v3_turbo(**kwargs: Any) -> WhisperLargeV3Turbo:
    """Return the ``mlx-audio/whisper-large-v3-turbo`` preset (fast multilingual Whisper).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`WhisperLargeV3Turbo`.

    Returns:
        A configured Whisper large-v3-turbo engine.
    """
    return WhisperLargeV3Turbo(**kwargs)


def create_whisper_tiny(**kwargs: Any) -> WhisperTiny:
    """Return the ``mlx-audio/whisper-tiny`` preset (smallest Whisper; smoke/tests).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`WhisperTiny`.

    Returns:
        A configured Whisper tiny engine.
    """
    return WhisperTiny(**kwargs)


def create_nemotron_asr_streaming_0_6b(**kwargs: Any) -> NemotronAsrStreaming06B:
    """Return the ``mlx-audio/nemotron-asr-streaming-0.6b`` preset (Nemotron; word timing).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`NemotronAsrStreaming06B`.

    Returns:
        A configured Nemotron ASR streaming 0.6B engine.
    """
    return NemotronAsrStreaming06B(**kwargs)


def create_sensevoice_small(**kwargs: Any) -> SenseVoiceSmall:
    """Return the ``mlx-audio/sensevoice-small`` preset (SenseVoice; language ID).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`SenseVoiceSmall`.

    Returns:
        A configured SenseVoice-Small engine.
    """
    return SenseVoiceSmall(**kwargs)


def create_cohere_asr(**kwargs: Any) -> CohereAsr:
    """Return the ``mlx-audio/cohere-asr`` preset (Cohere ASR; 14 languages).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`CohereAsr`.

    Returns:
        A configured Cohere ASR engine.
    """
    return CohereAsr(**kwargs)


def create_fun_asr_nano(**kwargs: Any) -> FunAsrNano:
    """Return the ``mlx-audio/fun-asr-nano`` preset (Fun-ASR-Nano; hotwords + ITN).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`FunAsrNano`.

    Returns:
        A configured Fun-ASR-Nano engine.
    """
    return FunAsrNano(**kwargs)


def create_voxtral_mini_3b(**kwargs: Any) -> VoxtralMini3B:
    """Return the ``mlx-audio/voxtral-mini-3b`` preset (Mistral Voxtral-Mini 3B).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`VoxtralMini3B`.

    Returns:
        A configured Voxtral-Mini 3B engine.
    """
    return VoxtralMini3B(**kwargs)


def create_canary_1b_v2(**kwargs: Any) -> Canary1BV2:
    """Return the ``mlx-audio/canary-1b-v2`` preset (NVIDIA Canary; ASR + translation).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`Canary1BV2`.

    Returns:
        A configured Canary 1B v2 engine.
    """
    return Canary1BV2(**kwargs)


def create_qwen2_audio_7b(**kwargs: Any) -> Qwen2Audio7B:
    """Return the ``mlx-audio/qwen2-audio-7b`` preset (Qwen2-Audio 7B Instruct).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`Qwen2Audio7B`.

    Returns:
        A configured Qwen2-Audio 7B engine.
    """
    return Qwen2Audio7B(**kwargs)


def create_glm_asr_nano(**kwargs: Any) -> GlmAsrNano:
    """Return the ``mlx-audio/glm-asr-nano`` preset (GLM-ASR-Nano; segment timing).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`GlmAsrNano`.

    Returns:
        A configured GLM-ASR-Nano engine.
    """
    return GlmAsrNano(**kwargs)


def create_granite_speech_1b(**kwargs: Any) -> GraniteSpeech1B:
    """Return the ``mlx-audio/granite-speech-1b`` preset (Granite Speech; ASR + translation).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`GraniteSpeech1B`.

    Returns:
        A configured Granite Speech 1B engine.
    """
    return GraniteSpeech1B(**kwargs)


def create_granite_speech_nar_2b(**kwargs: Any) -> GraniteSpeechNar2B:
    """Return the ``mlx-audio/granite-speech-nar-2b`` preset (Granite Speech NAR).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`GraniteSpeechNar2B`.

    Returns:
        A configured Granite Speech NAR 2B engine.
    """
    return GraniteSpeechNar2B(**kwargs)


def create_vibevoice_asr(**kwargs: Any) -> VibeVoiceAsr:
    """Return the ``mlx-audio/vibevoice-asr`` preset (VibeVoice-ASR; context-biased).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`VibeVoiceAsr`.

    Returns:
        A configured VibeVoice-ASR engine.
    """
    return VibeVoiceAsr(**kwargs)


def create_moonshine_tiny(**kwargs: Any) -> MoonshineTiny:
    """Return the ``mlx-audio/moonshine-tiny`` preset (Moonshine tiny; fast English).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`MoonshineTiny`.

    Returns:
        A configured Moonshine tiny engine.
    """
    return MoonshineTiny(**kwargs)


def create_mms_1b_all(**kwargs: Any) -> Mms1BAll:
    """Return the ``mlx-audio/mms-1b-all`` preset (Meta MMS-1B-all; multilingual CTC).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`Mms1BAll`.

    Returns:
        A configured MMS-1B-all engine.
    """
    return Mms1BAll(**kwargs)


def create_fireredasr2_aed(**kwargs: Any) -> FireRedAsr2Aed:
    """Return the ``mlx-audio/fireredasr2-aed`` preset (FireRedASR2-AED; beam search).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`FireRedAsr2Aed`.

    Returns:
        A configured FireRedASR2-AED engine.
    """
    return FireRedAsr2Aed(**kwargs)


def create_voxtral_realtime_4b(**kwargs: Any) -> VoxtralRealtime4B:
    """Return the ``mlx-audio/voxtral-realtime-4b`` preset (Voxtral-Mini 4B Realtime).

    Args:
        **kwargs: Keyword arguments forwarded to :class:`VoxtralRealtime4B`.

    Returns:
        A configured Voxtral-Mini 4B Realtime engine.
    """
    return VoxtralRealtime4B(**kwargs)


__all__ = [
    "create_canary_1b_v2",
    "create_cohere_asr",
    "create_fireredasr2_aed",
    "create_fun_asr_nano",
    "create_glm_asr_nano",
    "create_granite_speech_1b",
    "create_granite_speech_nar_2b",
    "create_mms_1b_all",
    "create_moonshine_tiny",
    "create_nemotron_asr_streaming_0_6b",
    "create_parakeet_tdt_0_6b_v3",
    "create_qwen2_audio_7b",
    "create_qwen3_asr_0_6b",
    "create_qwen3_asr_1_7b",
    "create_sensevoice_small",
    "create_vibevoice_asr",
    "create_voxtral_mini_3b",
    "create_voxtral_realtime_4b",
    "create_whisper_large_v3_turbo",
    "create_whisper_tiny",
]
