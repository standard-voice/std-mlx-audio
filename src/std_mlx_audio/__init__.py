# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard ASR engine plugin for MLX (Apple Silicon) ASR models.

A thin, typed adapter over the upstream ``mlx-audio`` package that exposes
EVERY MLX speech-to-text model family under a single engine (engine_id
``mlx-audio``), headlined by **Qwen3-ASR**. Each mlx-audio STT family is its own
preset:

* Qwen3-ASR (``qwen3-asr-0.6b`` / ``qwen3-asr-1.7b``) — multilingual headliner;
* Whisper (``whisper-large-v3-turbo`` / ``whisper-tiny``) — word timestamps;
* Parakeet (``parakeet-tdt-0.6b-v3``) / Nemotron
  (``nemotron-asr-streaming-0.6b``) — precise word/sentence timestamps;
* SenseVoice, Voxtral, Canary, GLM-ASR, Granite Speech (+ NAR), Fun-ASR,
  VibeVoice, Moonshine, MMS, FireRedASR2, Qwen2-Audio, Cohere ASR, Voxtral
  Realtime — the long tail of mlx-audio STT architectures.

The "one engine, many models" design lives in :mod:`std_mlx_audio.backends`: a
per-family :class:`~std_mlx_audio.backends.ModelBackend` adapter normalizes each
backend's distinct native ``generate`` call-shape and return type onto the one
constant Standard ASR result schema. Whisper and Qwen3-ASR keep bespoke
backends; Parakeet/Nemotron share
:class:`~std_mlx_audio.backends.AlignedResultBackend`; the remaining ~14
``STTOutput`` families are served by a single data-driven
:class:`~std_mlx_audio.backends.GenericSttBackend` parameterized by a
:class:`~std_mlx_audio.backends.SttFamilySpec`. Batch transcription maps onto
``mlx_audio.stt`` ``model.generate``; streaming is a windowed re-decode session
(only for families that emit real segment/token timing) with honest,
conservative stability semantics (see :mod:`std_mlx_audio._streaming`).

Public surface:

* Engine classes — one per preset (e.g. :class:`Qwen3Asr06B`, :class:`MoonshineTiny`).
* Config / params models — :class:`MlxAudioConfig`, :class:`MlxAudioParams`.
* Properties — :class:`MlxAudioProperties` (+ the original per-preset subclasses).
* Backend adapters — :class:`Qwen3AsrBackend`, :class:`WhisperBackend`,
  :class:`AlignedResultBackend`, :class:`GenericSttBackend` (+ :class:`SttFamilySpec`).
* Entry-point factories — ``create_qwen3_asr_0_6b`` ... ``create_voxtral_realtime_4b``.
"""

from __future__ import annotations

from ._config import MlxAudioConfig, MlxAudioParams, MlxDtype
from ._metadata import (
    MlxAudioProperties,
    ParakeetTdt06BV3Properties,
    Qwen3Asr06BProperties,
    Qwen3Asr17BProperties,
    WhisperLargeV3TurboProperties,
    WhisperTinyProperties,
)
from ._streaming import MlxAudioStreamingSession
from .backends import (
    AlignedResultBackend,
    GenericSttBackend,
    ModelBackend,
    Qwen3AsrBackend,
    SttFamilySpec,
    WhisperBackend,
)
from .engine import (
    Canary1BV2,
    CohereAsr,
    FireRedAsr2Aed,
    FunAsrNano,
    GlmAsrNano,
    GraniteSpeech1B,
    GraniteSpeechNar2B,
    MlxAudioASR,
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
from .entrypoint import (
    create_canary_1b_v2,
    create_cohere_asr,
    create_fireredasr2_aed,
    create_fun_asr_nano,
    create_glm_asr_nano,
    create_granite_speech_1b,
    create_granite_speech_nar_2b,
    create_mms_1b_all,
    create_moonshine_tiny,
    create_nemotron_asr_streaming_0_6b,
    create_parakeet_tdt_0_6b_v3,
    create_qwen2_audio_7b,
    create_qwen3_asr_0_6b,
    create_qwen3_asr_1_7b,
    create_sensevoice_small,
    create_vibevoice_asr,
    create_voxtral_mini_3b,
    create_voxtral_realtime_4b,
    create_whisper_large_v3_turbo,
    create_whisper_tiny,
)

__all__ = [
    "AlignedResultBackend",
    "Canary1BV2",
    "CohereAsr",
    "FireRedAsr2Aed",
    "FunAsrNano",
    "GenericSttBackend",
    "GlmAsrNano",
    "GraniteSpeech1B",
    "GraniteSpeechNar2B",
    "MlxAudioASR",
    "MlxAudioConfig",
    "MlxAudioParams",
    "MlxAudioProperties",
    "MlxAudioStreamingSession",
    "MlxDtype",
    "Mms1BAll",
    "ModelBackend",
    "MoonshineTiny",
    "NemotronAsrStreaming06B",
    "ParakeetTdt06BV3",
    "ParakeetTdt06BV3Properties",
    "Qwen2Audio7B",
    "Qwen3Asr06B",
    "Qwen3Asr06BProperties",
    "Qwen3Asr17B",
    "Qwen3Asr17BProperties",
    "Qwen3AsrBackend",
    "SenseVoiceSmall",
    "SttFamilySpec",
    "VibeVoiceAsr",
    "VoxtralMini3B",
    "VoxtralRealtime4B",
    "WhisperBackend",
    "WhisperLargeV3Turbo",
    "WhisperLargeV3TurboProperties",
    "WhisperTiny",
    "WhisperTinyProperties",
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
