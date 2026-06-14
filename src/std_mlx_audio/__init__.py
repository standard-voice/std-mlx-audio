# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standard ASR engine plugin for MLX (Apple Silicon) ASR models.

A thin, typed adapter over the upstream ``mlx-audio`` package that exposes
MULTIPLE MLX speech-to-text model families under a single engine
(engine_id ``mlx-audio``), headlined by **Qwen3-ASR**:

* Qwen3-ASR (``qwen3-asr-0.6b`` / ``qwen3-asr-1.7b``) — multilingual, the
  headliner;
* Parakeet (``parakeet-tdt-0.6b-v3``) — precise word/sentence timestamps;
* Whisper (``whisper-large-v3-turbo`` / ``whisper-tiny``).

The "one engine, many models" design lives in :mod:`std_mlx_audio.backends`: a
per-family :class:`~std_mlx_audio.backends.ModelBackend` adapter normalizes each
backend's distinct native ``generate`` call-shape and return type onto the one
constant Standard ASR result schema. Batch transcription maps onto
``mlx_audio.stt`` ``model.generate``; streaming is a windowed re-decode session
with honest, conservative stability semantics
(see :mod:`std_mlx_audio._streaming`).

Public surface:

* Engine classes — one per preset (e.g. :class:`Qwen3Asr06B`).
* Config / params models — :class:`MlxAudioConfig`, :class:`MlxAudioParams`.
* Properties — :class:`MlxAudioProperties` and per-preset subclasses.
* Backend adapters — :class:`Qwen3AsrBackend`, :class:`WhisperBackend`,
  :class:`ParakeetBackend`.
* Entry-point factories — ``create_qwen3_asr_0_6b`` ... ``create_whisper_tiny``.
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
    ModelBackend,
    ParakeetBackend,
    Qwen3AsrBackend,
    WhisperBackend,
)
from .engine import (
    MlxAudioASR,
    ParakeetTdt06BV3,
    Qwen3Asr06B,
    Qwen3Asr17B,
    WhisperLargeV3Turbo,
    WhisperTiny,
)
from .entrypoint import (
    create_parakeet_tdt_0_6b_v3,
    create_qwen3_asr_0_6b,
    create_qwen3_asr_1_7b,
    create_whisper_large_v3_turbo,
    create_whisper_tiny,
)

__all__ = [
    "MlxAudioASR",
    "MlxAudioConfig",
    "MlxAudioParams",
    "MlxAudioProperties",
    "MlxAudioStreamingSession",
    "MlxDtype",
    "ModelBackend",
    "ParakeetBackend",
    "ParakeetTdt06BV3",
    "ParakeetTdt06BV3Properties",
    "Qwen3Asr06B",
    "Qwen3Asr06BProperties",
    "Qwen3Asr17B",
    "Qwen3Asr17BProperties",
    "Qwen3AsrBackend",
    "WhisperBackend",
    "WhisperLargeV3Turbo",
    "WhisperLargeV3TurboProperties",
    "WhisperTiny",
    "WhisperTinyProperties",
    "create_parakeet_tdt_0_6b_v3",
    "create_qwen3_asr_0_6b",
    "create_qwen3_asr_1_7b",
    "create_whisper_large_v3_turbo",
    "create_whisper_tiny",
]
