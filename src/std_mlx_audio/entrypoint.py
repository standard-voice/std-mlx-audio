# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Entry-point factory functions for the std-mlx-audio plugin.

One factory per model preset (spec IC.7: model selection = entry-point preset,
never an init ``model`` field), so ``standard-asr models list`` / the registry /
a settings UI can enumerate every available model under the single ``mlx-audio``
engine. Each factory's return annotation is the **concrete** preset class (NOT
the ``StandardASR`` protocol) so the registry can resolve the class — and read
its class-level ``properties`` / ``declared_capabilities`` /
``provider_params_type`` — WITHOUT instantiating the engine
(``ModelRegistry.engine_class``; adapting_engine.md "Publish").
"""

from __future__ import annotations

from typing import Any

from .engine import (
    ParakeetTdt06BV3,
    Qwen3Asr06B,
    Qwen3Asr17B,
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


__all__ = [
    "create_parakeet_tdt_0_6b_v3",
    "create_qwen3_asr_0_6b",
    "create_qwen3_asr_1_7b",
    "create_whisper_large_v3_turbo",
    "create_whisper_tiny",
]
