# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The portable guidance ``prompt``: Qwen3-ASR's context and Whisper's ``initial_prompt``.

Written from the ways it can go wrong:

* a model declares ``guidance.prompt`` but the prompt never reaches it (the
  cardinal sin: a silently ignored, gated-and-passed request);
* the prompt reaches the wrong native slot, or overrides a Qwen ``system_prompt``
  the caller set explicitly;
* audio with no speech comes back as the prompt itself, and is reported as a
  transcript, in a file or in a streaming window;
* the guard against that drops real speech that uses the prompt's words.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from standard_asr import AudioArray, RuntimeParams, TranscriptionEvent
from standard_asr.audio.format import AudioFormat

from std_mlx_audio import MlxAudioConfig, MlxAudioParams, ParakeetTdt06BV3, Qwen3Asr06B, WhisperTiny
from std_mlx_audio._guidance import echoes_prompt  # pyright: ignore[reportPrivateUsage]
from std_mlx_audio.backends import Qwen3AsrBackend, WhisperBackend

from .conftest import FakeLoader, FakeSTTOutput, silent_pcm

_CONFIG = MlxAudioConfig()
_FMT = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)
# Chinese punctuation is full-width; the guard has to see past it.
_CONTEXT = "詞彙：Jezo、隨手記、Standard ASR、待辦。待辦：整理鵝鑾鼻露營裝備。"  # noqa: RUF001


# --------------------------------------------------------------------------- #
# Declared where it's honored
# --------------------------------------------------------------------------- #
def test_qwen3_declares_prompt_guidance_in_both_modes() -> None:
    caps = Qwen3Asr06B.declared_capabilities
    assert caps.supports("batch.guidance.prompt") is True
    assert caps.supports("streaming.guidance.prompt") is True
    # Only the free-text context; Qwen3-ASR has no term-boost channel.
    assert caps.supports("batch.guidance.phrase_hints") is False
    assert caps.supports("streaming.guidance.phrase_hints") is False


def test_parakeet_still_declares_no_guidance() -> None:
    assert ParakeetTdt06BV3.declared_capabilities.supports("streaming.guidance.prompt") is False


# --------------------------------------------------------------------------- #
# The prompt reaches the model's own slot
# --------------------------------------------------------------------------- #
def test_qwen3_prompt_is_its_context() -> None:
    kw = Qwen3AsrBackend().generate_kwargs(
        resolved_language=None,
        want_words=False,
        params=MlxAudioParams(),
        config=_CONFIG,
        prompt=_CONTEXT,
    )
    assert kw["system_prompt"] == _CONTEXT


def test_qwen3_explicit_system_prompt_wins_over_the_prompt() -> None:
    kw = Qwen3AsrBackend().generate_kwargs(
        resolved_language=None,
        want_words=False,
        params=MlxAudioParams(system_prompt="explicit"),
        config=_CONFIG,
        prompt=_CONTEXT,
    )
    assert kw["system_prompt"] == "explicit"


def test_qwen3_no_prompt_no_system_prompt() -> None:
    kw = Qwen3AsrBackend().generate_kwargs(
        resolved_language=None, want_words=False, params=MlxAudioParams(), config=_CONFIG
    )
    assert "system_prompt" not in kw


def test_whisper_prompt_is_its_initial_prompt() -> None:
    kw = WhisperBackend().generate_kwargs(
        resolved_language=None,
        want_words=False,
        params=MlxAudioParams(),
        config=_CONFIG,
        prompt="Jezo",
    )
    assert kw["initial_prompt"] == "Jezo"


def test_batch_prompt_reaches_generate(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(
        output=FakeSTTOutput(text="我在 Jezo 的隨手記裡寫了一個待辦", language=["Chinese"])
    )
    result = Qwen3Asr06B().transcribe(
        AudioArray(np.zeros(16000, dtype=np.float32), 16000), RuntimeParams(prompt=_CONTEXT)
    )
    assert loader.model.generate_calls[0]["system_prompt"] == _CONTEXT
    # Speech that uses the context's words is kept.
    assert result.text == "我在 Jezo 的隨手記裡寫了一個待辦"


# --------------------------------------------------------------------------- #
# Silence handed back as the prompt
# --------------------------------------------------------------------------- #
def test_batch_drops_the_prompt_handed_back(fake_loader: Callable[..., FakeLoader]) -> None:
    fake_loader(output=FakeSTTOutput(text=_CONTEXT, language=["Chinese"]))
    result = Qwen3Asr06B().transcribe(
        AudioArray(np.zeros(16000, dtype=np.float32), 16000), RuntimeParams(prompt=_CONTEXT)
    )
    assert result.text == ""
    assert result.segments is None


async def test_streaming_drops_windows_that_are_the_prompt(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    def output_fn(_audio: Any, kwargs: dict[str, Any]) -> FakeSTTOutput:
        # What Qwen3-ASR 0.6B does with a context and no speech.
        text = kwargs.get("system_prompt") or ""
        return FakeSTTOutput(
            text=text,
            segments=[{"text": text, "language": "Chinese", "start": 0.0, "end": 1.0}]
            if text
            else [],
        )

    loader = fake_loader(output_fn=output_fn)
    events: list[TranscriptionEvent] = []
    async with Qwen3Asr06B().start_transcription(
        audio_format=_FMT, params=RuntimeParams(prompt=_CONTEXT)
    ) as session:
        session.feed([silent_pcm(1.0)] * 6)
        async for event in session:
            events.append(event)
    assert any(call.get("system_prompt") == _CONTEXT for call in loader.model.generate_calls)
    assert all(e.type in ("progress", "done") for e in events)
    assert session.result().text == ""


async def test_whisper_streaming_passes_the_prompt(fake_loader: Callable[..., FakeLoader]) -> None:
    loader = fake_loader(output=FakeSTTOutput(text="", segments=[]))
    async with WhisperTiny().start_transcription(
        audio_format=_FMT, params=RuntimeParams(prompt="Jezo")
    ) as session:
        session.feed([silent_pcm(1.0)] * 3)
        async for _ in session:
            pass
    assert any(call.get("initial_prompt") == "Jezo" for call in loader.model.generate_calls)


# --------------------------------------------------------------------------- #
# The guard itself
# --------------------------------------------------------------------------- #
def test_the_whole_prompt_is_an_echo() -> None:
    assert echoes_prompt(_CONTEXT, _CONTEXT)


def test_spacing_and_punctuation_dont_hide_an_echo() -> None:
    assert echoes_prompt(
        "詞彙: Jezo, 隨手記, standard ASR, 待辦. 待辦: 整理鵝鑾鼻露營裝備", _CONTEXT
    )


def test_most_of_the_prompt_is_an_echo() -> None:
    # Measured: 405 of a 435-character context came back on low noise.
    assert echoes_prompt(_CONTEXT[: int(len(_CONTEXT) * 0.8)], _CONTEXT)


def test_speech_using_the_prompts_words_is_not_an_echo() -> None:
    assert not echoes_prompt("幫我把整理鵝鑾鼻露營裝備排到明天", _CONTEXT)
    assert not echoes_prompt("Jezo", _CONTEXT)
    assert not echoes_prompt("隨手記", _CONTEXT)


def test_a_term_or_two_is_too_short_to_tell() -> None:
    assert not echoes_prompt("Jezo", "Jezo")
    assert not echoes_prompt("Jezo 隨手記", "Jezo、隨手記")


def test_nothing_said_or_no_prompt_is_not_an_echo() -> None:
    assert not echoes_prompt("", _CONTEXT)
    assert not echoes_prompt("。", _CONTEXT)
    assert not echoes_prompt(_CONTEXT, None)
    assert not echoes_prompt(_CONTEXT, "")
