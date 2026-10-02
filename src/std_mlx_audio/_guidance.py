# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Guarding against a decode that hands the guidance prompt back.

Measured on Qwen3-ASR 0.6B (4-bit MLX): given a context in its system turn and
audio with no speech in it (silence, or low room noise), the model returns the
context itself, word for word, as the transcript. A 47-character term list and
a 435-character paragraph both came back whole. Without a context it returns
nothing or a stray word. Whisper is known to do the same with
``initial_prompt``. The guidance contract calls a prompt background knowledge,
never text to emit, so an output that is the prompt is not a transcript.

The check is deliberately narrow, so speech that happens to use the prompt's
words is never dropped: the output, with spacing and punctuation removed, must
be a stretch of the prompt at least half its length, and the prompt must be
long enough (12 such characters) for that to mean anything.
"""

from __future__ import annotations

import unicodedata

#: Below this many letters and digits, a prompt is one or two terms, and a user
#: saying them would be indistinguishable from an echo.
_MIN_PROMPT_CHARS = 12
#: The share of the prompt an output has to reproduce to count as an echo.
_MIN_SHARE = 0.5


def _letters(text: str) -> str:
    """Letters and digits only, case-folded, so spacing and punctuation don't matter."""
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKC", text).casefold()
        if unicodedata.category(ch)[0] in "LN"
    )


def echoes_prompt(text: str, prompt: str | None) -> bool:
    """Whether a decoded ``text`` is the guidance ``prompt`` handed back rather than speech.

    Args:
        text: What the model returned for a window or a file.
        prompt: The guidance prompt it was given, or ``None``.

    Returns:
        ``True`` when ``text`` is a stretch of ``prompt`` covering at least half
        of it, and ``prompt`` is long enough for that to be a signal.
    """
    if not prompt:
        return False
    said = _letters(text)
    given = _letters(prompt)
    if len(given) < _MIN_PROMPT_CHARS or not said:
        return False
    return said in given and len(said) >= _MIN_SHARE * len(given)
