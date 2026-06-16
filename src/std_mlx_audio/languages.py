# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""BCP-47 <-> backend language mapping for the MLX ASR engine.

The MLX backends name languages differently, so the standard layer's single
BCP-47 vocabulary (``selectable_languages`` / ``language`` /
``detected_language`` are BCP-47, spec G.1.3) must be translated per backend:

* **Qwen3-ASR** (mlx-audio ``qwen3_asr``) validates ``language=`` against English
  *name* strings (``"Chinese"``, ``"English"``, ``"Cantonese"`` ...) — its
  ``config.support_languages`` is a list of names, and it auto-detects when
  ``language=None``, reporting the detected name(s) back in the output.
* **Whisper** (mlx-audio ``whisper``) takes ISO-639 *codes* (``"en"``, ``"ja"``)
  and reports a code back as ``output.language``.
* **Parakeet** (mlx-audio ``parakeet``) takes no language argument at all — the
  model is fixed-language (v3 covers 25 European languages); we expose its
  selectable set but never pass a language through.

This module is the single source of truth that converts a resolved BCP-47 tag
into whichever shape the active backend needs, and converts a backend-reported
language back into BCP-47 for ``detected_language``. Keeping it in one table
avoids the "same value means different things on different backends" trap (spec
Runtime §6).
"""

from __future__ import annotations

from standard_asr.language import normalize_bcp47

#: BCP-47 primary subtag -> (Whisper ISO code, Qwen3-ASR English name).
#:
#: Covers the 30 language varieties Qwen3-ASR advertises (verified against the
#: ``support_languages`` list in the mlx-community Qwen3-ASR ``config.json``).
#: ``yue`` (Cantonese) is a distinct BCP-47 subtag from ``zh`` (Mandarin) and
#: Qwen treats them separately, so both are listed. We key on the *primary*
#: subtag (``normalize_bcp47(tag).split("-")[0]``) so region-qualified tags such
#: as ``en-US`` / ``zh-CN`` resolve to the same engine language. The ISO code and
#: the English name happen to share the subtag for most entries, but keeping both
#: explicit keeps each backend's surface honest.
_LANGUAGE_TABLE: dict[str, tuple[str, str]] = {
    "zh": ("zh", "Chinese"),
    "yue": ("yue", "Cantonese"),
    "en": ("en", "English"),
    "ar": ("ar", "Arabic"),
    "de": ("de", "German"),
    "fr": ("fr", "French"),
    "es": ("es", "Spanish"),
    "pt": ("pt", "Portuguese"),
    "id": ("id", "Indonesian"),
    "it": ("it", "Italian"),
    "ko": ("ko", "Korean"),
    "ru": ("ru", "Russian"),
    "th": ("th", "Thai"),
    "vi": ("vi", "Vietnamese"),
    "ja": ("ja", "Japanese"),
    "tr": ("tr", "Turkish"),
    "hi": ("hi", "Hindi"),
    "ms": ("ms", "Malay"),
    "nl": ("nl", "Dutch"),
    "sv": ("sv", "Swedish"),
    "da": ("da", "Danish"),
    "fi": ("fi", "Finnish"),
    "pl": ("pl", "Polish"),
    "cs": ("cs", "Czech"),
    "fil": ("fil", "Filipino"),
    "fa": ("fa", "Persian"),
    "el": ("el", "Greek"),
    "ro": ("ro", "Romanian"),
    "hu": ("hu", "Hungarian"),
    "mk": ("mk", "Macedonian"),
}

#: Reverse map: Qwen English name (lower-cased) -> BCP-47 primary subtag.
_NAME_TO_BCP47: dict[str, str] = {
    name.lower(): bcp47 for bcp47, (_code, name) in _LANGUAGE_TABLE.items()
}
#: Reverse map: Whisper ISO code (lower-cased) -> BCP-47 primary subtag.
_CODE_TO_BCP47: dict[str, str] = {
    code.lower(): bcp47 for bcp47, (code, _name) in _LANGUAGE_TABLE.items()
}

#: The full BCP-47 inventory Qwen3-ASR supports, plus the reserved ``"auto"``
#: directive (multilingual auto-detection = ``language=None`` upstream). Used by
#: the Qwen3-ASR presets' ``selectable_languages``.
QWEN_SELECTABLE_LANGUAGES: list[str] = ["auto", *_LANGUAGE_TABLE.keys()]
#: BCP-47 tags Qwen3-ASR can detect (everything it supports).
QWEN_DETECTABLE_LANGUAGES: list[str] = list(_LANGUAGE_TABLE.keys())

#: Whisper supports ~99 languages; we surface a representative multilingual
#: subset for the discovery UI / candidate validation (the model still detects
#: any of its languages under ``auto``). Intersecting with our table keeps the
#: round-trip (code -> BCP-47 -> code) lossless.
WHISPER_SELECTABLE_LANGUAGES: list[str] = ["auto", *_LANGUAGE_TABLE.keys()]
WHISPER_DETECTABLE_LANGUAGES: list[str] = list(_LANGUAGE_TABLE.keys())

#: Parakeet TDT v3 covers these 25 European languages (NVIDIA model card). It is
#: not user-selectable at runtime (no language arg), so there is no ``"auto"``
#: directive — the model decides. Declared for discovery only.
PARAKEET_V3_LANGUAGES: list[str] = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk",
]  # fmt: skip

# --------------------------------------------------------------------------- #
# Per-family language inventories for the generic STTOutput backends.
#
# Each family below takes an ISO-639 code directly in its ``generate`` call (the
# model maps the code to whatever internal surface it needs), so the adapter
# passes the BCP-47 *primary subtag* through unchanged (see ``to_iso_subtag``).
# The inventories are verified against each model's source/config in
# mlx-audio 0.4.4; declaring a language the model does not accept would let the
# standard layer green-light a selection the engine cannot honor, so these lists
# are the honest, source-checked supported sets (subsets where noted).
# --------------------------------------------------------------------------- #

#: SenseVoice's selectable languages: ``"auto"`` + the five it accepts and can
#: also *detect* and report back (``config`` language-id map). Verified against
#: ``sensevoice.py`` (lid map: zh/en/yue/ja/ko).
SENSEVOICE_LANGUAGES: list[str] = ["auto", "zh", "en", "yue", "ja", "ko"]
#: SenseVoice reports a detected language (argmax over the language-id logits);
#: these are the codes it can emit (everything but ``"auto"``).
SENSEVOICE_DETECTABLE_LANGUAGES: list[str] = ["zh", "en", "yue", "ja", "ko"]

#: Cohere ASR validates ``language`` against this fixed set (no auto-detect
#: directive — you MUST pick one). Verified against ``cohere_asr/config.py``.
COHERE_LANGUAGES: list[str] = [
    "en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko",
]  # fmt: skip

#: Fun-ASR-Nano accepts these ISO codes (mapped to Chinese prompt names
#: internally) and also auto-detects when language is omitted. A representative
#: subset of its supported set (the source also lists Chinese topolect codes
#: gan/hak/hsn/nan/wuu/cjy that we omit from the discovery surface).
FUN_ASR_LANGUAGES: list[str] = ["auto", "zh", "en", "ja", "yue"]
#: Languages Fun-ASR-Nano recognizes in auto mode (its set minus the ``"auto"``
#: directive) — required as ``detectable_languages`` because ``"auto"`` is
#: selectable. (Fun-ASR does not *report* which it picked, so the result's
#: ``detected_language`` stays ``None``; this is the candidate-validation set.)
FUN_ASR_DETECTABLE_LANGUAGES: list[str] = ["zh", "en", "ja", "yue"]

#: Voxtral's supported transcription languages (Mistral Voxtral model card). It
#: takes an ISO code with default ``"en"`` and has no ``"auto"`` directive in its
#: ``generate`` surface, so a concrete default is required.
VOXTRAL_LANGUAGES: list[str] = ["en", "es", "fr", "pt", "hi", "de", "nl", "it"]

#: Canary covers these 25 European languages (its ``config.supported_languages``)
#: as both transcription source and translation target. No ``"auto"`` directive
#: (you specify the source language; default ``"en"``).
CANARY_LANGUAGES: list[str] = list(PARAKEET_V3_LANGUAGES)

#: Granite Speech can translate the transcript into one of these targets via the
#: ``target_language`` provider param (its ``LANGUAGE_CODES`` map). The spoken
#: language itself is not selectable (the model auto-handles the input).
GRANITE_TRANSLATE_LANGUAGES: list[str] = ["en", "fr", "de", "es", "pt", "ja"]


def _primary_subtag(bcp47_tag: str) -> str:
    """Return the lower-cased primary subtag of a BCP-47 tag.

    Args:
        bcp47_tag: A BCP-47 language tag (e.g. ``"en-US"``).

    Returns:
        The lower-cased primary subtag (e.g. ``"en"``).
    """
    return normalize_bcp47(bcp47_tag).split("-", maxsplit=1)[0].lower()


def to_qwen_name(bcp47_tag: str) -> str | None:
    """Translate a BCP-47 tag to the Qwen3-ASR English language name.

    Args:
        bcp47_tag: A resolved BCP-47 tag (never ``"auto"``; callers resolve
            auto-detection to ``language=None`` before reaching the backend).

    Returns:
        The English language name (e.g. ``"Chinese"``), or ``None`` if the tag is
        not in the supported inventory (the caller should then omit the language
        and let the model auto-detect rather than send an invalid name).
    """
    entry = _LANGUAGE_TABLE.get(_primary_subtag(bcp47_tag))
    return entry[1] if entry is not None else None


def to_whisper_code(bcp47_tag: str) -> str | None:
    """Translate a BCP-47 tag to the Whisper ISO-639 language code.

    Args:
        bcp47_tag: A resolved BCP-47 tag (never ``"auto"``).

    Returns:
        The Whisper ISO code (e.g. ``"en"``), or ``None`` if unsupported by our
        table (the caller omits the code and lets Whisper auto-detect).
    """
    entry = _LANGUAGE_TABLE.get(_primary_subtag(bcp47_tag))
    return entry[0] if entry is not None else None


def to_iso_subtag(bcp47_tag: str) -> str:
    """Return the ISO-639 primary subtag of a resolved BCP-47 tag.

    Used by the generic STTOutput backends whose ``generate`` takes an ISO code
    directly (SenseVoice, Cohere, Voxtral, Fun-ASR, Canary). Unlike
    :func:`to_whisper_code` / :func:`to_qwen_name`, this does not consult the
    Qwen/Whisper translation table — it simply strips region/script subtags
    (``"en-US"`` -> ``"en"``), because these models accept the bare ISO code and
    the standard layer has already gated the value against the family's declared
    ``selectable_languages`` (so it is a code the model supports).

    Args:
        bcp47_tag: A resolved BCP-47 tag (never ``"auto"``; callers resolve
            auto-detection to ``None`` before reaching the backend).

    Returns:
        The lower-cased ISO-639 primary subtag (e.g. ``"en"``).
    """
    return _primary_subtag(bcp47_tag)


def from_backend_language(value: str | None) -> str | None:
    """Translate a backend-reported language into a BCP-47 tag.

    Different backends report the detected language differently: Whisper reports
    an ISO code (``"en"``); Qwen3-ASR reports an English name and, for
    mixed-language audio, the mlx-audio output's ``language`` is a *list* of
    names. The caller passes a single string (the dominant language); we map it
    via the ISO-code table first, then the English-name table. An unrecognized
    value yields ``None`` rather than a fabricated tag —
    ``validate_detected_language`` rejects ``"auto"`` and non-BCP-47 strings, so
    honesty beats guessing.

    Args:
        value: The language string from the backend response, or ``None``.

    Returns:
        A BCP-47 primary subtag, or ``None`` if it cannot be mapped.
    """
    if not value:
        return None
    first = value.split(",", maxsplit=1)[0].strip()
    if not first:
        return None
    lowered = first.lower()
    return _CODE_TO_BCP47.get(lowered) or _NAME_TO_BCP47.get(lowered)


__all__ = [
    "CANARY_LANGUAGES",
    "COHERE_LANGUAGES",
    "FUN_ASR_DETECTABLE_LANGUAGES",
    "FUN_ASR_LANGUAGES",
    "GRANITE_TRANSLATE_LANGUAGES",
    "PARAKEET_V3_LANGUAGES",
    "QWEN_DETECTABLE_LANGUAGES",
    "QWEN_SELECTABLE_LANGUAGES",
    "SENSEVOICE_DETECTABLE_LANGUAGES",
    "SENSEVOICE_LANGUAGES",
    "VOXTRAL_LANGUAGES",
    "WHISPER_DETECTABLE_LANGUAGES",
    "WHISPER_SELECTABLE_LANGUAGES",
    "from_backend_language",
    "to_iso_subtag",
    "to_qwen_name",
    "to_whisper_code",
]
