# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Init config and provider-params models for the MLX ASR engine.

Two pydantic models:

* :class:`MlxAudioConfig` — init configuration (spec IC.1). The model is selected
  by the entry-point preset (spec IC.7), never a field here. Standard
  "relevant-only" axes (download root) come from the standard mixin so the
  auto-UI renders them; engine-specific init knobs (``revision``, the
  streaming window) are declared directly. There is no device field: MLX always runs on the
  Apple-Silicon GPU/Metal (no CPU/GPU choice to expose), so declaring a
  ``DeviceConfigMixin`` would advertise a knob that does not exist (spec IC.5 —
  a field present means it applies).
* :class:`MlxAudioParams` — per-request decoding knobs that are MLX-native and
  NOT in the portable standard set (sampling ``temperature`` / ``top_p`` /
  ``top_k``, ``repetition_penalty``, ``max_tokens``, ``system_prompt``). They
  live in a :class:`ProviderParams` subclass (spec RT §3.2); passing them to a
  different engine raises ``InvalidProviderParamError``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from standard_asr.contract.params import ProviderParams
from standard_asr.engine import (
    BaseConfig,
    DownloadConfigMixin,
    LanguageConfigMixin,
    secret_field,
)


class MlxAudioConfig(
    DownloadConfigMixin,
    LanguageConfigMixin,
    BaseConfig[Literal["mlx-audio"]],
):
    """Init configuration for the MLX ASR engine.

    The model is selected by the entry-point preset (spec IC.7), NOT by a field
    here. ``model_path`` is only an optional *local checkpoint override* (spec
    IC.7 weights/path): point it at a local MLX model directory to load your own
    converted weights instead of the preset's Hub repo. ``None`` (default) loads
    the preset.

    Standard axes via mixins (field present => applicable, spec IC.5):

    * ``download_root`` (:class:`DownloadConfigMixin`) — cache directory; the
      lazy loader resolves it against the spec IC.9 precedence.
    * ``default_language`` / ``default_candidate_languages``
      (:class:`LanguageConfigMixin`) — the language axis (spec LANG R1 requires
      ``default_language`` because the engine exposes ``selectable_languages``).

    Note there is deliberately **no** ``device`` field: MLX runs on Apple Silicon
    Metal unconditionally, so there is no CPU/GPU axis to expose (advertising one
    would be a phantom knob; spec IC.5).

    Args:
        engine: Discriminator value (entry-point-derived; never hand-written).
        model_path: Optional LOCAL MLX checkpoint directory overriding the
            preset's model (spec IC.7 weights/path). The model is chosen by the
            preset, not by this field; ``None`` loads the preset's Hub repo.
        local_files_only: Never download; require a cached/local model.
        revision: Optional Hugging Face model revision (branch/tag/commit).
        hf_token: Optional Hugging Face access token for gated/private model
            repositories. Secret (masked in repr / dumps / ``/v1/models``).
    """

    engine: Literal["mlx-audio"] = "mlx-audio"

    # The language axis default. The backends auto-detect on `None`/`"auto"`; we
    # default to "auto" so a zero-config engine just works (spec LANG R1).
    default_language: str | None = Field(
        default="auto", description="Default language (BCP-47) or 'auto' for detection."
    )

    model_path: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional local MLX checkpoint directory overriding the preset's "
            "model (spec IC.7 weights/path). The model is selected by the "
            "entry-point preset, not by this field; None loads the preset's repo."
        ),
    )
    local_files_only: bool = Field(default=False, description="Disable downloads when True.")
    revision: str | None = Field(default=None, description="Optional HF model revision.")
    hf_token: SecretStr | None = secret_field(
        description="Hugging Face access token for gated/private model repos (secret)."
    )

    # --- Windowed-streaming knobs (see ``_streaming.py``) --------------------- #
    # These tune the re-decode-the-window streaming strategy. The MLX backends are
    # batch decoders, so "streaming" is synthesized by re-running the model over a
    # window of not-yet-committed audio; these control its latency / compute /
    # stability trade-off.
    redecode_interval_s: float = Field(
        default=1.5,
        gt=0.0,
        description=(
            "Streaming: minimum seconds of new audio between two re-decodes of the "
            "window. A lower bound, not a fixed cadence: each decode takes all the "
            "audio that has arrived, so after a slow decode the next one covers more "
            "than this; less audio is decoded once it has waited this long. This is "
            "the one setting for how often the window is decoded (there is no rest "
            "between passes). Smaller = snappier partials but more compute. The "
            "session keeps up only while its sustained end-to-end capacity (every "
            "decode, counting the window decoded again on every pass, plus settling "
            "and other work on the event loop) beats the rate at which audio arrives."
        ),
    )
    settle_margin_s: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Streaming: a segment is finalized (and its audio dropped from the "
            "window) once it ends at least this many seconds before the end of the "
            "decoded audio. Smaller = text settles sooner but is likelier to still "
            "change. Qwen3-ASR returns one segment per chunk_duration (1200 s by "
            "default), which in a shorter window never ends before the window does, "
            "so there this has no effect, except that 0 finalizes the whole window "
            "on every decode; such a model's commits come from the window cap, "
            "pauses (commit_pause_s), and the end of the input."
        ),
    )
    max_window_s: float | None = Field(
        default=30.0,
        gt=0.0,
        description=(
            "Streaming: cap on the window length in seconds; no decode covers more "
            "than this, and it also bounds the audio waiting to be decoded (the cap "
            "plus the one chunk that crossed it; the float32 window, at most twice "
            "the cap, and the byte copy a decode pass takes out of that audio are "
            "separate buffers; these bound the session's buffers, not "
            "the process's memory). When the "
            "window reaches it, bounded heads are committed until less than the cap "
            "remains: at the decoder's segment boundaries for a model that returns "
            "them inside a window this long, otherwise in the longest pause of the "
            "last ten seconds before the cap (at the least loud point if there is "
            "none). Must be at least 0.02 s. None disables the cap: the window then "
            "grows without limit until a settled segment, a pause, or the end of "
            "the input commits it (waiting audio stays bounded at 30 s plus one "
            "chunk)."
        ),
    )
    commit_pause_s: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Streaming: when set, a pause of at least this many seconds after speech "
            "commits the audio before it right away (decoded once and finalized), "
            "without waiting for the window cap. Only for a model that returns no "
            "segment boundaries inside the window (Qwen3-ASR at its default "
            "chunk_duration). Keeps decodes short and delivers a final soon after "
            "the speaker pauses, at the cost of more seams. None commits at settled "
            "segments, the window cap, and the end of the input."
        ),
    )


class MlxAudioParams(ProviderParams):
    """Engine-specific decoding knobs for the MLX backends (non-portable).

    These map onto the ``model.generate(...)`` arguments shared by the MLX
    *generative* STT backends (Qwen3-ASR is an audio-conditioned LLM, so it has a
    full sampler). Backends that do not expose a given knob ignore it — e.g.
    Whisper has its own temperature-fallback schedule and Parakeet is a
    non-autoregressive decoder with no sampler, so for those backends only the
    fields they understand are forwarded (the adapter maps per backend; see
    ``backends.py``). Setting any of these locks the request to this engine:
    handing this object to another engine raises ``InvalidProviderParamError``
    (spec RT §3.2, swap-safety via exact-type match — so this class MUST stay a
    distinct terminal type).

    Args:
        temperature: Sampling temperature (0.0 = greedy/deterministic, the
            default — important for reproducible ASR). Honored by the Qwen3-ASR
            backend; Whisper takes it as the first step of its fallback schedule.
        top_p: Nucleus-sampling probability mass (Qwen3-ASR). Only meaningful
            when ``temperature > 0``.
        top_k: Top-k sampling cutoff (Qwen3-ASR; ``0`` = disabled).
        repetition_penalty: Penalty (>1) on recently generated tokens to curb
            looping (Qwen3-ASR). ``None`` disables the logits processor.
        repetition_context_size: How many recent tokens the repetition penalty
            considers (Qwen3-ASR).
        max_tokens: Hard cap on generated tokens for the autoregressive backends
            (Qwen3-ASR). Guards against runaway decoding on long/degenerate
            audio.
        system_prompt: The system turn of the Qwen3-ASR chat template, verbatim.
            The portable ``prompt`` fills the same slot (it's Qwen3-ASR's
            context); when both are given, this one wins.
        chunk_duration: Max seconds of audio per decode chunk for the chunking
            backends (Qwen3-ASR default 1200s = 20 min; Whisper/Parakeet have
            their own internal windowing). Long files are split and concatenated.
        hotwords: Domain/biasing terms to favor during decoding. Honored by the
            Fun-ASR backend (its ``hotwords`` argument); ignored by backends
            without a hotword channel. ``None`` leaves the model's default.
        use_itn: Toggle inverse text normalization (spelling out vs. rendering
            numerals/dates/punctuation as digits/symbols). Honored by the
            SenseVoice and Fun-ASR backends; ``None`` keeps each model's default.
        target_language: Translation target as a BCP-47 tag. When set, the
            speech-translation backends (Canary, Granite Speech) emit the
            transcript in this language instead of the spoken one; ``None``
            transcribes in the spoken language. Ignored by transcription-only
            backends. The detected/source language is still reported in
            ``result.detected_language``.
        beam_size: Beam width for the beam-search backends (FireRedASR2).
            ``None`` keeps the model default; ignored by greedy/sampling
            backends.
        context: Free-text context/biasing prompt for the VibeVoice backend
            (domain hints, expected vocabulary). ``None`` leaves it unset;
            ignored by backends without a context channel. Distinct from the
            Qwen-specific ``system_prompt`` and the portable Whisper ``prompt``.
    """

    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    repetition_context_size: int = Field(default=100, ge=1)
    max_tokens: int = Field(default=8192, ge=1)
    system_prompt: str | None = None
    chunk_duration: float = Field(default=1200.0, gt=0.0)
    hotwords: list[str] | None = None
    use_itn: bool | None = None
    target_language: str | None = None
    beam_size: int | None = Field(default=None, ge=1)
    context: str | None = None


__all__ = ["MlxAudioConfig", "MlxAudioParams"]
