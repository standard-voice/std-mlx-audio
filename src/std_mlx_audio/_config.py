# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Init config and provider-params models for the MLX ASR engine.

Two pydantic models:

* :class:`MlxAudioConfig` — init configuration (spec IC.1). The model is selected
  by the entry-point preset (spec IC.7), never a field here. Standard
  "relevant-only" axes (download root) come from the standard mixin so the
  auto-UI renders them; engine-specific init knobs (``dtype``, ``quantization``
  hint) are declared directly. There is no device field: MLX always runs on the
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
from standard_asr import (
    BaseConfig,
    DownloadConfigMixin,
    LanguageConfigMixin,
    secret_field,
)
from standard_asr.runtime_params import ProviderParams

#: MLX compute dtypes we let the loader request. ``"auto"`` keeps the dtype
#: baked into the (often pre-quantized) checkpoint — the right default, since the
#: mlx-community repos ship already quantized (4bit/8bit) and re-casting them is
#: usually wrong. ``float16`` / ``bfloat16`` / ``float32`` force a cast.
MlxDtype = Literal["auto", "float16", "bfloat16", "float32"]


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
        dtype: MLX compute dtype. ``"auto"`` (default) keeps the checkpoint's
            baked-in dtype (correct for the pre-quantized mlx-community repos);
            the others force a cast.
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
        description=(
            "Optional local MLX checkpoint directory overriding the preset's "
            "model (spec IC.7 weights/path). The model is selected by the "
            "entry-point preset, not by this field; None loads the preset's repo."
        ),
    )
    dtype: MlxDtype = Field(
        default="auto",
        description=(
            "MLX compute dtype. 'auto' keeps the checkpoint's baked-in dtype "
            "(correct for pre-quantized repos); others force a cast."
        ),
    )
    local_files_only: bool = Field(default=False, description="Disable downloads when True.")
    revision: str | None = Field(default=None, description="Optional HF model revision.")
    hf_token: SecretStr | None = secret_field(
        description="Hugging Face access token for gated/private model repos (secret)."
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
        system_prompt: Optional system prompt for the Qwen3-ASR decoder (e.g. to
            bias domain/formatting). Distinct from the portable ``prompt`` (which
            maps to Whisper's ``initial_prompt``); kept here because it is a
            Qwen-specific chat-template slot with no portable equivalent.
        chunk_duration: Max seconds of audio per decode chunk for the chunking
            backends (Qwen3-ASR default 1200s = 20 min; Whisper/Parakeet have
            their own internal windowing). Long files are split and concatenated.
    """

    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    repetition_context_size: int = Field(default=100, ge=1)
    max_tokens: int = Field(default=8192, ge=1)
    system_prompt: str | None = None
    chunk_duration: float = Field(default=1200.0, gt=0.0)


__all__ = ["MlxAudioConfig", "MlxAudioParams", "MlxDtype"]
