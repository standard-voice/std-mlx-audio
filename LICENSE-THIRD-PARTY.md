# Third-party licenses (dependency & license isolation)

`std-mlx-audio` is licensed **Apache-2.0** (see `LICENSE`). This file documents
the licenses of the libraries it depends on, so that application developers can
make an informed, license-aware choice when installing this plugin (Standard ASR
goal **G.4.2**: "dependency & license isolation — applications choose plugins per
their license and cost needs, with clear license responsibility boundaries").

This adapter does **not** vendor, copy, or re-license any of the code below — it
declares them as ordinary runtime dependencies and they are installed and
licensed under their own terms.

## Runtime dependencies

| Package | License | Role |
| --- | --- | --- |
| [mlx-audio](https://github.com/Blaizzy/mlx-audio) | **MIT** | The upstream MLX inference backend we adapt (`mlx_audio.stt.load` / `model.generate`). Unifies Qwen3-ASR, Whisper, Parakeet and others. |
| [mlx](https://github.com/ml-explore/mlx) | **MIT** | Apple's array framework / Metal compute runtime (transitive via mlx-audio). Apple-Silicon-only. |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | **MIT** | MLX LLM sampling utilities used by the Qwen3-ASR text decoder (transitive). |
| [transformers](https://github.com/huggingface/transformers) | **Apache-2.0** | Tokenizers / feature extractors for some models (transitive). |
| [tokenizers](https://github.com/huggingface/tokenizers) | **Apache-2.0** | Fast tokenizers (transitive). |
| [sentencepiece](https://github.com/google/sentencepiece) | **Apache-2.0** | Tokenizer for Qwen-family models (the `[stt]` extra). |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | **Apache-2.0** | Model download/cache (transitive). |
| [miniaudio](https://github.com/irmen/pyminiaudio) | **MIT** | Audio file decoding inside mlx-audio (transitive). |
| [scipy](https://scipy.org/) | **BSD-3-Clause** | Resampling inside mlx-audio (transitive). |
| [standard-asr](https://github.com/standard-voice/standard_asr) | **Apache-2.0** | The protocol this plugin implements. |
| [numpy](https://numpy.org/) | **BSD-3-Clause** | Waveform arrays. |
| [pydantic](https://github.com/pydantic/pydantic) | **MIT** | Config / result models (transitive via standard-asr). |

## Model weights

Model weights are downloaded at runtime from the Hugging Face Hub and are
published by their respective authors under their own terms — they are **not**
distributed with this package (they are fetched on first use, subject to the
download policy `STANDARD_ASR_ALLOW_DOWNLOAD`). Consult each model's card for its
exact terms:

| Model (HF repo) | Weights license (per model card) |
| --- | --- |
| `mlx-community/Qwen3-ASR-0.6B-4bit`, `Qwen3-ASR-1.7B-8bit` | **Apache-2.0** (Qwen3-ASR) |
| `mlx-community/parakeet-tdt-0.6b-v3` | **CC-BY-4.0** (NVIDIA Parakeet — attribution required) |
| `mlx-community/whisper-large-v3-turbo`, `whisper-tiny` | **MIT** (OpenAI Whisper) |

> **Note on Parakeet weights (CC-BY-4.0):** the *code* in this plugin and in
> mlx-audio is permissively licensed, but the Parakeet model **weights** are
> CC-BY-4.0, which requires attribution if you redistribute outputs in a context
> where that license applies. The Qwen3-ASR and Whisper weights are Apache-2.0 /
> MIT respectively. Pick the model whose weight license matches your use.

## Note on MLX / Apple Silicon

`mlx` and `mlx-lm` require Apple Silicon (arm64) with Metal. The wheel for this
plugin installs on any platform, but inference only runs on a Metal-capable Mac.
This is a hardware boundary, not a license one.
