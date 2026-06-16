<!--
SPDX-FileCopyrightText: 2026 Standard Voice Contributors
SPDX-License-Identifier: Apache-2.0
-->

# std-mlx-audio

> ⚠️ **Experimental — for protocol testing.** This is an experimental Standard ASR engine plugin, published to exercise and validate the [Standard ASR](https://github.com/standard-voice/standard_asr) interface. Expect breaking changes; it is not production-ready.

**A [Standard ASR](https://github.com/standard-voice/standard_asr) engine plugin
for Apple-Silicon-native (MLX) speech-to-text — one engine, many models,
headlined by Qwen3-ASR.**

`std-mlx-audio` adapts the upstream [`mlx-audio`](https://github.com/Blaizzy/mlx-audio)
backend so any Standard ASR application can run **every MLX speech-to-text model
family** on a Mac with no per-engine integration work — Qwen3-ASR, Whisper,
Parakeet, Nemotron, SenseVoice, Voxtral, Canary, GLM-ASR, Granite Speech,
Fun-ASR, VibeVoice, Moonshine, MMS, FireRedASR2, and Qwen2-Audio. Install it, and
every Standard ASR app, the CLI, and the web server can use these models
immediately.

> **Apple Silicon only.** MLX requires an arm64 Mac with Metal. The wheel
> installs anywhere, but inference runs only on a supported Mac.

## Models

All models live under one engine (`engine_id = "mlx-audio"`), each as its own
entry-point key. **Timestamps** = word (W) / segment (S) / none (—);
**Streaming** = windowed re-decode where the model emits real timing, else batch
only.

| Model key | HF repo | Timestamps | Stream | Notes |
| --- | --- | --- | --- | --- |
| `mlx-audio/qwen3-asr-0.6b` | `mlx-community/Qwen3-ASR-0.6B-4bit` | S | ✓ | **Headliner.** 30-language; fast. Smallest Qwen3-ASR. |
| `mlx-audio/qwen3-asr-1.7b` | `mlx-community/Qwen3-ASR-1.7B-8bit` | S | ✓ | Higher-accuracy Qwen3-ASR (~3.4 GB). |
| `mlx-audio/whisper-large-v3-turbo` | `openai/whisper-large-v3-turbo` | W·S | ✓ | Fast multilingual Whisper; word timestamps + prompt. |
| `mlx-audio/whisper-tiny` | `openai/whisper-tiny` | W·S | ✓ | Smallest Whisper; smoke/tests. |
| `mlx-audio/parakeet-tdt-0.6b-v3` | `mlx-community/parakeet-tdt-0.6b-v3` | W·S | ✓ | 25 EU languages; precise word/sentence timestamps (weights **CC-BY-4.0**). |
| `mlx-audio/nemotron-asr-streaming-0.6b` | `mlx-community/nemotron-3.5-asr-streaming-0.6b` | W·S | ✓ | NVIDIA Nemotron; English; word timestamps. |
| `mlx-audio/sensevoice-small` | `mlx-community/SenseVoiceSmall` | — | — | zh/en/yue/ja/ko; language detection + ITN. |
| `mlx-audio/cohere-asr` | `appautomaton/cohere-asr-mlx` | S | ✓ | 14-language; VAD segment timing. ⚠ public weights in a repo subfolder — may need a local `model_path` (see VERIFICATION.md). |
| `mlx-audio/fun-asr-nano` | `mlx-community/Fun-ASR-Nano-2512` | S | ✓ | zh/en/ja; hotwords + ITN; per-chunk timing. |
| `mlx-audio/glm-asr-nano` | `mlx-community/GLM-ASR-Nano-2512-4bit` | S | ✓ | Compact ASR; per-chunk segment timing. |
| `mlx-audio/canary-1b-v2` | `TechHara/canary-1b-v2-mlx-q4` | — | — | 25 EU languages; speech translation (`target_language`). |
| `mlx-audio/granite-speech-1b` | `mlx-community/granite-4.0-1b-speech-5bit` | — | — | IBM Granite; ASR + translation (`target_language`). |
| `mlx-audio/granite-speech-nar-2b` | `mlx-community/granite-speech-4.1-2b-nar-mlx` | — | — | Fast non-autoregressive ASR. |
| `mlx-audio/voxtral-mini-3b` | `mlx-community/Voxtral-Mini-3B-2507-bf16` | — | — | Mistral Voxtral; multilingual. Large (~9 GB); decoded-array input only. |
| `mlx-audio/voxtral-realtime-4b` | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | — | — | English. Batch (native streaming not yet wired). |
| `mlx-audio/qwen2-audio-7b` | `mlx-community/Qwen2-Audio-7B-Instruct-4bit` | — | — | Qwen2-Audio audio-LLM transcription. |
| `mlx-audio/vibevoice-asr` | `mlx-community/VibeVoice-ASR-4bit` | — | — | Microsoft VibeVoice; context-biased (`context`). |
| `mlx-audio/moonshine-tiny` | `UsefulSensors/moonshine-tiny` | — | — | Tiny fast English ASR (~27 M). |
| `mlx-audio/fireredasr2-aed` | `mlx-community/FireRedASR2-AED-mlx` | — | — | Chinese/English; beam search (`beam_size`). |
| `mlx-audio/mms-1b-all` | `facebook/mms-1b-all` | — | — | Meta MMS multilingual CTC. ⚠ ~29 GB; `model_type` pinned to `mms`. |

Each model declares its **own** honest capabilities (word vs. segment timestamps,
whether language is runtime-selectable, whether it detects/reports a language,
whether it streams). A model that produces no real timing returns text only and
declares no timestamps rather than fabricating spans, and a model with no
language axis declares an empty `selectable_languages`. Query any model with
`standard-asr show <key>` — no instantiation, no download.

> **Model selection is by preset.** Each key binds a fixed HF repo to the right
> backend; the loader auto-detects the family and the engine **fails loudly** if
> a `model_path` override resolves to a family the preset's backend cannot run
> (never a silent wrong transcript). To run another repo of a supported family,
> point `model_path` at it on the matching preset.

## Install

```bash
pip install std-mlx-audio          # on an Apple-Silicon Mac
# or, in a uv project:
uv add std-mlx-audio
```

This pulls `mlx-audio[stt]` (which pulls `mlx`, `mlx-lm`, `transformers`) and
`standard-asr`. Model weights download from the Hugging Face Hub on first use
(set `STANDARD_ASR_ALLOW_DOWNLOAD=1` if your environment disables downloads).

## Use

### CLI (no code)

```bash
standard-asr list                                 # see all 20 models
standard-asr show mlx-audio/qwen3-asr-0.6b        # capabilities + params schema
standard-asr transcribe mlx-audio/qwen3-asr-0.6b path/to/audio.wav
```

### Python

```python
from standard_asr import RuntimeParams, discover_models

engine = discover_models().create("mlx-audio/qwen3-asr-0.6b")
result = engine.transcribe("meeting.m4a", RuntimeParams(language="en"))
print(result.text)

# Switch models with one string — same code, same result schema:
parakeet = discover_models().create("mlx-audio/parakeet-tdt-0.6b-v3")
words = parakeet.transcribe("meeting.m4a", RuntimeParams(word_timestamps="word")).words
```

### Streaming (windowed)

```python
import asyncio
from standard_asr import RuntimeParams, discover_models
from standard_asr.audio_format import AudioFormat

async def main() -> None:
    engine = discover_models().create("mlx-audio/qwen3-asr-0.6b")
    fmt = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)
    async with engine.start_transcription(audio_format=fmt, params=RuntimeParams()) as session:
        session.feed(pcm_chunks)            # iterable of 16 kHz mono pcm_s16le bytes
        async for event in session:
            if event.type in ("partial", "final"):
                print(event.type, event.text)

asyncio.run(main())
```

> **Streaming is a windowed re-decode**, not a native low-latency recognizer
> (none of these MLX backends expose incremental decoding). Capabilities are
> declared accordingly: partials may be rewritten (`stable_until=0`), no
> re-segmentation, no reconnect. See `docs/DESIGN.md`.

## Configuration

Init config (`MlxAudioConfig`) — set via `create(...)` kwargs or
`STANDARD_ASR_MLX_AUDIO__<FIELD>` env vars:

| Field | Default | Meaning |
| --- | --- | --- |
| `default_language` | `"auto"` (`"en"` for Parakeet) | BCP-47 tag or `"auto"`. |
| `dtype` | `"auto"` | `auto` keeps the checkpoint dtype (best for pre-quantized repos); else `float16`/`bfloat16`/`float32`. |
| `model_path` | `None` | Local MLX checkpoint dir overriding the preset's repo. |
| `local_files_only` | `False` | Never download; require cached weights. |
| `revision` | `None` | HF revision (branch/tag/commit). |
| `hf_token` | `None` | HF token for gated repos (secret, masked everywhere). |

There is no `device` field — MLX runs on Metal unconditionally.

Per-request decode knobs (`MlxAudioParams`, the engine's provider params):
`temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_tokens`,
`system_prompt` (Qwen3-ASR), `chunk_duration`, plus per-family knobs `hotwords`
(Fun-ASR), `use_itn` (SenseVoice/Fun-ASR), `target_language` (speech translation
on Canary/Granite Speech), `beam_size` (FireRedASR2), and `context` (VibeVoice).
Each backend honors only the subset it supports; the rest are ignored.

## Adding a model

Every preset is a few lines. Pick any STT repo mlx-audio supports, bind it to a
backend, and add an entry point:

```python
# engine.py
class Qwen3Asr2B(MlxAudioASR):
    hf_repo = "mlx-community/Qwen3-ASR-2B-8bit"
    backend = Qwen3AsrBackend()
    properties = Qwen3Asr2BProperties()         # model_name must match the key
    declared_capabilities = _QWEN_CAPABILITIES
```

```toml
# pyproject.toml
[project.entry-points."standard_asr.models"]
"mlx-audio/qwen3-asr-2b" = "std_mlx_audio.entrypoint:create_qwen3_asr_2b"
```

Most new families return the shared `STTOutput` shape, so they need only a new
`SttFamilySpec` (declarative data — language axis, timing honesty, decode knobs)
bound to the generic `GenericSttBackend` — no new backend code. A genuinely new
return type (like Parakeet/Nemotron's `AlignedResult`) is a new `ModelBackend`
(one `generate_kwargs` + `to_result`) — no change to the engine. See
`docs/DESIGN.md`.

## Development

```bash
uv sync
uv run pytest                 # 102 tests, 100% coverage (mocks MLX; no downloads)
uv run ruff check src/ tests/
uv run pyright src/           # strict
uv run standard-asr compliance run mlx-audio/qwen3-asr-0.6b
# Real inference (Apple Silicon; downloads weights on first run):
uv run python scripts/verify_inference.py path/to/audio.m4a mlx-audio/qwen3-asr-0.6b
```

See `VERIFICATION.md` for verified real-inference results and
`docs/STANDARD_ASR_FINDINGS.md` for protocol findings.

## Licensing

This plugin is **Apache-2.0**. It does not vendor upstream code — `mlx-audio`
(MIT), `mlx` (MIT), and the model weights are ordinary dependencies under their
own terms. **Model weight licenses differ:** Qwen3-ASR **Apache-2.0**, Whisper
**MIT**, Parakeet **CC-BY-4.0** (attribution required). See
`LICENSE-THIRD-PARTY.md`.
