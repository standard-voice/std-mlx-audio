# Design — std-mlx-audio

## 1. What this plugin is

`std-mlx-audio` is a Standard ASR engine plugin that exposes **multiple
Apple-Silicon-native (MLX) speech-to-text models under one engine**
(`engine_id = "mlx-audio"`), headlined by **Qwen3-ASR**. It adapts the upstream
[`mlx-audio`](https://github.com/Blaizzy/mlx-audio) package (MIT) so that any
Standard ASR application can use Qwen3-ASR, Parakeet, or Whisper on a Mac with no
per-engine integration work.

Registered model keys (`mlx-audio/<model>`):

| Key | Family | Repo | Role |
| --- | --- | --- | --- |
| `mlx-audio/qwen3-asr-0.6b` | Qwen3-ASR | `mlx-community/Qwen3-ASR-0.6B-4bit` | Headliner; fast |
| `mlx-audio/qwen3-asr-1.7b` | Qwen3-ASR | `mlx-community/Qwen3-ASR-1.7B-8bit` | Headliner; higher accuracy |
| `mlx-audio/parakeet-tdt-0.6b-v3` | Parakeet | `mlx-community/parakeet-tdt-0.6b-v3` | Word/sentence timestamps |
| `mlx-audio/whisper-large-v3-turbo` | Whisper | `openai/whisper-large-v3-turbo` | Fast multilingual Whisper |
| `mlx-audio/whisper-tiny` | Whisper | `openai/whisper-tiny` | Smallest; smoke/tests |

## 2. Adapter vs. vendor/fork — decision: **thin adapter**

The project owner explicitly allowed vendoring/forking upstream (mlx-audio is
MIT). We chose **a thin adapter** anyway, because the cost/benefit clearly favors
it here:

- **mlx-audio already is the multi-model layer we'd otherwise build.** A single
  `mlx_audio.stt.load(repo)` auto-detects the architecture and returns a model
  with a uniform `model.generate(...)`. It supports Qwen3-ASR, Whisper, Parakeet
  and ~20 other STT architectures. Vendoring would mean copying and then tracking
  a large, actively-maintained backend (encoders, decoders, tokenizers, Metal
  kernels) for **zero** behavioral gain.
- **The interesting engineering is the normalization layer, not the inference.**
  What Standard ASR needs is honest capability declaration and a constant result
  schema across heterogeneous engines. That is exactly what this plugin adds on
  top of mlx-audio — and it is small, pure, and fully testable without MLX.
- **License isolation is satisfied by dependency, not vendoring.** mlx-audio
  (MIT), mlx (MIT), and the model weights (Apache-2.0 for Qwen3-ASR, MIT for
  Whisper, CC-BY-4.0 for Parakeet) stay in their own packages with their own
  terms (`LICENSE-THIRD-PARTY.md`), which is precisely Standard ASR goal G.4.2.

We do depend on a couple of upstream behaviors that are not part of a stable API
(the exact `generate` kwargs and return shapes per family). Those are pinned
behind the per-family backend adapters (§3) and the `mlx-audio>=0.4.4,<0.5`
version bound, so an upstream change is contained to one small module.

## 3. Multi-model architecture — the "one engine, many models" core

The deliberate test target was: **expose several models under one engine and
report how well the protocol supports a model family.** The challenge is that the
three families are genuinely heterogeneous:

| | Qwen3-ASR | Whisper | Parakeet |
| --- | --- | --- | --- |
| Native return | `STTOutput` | `STTOutput` | `AlignedResult` (different type!) |
| Language arg | English **name** (`"Chinese"`) | ISO **code** (`"ja"`) | **none** (fixed-language) |
| Word timing | no (segment only) | yes (when asked) | yes (always, token-level) |
| Guidance | `system_prompt` (chat slot) | `initial_prompt` | none |
| Sampler | full LLM sampler | temperature schedule | none |
| Runtime language override | yes | yes | **no** |

A faster-whisper-style "all presets share one `_transcribe`" does **not** fit —
the presets are not interchangeable. So the design splits into two layers:

1. **`engine.py` — a family-agnostic engine.** `MlxAudioASR` (an `EngineBase`
   subclass) owns the parts that are identical for every model: pure `__init__`,
   lazy loading via `mlx_audio.stt.load` (with the download policy and
   `local_files_only`), audio negotiation (it receives a negotiated 16 kHz mono
   waveform / path / bytes), language-axis resolution, the batch error contract,
   and the windowed streaming session. Each preset is a tiny subclass that binds
   four class vars: `hf_repo`, `backend`, `properties`, `declared_capabilities`
   (+ optional `default_config_overrides`).

2. **`backends.py` — a per-family `ModelBackend` adapter.** This is where all the
   heterogeneity lives. A `ModelBackend` does two things:
   - `generate_kwargs(...)` — translate the resolved BCP-47 language into the
     family's native surface (name vs. code vs. nothing) and build the
     family-specific `generate` kwargs.
   - `to_result(native, ...)` — normalize the family's native return value
     (`STTOutput` *or* `AlignedResult`) onto the one constant
     `TranscriptionResult` schema, honoring the word-timestamp null semantics
     (`words=None` ⇒ not requested; spec TR.3).

   Three implementations exist: `Qwen3AsrBackend`, `WhisperBackend`,
   `ParakeetBackend`. Adding a fourth family (e.g. Voxtral) is **a new
   `ModelBackend` + an entry point** — no change to the engine, config, or
   pipeline.

3. **`_metadata.py` — per-family Properties + Capabilities.** Each family
   declares its *own* honest, fail-closed capabilities: Qwen3-ASR declares only
   `word_timestamps=["segment"]`; Whisper declares `["word","segment"]` + prompt
   guidance; Parakeet declares `["word","segment"]` but
   `language.runtime_override=false` (a fixed-language model). This is the
   protocol carrying a *family*: same engine_id, different per-model capabilities,
   discovered without instantiation.

4. **Config is one model, with per-preset defaults.** `MlxAudioConfig` is shared;
   a preset can seed defaults via `default_config_overrides` (Parakeet sets
   `default_language="en"` because its `selectable_languages` has no `"auto"`
   directive and LANG R1 requires a valid default — see findings). There is no
   `device` field: MLX runs on Metal unconditionally (a `device` axis would be a
   phantom knob; spec IC.5).

### Provider params

`MlxAudioParams` is the single terminal `ProviderParams` type for the whole
engine. It carries the MLX generation knobs shared by the *generative* backends
(`temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_tokens`,
`system_prompt`, `chunk_duration`). Each backend forwards only the subset it
understands (Parakeet forwards none). Keeping one params type per engine respects
the swap-safety rule (spec RT §3.2: exact-type match), while the per-backend
mapping keeps each family honest.

## 4. Streaming — windowed re-decode (honest)

None of these MLX backends expose a native incremental decoder, so streaming is a
**re-decode-the-window** session (the same approach `std-faster-whisper` uses for
a batch engine): accumulate fed PCM, periodically re-decode the whole window via
the bound backend, finalize sentences that have settled behind the frontier, and
emit the tail as one moving `partial`. The declared streaming capabilities match
exactly: `emits_partials=true`, `word_stability=false` (`stable_until=0`
always), `re_segments=false`, `reconnect=unsupported`, `finality=final`,
`timestamps=post_align`.

**MLX threading constraint:** MLX arrays (incl. model weights) are bound to the
Metal stream of the thread that created them, and a stream cannot be used from
another thread. The model loads on the event-loop thread, so the decode runs
**inline** on that thread (offloading to `asyncio.to_thread` raises
`RuntimeError: There is no Stream(gpu, N) in current thread`). We `await
asyncio.sleep(0)` before each decode so the audio pump makes progress, then
decode inline. This blocks the loop for the decode duration (1–2 s), which is
acceptable for a coarse windowed re-decode and documented honestly.

## 5. Testing strategy

- **Unit suite (no MLX, no downloads):** a fake `mlx_audio.stt.load` returns
  shape-accurate fakes (`STTOutput`-like / `AlignedResult`-like). This exercises
  the real adapter logic (language mapping, output normalization, batch +
  streaming engine paths, config/env, download policy) at 100 % coverage and runs
  on any platform.
- **Real-inference verification (opt-in):** `scripts/verify_inference.py` runs
  actual MLX inference on a Mac over a real file (batch + word timestamps +
  windowed streaming). See `VERIFICATION.md`.
