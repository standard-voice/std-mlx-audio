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
**Stream** = windowed re-decode streaming (see [Streaming](#streaming-windowed)),
else batch only.

| Model key | HF repo | Timestamps | Stream | Notes |
| --- | --- | --- | --- | --- |
| `mlx-audio/qwen3-asr-0.6b` | `mlx-community/Qwen3-ASR-0.6B-4bit` | S¹ | ✓ | **Headliner.** 30-language; fast. Smallest Qwen3-ASR. |
| `mlx-audio/qwen3-asr-1.7b` | `mlx-community/Qwen3-ASR-1.7B-8bit` | S¹ | ✓ | Higher-accuracy Qwen3-ASR (~3.4 GB). |
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

¹ Qwen3-ASR returns one segment per decoded chunk, and a chunk is the whole
input up to 20 minutes (`chunk_duration`). Its segment timestamps are therefore
the bounds of the input, not of sentences or phrases. (mlx-audio pads input
shorter than one second and reports the padded length; the plugin clamps
streaming timestamps to the audio actually decoded.) In streaming, each `final`
spans the audio committed at one pause or window cut.

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

> **Not yet published to PyPI** — install from GitHub. Apple Silicon (arm64 +
> Metal) is required for inference.

```bash
uv pip install git+https://github.com/standard-voice/std-mlx-audio
```

This pulls `mlx-audio[stt]` (which pulls `mlx`, `mlx-lm`, `transformers`) and
`standard-asr` (from GitHub `main`). Model weights download from the Hugging Face
Hub on first use (set `STANDARD_ASR_ALLOW_DOWNLOAD=1` if your environment disables
downloads). Once published to PyPI this becomes `uv pip install std-mlx-audio`.

**Artifact lifecycle.** `standard-asr status mlx-audio/<model>`
reports whether the preset's snapshot is cached and provably complete (a
sharded checkpoint is ready only when every shard is present -- named by the
safetensors index, or by the shards' own `-NNNNN-of-NNNNN` names when the
index has not arrived yet). Non-weight files enter the ready closure on two
verified axes. Files a loader reads by one fixed name and silently corrupts
without (the MMS base weights and CTC vocab, the SenseVoice bpe and
normalization stats, the FireRed dict and cmvn, the Moonshine tokenizer,
the Cohere tokenizer config) gate every checkpoint, an operator
`model_path` included. Files whose absence provably breaks inference for
the preset's own repo layout (the Whisper and Qwen3 processor configs, the
GLM, Granite NAR, and tekken tokenizers, the Canary SentencePiece model)
gate only the Hub snapshot: the flexible upstream loaders accept
alternative local layouts an exact-name check would wrongly reject. Each
declared file is verified as a single point of failure against the
installed loader, by per-file ablation. VibeVoice's tokenizer lives in a
separate Hub repo its upstream loader fetches at load time; the preset
models it as a companion, so status requires it cached, `pull` acquires
it, and a load under a no-download policy refuses instead of letting the
fetch bypass the policy. `standard-asr pull` acquires or repairs the
snapshot without loading or priming a model.
`pull --refresh` re-resolves a mutable revision and verifies against the
source that the re-resolution happened: the downloader alone silently falls
back to the local cache when the source is unreachable, and a refresh fails
rather than report the stale cache as fresh (a pinned 40-hex commit is
immutable and a no-op). The implicit first-use load re-verifies completeness
after its online resolution for the same reason, so the non-strict loader
never receives a fragment. With `local_files_only=true` the engine refuses
every network transfer, including a refresh.

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
from standard_asr.audio.format import AudioFormat

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
> (this plugin decodes every family through its batch `generate`). The session
> keeps a window of the audio it has not committed yet. Once at least
> `redecode_interval_s` of new audio has arrived (or less, once it has waited
> that long), it decodes the whole window again, taking all the audio that has
> arrived, so a decode that took long is followed by one that covers the
> backlog. It keeps up only while its sustained end-to-end capacity (every
> decode, counting that each pass decodes the whole window again, plus settling,
> conversion and other work on the event loop) beats
> the rate at which audio arrives; otherwise it falls behind. Audio waiting to
> be decoded is bounded at one `max_window_s` of audio plus the chunk that
> crossed it, so a client that sends faster than that is held back by the
> library's queue; this bounds the session's own buffers, not the process's
> memory.
> A `partial` may be rewritten until its audio is committed (`stable_until=0`).
> Audio is committed (emitted as `final` and dropped from the window) at a
> settled segment boundary, for models that return segment boundaries (such as
> Whisper and Parakeet). Qwen3-ASR returns one segment per `chunk_duration`
> (1200 s by default), so within a window it normally has none: its audio is
> committed at a pause of at least `commit_pause_s` (when set) or, when the
> window reaches `max_window_s`, in the longest pause of the ten seconds before
> the cap; the audio before that point is decoded once more on its own. The end
> of the input commits the rest. No decode covers more than `max_window_s`.
> With a cap, a session working through a backlog (it fell behind, or input was
> queued all at once: a whole file through `audio=`, or a client pushing faster
> than real time) commits bounded heads and emits `final` events and no
> `partial` until it has caught up; with `max_window_s=None` there are no cap heads
> (a pause commit still commits a head, but only a cap commit starts this
> backlog mode), and queued input still produces partials. The library's
> `max_idle` deadline is a content deadline, not silence detection: an ordinary
> decode before the end of the input emits a `partial` whenever its unsettled
> tail has text, even unchanged text, and every `final` counts as content too
> (with `settle_margin_s=0` a pass can emit `final` events and no `partial`), so
> a decoder that keeps returning earlier text over silence keeps the session
> alive; detecting the end of an utterance (endpointing) is the application's
> job. `max_idle` measures time without a delivered content event. Enforcement
> is cooperative: inline decoding blocks the event loop, and queueing and task
> scheduling can delay termination by several decodes. Neither the configured
> duration nor that duration plus one decode is a guaranteed termination bound.
> Every commit is a seam: the next
> window is decoded without the earlier text, so a word next to a seam can still
> be lost or doubled, and a cut in a mid-sentence pause can end the text with a
> period. Pauses are found from the audio's energy, which works on clean
> recordings and is untested on noisy microphones and human pauses. Capabilities
> are declared accordingly: no re-segmentation, no reconnect. The decode runs on
> the event-loop thread (MLX arrays are bound to the thread that created them),
> so it blocks that loop while it runs. See `docs/DESIGN.md`.

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
| `redecode_interval_s` | `1.5` | Streaming: minimum seconds of new audio between two decodes of the window, and the one setting for how often the window is decoded (there is no rest between passes). A lower bound: each decode takes all the audio that has arrived. Less audio is decoded once it has waited this long. |
| `settle_margin_s` | `2.0` | Streaming: finalize a segment once it ends this many seconds before the end of the decoded audio. For Qwen3-ASR, whose one segment spans each `chunk_duration` (1200 s by default), this has no effect in a shorter window, except that `0` finalizes the whole window on every decode. |
| `max_window_s` | `30.0` | Streaming: window cap; no decode covers more, and it also bounds the audio waiting to be decoded (the cap plus one chunk; the float32 window, at most twice the cap, and the byte copy a decode pass takes out of that audio are separate buffers). Heads are committed at segment boundaries for models that return them inside a window this long; otherwise in the longest pause of the last ten seconds before the cap (the least loud point if there is none). At least `0.02`. `None` disables the cap: the window then grows without limit until a settled segment, a pause, or the end of the input commits it. |
| `commit_pause_s` | `None` | Streaming: commit the audio before a pause of at least this many seconds after speech, without waiting for the cap. Only for models without segment boundaries inside the window (Qwen3-ASR at its default `chunk_duration`). `None` commits at settled segments, the cap, and the end of the input; it is the default because pause commits multiplied the false sentence breaks on synthetic speech (see `docs/DESIGN.md` §4). |

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
(one `generate_kwargs` + `to_result` + `single_segment_span`) — no change to the
engine. See `docs/DESIGN.md`.

> **Migration note for existing `ModelBackend` implementations.** The protocol
> now requires `single_segment_span(params)`. Return `None` if the family splits
> its output at what it hears (sentences, pauses), whatever the input's length.
> Return the chunk length in seconds if the family returns one segment spanning
> its whole input up to some length (as Qwen3-ASR does up to `chunk_duration`).
> The streaming session uses it to decide, before any decode, whether a window
> can hold a segment boundary, and so where to commit audio; a wrong answer
> costs a wasted decode at the cap or turns pause commits off. A
> `SttFamilySpec` sets it through `single_segment_span_s` (default `None`).

## Development

```bash
uv sync
uv run pytest                 # mocks MLX; no downloads
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
