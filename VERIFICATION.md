# Verification — std-mlx-audio

This document records the **real, local inference** verification of `std-mlx-audio`
on Apple Silicon, plus the protocol-compliance and test-suite results. Everything
below is reproducible with the exact commands shown.

## Environment

| Item | Value |
| --- | --- |
| Machine | Apple **M5 Max**, `arm64`, macOS 27 (Metal) |
| Python | 3.12.11 (pinned via `.python-version`) |
| `mlx` | 0.31.2 |
| `mlx-audio` | 0.4.4 (`[stt]`) |
| `mlx-lm` | 0.31.3 |
| `transformers` | 5.12.0 |
| `numpy` | 2.4.6 |
| `pydantic` | 2.13.4 |
| `standard-asr` | 0.1.0 (git `main`) |
| Test audio | `standard_asr/reference/standard_asr_test_audio_english.m4a` (~57 s, English); `standard_asr_user/harvard.wav` (Harvard sentences) for the generic-family runs in §4b |

Setup:

```bash
cd std-mlx-audio
uv python pin 3.12
uv sync                      # resolves standard-asr from the branch + mlx-audio[stt]
```

## 1. Discovery — 20 models under one engine (`standard-asr list`)

```bash
uv run standard-asr list
```

All 20 mlx-audio STT families resolve under the single `engine_id = "mlx-audio"`:
the original five (`qwen3-asr-0.6b`, `qwen3-asr-1.7b`, `parakeet-tdt-0.6b-v3`,
`whisper-large-v3-turbo`, `whisper-tiny`) plus `nemotron-asr-streaming-0.6b`,
`sensevoice-small`, `cohere-asr`, `fun-asr-nano`, `voxtral-mini-3b`,
`canary-1b-v2`, `qwen2-audio-7b`, `glm-asr-nano`, `granite-speech-1b`,
`granite-speech-nar-2b`, `vibevoice-asr`, `moonshine-tiny`, `mms-1b-all`,
`fireredasr2-aed`, `voxtral-realtime-4b`. `standard-asr show <key>` prints each
model's capabilities and params schema **without instantiating** the engine.

## 2. Compliance — all 20 models pass

`standard.compliance.check_entrypoints()` validates **every** discovered model
(entry-point metadata, capability declarations, `model_id == key`) and is asserted
green for all 20 in the test suite (`tests/test_entrypoints.py`). The original
five also pass the CLI `standard-asr compliance run mlx-audio/<key>` end to end
(`[OK] Compliance run passed.`); the streaming event-sequence contract is covered
via `standard_asr.compliance.check_event_sequence` (see §5).

## 3. Real transcript — Qwen3-ASR (the headliner)

```bash
STANDARD_ASR_ALLOW_DOWNLOAD=1 \
  uv run python scripts/verify_inference.py \
  ../standard_asr/reference/standard_asr_test_audio_english.m4a \
  mlx-audio/qwen3-asr-0.6b
```

**BATCH** (`mlx-audio/qwen3-asr-0.6b`, via discovery → create → transcribe):

- `detected_language = en`, 1 segment, ~597 chars
- wall time ≈ **1.3 s** (model preloaded) / ≈ 2.1 s including lazy load, for ~57 s of audio
- Transcript (first sentence, verbatim):

  > "This is a crazy interesting test for testing the capabilities and initial prototype of standard ASR package."

  (Full text correctly transcribes the ~57 s clip, including "Faster Whisper and
  QN3 ASR", through to "…now is the time to put the design into test. Complete.")

**STREAMING** (windowed re-decode, 1 s chunks fed live): 11 `partial` → 1 `final`
→ `done`; partials show the text growing and `stable_until = 0` (honest — the
re-decode may rewrite any earlier text). Reduced stream text matches the batch
transcript.

## 4. Multi-model proof — a second (and third) family runs

**Parakeet TDT 0.6B v3** (`mlx-audio/parakeet-tdt-0.6b-v3`) — a *different native
return type* (`AlignedResult`, not `STTOutput`) and the word-timestamp specialist:

```bash
STANDARD_ASR_ALLOW_DOWNLOAD=1 uv run python scripts/verify_inference.py \
  ../standard_asr/reference/standard_asr_test_audio_english.m4a \
  mlx-audio/parakeet-tdt-0.6b-v3
```

- BATCH: 8 segments, wall ≈ **1.8 s**; first sentence verbatim:
  > "This is a crazy interesting test for testing the capabilities and initial prototype of standard ASR package."
- BATCH + WORD TIMESTAMPS: **189 words** with per-token start/end (e.g.
  `[0.32-0.64] ' This'`).
- STREAMING: 11 `partial` → **8 `final`** → `done` (segments finalize
  sentence-by-sentence as the window settles).

**Whisper tiny** (`mlx-audio/whisper-tiny`) — a third family (encoder-decoder):

- BATCH: `detected_language = en`, wall ≈ 2.2 s, correct transcript.
- BATCH + WORD TIMESTAMPS: 105 words **with probabilities** (e.g.
  `[1.18-1.56] ' crazy' (p=0.986)`).
- STREAMING: partial → final progression across multiple segments.

> All three families (Qwen3-ASR, Parakeet, Whisper) run real inference through
> the **same** engine, the same `transcribe` / `start_transcription` calls, and
> the same constant result schema — only the entry-point key differs. This is the
> "one engine, many models" target.

## 4b. Generic-family coverage — the new presets run on real weights

The 15 additional families normalize onto two output shapes (`STTOutput` and
`AlignedResult`), so they are served by one data-driven `GenericSttBackend`
(parameterized by an `SttFamilySpec`) plus the shared `AlignedResultBackend`.
Three representative models — one per distinct adapter behavior — were run end to
end on real weights against `standard_asr_user/harvard.wav` (Harvard sentences):

```bash
STANDARD_ASR_ALLOW_DOWNLOAD=1 uv run python scripts/verify_inference.py \
  ../standard_asr_user/harvard.wav mlx-audio/<key>
```

- **`moonshine-tiny`** (generic, text-only, no language axis) — correct transcript
  ("The stale smell of old beer lingers. …"); `detected_language=None`,
  `segments=None`, `words=None` (honest: placeholder timing not surfaced). ~6 s.
- **`sensevoice-small`** (generic, ISO language + language detection) — correct
  transcript with `language="en"` forwarded and **`detected_language="en"`
  reported** back. ~16 s.
- **`nemotron-asr-streaming-0.6b`** (`AlignedResult`, word timing) — correct
  transcript, **5 segments / 112 words** with real per-token timestamps. ~18 s.

These exercise every new code path: ISO language pass-through, detected-language
mapping, the text-only (no-fabricated-timing) path, and the aligned token→word
mapping. The remaining families share these exact paths and are wired with the
verified repos below; each is runnable via the same `verify_inference.py` command.

### Verified model repos (and caveats)

Each preset's `hf_repo` was confirmed to exist on the Hugging Face Hub with a
`config.json` whose `model_type` matches the mlx-audio family. Caveats worth
knowing before first use:

| Preset | HF repo | Caveat |
| --- | --- | --- |
| `sensevoice-small` | `mlx-community/SenseVoiceSmall` | — |
| `nemotron-asr-streaming-0.6b` | `mlx-community/nemotron-3.5-asr-streaming-0.6b` | — |
| `fun-asr-nano` | `mlx-community/Fun-ASR-Nano-2512` | — |
| `glm-asr-nano` | `mlx-community/GLM-ASR-Nano-2512-4bit` | — |
| `granite-speech-1b` | `mlx-community/granite-4.0-1b-speech-5bit` | — |
| `granite-speech-nar-2b` | `mlx-community/granite-speech-4.1-2b-nar-mlx` | `custom_code`; bf16 (~4.5 GB) |
| `voxtral-mini-3b` | `mlx-community/Voxtral-Mini-3B-2507-bf16` | bf16 only (~9 GB); **upstream-blocked** — transformers' VoxtralProcessor needs the Mistral stack (librosa + mistral_common[audio] + more) that does not converge; use `voxtral-realtime-4b` (see §6) |
| `voxtral-realtime-4b` | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | batch only (native streaming not wired); runs out of the box |
| `qwen2-audio-7b` | `mlx-community/Qwen2-Audio-7B-Instruct-4bit` | 4-bit (~6.6 GB); preset injects a strict transcription prompt (§6) |
| `vibevoice-asr` | `mlx-community/VibeVoice-ASR-4bit` | 8 B model; preset rebuilds text from its diarization-JSON segments (§6) |
| `fireredasr2-aed` | `mlx-community/FireRedASR2-AED-mlx` | — |
| `canary-1b-v2` | `CogniSoftOrg/canary-1b-v2-mlx-bf16` | bf16 (~3.2 GB); **switched from `TechHara/...q4`**, which ships only `tokens.json` (no SentencePiece) and fails with `Tokenizer not loaded` (§6) |
| `moonshine-tiny` | `UsefulSensors/moonshine-tiny` | upstream PyTorch repo (no mlx-community port; sanitized at load) |
| `cohere-asr` | `appautomaton/cohere-asr-mlx` | weights in a `mlx-int8/` subfolder — the preset now loads from it via `hf_subfolder` (§6); **but this int8 checkpoint decodes poorly** (the only public cohere MLX repo) |
| `mms-1b-all` | `facebook/mms-1b-all` | **~29 GB** (base + ~1100 adapters); config says `wav2vec2`, so the preset pins `load_model_type="mms"` |

## 5. Test suite

```bash
uv run pytest          # 113 tests, 99% (only pre-existing _streaming.py lines)
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pyright src/    # strict: 0 errors
```

The unit suite **mocks the MLX model** (a fake `mlx_audio.stt.load`) and never
downloads weights or requires MLX at test time; it covers all backend adapters
(Qwen3-ASR, Whisper, the aligned-output backend, and the generic `STTOutput`
backend across its language/timing/translation/list-input variants), the
batch/streaming engine paths, the loaded-model family check, the `model_type`
load override, config/env, the download policy, language mapping, and the
streaming event-sequence contract (`check_event_sequence`). Real inference is the
separate, opt-in `scripts/verify_inference.py` above.

## 6. Full functional audit + fixes (2026-06-16)

A download → test → delete sweep ran **every** model end to end (real weights,
`jfk.flac`), beyond the representative subset in §3–§4b. It surfaced six models
that had never been run before; each is now fixed or documented honestly:

| model | symptom | fix |
| --- | --- | --- |
| `vibevoice-asr` | `text` was raw diarization JSON | `SttFamilySpec.text_from_segments` rebuilds the transcript from the parsed `STTOutput.segments` (stays text-only; segments not emitted) — ✅ clean |
| `qwen2-audio-7b` | `text` was a chatty LLM wrapper ("The speech is in English, with the transcription being: …") | `SttFamilySpec.default_prompt` injects a strict transcription prompt; a provider `system_prompt` overrides — ✅ clean |
| `canary-1b-v2` | `RuntimeError: Tokenizer not loaded` | `CanaryTokenizer` is SentencePiece-based; the old `TechHara/...q4` repo ships only `tokens.json`. Switched to `CogniSoftOrg/canary-1b-v2-mlx-bf16` (ships `tokenizer.model`) — ✅ clean |
| `granite-speech-nar-2b` | `ModuleNotFoundError: soundfile`, then `ValueError: audio must be 16000 Hz` | its `_load_waveform` accepts an `mx.array` directly but refuses to resample a path; `array_only=True` feeds it a decoded 16 kHz array — ✅ clean |
| `cohere-asr` | `FileNotFoundError: Config not found` | weights live in `mlx-int8/`; new `MlxAudioASR.hf_subfolder` snapshot-downloads + loads the subfolder. Loads + runs now, **but the only public (int8) checkpoint decodes garbage** — ⚠️ upstream checkpoint quality |
| `voxtral-mini-3b` | `ValueError` → `TypeError` → `ImportError` → `NameError` → `ImportError` | input-shape bug fixed via `SttFamilySpec.wants_path` (engine materializes a temp WAV → path), but transformers' VoxtralProcessor then pulls in an unconverging Mistral dep stack (librosa, mistral_common[audio], …). **Left as-is; use `voxtral-realtime-4b`** — ❌ upstream-blocked |

New engine mechanisms (all unit-tested): `SttFamilySpec.wants_path` /
`.default_prompt` / `.text_from_segments`; `MlxAudioASR.hf_subfolder` +
`_resolve_subfolder`; `_array_to_wav_tempfile`.

**Net real-inference status of all 20 models:** 17 clean, 1 loads-but-poor
(`cohere-asr`, int8 checkpoint), 1 upstream-blocked (`voxtral-mini-3b`); the 20th,
`voxtral-realtime-4b`, runs (batch). For live mic specifically, 8 of the 9
streaming-capable models work (all but `cohere-asr`).

## Notes / honest caveats

- **Streaming is a windowed re-decode**, not a native low-latency recognizer
  (none of these MLX backends expose incremental decoding in mlx-audio). The
  capabilities are declared to match exactly that: `word_stability=false`,
  `re_segments=false`, `reconnect=unsupported`, `finality=final`,
  `timestamps=post_align`. See `docs/STANDARD_ASR_FINDINGS.md`.
- **MLX is thread-bound**: the streaming decode runs inline on the event-loop
  thread (offloading to a worker thread raises `RuntimeError: There is no
  Stream(gpu, N) in current thread`). Documented in the findings.
- **Whisper presets point at the `openai/*` repos**, not `mlx-community/whisper-*`,
  because the mlx-community Whisper repos omit the `WhisperProcessor` files that
  mlx-audio needs at runtime. Documented in the findings.
- **Backend is verified against the loaded model's family.** mlx-audio's `load`
  auto-detects the family from `config.json`; each preset binds one backend, so
  after loading, the engine asserts the loaded family is one the backend handles
  and raises a `DiscoveryError` otherwise. A `model_path` override pointing at a
  different family fails loudly instead of being run through the wrong adapter
  (which would silently produce a wrong transcript).
- **Text-only families are batch only.** The windowed streaming strategy settles
  on real segment/token timing; families that emit no real timing (SenseVoice,
  Voxtral, Canary, Qwen2-Audio, Granite Speech, Moonshine, MMS, FireRedASR2,
  VibeVoice, Voxtral-Realtime) declare `streaming` unsupported rather than emit a
  degenerate stream. The timing-bearing families (Cohere, Fun-ASR, GLM-ASR) and
  the aligned families (Parakeet, Nemotron) stream via re-decode like Qwen3-ASR.
- Larger models download on first run: `Qwen3-ASR-1.7B-8bit` (~3.4 GB) and
  `whisper-large-v3-turbo` (OpenAI repo) are heavier; the 0.6B Qwen, Parakeet
  v3, and whisper-tiny are the fast paths used above.
