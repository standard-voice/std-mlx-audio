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
| `standard-asr` | 0.1.0 (git `refactor/v0.1.0-redesign`) |
| Test audio | `standard_asr/reference/standard_asr_test_audio_english.m4a` (~57 s, English) |

Setup:

```bash
cd std-mlx-audio
uv python pin 3.12
uv sync                      # resolves standard-asr from the branch + mlx-audio[stt]
```

## 1. Discovery — five models under one engine (`standard-asr list`)

```bash
uv run standard-asr list
```

```
 - mlx-audio/parakeet-tdt-0.6b-v3    engine=mlx-audio  model=parakeet-tdt-0.6b-v3
 - mlx-audio/qwen3-asr-0.6b          engine=mlx-audio  model=qwen3-asr-0.6b
 - mlx-audio/qwen3-asr-1.7b          engine=mlx-audio  model=qwen3-asr-1.7b
 - mlx-audio/whisper-large-v3-turbo  engine=mlx-audio  model=whisper-large-v3-turbo
 - mlx-audio/whisper-tiny            engine=mlx-audio  model=whisper-tiny
```

`standard-asr show mlx-audio/qwen3-asr-0.6b` prints the capabilities and
params schema **without instantiating** the engine (instantiation-free discovery).

## 2. Compliance — all five models pass (`standard-asr compliance run`)

```bash
for m in qwen3-asr-0.6b qwen3-asr-1.7b parakeet-tdt-0.6b-v3 \
         whisper-large-v3-turbo whisper-tiny; do
  uv run standard-asr compliance run "mlx-audio/$m"
done
```

Each prints `[OK] Compliance run passed.` (entry-point metadata, capability
declarations, `model_id` match, and streaming param-gating). The streaming
event-sequence contract is covered in the test suite via
`standard_asr.compliance.check_event_sequence` (see §5).

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

## 5. Test suite — passes at 100 % coverage

```bash
uv run pytest          # 69 tests, 100% line+branch coverage
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pyright src/    # strict: 0 errors
```

The unit suite **mocks the MLX model** (a fake `mlx_audio.stt.load`) and never
downloads weights or requires MLX at test time; it covers all three backend
adapters, the batch/streaming engine paths, config/env, the download policy,
language mapping, and the streaming event-sequence contract
(`check_event_sequence`). Real inference is the separate, opt-in
`scripts/verify_inference.py` above.

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
- Larger models download on first run: `Qwen3-ASR-1.7B-8bit` (~3.4 GB) and
  `whisper-large-v3-turbo` (OpenAI repo) are heavier; the 0.6B Qwen, Parakeet
  v3, and whisper-tiny are the fast paths used above.
