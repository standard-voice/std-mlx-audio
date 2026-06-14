<!--
SPDX-FileCopyrightText: 2026 Standard Voice Contributors
SPDX-License-Identifier: Apache-2.0
-->

# Standard ASR v0.1.0 — plugin-author findings (MLX, one-engine-many-models)

Findings from building `std-mlx-audio` as a fully independent plugin against
`standard-asr @ refactor/v0.1.0-redesign`. This plugin's deliberate stress test
was **"one engine, many models"**: a single `engine_id` (`mlx-audio`) exposing
three genuinely heterogeneous model families (Qwen3-ASR, Parakeet, Whisper) with
different native APIs, return types, languages, and capabilities. The headline
result is positive — the protocol modeled the family cleanly, and a complete
batch + streaming engine with discovery, compliance, real inference on all three
families, 100 % test coverage, and pyright-strict typing came together in one
sitting. The items below are ordered by impact.

Severity legend: **[High]** blocks or silently misleads · **[Med]** real friction
· **[Low]** papercut.

Most items below are **about MLX / mlx-audio**, not Standard ASR — included
because they are exactly the integration realities an engine author hits. The
Standard-ASR-specific findings are tagged **[std-asr]**.

---

## 1. [Med] [std-asr] "One engine, many models" works well — but capabilities/properties are *per model*, and that story could be documented

**What happened (the good part).** The protocol carried a heterogeneous model
family with no contortions. Each preset is a tiny `EngineBase` subclass binding
four class vars (`hf_repo`, `backend`, `properties`, `declared_capabilities`), and
the three families declare *different* capabilities under the *same* engine:

- Qwen3-ASR: `word_timestamps=["segment"]`, no guidance, language override yes.
- Whisper: `word_timestamps=["word","segment"]`, prompt guidance, override yes.
- Parakeet: `word_timestamps=["word","segment"]`, **`language.runtime_override=false`**
  (fixed-language model), no guidance.

`models list` / `models show` / compliance all treated these as five independent
models keyed by `mlx-audio/<model>`, read instantiation-free. This is genuinely
the USB-C promise for a *model family*, and it's a strength worth showcasing.

**The friction.** All the guidance (`adapting_engine.md`, the faster-whisper
template) implies **one Properties + one Capabilities per engine**. There is no
worked example of an engine whose presets differ in *capabilities* (not just
weights). I had to infer that the per-preset class is the right seam for
divergent capabilities, and that nothing in the core assumes a single
engine-wide capability set. It works, but a short "model family" section —
"presets MAY declare different capabilities/properties; bind them per preset
class; the registry reads each independently" — would save the next author the
inference. (Also: per-engine *config* is one model, so per-preset config
*defaults* needed a small `default_config_overrides` hook on my side; see #2.)

## 2. [Med] [std-asr] LANG R1 (`default_language` ∈ `selectable_languages`) collides with a fixed-language model whose set has no `"auto"`

**What happened.** My engine-wide config defaults `default_language="auto"`. For
the Qwen3-ASR/Whisper presets that's fine (`"auto"` is in their
`selectable_languages`). But Parakeet is fixed-language: it has no auto-detect
directive, so its `selectable_languages` is just the 25 supported tags **without**
`"auto"`. Compliance then failed, correctly but surprisingly:

```
[FAIL] mlx-audio/parakeet-tdt-0.6b-v3 [language_config_invalid]:
  default_language 'auto' is not in selectable_languages [...] (spec LANG R1).
```

**Why it mattered.** The engine has *one* config class but the presets have
*different* valid language defaults. LANG R1 is right to require a valid default,
but the interaction with one-engine-many-models isn't obvious. I solved it with a
per-preset `default_config_overrides = {"default_language": "en"}`, applied in
`__init__` before `from_env` (explicit kwargs still win). The value is inert at
inference (Parakeet ignores language) but satisfies the axis totality invariant.

**Suggestion.** Document the pattern for per-preset config defaults in a
multi-model engine, and/or let a model whose set has no `"auto"` use `None` as a
"no detection, model decides" default without tripping LANG R1 (today `None`
raises a *different* ConfigError because the axis is non-empty). A fixed-language
model genuinely has "no selectable default"; the spec currently forces a
semantically-inert pick.

## 3. [Low] [std-asr] Provider-params for a multi-backend engine: one terminal type, but the knobs apply to *different subsets* of models

**What happened.** Swap-safety requires exactly one terminal `ProviderParams`
type per engine (exact-type match). With one engine spanning three backends, my
`MlxAudioParams` necessarily carries the *union* of native knobs
(`temperature`/`top_p`/… for Qwen3-ASR; `chunk_duration` for the chunkers;
`system_prompt` for Qwen only). Parakeet understands *none* of them. So a caller
can set `top_k=50` and target Parakeet, and it is silently ignored.

**Why it's only Low.** This is inherent to "one engine, many models" + one params
type, and the values are all decode hints (no correctness risk). But it's a small
honesty gap: the params model can't express "this field applies to backends X,Y
but not Z." A per-field "applies to" hint (advisory, for the auto-UI) would let a
settings UI grey out inapplicable knobs per selected model. Not blocking.

## 4. [High] MLX is thread-bound — `asyncio.to_thread` for the streaming decode crashes [not std-asr; document for engine authors]

**What happened.** I followed the faster-whisper template's streaming pattern,
which offloads the blocking decode to `asyncio.to_thread` so the event loop
doesn't stall. With MLX that raises:

```
RuntimeError: There is no Stream(gpu, 1) in current thread.
```

MLX arrays (incl. model weights) are bound to the Metal *stream* of the thread
that created them, and a stream **cannot** be used from another thread (I tried
`set_default_device`, `set_default_stream`, a dedicated worker with its own
stream — all fail, because the *weights* live on the load thread's stream).

**Resolution.** Run the decode **inline** on the event-loop thread (the model is
loaded there), with an `await asyncio.sleep(0)` before each decode so the audio
pump progresses. This blocks the loop for the decode duration, which is fine for a
coarse windowed re-decode.

**Why it's a finding for Standard ASR.** The streaming base + the faster-whisper
template both nudge authors toward `to_thread`. A one-line note in
`adapting_engine.md` — "if your runtime is thread-affine (MLX, some CUDA
contexts), decode inline or on a single dedicated thread that *owns* the model"
— would save the next GPU-framework author this crash.

## 5. [Med] mlx-community Whisper repos omit the processor mlx-audio needs [not std-asr]

**What happened.** `mlx-community/whisper-large-v3-turbo` and
`mlx-community/whisper-tiny` ship only `config.json` + `weights.safetensors`. But
mlx-audio's Whisper backend calls `WhisperProcessor.from_pretrained(repo)` in its
post-load hook; with no processor files it sets `_processor=None` and then *every
transcription* raises `ValueError: Processor not found`. Discovery, compliance
(instantiation-free), and `__init__` all pass — the failure only appears at the
first real transcribe.

**Resolution.** Point the Whisper presets at the **OpenAI** repos
(`openai/whisper-large-v3-turbo`, `openai/whisper-tiny`), which ship the full
processor; mlx-audio loads/quantizes them on first use and they transcribe
correctly. (Qwen3-ASR and Parakeet mlx-community repos are complete — this is
Whisper-specific.)

**Why it's a finding for Standard ASR.** It's a good motivating example for an
*optional* `prepare()`/preflight in compliance that actually loads weights behind
a flag (`compliance run --include-load`), so a "looks discoverable, fails at
first transcribe" model is caught in CI rather than in production. Today
`--no-instantiate` and the default both stop short of a real load.

## 6. [Low] Per-backend native shapes are untyped (`STTOutput` vs `AlignedResult`) [not std-asr]

mlx-audio returns `STTOutput` (with `segments: List[dict]`, untyped) for
Qwen3-ASR/Whisper and a different `AlignedResult` for Parakeet. My adapter
normalizes both to the constant `TranscriptionResult` and clamps the occasional
inverted/negative span so a stray backend timestamp can't make `Segment`/`Word`
construction (rightly strict: `end>=start>=0`, no NaN/Inf) reject a whole
transcript. The Standard result models being strict here is a **plus** — it
turned "silently wrong timestamps" into "clamp + keep the transcript," and the
clamping is in one small helper.

## 7. [Low] [std-asr] `models show` capability JSON is verbose for a quick check

`models show` dumps the full capability tree as JSON. For a human eyeballing
"does this model do word timestamps?" a one-line capability summary (like
`models list` has) would help. Minor; the JSON is correct and complete.

---

## What worked well (credit where due)

- **`EngineBase` template method** — implementing only `_transcribe` /
  `_start_transcription` and getting audio negotiation, param gating, language
  resolution, the sync bridge, and the error contract for free is excellent. The
  family-agnostic engine + per-family backend split fell out naturally.
- **Fail-closed capabilities + instantiation-free discovery** — declaring
  per-model capabilities and having `models show` / the registry read them
  without constructing the engine is exactly right for a multi-model engine.
- **Honest streaming model** — `stable_until` / `finality` / `re_segments` /
  `reconnect` let me describe a windowed re-decode *truthfully* instead of
  pretending it's a native incremental recognizer. The `check_event_sequence`
  helper validated my event stream in tests.
- **Strict result models** — rejecting inverted/NaN spans caught real backend
  quirks at the boundary instead of downstream.
- **`from_env` + `SecretStr`** — zero-config env fallback and a masked HF token
  for free.
