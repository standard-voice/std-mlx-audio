<!--
SPDX-FileCopyrightText: 2026 Standard Voice Contributors
SPDX-License-Identifier: Apache-2.0
-->

# Standard ASR v0.1.0 — plugin-author findings (MLX, one-engine-many-models)

Findings from building `std-mlx-audio` as a fully independent plugin against
`standard-asr @ main`. This plugin's deliberate stress test
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

## 8. [High] [std-asr] A streaming session fell behind for good on long input, and the library gives an adapter no way to see how much audio is waiting

*Added 2026-10-02.*

**What happened.** The session decoded its window after every
`redecode_interval_s` of consumed audio. With an application that set the
interval to 0.3 s, Qwen3-ASR 1.7B needed more than 0.3 s per decode once the
window passed about 12 s, so the session lost ground with every decode: 131 to
172 s behind after 291 s of speech on an M5 Max (three runs). Separately,
Qwen3-ASR returns one segment spanning its whole input, so no segment ever
settled; only the 30 s cap committed audio, and it committed the whole window,
including the audio just heard, at an arbitrary sample. Every one of those cuts
damaged the text next to it (a lost or doubled word, or a false sentence break).

**Resolution (plugin).** A task owned by the session reads `audio_chunks()` into
a bounded inbox as chunks arrive (one `max_window_s` of audio; while it is full
the task reads nothing more, so the library's queue still pushes back on the
client; because a chunk is read whole, the inbox can overshoot the bound by one
chunk), and each decode takes everything in the inbox, so a slow decode is
followed by one that covers the backlog (the interval is now a lower bound).
Bounded heads are committed until the window is under the cap, so no decode
covers more than `max_window_s`. Backends that return segment boundaries still
commit at them; for Qwen3-ASR at its default `chunk_duration`, audio is committed
in pauses found in the window's energy profile (at a pause when `commit_pause_s`
is set, or before the cap) instead of at the cap's sample. Positions are integer
samples. See `_streaming.py` and `docs/DESIGN.md` §4.

Earlier versions of this fix were reviewed twice. The first inferred the backlog
from the clock (chunks read without time passing had queued during a decode),
which broke in three ways: time another session spent on the same event loop
looked like idle time, so the lag grew without bound again (560 s after 240 s of
audio in a simulated schedule); audio read during the wait could stay undecoded
while the client sent nothing; and the cap bounded neither the window nor the
decode when much audio arrived at once. The second owned an intake but did not
bound its buffer, which removed the library's backpressure.

**Why it's a finding for Standard ASR.** The only input surface,
`TranscriptionSession.audio_chunks`, yields one chunk at a time. An adapter that
needs to know how much audio is waiting has to take it out of the library's
queue: run its own intake task beside the producer, keep the audio in a buffer
of its own (which it must bound, or the library's backpressure stops reaching
the client), and handle that task's cancellation and errors (the base session
gives a subclass no place to own such a task beyond `_produce` and `_close`).
What the library lacks is a way to see how much audio is waiting without taking
it out of the queue: a pending-sample count, or a non-blocking "everything that
has arrived" read, would let a re-decode adapter take the backlog while the
library keeps buffering and backpressure. The spec also does not ask a streaming
engine to keep up with real-time input, or to say so when it falls behind; a
compliance probe that reports decode cost against audio position would have
caught this. Finally, with the old pacing, a session that fell far behind made
the reference server stop answering WebSocket keepalive pings and the connection
dropped (observed twice; the cause is not confirmed).

A smaller, related point: Qwen3-ASR declares the `"segment"` word-timestamp
granularity because it can always return its one input-spanning segment (spec
TR.3). That declaration is true but tells an application nothing about sentence
timing. `check_event_sequence(..., capabilities=...)` checks only that a stream
does not exceed its declaration, so it cannot flag this; whether such a model
should declare `"segment"` is a protocol question.

**Deadlines.** A deadline guarantee (`max_idle`, `max_session_seconds`) needs enforcement in the library and isolation of the engine's work from the event loop; this adapter, which decodes inline, cannot guarantee the library's deadlines through its pacing or a different queue size. The library checks deadlines only in the consumer's loop (`max_session_seconds` on every received event, `max_idle` only when the wait for the next event times out), so a plugin that decodes inline (as MLX requires, see item 4) delays them by several decodes, depending on queue size and scheduling (`docs/DESIGN.md` §4).

---

## 9. [High] Several family loaders degrade SILENTLY when non-weight assets are missing [not std-asr]

The `post_load_hook` of several families loads its tokenizer or
normalization assets with `if path.exists()` (or `try/except: pass`) and the
decode path then falls back without any signal: SenseVoice and MMS emit the
numeric token ids as the transcript, FireRedASR2 returns an empty string,
Moonshine joins per-id characters, and Cohere-ASR decodes with an empty
special-token set (the language, task, and speaker tags leak into the text)
when its tokenizer config is absent. A partial snapshot (config + weights,
assets missing) therefore transcribes silently wrong. Verified per family
in the installed mlx-audio; the plugin closes it by declaring each verified
family's silent-corruption files in `required_checkpoint_files` (status
reports `incomplete` for any checkpoint, `model_path` included, and the
implicit-load recheck refuses the fragment). The families that instead fail
loudly at load (Qwen3, GLM, Granite x2, Voxtral x2, Fun-ASR, Qwen2-Audio)
or at the first generate (Whisper, Canary) get `required_snapshot_files`:
per-file-ablated single points of failure that gate only the Hub snapshot,
so status stays honest for an interrupted download and plain `pull`
repairs it, while alternative local layouts (Canary's `tokens.txt`, a
vocab + merges tokenizer) are not falsely rejected. Two upstream details
worth knowing: FireRed's `train_bpe1000.model` is loaded into a field no
decode path reads (dead code), and Parakeet / Nemotron embed their
vocabularies in `config.json` (the repos' tokenizer files are conversion
by-products).

---

## 10. [Med] `DEFAULT_ALLOW_PATTERNS` omits SenseVoice's `am.mvn`; VibeVoice silently Hub-fetches its tokenizer [not std-asr]

Two acquisition gaps in upstream defaults. First, SenseVoice's feature
normalization stats live in `am.mvn`, its config carries no fallback, and no
default allow pattern matches `.mvn` -- every default-pattern snapshot loads
with normalization silently skipped. The plugin extends its snapshot filter
with `*.mvn`. Second, the VibeVoice-ASR checkpoint ships no tokenizer files
at all, and the upstream hook silently falls back to downloading the
`Qwen/Qwen2.5-7B` tokenizer from the Hub on every cold load -- a network
fetch that happens inside transformers, past the engine's
`local_files_only`. The plugin models that repo as the preset's companion
tokenizer: status requires it in the default Hugging Face cache, `pull`
acquires it there, and a load under a no-download policy refuses instead of
letting the upstream fetch bypass the policy. The residual (cached
tokenizer, downloads disabled, network reachable: transformers may still
revalidate against the Hub and fetch an updated file) stays documented on
the preset. Related upstream note: the hook resolves its three speech
marker tokens with `convert_tokens_to_ids`, which returns the
unknown-token id for an absent token without any check, so a tokenizer
from the wrong vocabulary silently misplaces the speech embeddings. A
three-line marker-id validation in the hook would make that loud; the
plugin does not second-guess a checkpoint's internal consistency (the
same line drawn for mixed-up weights).

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
