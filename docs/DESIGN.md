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
| Guidance | portable `prompt` as its context (the system turn, `system_prompt`) | portable `prompt` as `initial_prompt` | none |
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
   heterogeneity lives. A `ModelBackend` does three things:
   - `generate_kwargs(...)` — translate the resolved BCP-47 language into the
     family's native surface (name vs. code vs. nothing) and build the
     family-specific `generate` kwargs.
   - `to_result(native, ...)` — normalize the family's native return value
     (`STTOutput` *or* `AlignedResult`) onto the one constant
     `TranscriptionResult` schema, honoring the word-timestamp null semantics
     (`words=None` ⇒ not requested; spec TR.3).
   - `single_segment_span(params)` — say how the family splits its segments:
     `None` when it splits at what it hears (sentences, pauses), else the
     internal chunk length up to which it returns one segment spanning the input
     (Qwen3-ASR's `chunk_duration`). The streaming session reads it, through
     `has_inner_boundaries`, to decide where to commit audio (§4).

   Four implementations exist: `Qwen3AsrBackend`, `WhisperBackend`,
   `AlignedResultBackend` (Parakeet, Nemotron) and the data-driven
   `GenericSttBackend` (the other `STTOutput` families, each described by an
   `SttFamilySpec`). Adding a family is **a new `SttFamilySpec`, or a new
   `ModelBackend`, + an entry point** — no change to the engine, config, or
   pipeline.

3. **`_metadata.py` — per-family Properties + Capabilities.** Each family
   declares its *own* honest, fail-closed capabilities: Qwen3-ASR declares
   `word_timestamps=["segment"]` + prompt guidance (its context); Whisper declares
   `["word","segment"]` + prompt guidance; Parakeet declares `["word","segment"]` but
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

This plugin decodes every family through its batch `generate` call (some
upstream models also have streaming entry points, such as Nemotron's
cache-aware `stream_generate`; the plugin does not use them today), so streaming
is a **re-decode-the-window** session (the same approach `std-faster-whisper` uses for
a batch engine): accumulate fed PCM in a window of not-yet-committed audio,
periodically re-decode the whole window via the bound backend, commit the front
of the window as `final` events, and emit the rest as one moving `partial`. The
declared streaming capabilities match exactly: `emits_partials=true`,
`partial_stability=false` (every `partial` has an empty `stable_text`, because the
next pass may rewrite any text not yet finalized), `re_segments=false`,
`reconnect=unsupported`, `finality=final`, `timestamps=post_align`.

**Intake.** A task owned by the session reads the fed audio into an inbox as it
arrives, so the session knows how much audio is waiting. The inbox is bounded
(`max_window_s` of audio, or 30 s with no cap): while it is at or over the bound
the intake reads nothing more, the library's bounded queue fills, and the client
is held back, as it would be without an intake. Because a chunk is read whole,
the inbox can hold the bound plus one chunk. The float32 window (at most twice
the cap) and the byte copy a pass takes out of the inbox (up to the inbox's
contents) are separate buffers.
These are bounds on the session's own buffers, not on the process's memory. Only whole 16-bit samples leave the inbox; a
byte of a sample split across chunks waits for the rest, and one left at the end
of the input is dropped with a `partial_sample_dropped` diagnostic.

**Pacing.** A pass is due when the inbox holds `redecode_interval_s` of audio,
when it is full, when its oldest sample has waited `redecode_interval_s` (so
audio is decoded even when the client stops sending), or when the input ends.
`redecode_interval_s` is the one setting for how often the window is decoded.
The pass yields to the loop briefly so audio already on its way reaches the
inbox (until a 10 ms step passes with no new chunk and no other work blocking
the loop, at most five steps), then takes the whole inbox, and it yields once
more before each decode so the audio pump and other sessions can run. A fixed
cadence (one decode per interval of audio) falls further behind with every
decode once a decode takes longer than the interval; taking the whole backlog
does not, wherever the time went (this session's decode or another session's on
the same event loop).

**Why there is no rest between passes.** Three rules for resting after a pass
were tried, each meant to leave the loop free for other work. They were
compared in a simulation: eight configurations (decode cost `r × window` for
r = 0.3, 0.5, 0.8, 0.9; caps of 10 s and 30 s; interval 0.3 s; 600 s of live
audio), each run under each policy.

1. *Half the previous decode time after every pass* (the first rework). It
   reserved a third of the loop even when the session was behind: at
   `r = 0.5`, cap 10 s, it reached 159 s of lag against 9 s with no rest.
2. *The same rest, skipped when behind* (a full inbox, a cap's worth of audio
   waiting, or a pass whose decodes took as long as the audio it took in). Fine
   on linear decode costs, but with a cost of `0.8 × window²`, one-second chunks
   and a settle margin of 0, a schedule that keeps up without any rest (`done`
   0.8 s after the end of the input) ran away: 366 s late after 120 s of input.
   Audio keeps arriving during a rest, so a rest longer than the time the pass
   gained makes the next window bigger.
3. *A rest bounded by that gain*, `min(0.5 × D, N − W)` with `D` the pass's
   decode time, `W` its wall time and `N` the audio it took in (a bound of
   `N − D` instead still drifted 0.01 s per pass on settling overhead and ended
   83 s late). It passed the tested sweep: in every configuration it stayed
   within one pass of no rest, on both the lag of the newest partial and the lag
   over all events. The differences went both ways. At `r = 0.3`, cap 30 s, the
   bounded rest did better: worst lag 8.56 s (partials) and 14.58 s (all
   events) against 9.02 s and 15.04 s, `done` 6.83 s after the end against
   7.51 s, and 0.906 s of decoding per second of audio against 1.008 s. At
   `r = 0.3`, cap 10 s, no rest finished sooner: `done` 3.75 s after the end
   against 4.72 s.

The rest was removed to take scheduling complexity out of the session, accepting
those bounded differences, in both directions and within one pass, in the tested
configurations. The real-model runs of the two policies were made on machines
under different load and do not compare them. The simulation covers decode cost,
waits and settling only; it leaves out the time spent on PCM conversion, energy
analysis, event handling and transport, and the machine's thermal state.

**Backlog.** When a pass commits heads at the cap and the inbox has filled
again meanwhile (one interval of audio, full, or audio left after the end of the
input), the pass takes the inbox again and keeps committing heads; it decodes the
rest of the window only once the backlog is absorbed. While a session works
through a backlog it emits `final` events for the committed heads and no
`partial` until it has caught up. A pass that commits no head behaves as
before. Backlog rounds follow only cap heads (a pause commit also commits a head,
but does not start a backlog round), so with `max_window_s=None` queued input is
decoded window by window and still produces partials. Without this, each pass took at most one cap of audio and decoded the
remaining window again for its `partial`, which made a session that had fallen
behind fall further behind than it needed to.

The session keeps up only while its sustained end-to-end service capacity is
greater than the rate at which audio arrives. That capacity covers every decode
(each pass decodes the whole window again, so a window near the cap costs a
decode near the cap's length however little audio is new; a cap commit adds the
head's decode; during a backlog only heads are decoded), settling, PCM
conversion, event handling, and any other work on the event loop. A decode time
below one second per second of audio is necessary, not sufficient. In the
simulation, a decode costing half the window's length already needs about one
second of decoding per second of audio. When the capacity falls short, the
session falls behind; the cap still bounds every decode and the inbox bound
holds the client back.

**Where to commit.** Audio leaves the window as `final` events:

- **Settled segments** (normal pass): leading segments that end
  `settle_margin_s` before the end of the decoded window.
- **The cap**: the inbox moves into the window in portions of at most one cap,
  and while the window is at least `max_window_s` long, bounded heads are
  committed until less than the cap remains, also on the final pass and for a
  single huge chunk. No decode covers more than the cap, and the window never
  holds more than two caps. Whether the backend can return a segment boundary
  inside a window of the cap's length is decided from the model family and its
  configuration before any decode (`backends.has_inner_boundaries`): Whisper,
  Parakeet and Nemotron split at what they hear; Qwen3-ASR, Fun-ASR, GLM-ASR and
  Cohere return one segment per internal chunk (Qwen3-ASR's `chunk_duration`,
  1200 s by default and never less than the 1 s mlx-audio's splitter enforces;
  Fun-ASR 1200 s; GLM-ASR 30 s; Cohere 35 s, its clip limit when decoding
  without voice-activity detection, which is mlx-audio's default and which the
  plugin does not turn on), so they have inner boundaries only when that chunk
  is shorter than the cap. Above the chunk length, Qwen3-ASR's boundaries are
  energy-selected cut points near each chunk's end, not a fixed grid. With boundaries, the first
  `max_window_s` is decoded and committed up to its last settled segment, or
  else its last inner boundary, if that covers at least half the cap. Without
  boundaries, or when no boundary covers half the cap, the window is cut in the
  middle of the longest pause in the last ten seconds before the cap (never
  before half the cap, not in the last second of the window), and the head is
  decoded once more on its own. A longer pause is more likely to end a sentence
  than to sit inside one. With no pause there, the cut goes to the least loud
  point; a cap of a second or two leaves no room for a search and cuts at the
  cap itself, which can commit the whole window.
- **Pauses** (only without inner boundaries, when `commit_pause_s` is set): a
  pause of at least that length after speech commits the audio up to its
  middle. On the 291 s synthetic recording with Qwen3-ASR 1.7B (interval 0.3 s),
  0.3 s pause commits gave a `done` 0.06 s sooner, a worst partial lag 1.54 s
  lower and 35 s less total decode time than commits at the cap alone, against
  24 false sentence breaks in 40 seams instead of 5 in 11. The default stays
  `None` because of that seam count; the time until words become final was not
  measured.
- **The end of the input**: everything left is decoded and finalized.

Pauses come from a cheap energy profile (20 ms frame RMS). A frame is quiet when
it is within 10% of the way from the window's noise level (5th percentile) to
its speech level (99th percentile). The profile has limits: speech less than
12 dB above a steady noise floor shows no pause at all (the cap then falls back
to the least loud point), pauses filling under 5% of the window are not measured
as the noise level, and speech filling under 1% of the window is not seen. It
has been checked on synthetic speech and synthetic noise only, not on
microphone noise or human pauses.

Every commit is a seam: the next window is decoded without the earlier text, so
a word next to a seam can still be lost or doubled. On the 291 s synthetic
recordings measured (macOS `say`), the old cap, which cut the whole window at an
arbitrary sample, damaged the text at 9 of 9 cuts with the application's settings
and 7 of 9 with the defaults; no quiet-point cut lost or doubled a word, but a
cut in a comma pause still made the model end the head with a period ("heavy.
So" for "heavy, so"). These are results on these recordings, not guarantees.

The library's `max_idle` deadline is a content deadline, not silence detection.
An ordinary decode before the end of the input emits a `partial` whenever the
unsettled tail of its segments has text, even when the text is unchanged, and
every `final` counts as content too (with `settle_margin_s=0` every segment can
settle, so a pass can emit `final` events and no `partial`). A decoder that
keeps returning earlier text over silence therefore keeps the session alive
under `max_idle`. A run of decodes that return no text lets it expire: they emit
`progress`, apart from one clearing `partial` when text was shown and one empty
`final` that closes an open partial, each of which counts as content once.
Detecting the end of an utterance (endpointing) is the application's job and
has not been measured here.

`max_idle` measures time without a delivered content event. Enforcement is
cooperative: inline decoding blocks the event loop, and queueing and task
scheduling can delay termination by several decodes. Neither the configured
duration nor that duration plus one decode is a guaranteed termination bound.
The library checks its deadlines in the consumer's loop (`max_session_seconds`
whenever the consumer receives an event, `max_idle` only when its wait for the
next event times out), and that loop runs only between decodes. With a 5 s
`max_idle`, 2 s decodes and a 120 s silent backlog, the session ends at 6.02 s
with `audio_queue_maxsize=2` and at 12.00 s with the default queue of 256
chunks. The round-5 review measured 6.00 s and 12.00 s on the code before this
rework, and 12.00 s on a producer with no plugin code at all, so this is the library's design
combined with inline decoding, not something this session added. A stronger
guarantee needs deadline enforcement in the library and decoding isolated from
the event loop; a different queue default in the plugin would not give one.
`test_deadlines_still_apply_during_a_backlog` records both schedules.

Positions are integer sample counts and become seconds only inside events, so
the audio cursor never steps back by a rounding error. `audio_processed_until`
is the end of audio a successful decode covered, never audio merely buffered.

**MLX threading constraint:** MLX arrays (incl. model weights) are bound to the
Metal stream of the thread that created them, and a stream cannot be used from
another thread. The model loads on the event-loop thread, so the decode runs
**inline** on that thread (offloading to `asyncio.to_thread` raises
`RuntimeError: There is no Stream(gpu, N) in current thread`). We `await
asyncio.sleep(0)` before each decode so the audio pump makes progress, then
decode inline. This blocks the loop for the decode duration: for Qwen3-ASR 1.7B
on an M5 Max, typically 0.1–0.7 s, growing with the window, with spikes of
1.9–2.1 s in the saved runs (more when other work shares the GPU). That is
acceptable for a coarse windowed re-decode and documented honestly.

## 5. Testing strategy

- **Unit suite (no weights, no downloads):** a fake `mlx_audio.stt.load` returns
  shape-accurate fakes (`STTOutput`-like / `AlignedResult`-like). This exercises
  the real adapter logic (language mapping, output normalization, batch +
  streaming engine paths, config/env, download policy). It
  imports `mlx_audio` and MLX, so it needs them installed, but loads no model.
  Most streaming tests run on a virtual-time event loop (`tests/conftest.py`):
  waits, the session's deadlines, fake decode times and simulated work by other
  sessions share one clock. The simulation leaves out the time spent on PCM
  conversion, energy analysis, event handling and transport, and the machine's
  thermal state; a few tests run on a real event loop.
- **Real-inference verification (opt-in):** `scripts/verify_inference.py` runs
  actual MLX inference on a Mac over a real file (batch + word timestamps +
  windowed streaming). See `VERIFICATION.md`.
