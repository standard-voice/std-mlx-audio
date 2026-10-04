# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pacing, commit-point and robustness tests for the windowed MLX session.

Almost every test runs on a virtual-time event loop
(``conftest.VirtualTimeLoop``): waits, the session's own deadlines (through
``on_virtual_clock``), fake decodes and simulated work by other sessions all
move one clock, so minutes of live audio run in milliseconds. A feeder sends
each chunk at its real-time arrival; a fake decoder blocks the loop for a chosen
time by advancing the clock. The simulation leaves out everything but that
clock: the cost of PCM conversion, energy analysis, event handling and
transport. A few tests at the end run on a real event loop.

Sessions built by the ``_session`` helper run with ``strict_lifecycle=True``,
and the drive helpers require them to end with no recorded diagnostics, so the
runtime guard cannot quietly repair an invalid stream. Two ``done_timeout``
tests build their sessions with the library defaults, and one test expects the
``partial_sample_dropped`` diagnostic.

The default fake decoder behaves like Qwen3-ASR: one segment spanning whatever
audio it is given (no text for silence), so no segment ever settles. Whisper-
and Parakeet-like fakes return real segment boundaries.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from itertools import pairwise
from typing import Any, cast

import numpy as np
import pytest
from standard_asr import RuntimeParams, TranscriptionEvent
from standard_asr.compliance import check_event_sequence
from standard_asr.contract.exceptions import StreamClosedError

from std_mlx_audio import ParakeetTdt06BV3, Qwen3Asr06B, WhisperTiny
from std_mlx_audio._streaming import (
    MlxAudioStreamingSession,
    _frame_rms,
    _pause_cut,
)

from .conftest import (
    FAKE_SNAPSHOT_DIR,
    FakeAlignedResult,
    FakeAlignedSentence,
    FakeAlignedToken,
    FakeLoader,
    FakeSTTOutput,
    VirtualTimeLoop,
    on_virtual_clock,
    run_virtual,
)

_RATE = 16000
Timed = list[tuple[float, TranscriptionEvent]]


# --------------------------------------------------------------------------- #
# Audio and fakes
# --------------------------------------------------------------------------- #
def _tone(seconds: float, amp: float = 0.3) -> np.ndarray:
    """Return a loud 220 Hz tone (stands in for speech)."""
    t = np.arange(round(seconds * _RATE), dtype=np.float32) / _RATE
    return (amp * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * _RATE), dtype=np.float32)


def _noise(seconds: float, rms: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(round(seconds * _RATE)) * rms).astype(np.float32)


def _pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _chunks(audio: np.ndarray, chunk_s: float = 0.1) -> list[bytes]:
    """Split a float waveform into ``pcm_s16le`` chunks of ``chunk_s``."""
    pcm = _pcm(audio)
    step = round(chunk_s * _RATE) * 2
    return [pcm[i : i + step] for i in range(0, len(pcm), step)]


def _spanning(audio: Any, *, pad_to_one_second: bool = False) -> FakeSTTOutput:
    """Qwen3-ASR-like output: one segment spanning the input; no text for silence.

    With ``pad_to_one_second``, input shorter than a second reports a one-second
    segment, as mlx-audio's Qwen3-ASR does after padding it.
    """
    samples = np.asarray(audio)
    if not np.any(np.abs(samples) > 1e-3):
        return FakeSTTOutput(text="", segments=[])
    secs = samples.size / _RATE
    if pad_to_one_second:
        secs = max(secs, 1.0)
    text = f"[{samples.size / _RATE:.2f}s]"
    return FakeSTTOutput(text=text, segments=[{"text": text, "start": 0.0, "end": secs}])


def _costing(loop: VirtualTimeLoop, cost: Callable[[float], float]) -> Callable[..., Any]:
    """A spanning decoder that blocks the loop for ``cost(window seconds)``."""

    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        loop.advance(cost(len(audio) / _RATE))
        return _spanning(audio)

    return output_fn


def _session(engine: Any = None, **knobs: Any) -> MlxAudioStreamingSession:
    kwargs: dict[str, Any] = {
        "redecode_interval_s": 1.0,
        "settle_margin_s": 0.4,
        "max_window_s": 10.0,
        **knobs,
    }
    params = kwargs.pop("params", RuntimeParams())
    return MlxAudioStreamingSession(
        engine if engine is not None else Qwen3Asr06B(), params, strict_lifecycle=True, **kwargs
    )


class _PartialLedger:
    """Check the partial lifecycle from the raw events the session produces.

    The ledger wraps the session's ``_produce`` and records each event before
    the library sees it (the library coalesces partials, which would hide a
    duplicate). From those raw events alone, not from the session's own state,
    it keeps the open segment id, the text the application shows for it, and
    the ids already finalized, and checks every event as it arrives:

    * a ``partial`` or ``final`` never names a finalized id, and never arrives
      while a different id is still open;
    * an empty ``partial`` clears text: it is allowed only while the open id
      shows non-empty text, so once per withdrawal;
    * an empty ``final`` closes an open partial: it is allowed only while its
      id is open;
    * nothing follows ``done`` or ``error``, and ``done`` leaves no id open.

    It also wraps ``_decode_pass`` and, as extra checks, compares the ledger
    with the session's state (``_partial_shown`` and the open id). Inside a
    pass the session can be ahead, because a batch of events is built before
    it is yielded, so after each event the comparison runs only when the two
    agree; a non-empty open partial's audio must then still be in the window.
    On every exit from a pass, normal or by an exception (a decode or intake
    failure), the two must agree and the same audio check runs. When the pass
    is cancelled or closed (the application left, or the library ended the
    stream at a deadline), the session may hold events it built but never
    yielded, so only the per-event rules apply. After an abnormal end, the
    partial the application saw last stays open in the ledger: the library's
    terminal ``error``, or the application's leaving, ends it. ``ended``
    records how the producer stopped.

    Problems are collected and reported by :meth:`check`, not raised inside the
    session, where the base session would turn them into an ``engine_error``.
    """

    def __init__(self, session: MlxAudioStreamingSession) -> None:
        self.session = session
        self.shown: dict[str, str] = {}  # open id -> the text it shows
        self.finalized: set[str] = set()
        self.last_partial: dict[str, TranscriptionEvent] = {}
        # "done", "error", the name of the exception that stopped the producer
        # (an intake failure, or CancelledError when the session is torn down),
        # or "closed".
        self.ended: str | None = None
        self.problems: list[str] = []
        real_produce = session._produce  # type: ignore[attr-defined]
        real_pass = session._decode_pass  # type: ignore[attr-defined]

        async def produce() -> AsyncIterator[TranscriptionEvent]:
            events = real_produce()
            try:
                async for event in events:
                    self.record(event)
                    yield event
            except BaseException as exc:
                self.stop(type(exc).__name__)
                raise
            finally:
                self.stop("closed")
                await events.aclose()

        async def checked_pass(**kwargs: Any) -> AsyncIterator[TranscriptionEvent]:
            left = False
            try:
                async for event in real_pass(**kwargs):
                    yield event
                    self.compare(pass_done=False)
            except (asyncio.CancelledError, GeneratorExit):
                left = True  # the application left, or the session was torn down
                raise
            finally:
                self.compare(pass_done=True, left=left)

        session._produce = produce  # type: ignore[method-assign]
        session._decode_pass = checked_pass  # type: ignore[method-assign]

    def problem(self, text: str) -> None:
        self.problems.append(text)

    def record(self, event: TranscriptionEvent) -> None:
        """Apply one raw event to the ledger and check it."""
        if self.ended is not None:
            self.problem(f"{event.type} after the stream ended ({self.ended})")
        sid = event.segment_id
        if event.type in ("partial", "final") and sid is not None:
            if sid in self.finalized:
                self.problem(f"{sid}: {event.type} after its final")
            for other in self.shown.keys() - {sid}:
                self.problem(f"{sid}: {event.type} while {other} is still open")
            if event.type == "partial":
                if not event.text and not self.shown.get(sid):
                    self.problem(f"{sid}: an empty partial clears nothing")
                self.shown[sid] = event.text
                self.last_partial[sid] = event
            else:
                if not event.text and sid not in self.shown:
                    self.problem(f"{sid}: an empty final closes no open partial")
                self.shown.pop(sid, None)
                self.finalized.add(sid)
        elif event.type == "done":
            self.ended = "done"
            for sid in self.shown:
                self.problem(f"{sid} is still open at done")
            if self.session._partial_shown is not None:  # type: ignore[attr-defined]
                self.problem("the session still shows a partial at done")
        elif event.type == "error":
            self.ended = "error"

    def stop(self, how: str) -> None:
        if self.ended is None:
            self.ended = how

    def compare(self, *, pass_done: bool, left: bool = False) -> None:
        """Compare the ledger with the session's state (see the class docstring)."""
        session = self.session
        shown = session._partial_shown  # type: ignore[attr-defined]
        open_id = f"seg-{session._finalized_count}"  # type: ignore[attr-defined]
        expected = {} if shown is None else {open_id: shown}
        if left:
            return
        if self.shown != expected:
            # A batch is built before it is yielded, so inside a pass the
            # session can be ahead of the ledger until the batch is out.
            if pass_done:
                self.problem(f"the session shows {expected}, the events {self.shown}")
            return
        for sid, text in self.shown.items():
            partial = self.last_partial.get(sid)
            if not text or partial is None or partial.start is None:
                continue
            # One sample of tolerance: the partial's start and the trim each
            # round a time to samples once.
            if round(partial.start * _RATE) < session._origin - 1:  # type: ignore[attr-defined]
                self.problem(f"{sid} shows {text!r} but its audio was dropped")

    def check(self, events: list[TranscriptionEvent]) -> None:
        """Report every problem; also check the delivered ``events``.

        On the delivered stream: an empty ``partial`` follows a non-empty one of
        the same id, and an empty ``final`` follows a partial of its id. When it
        holds ``done``, even one the library added after the producer returned
        without yielding it, the ledger has no id open: ``done`` is a normal
        end, so the abnormal-end allowance does not apply.
        """
        assert not self.problems, self.problems
        assert self.ended is not None, "the watched producer never ended"
        if any(event.type == "done" for event in events):
            assert not self.shown, f"delivered done with {sorted(self.shown)} still open"
        previous: dict[str, TranscriptionEvent] = {}
        for event in events:
            sid = event.segment_id
            if event.type not in ("partial", "final") or sid is None:
                continue
            before = previous.get(sid)
            if event.type == "partial" and not event.text:
                assert before is not None and before.type == "partial" and before.text, (
                    f"delivered: empty partial for {sid} clears nothing"
                )
            if event.type == "final" and not event.text:
                assert before is not None and before.type == "partial", (
                    f"delivered: empty final for {sid} closes no partial"
                )
            previous[sid] = event


def _watch_partials(
    session: MlxAudioStreamingSession,
) -> Callable[[list[TranscriptionEvent]], None]:
    """Install a :class:`_PartialLedger` on ``session``; return its check."""
    return _PartialLedger(session).check


async def _drive(
    loop: VirtualTimeLoop,
    session: MlxAudioStreamingSession,
    chunks: list[bytes],
    *,
    chunk_s: float = 0.1,
    gaps: dict[int, float] | None = None,
    end: bool = True,
    end_at: float | None = None,
) -> Timed:
    """Send each chunk at its real-time arrival, then end; collect timed events.

    Args:
        loop: The virtual-time loop.
        session: The session under test.
        chunks: The chunks to send.
        chunk_s: The duration of each chunk (sets the arrival times).
        gaps: Extra seconds with nothing sent, before chunk ``i``.
        end: Whether to call ``end_audio`` after the last chunk.
        end_at: When set, the time to call ``end_audio`` at (after the chunks).

    Returns:
        ``(virtual time, event)`` for every event delivered.
    """
    out: Timed = []
    on_virtual_clock(session, loop)
    at_end = _watch_partials(session)

    async def feeder() -> None:
        due = 0.0
        try:
            for i, chunk in enumerate(chunks):
                due += (gaps or {}).get(i, 0.0) + chunk_s
                await asyncio.sleep(max(0.0, due - loop.time()))
                await session.send_audio(chunk)
            if end:
                if end_at is not None:
                    await asyncio.sleep(max(0.0, end_at - loop.time()))
                await session.end_audio()
        except StreamClosedError:
            pass  # the session ended early (an error); stop sending

    async with session:
        task = asyncio.ensure_future(feeder())
        async for event in session:
            out.append((loop.time(), event))
        await task
    assert not session.diagnostics(), session.diagnostics()
    at_end([event for _, event in out])
    return out


async def _drive_queued(session: MlxAudioStreamingSession, chunks: list[bytes]) -> Timed:
    """Feed every chunk at once (a whole file); collect events."""
    out: Timed = []
    on_virtual_clock(session, cast(VirtualTimeLoop, asyncio.get_running_loop()))
    at_end = _watch_partials(session)
    async with session:
        session.feed(chunks)
        async for event in session:
            out.append((asyncio.get_running_loop().time(), event))
    assert not session.diagnostics(), session.diagnostics()
    at_end([event for _, event in out])
    return out


def _events(timed: Timed) -> list[TranscriptionEvent]:
    return [event for _, event in timed]


def _lags(timed: Timed) -> list[float]:
    return [t - e.audio_processed_until for t, e in timed if e.audio_processed_until is not None]


def _sizes(loader: FakeLoader) -> list[float]:
    return [len(c["audio"]) / _RATE for c in loader.model.generate_calls]


def _assert_contract(events: list[TranscriptionEvent], engine_cls: Any = Qwen3Asr06B) -> None:
    report = check_event_sequence(events, capabilities=engine_cls.declared_capabilities)
    assert report.passed, [i.message for i in report.issues]
    finals = [e for e in events if e.type == "final"]
    for before, after in pairwise(finals):
        assert after.start is not None and before.end is not None
        assert after.start >= before.end - 1e-9, (before, after)


# --------------------------------------------------------------------------- #
# Pacing: keeping up with live input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("decode_s", [0.5, 2.0])
def test_lag_stays_about_one_decode_under_a_real_time_feed(
    fake_loader: Callable[..., FakeLoader], decode_s: float
) -> None:
    """Decodes slower than the interval must not make the session fall behind.

    With a 0.3 s interval and 0.5 s per decode, a loop that decodes after every
    0.3 s of audio loses 0.2 s per decode: about 40 s behind after a minute
    (342 s with 2.0 s decodes). The session takes all arrived audio each time.
    """

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_costing(loop, lambda _w: decode_s))
        session = _session(redecode_interval_s=0.3, max_window_s=None)
        return await _drive(loop, session, _chunks(_tone(60.0)))

    timed = run_virtual(main)
    assert max(_lags(timed)) <= 2 * decode_s, f"fell behind: {max(_lags(timed)):.1f}s"
    assert timed[-1][1].type == "done"
    _assert_contract(_events(timed))


def test_decode_cost_growing_with_the_window_stays_bounded_by_the_cap(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    box: dict[str, FakeLoader] = {}

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["loader"] = fake_loader(output_fn=_costing(loop, lambda w: 0.07 + 0.02 * w))
        session = _session(redecode_interval_s=0.3, max_window_s=30.0)
        return await _drive(loop, session, _chunks(_tone(180.0)))

    timed = run_virtual(main)
    assert max(_sizes(box["loader"])) <= 30.0 + 1e-9
    assert max(_lags(timed)) <= 2.0, max(_lags(timed))
    _assert_contract(_events(timed))


def test_a_decoder_slower_than_real_time_falls_behind_but_each_decode_stays_bounded(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # 1.2 s of decode per second of audio, even for heads alone: no schedule can
    # keep up. The lag grows, but the cap still bounds every decode.
    box: dict[str, FakeLoader] = {}

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["loader"] = fake_loader(output_fn=_costing(loop, lambda w: 0.1 + 1.2 * w))
        session = _session(redecode_interval_s=0.3, max_window_s=10.0)
        return await _drive(loop, session, _chunks(_tone(180.0)))

    timed = run_virtual(main)
    assert max(_sizes(box["loader"])) <= 10.0 + 1e-9
    assert timed[-1][1].type == "done"
    lags = [(t, t - e.audio_processed_until) for t, e in timed if e.audio_processed_until]
    early = max(lag for t, lag in lags if t < 90.0)
    assert max(lag for _, lag in lags) > early + 10.0  # documented: scheduling cannot fix this
    _assert_contract(_events(timed))


def test_other_work_on_the_same_loop_does_not_make_the_lag_grow(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    """Another session's decode runs on the loop after each of ours.

    Interval 0.3 s, own decode 0.5 s, then 0.5 s of other work: the review's
    counterexample, where inferring arrival from idle time reached 560 s of lag
    after 240 s of audio.
    """

    async def main(loop: VirtualTimeLoop) -> Timed:
        due = asyncio.Event()

        def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
            loop.advance(0.5)
            due.set()
            return _spanning(audio)

        async def other_session() -> None:
            while True:
                await due.wait()
                due.clear()
                await asyncio.sleep(0)
                loop.advance(0.5)

        fake_loader(output_fn=output_fn)
        other = asyncio.ensure_future(other_session())
        timed = await _drive(loop, _session(redecode_interval_s=0.3), _chunks(_tone(240.0)))
        other.cancel()
        return timed

    timed = run_virtual(main)
    assert max(_lags(timed)) <= 2.5, max(_lags(timed))
    late = [t - e.audio_processed_until for t, e in timed if t > 200 and e.audio_processed_until]
    assert max(late) <= 2.5  # not growing with session length
    _assert_contract(_events(timed))


def test_fast_decoder_decodes_once_per_interval(fake_loader: Callable[..., FakeLoader]) -> None:
    box: dict[str, FakeLoader] = {}

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["loader"] = fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
        return await _drive(loop, _session(max_window_s=None), _chunks(_tone(5.0)))

    timed = run_virtual(main)
    # The last chunk and the end of the input arrive together: the 5 s decode is
    # the final pass.
    assert _sizes(box["loader"]) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sum(e.type == "partial" for e in _events(timed)) == 4


def test_queued_whole_input_is_decoded_in_bounded_heads(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    box: dict[str, FakeLoader] = {}

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["loader"] = fake_loader(output_fn=_costing(loop, lambda _w: 0.5))
        return await _drive_queued(_session(max_window_s=30.0), _chunks(_tone(100.0)))

    timed = run_virtual(main)
    assert max(_sizes(box["loader"])) <= 30.0 + 1e-9
    finals = [e for e in _events(timed) if e.type == "final"]
    assert finals[-1].end == pytest.approx(100.0, abs=0.01)
    _assert_contract(_events(timed))


def test_audio_buffered_when_the_client_pauses_is_decoded_without_more_chunks(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # A 2 s decode covers the first 0.3 s; another 0.4 s arrives during it, and
    # then the client sends nothing for a minute (without ending the input).
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_costing(loop, lambda _w: 2.0))
        session = _session(redecode_interval_s=0.3, max_window_s=None)
        return await _drive(loop, session, _chunks(_tone(1.7)), gaps={7: 60.0})

    timed = run_virtual(main)
    during_gap = [e for t, e in timed if 3.0 < t < 60.0 and e.audio_processed_until]
    assert max(e.audio_processed_until for e in during_gap) == pytest.approx(0.7)
    assert timed[-1][1].type == "done"
    assert timed[-1][1].audio_processed_until == pytest.approx(1.7)
    _assert_contract(_events(timed))


def test_the_end_of_input_arriving_during_a_slow_decode_is_finalized_right_after_it(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_costing(loop, lambda _w: 2.0))
        session = _session(redecode_interval_s=0.3, max_window_s=None)
        return await _drive(loop, session, _chunks(_tone(0.7)))

    timed = run_virtual(main)
    done_at, done = timed[-1]
    assert done.type == "done" and done.audio_processed_until == pytest.approx(0.7)
    # One decode of 0.3 s ends at 2.3 s; the final pass starts right away.
    assert done_at == pytest.approx(4.3, abs=0.1)


def test_a_long_strict_session_keeps_the_cursor_monotonic(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The review's probe: float seconds made the cursor step back
    # 190.70000000000002 -> 190.7 and strict mode ended the session.
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_costing(loop, lambda _w: 0.5))
        session = _session(redecode_interval_s=0.3, settle_margin_s=0.4, max_window_s=30.0)
        return await _drive(loop, session, _chunks(_tone(240.0)))

    events = _events(run_virtual(main))
    assert events[-1].type == "done"
    assert events[-1].audio_processed_until == pytest.approx(240.0)
    cursors = [e.audio_processed_until for e in events if e.audio_processed_until is not None]
    assert cursors == sorted(cursors)
    _assert_contract(events)


# --------------------------------------------------------------------------- #
# The cap bounds every decode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("seconds", "chunk_s"), [(120.0, 120.0), (200.0, 40.0)])
def test_oversized_chunks_never_produce_an_oversized_decode(
    fake_loader: Callable[..., FakeLoader], seconds: float, chunk_s: float
) -> None:
    box: dict[str, FakeLoader] = {}

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["loader"] = fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
        session = _session(redecode_interval_s=0.3, max_window_s=30.0)
        return await _drive(loop, session, _chunks(_tone(seconds), chunk_s), chunk_s=chunk_s)

    timed = run_virtual(main)
    assert max(_sizes(box["loader"])) <= 30.0 + 1e-9
    finals = [e for e in _events(timed) if e.type == "final"]
    assert finals[-1].end == pytest.approx(seconds, abs=0.01)
    _assert_contract(_events(timed))


def test_an_interval_longer_than_the_cap_still_bounds_each_decode(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    box: dict[str, FakeLoader] = {}

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["loader"] = fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
        session = _session(redecode_interval_s=40.0, max_window_s=10.0)
        return await _drive(loop, session, _chunks(_tone(100.0)))

    timed = run_virtual(main)
    assert max(_sizes(box["loader"])) <= 10.0 + 1e-9
    _assert_contract(_events(timed))


def test_a_cap_below_one_frame_is_rejected(fake_loader: Callable[..., FakeLoader]) -> None:
    fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
    with pytest.raises(ValueError, match="max_window_s"):
        MlxAudioStreamingSession(Qwen3Asr06B(), RuntimeParams(), max_window_s=0.00001)


# --------------------------------------------------------------------------- #
# Quiet-point commits at the window cap (no segment boundaries)
# --------------------------------------------------------------------------- #
def _run_cut(
    fake_loader: Callable[..., FakeLoader], audio: np.ndarray, **knobs: Any
) -> tuple[list[TranscriptionEvent], FakeLoader]:
    box: dict[str, FakeLoader] = {}
    output_fn = knobs.pop("output_fn", lambda a, _kw: _spanning(a))

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["loader"] = fake_loader(output_fn=output_fn)
        return await _drive(loop, _session(**knobs), _chunks(audio))

    events = _events(run_virtual(main))
    _assert_contract(events)
    return events, box["loader"]


def test_cap_cut_lands_in_the_silence_between_bursts(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    audio = np.concatenate([_tone(7.0), _silence(0.6), _tone(5.0)])
    events, loader = _run_cut(fake_loader, audio)
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 2
    cut = finals[0].end
    assert cut is not None and 7.0 < cut < 7.6, cut
    # The head was decoded on its own, exactly up to the cut.
    assert any(abs(size - cut) < 1e-3 for size in _sizes(loader))
    assert finals[1].start == pytest.approx(cut, abs=1e-3)
    assert finals[1].end == pytest.approx(12.6, abs=1e-3)


def test_cap_cut_prefers_the_longest_pause(fake_loader: Callable[..., FakeLoader]) -> None:
    # A short pause (like a comma) and a longer one (like a sentence end).
    audio = np.concatenate([_tone(6.0), _silence(0.2), _tone(1.5), _silence(0.5), _tone(4.0)])
    events, _ = _run_cut(fake_loader, audio)
    first = next(e for e in events if e.type == "final")
    assert first.end == pytest.approx(7.95, abs=0.03)


def test_cap_cut_never_finalizes_the_audio_just_heard(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The only silence sits in the last second before the cap is reached, so it
    # is out of reach; the cut falls back to the least loud point earlier on.
    audio = np.concatenate([_tone(9.4), _silence(0.5), _tone(4.0)])
    events, _ = _run_cut(fake_loader, audio)
    first = next(e for e in events if e.type == "final")
    assert first.end is not None and first.end < 9.4 - 1e-6


def test_cap_still_bounds_the_window_without_any_quiet_point(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    events, loader = _run_cut(fake_loader, _tone(45.0))
    assert max(_sizes(loader)) <= 10.0 + 1e-6
    assert sum(e.type == "final" for e in events) >= 5


def test_cuts_keep_timestamps_and_the_cursor_absolute_and_monotonic(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    rng = np.random.default_rng(7)
    parts = [np.concatenate([_tone(rng.uniform(1.5, 4.0)), _silence(0.4)]) for _ in range(14)]
    audio = np.concatenate(parts)
    events, _ = _run_cut(fake_loader, audio)
    finals = [e for e in events if e.type == "final"]
    assert len(finals) >= 4
    assert [e.segment_id for e in finals] == [f"seg-{i}" for i in range(len(finals))]
    for before, after in pairwise(finals):
        assert after.start == pytest.approx(before.end, abs=1e-3)
    assert finals[0].start == 0.0
    cursors = [e.audio_processed_until for e in events if e.audio_processed_until is not None]
    assert cursors == sorted(cursors)
    assert cursors[-1] == pytest.approx(audio.size / _RATE, abs=1e-9)


def test_cap_shorter_than_the_search_range_cuts_at_the_cap(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    events, loader = _run_cut(fake_loader, _tone(4.0), max_window_s=1.0)
    assert max(_sizes(loader)) <= 1.0 + 1e-6
    assert sum(e.type == "final" for e in events) == 4


def test_a_padded_short_head_does_not_overlap_the_next_segment(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # mlx-audio's Qwen3-ASR reports a one-second segment for a shorter input. A
    # head shorter than a second must still end where the next segment starts.
    def padded(audio: Any, _kw: Any) -> FakeSTTOutput:
        return _spanning(audio, pad_to_one_second=True)

    audio = np.concatenate([_tone(0.2), _silence(0.3), _tone(3.0)])
    events, _ = _run_cut(
        fake_loader,
        audio,
        output_fn=padded,
        redecode_interval_s=0.5,
        commit_pause_s=0.25,
        max_window_s=None,
    )
    finals = [e for e in events if e.type == "final"]
    assert finals[0].end is not None and finals[0].end < 0.5
    assert finals[1].start == pytest.approx(finals[0].end)


# --------------------------------------------------------------------------- #
# Early commit at pauses
# --------------------------------------------------------------------------- #
def test_pause_commits_the_audio_before_it(fake_loader: Callable[..., FakeLoader]) -> None:
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
        audio = np.concatenate(
            [_silence(1.0), _tone(2.0), _silence(0.8), _tone(2.0), _silence(0.3), _tone(1.0)]
        )
        session = _session(redecode_interval_s=0.5, commit_pause_s=0.5)
        return await _drive(loop, session, _chunks(audio))

    timed = run_virtual(main)
    finals = [(t, e) for t, e in timed if e.type == "final"]
    # The leading second of silence follows no speech, and the 0.3 s gap is too
    # short: only the 0.8 s pause (3.0-3.8 s) commits. It commits as soon as it
    # is 0.5 s long, at the middle of what has been heard of it so far.
    assert len(finals) == 2
    first_at, first = finals[0]
    assert first.end == pytest.approx(3.25, abs=0.05)
    assert 3.5 <= first_at <= 3.6, first_at  # delivered right after the pause
    _assert_contract(_events(timed))


def test_without_commit_pause_a_pause_commits_nothing(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    audio = np.concatenate([_tone(2.0), _silence(0.8), _tone(2.0)])
    events, _ = _run_cut(fake_loader, audio, redecode_interval_s=0.5)
    assert [e.type for e in events if e.type == "final"] == ["final"]


def test_quiet_threshold_finds_pauses_over_a_noise_floor() -> None:
    # Speech at RMS 0.1 over a steady noise floor at RMS 0.02 (14 dB below).
    speech = np.concatenate([_tone(2.0, 0.14), _silence(1.0), _tone(2.0, 0.14)])
    cut = _pause_cut(_frame_rms(speech + _noise(5.0, 0.02)), 0.5)
    assert cut is not None and 2.2 * _RATE < cut < 2.8 * _RATE


def test_quiet_threshold_with_sparse_speech() -> None:
    # Speech fills 5% of the window; the rest is a quiet noise floor.
    sparse = np.concatenate([_tone(0.5, 0.14), _silence(9.5)]) + _noise(10.0, 0.002)
    assert _pause_cut(_frame_rms(sparse), 1.0) is not None


def test_quiet_threshold_sees_no_speech_in_noise_alone_or_at_low_snr() -> None:
    assert _pause_cut(_frame_rms(_noise(10.0, 0.02)), 0.5) is None
    # Speech within 12 dB of the noise floor is not told apart from it.
    low_snr = np.concatenate([_tone(2.0, 0.07), _silence(1.0), _tone(2.0, 0.07)])
    assert _pause_cut(_frame_rms(low_snr + _noise(5.0, 0.03)), 0.5) is None


def test_energy_helpers_handle_audio_shorter_than_a_frame() -> None:
    rms = _frame_rms(_silence(0.01))
    assert rms.size == 0
    assert _pause_cut(rms, 0.5) is None


# --------------------------------------------------------------------------- #
# Backends with segment boundaries keep committing at them
# --------------------------------------------------------------------------- #
def _fractions_output(fractions: list[float]) -> Callable[..., Any]:
    """A Whisper-like decoder that splits its input at the given fractions."""

    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        secs = len(audio) / _RATE
        cuts = [0.0, *(f * secs for f in fractions), secs]
        segs = [{"text": f"s{i} ", "start": a, "end": b} for i, (a, b) in enumerate(pairwise(cuts))]
        return FakeSTTOutput(text="".join(s["text"] for s in segs), language="en", segments=segs)

    return output_fn


def test_a_settled_boundary_wins_over_the_energy_cut(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The review's probe: interval and cap 10 s, segments [0, 8] and [8, 10],
    # settle margin 2 s. The first decode already crosses the cap.
    box: dict[str, FakeLoader] = {}

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["loader"] = fake_loader(output_fn=_fractions_output([0.8]))
        session = _session(
            WhisperTiny(), redecode_interval_s=10.0, settle_margin_s=2.0, max_window_s=10.0
        )
        return await _drive(loop, session, [_pcm(_tone(10.0))], chunk_s=10.0)

    timed = run_virtual(main)
    first = next(e for e in _events(timed) if e.type == "final")
    assert first.end == pytest.approx(8.0, abs=1e-3)
    assert _sizes(box["loader"])[0] == pytest.approx(10.0)
    _assert_contract(_events(timed), WhisperTiny)


def test_an_oversized_window_on_a_boundary_backend_is_committed_at_boundaries(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # 100 s arrives at once; every decode returns segments at 60% and 40%.
    box: dict[str, FakeLoader] = {}

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["loader"] = fake_loader(output_fn=_fractions_output([0.6]))
        session = _session(WhisperTiny(), settle_margin_s=2.0, max_window_s=10.0)
        return await _drive(loop, session, [_pcm(_tone(100.0))], chunk_s=100.0)

    timed = run_virtual(main)
    assert max(_sizes(box["loader"])) <= 10.0 + 1e-9
    finals = [e for e in _events(timed) if e.type == "final"]
    # Every commit lands on a boundary the decoder gave (6 s into each prefix).
    assert finals[0].end == pytest.approx(6.0)
    assert finals[-1].end == pytest.approx(100.0)
    _assert_contract(_events(timed), WhisperTiny)


def test_parakeet_like_sentences_crossing_the_cap_commit_at_sentence_ends(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeAlignedResult:
        secs = len(audio) / _RATE
        bounds = [*np.arange(0.0, secs, 3.0).tolist(), secs]
        sents = [
            FakeAlignedSentence(
                text=f"S{i}. ",
                start=a,
                end=b,
                tokens=[FakeAlignedToken(text=f"S{i}. ", start=a, end=b)],
            )
            for i, (a, b) in enumerate(pairwise(bounds))
            if b > a
        ]
        return FakeAlignedResult(text="".join(s.text for s in sents), sentences=sents)

    box: dict[str, FakeLoader] = {}

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["loader"] = fake_loader(output_fn=output_fn)
        session = _session(ParakeetTdt06BV3(), settle_margin_s=1.0, max_window_s=10.0)
        return await _drive(loop, session, _chunks(_tone(40.0), 20.0), chunk_s=20.0)

    timed = run_virtual(main)
    assert max(_sizes(box["loader"])) <= 10.0 + 1e-9
    ends = [e.end for e in _events(timed) if e.type == "final" and e.end is not None]
    # Commits during the cap loop land on the decoder's 3 s sentence grid.
    assert ends[0] == pytest.approx(3.0)
    assert ends[-1] == pytest.approx(40.0)
    _assert_contract(_events(timed), ParakeetTdt06BV3)


def test_settled_segments_still_commit_with_absolute_word_times(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    """Below the cap, settled segments commit; words shift to session time."""

    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        secs = len(audio) / _RATE
        segs = [
            {
                "text": f"w{i} ",
                "start": float(i),
                "end": float(i + 1),
                "words": [{"word": f"w{i}", "start": float(i), "end": i + 0.5, "probability": 1.0}],
            }
            for i in range(max(1, int(secs)))
        ]
        return FakeSTTOutput(text="".join(s["text"] for s in segs), language="en", segments=segs)

    box: dict[str, FakeLoader] = {}

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["loader"] = fake_loader(output_fn=output_fn)
        session = _session(
            WhisperTiny(),
            params=RuntimeParams(language="en", word_timestamps="word"),
            settle_margin_s=1.0,
            commit_pause_s=0.5,  # ignored: this backend gives boundaries
        )
        return await _drive(loop, session, _chunks(np.concatenate([_tone(6.0), _silence(6.0)])))

    timed = run_virtual(main)
    finals = [e for e in _events(timed) if e.type == "final"]
    assert len(finals) >= 10
    assert max(_sizes(box["loader"])) <= 3.0
    for event in finals:
        assert event.words and event.start is not None
        assert event.words[0].start == pytest.approx(event.start)
    _assert_contract(_events(timed), WhisperTiny)


# --------------------------------------------------------------------------- #
# Commits that decode to nothing, decode errors, and the intake task
# --------------------------------------------------------------------------- #
def test_silent_head_closes_the_open_partial(fake_loader: Callable[..., FakeLoader]) -> None:
    # The partial for seg-0 shows the tone that has started; the cut then
    # commits only the leading silence, which decodes to no text. seg-0 must be
    # closed (empty final) rather than left showing stale text.
    events, _ = _run_cut(fake_loader, np.concatenate([_silence(8.0), _tone(4.0)]))
    seg0 = [e for e in events if e.segment_id == "seg-0"]
    assert seg0[0].type == "partial" and seg0[0].text
    assert seg0[-1].type == "final" and seg0[-1].text == ""
    later = [e for e in events if e.segment_id == "seg-1"]
    assert later and later[-1].type == "final" and later[-1].text


def test_final_pass_closes_the_open_partial_when_it_decodes_to_nothing(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    calls = {"n": 0}

    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        calls["n"] += 1
        if calls["n"] > 2:  # the final pass (3 s) hears nothing after all
            return FakeSTTOutput(text="", segments=[])
        return _spanning(audio)

    events, _ = _run_cut(fake_loader, _tone(3.0), output_fn=output_fn)
    assert [(e.type, e.text) for e in events if e.type in ("partial", "final")][-1] == ("final", "")


def test_head_decode_error_surfaces_and_skips_the_tail(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    sizes: list[int] = []

    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        if sizes and len(audio) < sizes[-1]:
            raise RuntimeError("head decode failed")
        sizes.append(len(audio))
        return _spanning(audio)

    audio = np.concatenate([_tone(7.0), _silence(0.6), _tone(5.0)])
    events, loader = _run_cut(fake_loader, audio, output_fn=output_fn)
    assert events[-1].type == "error"
    # Window decodes at 1..9 s showed no boundaries, so at the cap the head is
    # cut at a pause and decoded; that decode fails, and no tail decode follows.
    assert len(loader.model.generate_calls) == 10


def test_a_failed_tail_does_not_claim_its_audio_was_processed(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # A 30 s window: the head commit succeeds, the tail decode fails. The head's
    # final may claim only the audio its decode covered.
    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        if len(audio) < 20 * _RATE:
            raise RuntimeError("tail failed")
        return _spanning(audio)

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=output_fn)
        session = _session(redecode_interval_s=30.0, max_window_s=30.0)
        audio = np.concatenate([_tone(24.0), _silence(0.5), _tone(5.5)])
        return await _drive(loop, session, [_pcm(audio)], chunk_s=30.0)

    events = _events(run_virtual(main))
    final = next(e for e in events if e.type == "final")
    assert final.end is not None and final.audio_processed_until is not None
    assert final.audio_processed_until <= 30.0 - 5.0  # not the whole window
    assert events[-1].type == "error"


def test_an_input_failure_in_the_intake_reaches_the_session(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    async def failing_source() -> AsyncIterator[bytes]:
        yield _pcm(_tone(0.5))
        raise OSError("microphone unplugged")

    async def main(loop: VirtualTimeLoop) -> list[TranscriptionEvent]:
        fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
        session = _session()
        ledger = _PartialLedger(session)
        out: list[TranscriptionEvent] = []
        async with session:
            session.feed(failing_source())
            async for event in session:
                out.append(event)
        assert session._intake is not None and session._intake.done()  # type: ignore[attr-defined]
        # The producer re-raised the intake's failure; the base turned it into
        # the terminal error. Everything emitted before it obeyed the rules.
        ledger.check(out)
        assert ledger.ended == "_InputSourceError"  # the library wraps the source's OSError
        return out

    events = run_virtual(main)
    assert events[-1].type == "error"
    assert events[-1].code == "input_source_error"


def test_leaving_the_session_early_stops_the_intake(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    async def main(loop: VirtualTimeLoop) -> bool:
        fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
        session = _session(redecode_interval_s=0.3)
        ledger = _PartialLedger(session)
        seen: list[TranscriptionEvent] = []
        async with session:
            for chunk in _chunks(_tone(2.0)):
                await session.send_audio(chunk)
            async for event in session:
                seen.append(event)
                if event.type == "partial":
                    break
        # The application left: the producer was cancelled, and the partial it
        # saw last stays open (nobody is listening any more).
        ledger.check(seen)
        assert ledger.ended == "CancelledError" and list(ledger.shown) == ["seg-0"]
        intake = session._intake  # type: ignore[attr-defined]
        return intake is not None and intake.done()

    assert run_virtual(main)


def test_a_prefix_without_a_settled_segment_commits_at_its_last_inner_boundary(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # Segments [0, 9.5] and [9.5, 10] of a 10 s prefix, settle margin 2 s: none
    # has settled, so the commit goes to the last boundary inside the prefix.
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_fractions_output([0.95]))
        session = _session(
            WhisperTiny(), redecode_interval_s=10.0, settle_margin_s=2.0, max_window_s=10.0
        )
        return await _drive(loop, session, [_pcm(_tone(10.0))], chunk_s=10.0)

    first = next(e for e in _events(run_virtual(main)) if e.type == "final")
    assert first.end == pytest.approx(9.5, abs=1e-3)


def test_a_prefix_without_any_boundary_falls_back_to_the_pause_cut(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # Whisper has boundaries, so the cap first decodes the 10 s prefix; this
    # decoder returns one spanning segment there, so the head is then cut at the
    # pause and decoded on its own.
    audio = np.concatenate([_tone(7.0), _silence(0.6), _tone(5.0)])
    events, loader = _run_cut(
        fake_loader, audio, engine=WhisperTiny(), settle_margin_s=0.4, max_window_s=10.0
    )
    sizes = _sizes(loader)
    prefix = sizes.index(10.0)  # the boundary probe ran ...
    assert 7.0 < sizes[prefix + 1] < 7.6  # ... and then the pause cut's head
    head = next(e for e in events if e.type == "final")
    assert head.end == pytest.approx(sizes[prefix + 1])
    assert max(sizes) <= 10.0 + 1e-9


def test_a_failed_prefix_decode_surfaces_an_error(fake_loader: Callable[..., FakeLoader]) -> None:
    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        raise RuntimeError("decode failed")

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=output_fn)
        session = _session(WhisperTiny(), redecode_interval_s=20.0, max_window_s=10.0)
        return await _drive(loop, session, [_pcm(_tone(20.0))], chunk_s=20.0)

    events = _events(run_virtual(main))
    assert [e.type for e in events] == ["error"]


def test_clamping_a_padded_segment_also_clamps_its_words(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        return FakeSTTOutput(
            text="hi",
            language="en",
            segments=[
                {
                    "text": "hi",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [{"word": "hi", "start": 0.1, "end": 0.9, "probability": 1.0}],
                }
            ],
        )

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=output_fn)
        session = _session(
            WhisperTiny(),
            params=RuntimeParams(language="en", word_timestamps="word"),
            redecode_interval_s=0.5,
        )
        return await _drive(loop, session, _chunks(_tone(0.5)))

    final = next(e for e in _events(run_virtual(main)) if e.type == "final")
    assert final.end == pytest.approx(0.5)
    assert final.words and final.words[0].end == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Bounded memory, backpressure, and the age of buffered audio
# --------------------------------------------------------------------------- #
def _peaks(session: MlxAudioStreamingSession, peaks: dict[str, float]) -> None:
    peaks["inbox"] = max(peaks.get("inbox", 0.0), len(session._inbox) / (2 * _RATE))  # type: ignore[attr-defined]
    peaks["window"] = max(peaks.get("window", 0.0), session._window.size / _RATE)  # type: ignore[attr-defined]


def test_memory_stays_bounded_when_input_is_queued_faster_than_decoded(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The review's probe: 120 s queued, cap 10 s, library queue of 2 chunks.
    peaks: dict[str, float] = {}
    box: dict[str, MlxAudioStreamingSession] = {}

    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        _peaks(box["s"], peaks)
        return _spanning(audio)

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=output_fn)
        box["s"] = _session(max_window_s=10.0, audio_queue_maxsize=2)
        return await _drive_queued(box["s"], _chunks(_tone(120.0)))

    timed = run_virtual(main)
    assert peaks["inbox"] <= 10.0 + 0.1  # the bound plus one chunk
    assert peaks["window"] <= 20.0 + 1e-9  # the cap plus one portion
    assert timed[-1][1].audio_processed_until == pytest.approx(120.0)
    # Queued input: bounded finals, not a partial after every interval.
    assert sum(e.type == "partial" for _, e in timed) <= 120.0 / 1.0 / 4


def test_an_oversized_chunk_enters_the_window_in_bounded_portions(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    peaks: dict[str, float] = {}
    box: dict[str, MlxAudioStreamingSession] = {}

    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        _peaks(box["s"], peaks)
        return _spanning(audio)

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=output_fn)
        box["s"] = _session(max_window_s=10.0)
        return await _drive(loop, box["s"], [_pcm(_tone(60.0))], chunk_s=60.0)

    timed = run_virtual(main)
    assert peaks["window"] <= 20.0 + 1e-9
    _assert_contract(_events(timed))


def test_a_full_inbox_makes_a_pass_due_whatever_the_interval(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # Interval 40 s, inbox bound 10 s: without the full-inbox rule, the intake
    # waits for room while the decoder waits for 40 s of audio.
    calls: list[float] = []

    async def main(loop: VirtualTimeLoop) -> Timed:
        def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
            calls.append(loop.time())
            return _spanning(audio)

        fake_loader(output_fn=output_fn)
        session = _session(redecode_interval_s=40.0, max_window_s=10.0, audio_queue_maxsize=2)
        return await _drive(loop, session, _chunks(_tone(30.0)))

    timed = run_virtual(main)
    assert calls[0] <= 10.5, calls[:3]
    assert timed[-1][1].type == "done"


def test_audio_below_one_interval_is_decoded_once_it_has_waited_one_interval(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The review's probe: interval 1.5 s, 0.5 s sent, input left open a minute.
    calls: list[float] = []

    async def main(loop: VirtualTimeLoop) -> list[TranscriptionEvent]:
        def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
            calls.append(loop.time())
            return _spanning(audio)

        fake_loader(output_fn=output_fn)
        session = on_virtual_clock(_session(redecode_interval_s=1.5, max_window_s=None), loop)
        check = _watch_partials(session)
        events: list[TranscriptionEvent] = []
        async with session:
            await asyncio.sleep(0.1)
            await session.send_audio(_pcm(_tone(0.5)))
            await asyncio.sleep(60.0)
            assert calls == [pytest.approx(1.6, abs=0.05)]  # one interval later, once
            await session.end_audio()
            async for event in session:
                events.append(event)
        check(events)
        return events

    events = run_virtual(main)
    assert events[-1].type == "done"
    assert events[-1].audio_processed_until == pytest.approx(0.5)


def test_a_lone_byte_never_makes_a_pass_due_and_is_reported_at_the_end(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    loader_box: dict[str, FakeLoader] = {}

    async def main(loop: VirtualTimeLoop) -> tuple[list[TranscriptionEvent], Any]:
        loader_box["l"] = fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
        session = on_virtual_clock(_session(redecode_interval_s=0.3), loop)
        check = _watch_partials(session)
        events: list[TranscriptionEvent] = []
        async with session:
            await session.send_audio(_pcm(_tone(0.5)) + b"\x01")
            await asyncio.sleep(30.0)  # the whole samples are decoded; the byte waits
            calls_before_end = len(loader_box["l"].model.generate_calls)
            await session.end_audio()
            async for event in session:
                events.append(event)
        assert calls_before_end == 1
        check(events)
        return events, session.diagnostics()

    events, diagnostics = run_virtual(main)
    assert events[-1].type == "done"
    assert events[-1].audio_processed_until == pytest.approx(0.5)
    assert [d.code for d in diagnostics] == ["partial_sample_dropped"]


def test_a_sample_split_across_chunks_survives_two_passes(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The review's probe: bytes 01 02 03, a pass, then 04 05 06.
    seen: list[list[int]] = []

    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        seen.append([round(float(x) * 32768) for x in np.asarray(audio)])
        return _spanning(audio)

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=output_fn)
        session = _session(redecode_interval_s=1 / _RATE, max_window_s=None)
        return await _drive(loop, session, [bytes([1, 2, 3]), bytes([4, 5, 6])], chunk_s=5.0)

    timed = run_virtual(main)
    assert seen[0] == [513]
    assert seen[-1] == [513, 1027, 1541]
    assert timed[-1][1].audio_processed_until == pytest.approx(3 / _RATE)


def test_a_sample_split_across_a_cap_commit_survives(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # 0.5 s plus one byte arrives, the cap commit runs on the whole samples, and
    # the sample completed by the next chunk lands in the next window intact.
    # Every sample value is distinct, so each decoded array must be an exact,
    # contiguous slice of the input.
    audio = np.arange(16000, dtype="<i2")
    raw = audio.tobytes()
    split = 8000 * 2 + 1
    seen: list[np.ndarray] = []

    def output_fn(a: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        seen.append(np.round(np.asarray(a) * 32768).astype(np.int64))
        return _spanning(a)

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=output_fn)
        session = _session(redecode_interval_s=0.25, max_window_s=0.4)
        return await _drive(loop, session, [raw[:split], raw[split:]], chunk_s=0.5)

    timed = run_virtual(main)
    assert timed[-1][1].audio_processed_until == pytest.approx(1.0)
    for decoded in seen:
        start = int(decoded[0])
        assert np.array_equal(decoded, audio[start : start + decoded.size]), start
    assert any(int(d[0]) <= 8000 < int(d[0]) + d.size for d in seen)


# --------------------------------------------------------------------------- #
# Boundaries decided from the backend, and real progress at the cap
# --------------------------------------------------------------------------- #
def test_a_short_first_window_does_not_hide_the_backends_boundaries(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The review's probe: one segment for a short window, [0, 8] and [8, 10] for
    # a full one. Boundaries come from the backend, not from the first decode.
    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        secs = len(audio) / _RATE
        if secs < 9.9:
            segs = [{"text": "a", "start": 0.0, "end": secs}]
        else:
            segs = [
                {"text": "a ", "start": 0.0, "end": 8.0},
                {"text": "b", "start": 8.0, "end": secs},
            ]
        return FakeSTTOutput(text="ab", language="en", segments=segs)

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=output_fn)
        session = _session(
            WhisperTiny(), redecode_interval_s=1.0, settle_margin_s=2.0, max_window_s=10.0
        )
        return await _drive(loop, session, [_pcm(_tone(1.0)), _pcm(_tone(9.0))], gaps={1: 8.0})

    first = next(e for e in _events(run_virtual(main)) if e.type == "final")
    assert first.end == pytest.approx(8.0)


@pytest.mark.parametrize("first_end", [0.02, 0.0])
def test_tiny_boundaries_cannot_turn_the_cap_loop_into_a_crawl(
    fake_loader: Callable[..., FakeLoader], first_end: float
) -> None:
    # The review's probe: every decode returns [0, 0.02] and [0.02, end] (or a
    # zero-length first segment). A full-cap decode must buy half a cap.
    box: dict[str, FakeLoader] = {}

    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        secs = len(audio) / _RATE
        segs = [
            {"text": "a ", "start": 0.0, "end": first_end},
            {"text": "b", "start": first_end, "end": secs},
        ]
        return FakeSTTOutput(text="ab", language="en", segments=segs)

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["l"] = fake_loader(output_fn=output_fn)
        session = _session(
            WhisperTiny(), redecode_interval_s=11.0, settle_margin_s=0.4, max_window_s=10.0
        )
        return await _drive(loop, session, [_pcm(_tone(11.0))], chunk_s=11.0)

    timed = run_virtual(main)
    sizes = _sizes(box["l"])
    assert len(sizes) <= 5, sizes
    assert sum(sizes) <= 40.0, sizes
    assert timed[-1][1].audio_processed_until == pytest.approx(11.0)


def test_qwen_with_a_short_chunk_duration_commits_at_its_chunk_boundaries(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # chunk_duration 4 s under a 10 s cap: Qwen3-ASR then returns inner
    # boundaries, and the cap commits at them instead of at a pause.
    from std_mlx_audio import MlxAudioParams

    def output_fn(audio: Any, kwargs: dict[str, Any]) -> FakeSTTOutput:
        secs = len(audio) / _RATE
        step = kwargs["chunk_duration"]
        bounds = [*np.arange(0.0, secs, step).tolist(), secs]
        segs = [{"text": "c ", "start": a, "end": b} for a, b in pairwise(bounds) if b > a]
        return FakeSTTOutput(text="c" * len(segs), segments=segs)

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=output_fn)
        params = RuntimeParams(provider_params=MlxAudioParams(chunk_duration=4.0))
        session = _session(params=params, redecode_interval_s=10.0, max_window_s=10.0)
        assert session._has_boundaries  # type: ignore[attr-defined]
        return await _drive(loop, session, [_pcm(_tone(10.0))], chunk_s=10.0)

    ends = [e.end for e in _events(run_virtual(main)) if e.type == "final"]
    assert ends[:2] == pytest.approx([4.0, 8.0])  # the prefix's settled chunks


# --------------------------------------------------------------------------- #
# Pacing near real time, the end of the input, and shutdown
# --------------------------------------------------------------------------- #
def test_throughput_near_real_time_does_not_build_a_growing_lag(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # Each decode costs half the window it decodes; with the window decoded again
    # on every pass this is close to the limit of what can keep up.
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_costing(loop, lambda w: 0.5 * w))
        session = _session(redecode_interval_s=0.3, max_window_s=10.0, done_timeout=None)
        return await _drive(loop, session, _chunks(_tone(300.0)))

    timed = run_virtual(main)
    lags = [(t, t - e.audio_processed_until) for t, e in timed if e.type == "partial"]
    early = max(lag for t, lag in lags if t < 150.0)
    late = max(lag for t, lag in lags if t >= 150.0)
    assert late <= early + 5.0, (early, late)  # within one cap decode
    assert max(lag for _, lag in lags) <= 15.0


def test_the_end_of_the_input_during_settling_is_finalized(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
        session = on_virtual_clock(_session(redecode_interval_s=0.3), loop)
        check = _watch_partials(session)
        out: Timed = []
        async with session:

            async def feeder() -> None:
                await session.send_audio(_pcm(_tone(0.3)))
                await asyncio.sleep(0.005)  # inside the first 10 ms settle step
                await session.send_audio(_pcm(_tone(0.2)))
                await session.end_audio()

            task = asyncio.ensure_future(feeder())
            async for event in session:
                out.append((loop.time(), event))
            await task
        check([e for _, e in out])
        return out

    timed = run_virtual(main)
    assert timed[-1][1].type == "done"
    assert timed[-1][1].audio_processed_until == pytest.approx(0.5)
    assert timed[-1][0] <= 0.05


def test_stopping_the_intake_retrieves_a_failure_that_lost_the_race(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    async def main(loop: VirtualTimeLoop) -> asyncio.Task[None]:
        fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
        session = _session()

        async def failed() -> None:
            raise OSError("late failure")

        task = asyncio.ensure_future(failed())
        await asyncio.sleep(0)
        session._intake = task  # type: ignore[attr-defined]
        await session._stop_intake()  # type: ignore[attr-defined]
        return task

    task = run_virtual(main)
    assert task.done() and task._log_traceback is False  # type: ignore[attr-defined]


def test_an_hour_long_pause_meets_the_librarys_done_timeout(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # Default deadlines: nothing arrives and nothing is left to decode, so the
    # base session's done_timeout (300 s) ends the stream, as documented.
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
        session = MlxAudioStreamingSession(Qwen3Asr06B(), RuntimeParams())
        on_virtual_clock(session, loop)
        ledger = _PartialLedger(session)
        out: Timed = []
        async with session:
            await session.send_audio(_pcm(_tone(0.7)))
            async for event in session:
                out.append((loop.time(), event))
        # The library ended the stream; the producer, waiting for audio, was
        # cancelled with the partial still open (the error tells the application).
        ledger.check([e for _, e in out])
        assert ledger.ended == "CancelledError" and list(ledger.shown) == ["seg-0"]
        return out

    timed = run_virtual(main)
    last_at, last = timed[-1]
    assert last.type == "error" and last.code == "done_timeout"
    assert 300.0 <= last_at <= 302.0
    assert any(
        e.type == "partial" and e.audio_processed_until == pytest.approx(0.7) for _, e in timed
    )


# --------------------------------------------------------------------------- #
# A few runs on a real event loop
# --------------------------------------------------------------------------- #
async def test_real_loop_session_completes_and_stops_its_intake(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
    session = _session(redecode_interval_s=0.3)
    check = _watch_partials(session)
    events: list[TranscriptionEvent] = []
    async with session:
        session.feed(_chunks(_tone(1.0)))
        async for event in session:
            events.append(event)
    check(events)
    assert events[-1].type == "done"
    assert events[-1].audio_processed_until == pytest.approx(1.0)
    assert session._intake is not None and session._intake.done()  # type: ignore[attr-defined]


async def test_real_loop_done_timeout_ends_a_silent_session(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
    session = MlxAudioStreamingSession(Qwen3Asr06B(), RuntimeParams(), done_timeout=0.2)
    check = _watch_partials(session)
    events: list[TranscriptionEvent] = []
    async with session:
        await session.send_audio(_pcm(_tone(0.1)))
        async for event in session:
            events.append(event)
    check(events)
    assert events[-1].type == "error" and events[-1].code == "done_timeout"


def test_a_steady_stream_of_small_chunks_cannot_hold_a_decode_back(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # A chunk every 5 ms: no 10 ms settle step is ever quiet, so settling stops
    # after its step limit and the pass goes ahead.
    calls: list[float] = []

    async def main(loop: VirtualTimeLoop) -> Timed:
        def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
            calls.append(loop.time())
            return _spanning(audio)

        fake_loader(output_fn=output_fn)
        session = _session(redecode_interval_s=0.3, max_window_s=None)
        return await _drive(loop, session, _chunks(_tone(2.0), 0.005), chunk_s=0.005)

    run_virtual(main)
    assert calls[0] <= 0.3 + 0.06


def test_a_backlog_decoded_with_the_intake_paused_is_not_a_stall(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # 120 s queued behind a two-chunk library queue, 2 s per decode, and a 5 s
    # done_timeout: the intake reads nothing for long stretches, but every head
    # decode emits an event, which the base counts as activity.
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_costing(loop, lambda _w: 2.0))
        session = _session(max_window_s=10.0, audio_queue_maxsize=2, done_timeout=5.0)
        return await _drive_queued(session, _chunks(_tone(120.0)))

    timed = run_virtual(main)
    assert timed[-1][1].type == "done"
    assert timed[-1][1].audio_processed_until == pytest.approx(120.0)


def _round_trip_losers(lo: float, hi: float) -> list[float]:
    """Ends ``k * 0.02`` in ``[lo, hi)`` that come back smaller via a sample count."""
    ends = [k * 0.02 for k in range(round(lo / 0.02), round(hi / 0.02))]
    return [e for e in ends if round(e * _RATE) / _RATE < e]


@pytest.mark.parametrize("path", ["settled", "inner"])
@pytest.mark.parametrize("before", [0, 2])
def test_a_boundary_end_that_does_not_survive_a_sample_round_trip_is_committed(
    fake_loader: Callable[..., FakeLoader], path: str, before: int
) -> None:
    # The boundary segment ends at a time like 276 * 0.02 == 5.5200000000000005,
    # which becomes 5.52 through a sample count. Choosing the finalized segments
    # by comparing with that time left the boundary segment out: its text was
    # lost, or (with nothing else to finalize) the session ended in an error.
    boundary = _round_trip_losers(5.5, 6.5)[0]
    earlier = _round_trip_losers(1.0, 5.0)[:before]
    ends = [*earlier, boundary]

    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        secs = len(audio) / _RATE
        bounds = [0.0, *ends, secs] if secs > boundary else [0.0, secs]
        segs = [
            {"text": f"T{i} ", "start": a, "end": b} for i, (a, b) in enumerate(pairwise(bounds))
        ]
        return FakeSTTOutput(text="".join(s["text"] for s in segs), language="en", segments=segs)

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=output_fn)
        margin = 2.0 if path == "settled" else 5.0  # 5 s: the boundary has not settled
        session = _session(
            WhisperTiny(), redecode_interval_s=10.0, settle_margin_s=margin, max_window_s=10.0
        )
        return await _drive(loop, session, [_pcm(_tone(10.0))], chunk_s=10.0)

    events = _events(run_virtual(main))
    assert events[-1].type == "done"
    finals = [e for e in events if e.type == "final"]
    committed = [e.text for e in finals[: len(ends)]]
    assert committed == [f"T{i} " for i in range(len(ends))]
    assert finals[len(ends) - 1].end == pytest.approx(boundary)
    _assert_contract(events, WhisperTiny)


def test_a_backlog_is_absorbed_with_finals_and_no_partial_until_caught_up(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # 60 s arrives at once behind a two-chunk library queue: each pass keeps
    # taking the refilled inbox and committing heads, and decodes the rest of the
    # window (the partial) only once the backlog is absorbed.
    peaks: dict[str, float] = {}
    box: dict[str, Any] = {}

    def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
        _peaks(box["s"], peaks)
        box["loop"].advance(0.5)
        return _spanning(audio)

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["loop"] = loop
        fake_loader(output_fn=output_fn)
        box["s"] = _session(max_window_s=10.0, audio_queue_maxsize=2)
        return await _drive_queued(box["s"], _chunks(_tone(60.0)))

    timed = run_virtual(main)
    events = _events(timed)
    first_partial = next(i for i, e in enumerate(events) if e.type in ("partial", "done"))
    finals_before = [e for e in events[:first_partial] if e.type == "final"]
    assert len(finals_before) >= 4  # heads committed back to back
    assert peaks["window"] <= 20.0 + 1e-9
    assert events[-1].type == "done"
    assert events[-1].audio_processed_until == pytest.approx(60.0)
    _assert_contract(events)


def test_the_end_of_the_input_during_a_backlog_round_still_reaches_the_final_pass(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # A 25 s chunk, then the end of the input while its heads are committed.
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_costing(loop, lambda _w: 1.0))
        session = _session(redecode_interval_s=0.3, max_window_s=10.0)
        return await _drive(loop, session, [_pcm(_tone(25.0)), _pcm(_tone(3.0))], chunk_s=1.0)

    timed = run_virtual(main)
    assert timed[-1][1].type == "done"
    assert timed[-1][1].audio_processed_until == pytest.approx(28.0)


def test_the_virtual_loop_fails_a_zero_cost_spin(monkeypatch: pytest.MonkeyPatch) -> None:
    # The review's mutation probe: "advancing" by zero and yielding forever must
    # not pass for progress.
    monkeypatch.setattr(VirtualTimeLoop, "_SPIN_BUDGET", 10)

    async def main(loop: VirtualTimeLoop) -> None:
        for _ in range(100):
            loop.advance(0.0)
            await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="spins"):
        run_virtual(main)


# --------------------------------------------------------------------------- #
# Round-3 review: rest bound, classification, empty output, intake failures
# --------------------------------------------------------------------------- #
def _late_after_end(timed: Timed, seconds: float) -> float:
    return timed[-1][0] - seconds


@pytest.mark.parametrize(
    ("cost", "seconds", "late_limit"),
    [
        (lambda w: 0.8 * w * w, 120.0, 1.0),  # the review's nonlinear counterexample
        (lambda w: 0.6 * w, 300.0, 1.5),  # a linear cost with the same coarse chunks
    ],
)
def test_one_second_chunks_keep_a_schedule_that_keeps_up(
    fake_loader: Callable[..., FakeLoader],
    cost: Callable[[float], float],
    seconds: float,
    late_limit: float,
) -> None:
    # One-second chunks, interval 1 s, settle margin 0: each window settles at
    # once. Resting longer than a pass gained on the input made the next window
    # bigger; with a superlinear cost that ran away (366 s late after 120 s).
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_costing(loop, cost))
        session = _session(
            redecode_interval_s=1.0, settle_margin_s=0.0, max_window_s=10.0, done_timeout=None
        )
        return await _drive(loop, session, _chunks(_tone(seconds), 1.0), chunk_s=1.0)

    timed = run_virtual(main)
    assert _late_after_end(timed, seconds) <= late_limit
    _assert_contract(_events(timed))


def test_cohere_is_classified_by_its_default_decode_and_takes_pause_commits(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # Cohere decodes without VAD by default: one segment up to its 35 s clip.
    # So under a 10 s cap it has no inner boundaries, and pause commits apply.
    from std_mlx_audio import CohereAsr

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
        session = _session(
            CohereAsr(model_path=FAKE_SNAPSHOT_DIR),
            redecode_interval_s=0.5,
            max_window_s=10.0,
            commit_pause_s=0.5,
        )
        assert session._has_boundaries is False  # type: ignore[attr-defined]
        audio = np.concatenate([_tone(2.0), _silence(1.0), _tone(2.0)])
        return await _drive(loop, session, _chunks(audio))

    finals = [e for e in _events(run_virtual(main)) if e.type == "final"]
    assert finals[0].end is not None and 2.2 <= finals[0].end <= 2.6


def _native_empty(audio: Any, _kwargs: Any = None) -> FakeSTTOutput:
    """Qwen3-ASR's native shape for silence: a chunk segment with empty text."""
    samples = np.asarray(audio)
    secs = samples.size / _RATE
    text = "" if not np.any(np.abs(samples) > 1e-3) else f"[{secs:.2f}s]"
    return FakeSTTOutput(text=text, segments=[{"text": text, "start": 0.0, "end": secs}])


def test_native_empty_segments_are_not_content_on_any_path(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # Silence on the normal pass, at the cap (head commits), in backlog rounds
    # and on the final pass: no partial or final with empty text appears.
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_native_empty)
        session = _session(redecode_interval_s=0.5, max_window_s=5.0, audio_queue_maxsize=2)
        return await _drive_queued(session, _chunks(_silence(40.0)))

    events = _events(run_virtual(main))
    assert [e.type for e in events if e.type in ("partial", "final")] == []
    assert events[-1].type == "done" and events[-1].audio_processed_until == pytest.approx(40.0)


@pytest.mark.parametrize(
    ("engine_cls", "prompt_key"), [(Qwen3Asr06B, "system_prompt"), (WhisperTiny, "initial_prompt")]
)
def test_prompt_echo_is_empty_output_through_windows_and_backlog(
    fake_loader: Callable[..., FakeLoader], engine_cls: Any, prompt_key: str
) -> None:
    prompt = "Vocabulary: Standard ASR, streaming transcription, and portable context."
    calls = 0
    box: dict[str, FakeLoader] = {}

    def output_fn(audio: Any, kwargs: dict[str, Any]) -> FakeSTTOutput:
        nonlocal calls
        calls += 1
        assert kwargs[prompt_key] == prompt
        text = "hello" if calls == 1 else prompt
        return FakeSTTOutput(
            text=text, segments=[{"text": text, "start": 0.0, "end": len(audio) / _RATE}]
        )

    async def main(loop: VirtualTimeLoop) -> Timed:
        box["loader"] = fake_loader(output_fn=output_fn)
        session = _session(
            engine_cls(), params=RuntimeParams(prompt=prompt), redecode_interval_s=0.5
        )
        # Live windows show text, clear it once, then emit progress. The queued
        # tail crosses several caps and must close the open partial only once.
        chunks = [_pcm(_tone(0.5))] * 3 + [_pcm(_tone(40.0))]
        return await _drive(loop, session, chunks, chunk_s=0.5)

    events = _events(run_virtual(main))
    content = [(e.type, e.text, e.segment_id) for e in events if e.type in ("partial", "final")]
    assert content == [
        ("partial", "hello", "seg-0"),
        ("partial", "", "seg-0"),
        ("final", "", "seg-0"),
    ]
    assert events[2].type == "progress"
    assert events[-1].type == "done"
    assert events[-1].audio_processed_until == pytest.approx(41.5)
    sizes = _sizes(box["loader"])
    assert max(sizes) <= 10.0
    assert sum(size >= 5.0 for size in sizes) >= 4
    _assert_contract(events, engine_cls)


def test_native_empty_segments_close_an_open_partial_with_one_empty_final(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    events, _ = _run_cut(
        fake_loader,
        np.concatenate([_silence(8.0), _tone(4.0)]),
        output_fn=lambda a, _k: _native_empty(a),
    )
    seg0 = [e for e in events if e.segment_id == "seg-0"]
    assert seg0[0].type == "partial" and seg0[0].text
    assert seg0[-1].type == "final" and seg0[-1].text == ""
    empties = [e for e in events if e.type in ("partial", "final") and not e.text]
    assert empties == [seg0[-1]]


def test_max_idle_ends_a_silent_session_that_returns_native_empty_segments(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_native_empty)
        session = _session(redecode_interval_s=0.5, max_idle=2.0)
        return await _drive(loop, session, _chunks(_silence(10.0)))

    timed = run_virtual(main)
    last_at, last = timed[-1]
    assert last.type == "error" and last.code == "stream_stalled"
    assert last_at == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize(
    ("deadline", "code", "queue", "ends_at"),
    [
        ("max_idle", "stream_stalled", 2, 6.02),
        ("max_session_seconds", "session_timeout", 2, 6.02),
        ("max_idle", "stream_stalled", None, 12.0),
        ("max_session_seconds", "session_timeout", None, 6.0),
    ],
)
def test_deadlines_still_apply_during_a_backlog(
    fake_loader: Callable[..., FakeLoader],
    deadline: str,
    code: str,
    queue: int | None,
    ends_at: float,
) -> None:
    # 120 s queued, 2 s per decode, silence (no content), a 5 s deadline. The
    # library checks deadlines in the consumer's loop: max_session_seconds each
    # time it receives an event, max_idle only when its wait for the next event
    # times out, and the decode blocks the loop. With a two-chunk queue the
    # intake refills the inbox only by taking turns with the feeder, two chunks
    # at a time, so the inbox is not yet full when the next pass starts;
    # settling then waits 10 ms of real time, and the consumer times out
    # during that wait. With the default queue the intake reads the queued
    # chunks without waiting, the inbox is full when the pass starts, settling
    # is skipped, and the one zero-time yield before each decode does not let
    # the consumer finish timing out: three more decodes run first. These are
    # observations of this schedule, not bounds (see "max_idle" in DESIGN.md).
    async def main(loop: VirtualTimeLoop) -> Timed:
        def output_fn(audio: Any, kwargs: dict[str, Any]) -> FakeSTTOutput:
            loop.advance(2.0)
            return _native_empty(audio)

        fake_loader(output_fn=output_fn)
        sizing = {} if queue is None else {"audio_queue_maxsize": queue}
        session = _session(max_window_s=10.0, **sizing, **{deadline: 5.0})
        return await _drive_queued(session, _chunks(_silence(120.0)))

    timed = run_virtual(main)
    assert timed[-1][1].type == "error" and timed[-1][1].code == code
    assert timed[-1][0] == pytest.approx(ends_at, abs=0.005)


def test_an_intake_failure_during_a_long_pass_ends_the_session_at_once(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The review's probe: one 120 s chunk, then the source fails; cap 10 s and
    # one second per decode. Checked before every decode, not once per pass.
    calls: list[float] = []

    async def source() -> AsyncIterator[bytes]:
        yield _pcm(_tone(120.0))
        raise OSError("microphone unplugged")

    async def main(loop: VirtualTimeLoop) -> list[TranscriptionEvent]:
        def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
            calls.append(loop.time())
            loop.advance(1.0)
            return _spanning(audio)

        fake_loader(output_fn=output_fn)
        session = on_virtual_clock(_session(max_window_s=10.0), loop)
        ledger = _PartialLedger(session)
        out: list[TranscriptionEvent] = []
        async with session:
            session.feed(source())
            async for event in session:
                out.append(event)
        assert session._intake is not None and session._intake.done()  # type: ignore[attr-defined]
        ledger.check(out)
        assert ledger.ended == "_InputSourceError"  # the library wraps the source's OSError
        return out

    events = run_virtual(main)
    assert events[-1].type == "error" and events[-1].code == "input_source_error"
    assert len(calls) <= 1


def test_leaving_the_session_during_a_long_pass_stops_the_intake(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    async def main(loop: VirtualTimeLoop) -> bool:
        fake_loader(output_fn=_costing(loop, lambda _w: 1.0))
        session = on_virtual_clock(_session(max_window_s=10.0, audio_queue_maxsize=2), loop)
        ledger = _PartialLedger(session)
        seen: list[TranscriptionEvent] = []
        async with session:
            session.feed(_chunks(_tone(120.0)))
            async for event in session:
                seen.append(event)
                if event.type == "final":
                    break
        ledger.check(seen)
        assert ledger.ended == "CancelledError"
        intake = session._intake  # type: ignore[attr-defined]
        return intake is not None and intake.done()

    assert run_virtual(main)


class _TrackedInbox(bytearray):
    """An inbox that records its largest size whenever the intake grows it."""

    peak = 0

    def extend(self, data: Any) -> None:  # type: ignore[override]
        super().extend(data)
        type(self).peak = max(type(self).peak, len(self))


@pytest.mark.parametrize(
    ("cap", "chunk_s", "bound_s"),
    [(10.0, 0.1, 10.0 + 0.1), (None, 0.1, 30.0 + 0.1), (10.0, 25.0, 25.0)],
)
def test_the_inbox_high_water_mark_is_the_bound_plus_one_chunk(
    fake_loader: Callable[..., FakeLoader], cap: float | None, chunk_s: float, bound_s: float
) -> None:
    # Measured where the intake grows the inbox, not at decode time. A chunk
    # larger than the bound cannot be refused, so it sets the high-water mark.
    _TrackedInbox.peak = 0

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_costing(loop, lambda _w: 0.5))
        session = _session(max_window_s=cap, audio_queue_maxsize=2)
        session._inbox = _TrackedInbox()  # type: ignore[attr-defined]
        return await _drive_queued(session, _chunks(_tone(120.0), chunk_s))

    timed = run_virtual(main)
    assert timed[-1][1].type == "done"
    assert _TrackedInbox.peak / (2 * _RATE) <= bound_s + 1e-9
    assert _TrackedInbox.peak / (2 * _RATE) >= min(bound_s, 10.0) - chunk_s - 1e-9


def test_words_running_past_the_audio_are_clamped_even_when_their_segment_fits() -> None:
    from standard_asr.contract.results import Segment, Word

    from std_mlx_audio._streaming import _clamp

    seg = Segment(start=0.0, end=0.5, text="hi", words=[Word(text="hi", start=0.1, end=0.8)])
    (out,) = _clamp([seg], 8000)
    assert out.end == 0.5 and out.words is not None and out.words[0].end == 0.5


def test_a_cap_cut_never_lands_below_half_the_cap(fake_loader: Callable[..., FakeLoader]) -> None:
    # A tone that gets steadily louder (no quiet point; the least loud point of
    # any range is its earliest frame) and a cap of 10.02 s: the cut may not fall
    # below half the cap, 5.01 s.
    ramp = np.linspace(0.2, 1.0, 12 * _RATE, dtype=np.float32)
    events, _ = _run_cut(fake_loader, _tone(12.0) * ramp, max_window_s=10.02)
    first = next(e for e in events if e.type == "final")
    assert first.end is not None and first.end >= 5.01


def _withdrawing(after_calls: int) -> Callable[..., Any]:
    """A decoder that recognizes text for its first calls, then withdraws it."""
    calls = {"n": 0}

    def output_fn(audio: Any, _kwargs: Any = None) -> FakeSTTOutput:
        calls["n"] += 1
        secs = len(audio) / _RATE
        text = "hello" if calls["n"] <= after_calls else ""
        return FakeSTTOutput(text=text, segments=[{"text": text, "start": 0.0, "end": secs}])

    return output_fn


def test_withdrawn_text_is_cleared_once_and_max_idle_still_fires(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # Text, then the model withdraws it: one partial with empty text clears what
    # the application shows; further empty results are progress only, so
    # max_idle still ends a session that hears nothing.
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_withdrawing(2))
        session = _session(redecode_interval_s=0.5, max_window_s=None, max_idle=3.0)
        return await _drive(loop, session, _chunks(np.concatenate([_tone(1.0), _silence(20.0)])))

    events = _events(run_virtual(main))
    content = [(e.type, e.text) for e in events if e.type in ("partial", "final")]
    assert content == [("partial", "hello"), ("partial", "hello"), ("partial", "")]
    clear_at = next(i for i, e in enumerate(events) if e.type == "partial" and not e.text)
    assert all(e.type == "progress" for e in events[clear_at + 1 : -1])
    assert events[-1].type == "error" and events[-1].code == "stream_stalled"


def test_a_cleared_partial_is_still_closed_by_the_final_pass(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_withdrawing(1))
        session = _session(redecode_interval_s=0.5, max_window_s=None)
        return await _drive(loop, session, _chunks(_tone(2.0)))

    events = _events(run_virtual(main))
    content = [(e.type, e.segment_id, e.text) for e in events if e.type in ("partial", "final")]
    assert content == [
        ("partial", "seg-0", "hello"),
        ("partial", "seg-0", ""),
        ("final", "seg-0", ""),
    ]
    assert events[-1].type == "done"
    _assert_contract(events)


def test_a_head_commit_that_decodes_to_nothing_closes_the_partial_without_a_clearing_partial(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The window decodes to text; the head cut at the cap decodes to nothing:
    # the open partial is closed by one empty final, with no clearing partial.
    events, _ = _run_cut(
        fake_loader, np.concatenate([_silence(8.0), _tone(4.0)]), output_fn=_native_empty
    )
    seg0 = [(e.type, e.text) for e in events if e.segment_id == "seg-0"]
    assert seg0[-1] == ("final", "")
    assert ("partial", "") not in seg0


def test_audio_read_during_a_pass_is_decoded_once_it_has_aged(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # Passes with several decodes (head commits at a 1 s cap, 1 s per decode)
    # read small chunks at their yields. The client then sends nothing for a
    # minute without ending the input, so neither a later chunk nor the end of
    # the input can trigger the last decode: only the age deadline can, about
    # one interval after the leftover audio arrived.
    calls: list[tuple[float, float]] = []

    async def main(loop: VirtualTimeLoop) -> Timed:
        def output_fn(audio: Any, _kwargs: dict[str, Any]) -> FakeSTTOutput:
            calls.append((loop.time(), len(audio) / _RATE))
            loop.advance(1.0)
            return _spanning(audio)

        fake_loader(output_fn=output_fn)
        session = _session(redecode_interval_s=1.0, max_window_s=1.0, done_timeout=None)
        chunks = _chunks(_tone(6.1), 0.3)
        return await _drive(loop, session, chunks, chunk_s=0.3, end_at=70.0)

    timed = run_virtual(main)
    covered = [t for t, e in timed if e.audio_processed_until == pytest.approx(6.1)]
    assert covered and covered[0] < 12.0, covered[:1]  # long before the end at 70 s
    assert max(t for t, _ in calls[:-1]) < 12.0  # no decode waited for the end
    assert timed[-1][1].type == "done"
    _assert_contract(_events(timed))


def test_a_pass_is_due_at_once_for_audio_already_older_than_the_interval(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # Audio read during a long pass can be older than the interval by the time
    # the pass ends; the wait must not start a new interval for it.
    async def main(loop: VirtualTimeLoop) -> float:
        fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
        session = _session(redecode_interval_s=1.5)
        loop.advance(10.0)
        session._inbox.extend(_pcm(_tone(0.1)))  # type: ignore[attr-defined]
        session._oldest_at = loop.time() - 2.0  # type: ignore[attr-defined]
        started = loop.time()
        await session._wait_until_due()  # type: ignore[attr-defined]
        return loop.time() - started

    assert run_virtual(main) < 0.001


def _text_then_empty(
    segments_for: Callable[[float], list[tuple[float, float]]],
) -> Callable[..., Any]:
    """A decoder that hears "hello" on its first call, then only empty segments."""
    calls = {"n": 0}

    def output_fn(audio: Any, _kwargs: Any = None) -> FakeSTTOutput:
        calls["n"] += 1
        secs = len(audio) / _RATE
        if calls["n"] == 1:
            segs = [{"text": "hello", "start": 0.0, "end": secs}]
        else:
            segs = [{"text": "", "start": a, "end": b} for a, b in segments_for(secs)]
        return FakeSTTOutput(text=segs[0]["text"], language="en", segments=segs)

    return output_fn


def test_an_empty_boundary_commit_closes_the_open_partial(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The review's probe: Whisper (boundaries), "hello" shown, then a silent
    # backlog whose decodes return empty segments [0, 0.6 w] and [0.6 w, w].
    # The first boundary commit must close seg-0 at once.
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_text_then_empty(lambda w: [(0.0, 0.6 * w), (0.6 * w, w)]))
        session = _session(WhisperTiny(), redecode_interval_s=0.5, max_window_s=10.0)
        chunks = [_pcm(_tone(0.5)), _pcm(_silence(120.0))]
        return await _drive(loop, session, chunks, chunk_s=0.5)

    events = _events(run_virtual(main))
    seg0 = [(e.type, e.text) for e in events if e.segment_id == "seg-0"]
    assert seg0 == [("partial", "hello"), ("final", "")]
    assert events[-1].type == "done"
    _assert_contract(events, WhisperTiny)


def test_a_final_pass_with_an_empty_window_still_closes_the_open_partial(
    fake_loader: Callable[..., FakeLoader],
) -> None:
    # The review's probe: a settle margin of 10 us lets an empty segment ending
    # 20 us before the window's end settle the whole window (the trim rounds to
    # the full sample count). The window is then empty when the input ends.
    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_text_then_empty(lambda w: [(0.0, w - 0.00002)]))
        session = _session(
            WhisperTiny(), redecode_interval_s=0.5, max_window_s=None, settle_margin_s=0.00001
        )
        chunks = [_pcm(_tone(0.5)), _pcm(_tone(0.5))]
        return await _drive(loop, session, chunks, chunk_s=0.5, end_at=3.0)

    events = _events(run_virtual(main))
    seg0 = [(e.type, e.text) for e in events if e.segment_id == "seg-0"]
    assert seg0 == [("partial", "hello"), ("partial", ""), ("final", "")]
    assert events[-1].type == "done"


def test_a_failed_virtual_run_cancels_its_tasks_and_keeps_its_error() -> None:
    children: list[asyncio.Task[None]] = []

    async def main(loop: VirtualTimeLoop) -> None:
        children.append(asyncio.ensure_future(asyncio.sleep(100.0)))
        await asyncio.sleep(0)
        raise ValueError("the test's own failure")

    with pytest.raises(ValueError, match="own failure"):
        run_virtual(main)
    assert children[0].cancelled()


# --------------------------------------------------------------------------- #
# The partial ledger catches lifecycle mutations (checks of the checker)
# --------------------------------------------------------------------------- #
def _run_withdrawal_once(fake_loader: Callable[..., FakeLoader]) -> Timed:
    """Text, then withdrawn; the final pass closes it (see the test above)."""

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=_withdrawing(1))
        session = _session(redecode_interval_s=0.5, max_window_s=None)
        return await _drive(loop, session, _chunks(_tone(2.0)))

    return run_virtual(main)


def test_the_ledger_catches_a_closing_final_replaced_by_progress(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mutation: _finalize_all still updates the session's state but returns
    # progress instead of the empty final. The session's own state then looks
    # closed; only the events show the partial was never closed.
    real = MlxAudioStreamingSession._finalize_all  # type: ignore[attr-defined]

    def mutant(self: MlxAudioStreamingSession, *args: Any) -> list[TranscriptionEvent]:
        return [
            TranscriptionEvent.progress(audio_processed_until=e.audio_processed_until)
            if e.type == "final" and not e.text
            else e
            for e in real(self, *args)
        ]

    monkeypatch.setattr(MlxAudioStreamingSession, "_finalize_all", mutant)
    with pytest.raises(AssertionError, match="seg-0 is still open at done"):
        _run_withdrawal_once(fake_loader)


def test_the_ledger_catches_a_duplicated_clearing_partial(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mutation: every clearing partial is emitted twice. The library coalesces
    # the duplicate before delivery, so only the raw events show it.
    real = MlxAudioStreamingSession._build_events  # type: ignore[attr-defined]

    def mutant(self: MlxAudioStreamingSession, *args: Any) -> list[TranscriptionEvent]:
        out: list[TranscriptionEvent] = []
        for e in real(self, *args):
            out.append(e)
            if e.type == "partial" and not e.text:
                out.append(e)
        return out

    monkeypatch.setattr(MlxAudioStreamingSession, "_build_events", mutant)
    with pytest.raises(AssertionError, match="seg-0: an empty partial clears nothing"):
        _run_withdrawal_once(fake_loader)


def test_the_ledger_catches_a_producer_that_skips_the_final_pass(
    fake_loader: Callable[..., FakeLoader], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mutation: _produce returns as soon as the input has ended, before the
    # final pass, so the library adds the done itself. The partial for seg-0 is
    # still open, and the session's own state agrees with the ledger.
    async def mutant(self: MlxAudioStreamingSession) -> AsyncIterator[TranscriptionEvent]:
        self._engine.ensure_loaded()  # type: ignore[attr-defined]
        self._intake = asyncio.ensure_future(self._take_in())  # type: ignore[attr-defined]
        try:
            while True:
                await self._wait_until_due()  # type: ignore[attr-defined]
                if not self._input_ended:  # type: ignore[attr-defined]
                    await self._settle_intake()  # type: ignore[attr-defined]
                self._raise_intake_failure()  # type: ignore[attr-defined]
                if self._input_ended:  # type: ignore[attr-defined]
                    return
                async for event in self._decode_pass(want_words=False, final_pass=False):  # type: ignore[attr-defined]
                    yield event
        finally:
            await self._stop_intake()  # type: ignore[attr-defined]

    async def main(loop: VirtualTimeLoop) -> Timed:
        fake_loader(output_fn=lambda audio, _kw: _spanning(audio))
        session = _session(redecode_interval_s=0.3)
        return await _drive(loop, session, _chunks(_tone(0.5)))

    monkeypatch.setattr(MlxAudioStreamingSession, "_produce", mutant)
    with pytest.raises(AssertionError, match=r"delivered done with \['seg-0'\] still open"):
        run_virtual(main)
