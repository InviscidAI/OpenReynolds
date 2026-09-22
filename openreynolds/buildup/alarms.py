"""Three named terminal states, two of which nothing was watching for.

`wedged` is the one every watcher has: the process is stuck or dead. The other two are
the ones the last round missed, and missing them cost three full clocks:

* **`no-progress`** -- turns advancing, executed-step count flat. The run replies and
  runs nothing. It was mistaken for ordinary budget exhaustion three times in a row
  because the only thing being counted was time.
* **`starved`** -- consecutive turns ending `max_tokens` with no text block at all.
  Adaptive thinking is spent from the same allowance as the words, so a hard prompt can
  reason through the whole reply budget and never open a text block. Measured on the
  Tesla valve: sixteen turns of sixteen, zero characters of text, nothing ran.

`K` is small because the failure was not marginal -- it was eight turns for eight. Three
consecutive turns of either is a run that is not going to recover by turn thirty, and
the cheapest thing to do with the clock is give it back with the reason attached rather
than let it run out and report `time`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .heartbeat import Beat

HEARTBEAT_STALE_S = 420.0
"""How old the last beat may be before the run counts as wedged.

Longer than one cell's window (`STEP_TIMEOUT_S`, 240 s) plus a model turn, because a
cell that outruns its window is polled at the next step and is a slow run rather than a
dead one. Shorter than the run's whole budget, or the alarm is just the clock."""

K = 3
"""Consecutive turns before `no-progress` or `starved` fires."""

PHASE_SLACK_S = 120.0
"""What a declared long operation gets on top of the time it said it would take.

A `checkMesh` that declares 600 s and takes 650 is slow, not wedged; one that is still
silent at 720 has stopped being a check. The slack is the model-call latency on either
side of it, and nothing more generous than that."""


@dataclass(frozen=True)
class Alarm:
    """A named terminal state, with the evidence that raised it."""

    name: str
    why: str
    turn: int = 0

    def as_dict(self) -> dict[str, object]:
        return {"alarm": self.name, "why": self.why, "turn": self.turn}


def evaluate(beats: Sequence[Beat], *, now: float, started_at: float,
             running: bool = True, stale_s: float = HEARTBEAT_STALE_S,
             k: int = K, steps_seen: int = 0) -> Alarm | None:
    """The first alarm these beats raise, or None if the run is behaving.

    `running` is the pidfile's answer, never a command-line match. A run whose process
    is gone and whose beats stop is finished, not wedged -- the supervisor reads its
    record for the ending; only a live-but-silent process is wedged.

    `steps_seen` is what the run has executed according to its own trail on disk, which
    the supervisor can read without the heartbeat. It is what separates the two ways of
    having no beats.
    """
    if not beats:
        if running and now - started_at > stale_s:
            # **Never beat at all, while plainly doing work, is our failure and not the
            # desk's.** The two used to be one alarm, and for a day they were the same
            # sentence: `Heartbeat.beat` would not accept the `phase` the loop had begun
            # sending, every call raised, `_beat` swallowed it because the watcher may not
            # end a run, and no heartbeat file was ever created. The supervisor then
            # killed every run slower than `stale_s` -- eight of twenty-six in one sweep,
            # at turn 0 and steps 0 and $0.00, with fifteen cells of real work in their
            # `cells.log`. Reported as `wedged`, which reads as a desk that hung.
            #
            # A desk that hung and a watcher that went blind are different findings and
            # only one of them is about the desk. So a run with steps on disk and no beats
            # anywhere is `unobserved`, which `record.TERMINAL` carries and a sweep
            # discards the way it discards `contaminated`: not averaged in, not retried
            # quietly, and counted out loud.
            if steps_seen > 0:
                return Alarm("unobserved",
                             f"no beat in {now - started_at:.0f}s while {steps_seen} "
                             "steps ran -- the run is working and nothing is watching it")
            return Alarm("wedged", f"no beat in {now - started_at:.0f}s of running")
        return None

    last = beats[-1]
    age = now - (last.at or started_at)
    # A run that has declared what it is doing gets the time it said it would take. The
    # loop is silent for real reasons -- a per-region `checkMesh`, a recovery replay -- and
    # a threshold that only knew about model turns would kill a run for being checked.
    allowance = stale_s if last.phase == "turn" else max(
        stale_s, float(last.expect_s or 0.0) + PHASE_SLACK_S)
    if running and age > allowance:
        return Alarm("wedged",
                     f"last beat is {age:.0f}s old"
                     + (f" during {last.phase}, which allowed {allowance:.0f}s"
                        if last.phase != "turn" else ""), last.turn)

    # Progress is a question about turns, so the marks that are only liveness are not
    # counted: three `checkMesh` marks in a row are not three turns that ran nothing.
    recent = [beat for beat in beats if beat.phase == "turn"][-k:]
    if len(recent) < k:
        return None
    turns_moved = recent[-1].turn > recent[0].turn

    # `starved` is asked first because it is the narrower diagnosis of the same
    # symptom: a run whose replies are all reasoning executes nothing, so its step count
    # is flat too and `no-progress` would also be true. Naming it `no-progress` would be
    # correct and useless -- it was the `max_tokens`-with-no-text pair that made the
    # failure obvious in one look, and that is the pair worth reporting.
    if turns_moved and all(beat.stop_reason == "max_tokens" and beat.text_chars == 0
                           for beat in recent):
        return Alarm("starved",
                     f"{k} turns to {recent[-1].turn} ended max_tokens with no text",
                     recent[-1].turn)
    if turns_moved and len({beat.steps for beat in recent}) == 1:
        return Alarm("no-progress",
                     f"{k} turns to {recent[-1].turn} with the step count flat at "
                     f"{recent[-1].steps}", recent[-1].turn)
    return None
