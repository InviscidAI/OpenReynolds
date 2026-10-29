"""A study's spending ceiling, from the agent's side.

The workspace service gives every study a ceiling and pauses it at 80%: from then on
the study's new sandboxes, model calls and GPU solves answer `409 study_paused` with
`{error, message, study_id, spent_cents, cap_cents}`, and `POST /v1/studies/{id}/cap
{cap_cents}` raises it (owner only; it must exceed what is spent, and one still under
spend/0.8 is taken but answered `paused: true`). Default ceilings are
the service's: 1000 cents for a quick study, 5000 for a full one.

Two things are done with that here, and nothing else:

- **The pause is the person's question.** Spending more of their money is not the
  model's call, so a pause is put to them through the same `Approver` every other
  question goes through, whatever mode the session is in, with what is spent, the
  ceiling, and the estimate given before the first solve. Yes raises the ceiling (to
  double, or to what they typed) and the call that met the pause goes again; no, or
  nobody there to answer, stops the study: the turn ends after the calls in flight, a
  plain final message is shown and kept in the transcript, and nothing more is sent to
  the model until the person speaks (`Loop.blocked_reason`).
- **The first solve is announced.** Before the session's first `job_start` the person
  is told the agent's estimate in dollars and the study's ceiling, as a message -- or
  as part of the job's question, when they chose to approve compute (`modes.PARTIAL`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from .backend.base import BackendError

STUDY_PAUSED = "study_paused"
PAUSE_AT = 0.8
"""The fraction of its ceiling at which the service pauses a study."""

DEFAULT_CAP_CENTS = {"quick": 1000, "full": 5000}
"""The service's default ceilings by study style; a study with no style is full."""

MAX_RAISES_PER_CALL = 3
"""How many times one call may meet a pause and have it lifted before it is given up
on. A raise that does not lift the pause (the service counting spend this side has not
seen) would otherwise ask the person forever about the same call."""


def default_cap_cents(style: str | None) -> int:
    return DEFAULT_CAP_CENTS.get((style or "").strip().lower(), DEFAULT_CAP_CENTS["full"])


def dollars(cents: float) -> str:
    return f"${cents / 100:,.2f}"


@dataclass(frozen=True)
class Paused:
    """What a `409 study_paused` said."""

    study_id: str
    spent_cents: float
    cap_cents: float
    message: str = ""


def paused(exc: BaseException) -> Paused | None:
    """The pause `exc` reports, from a workspace call (`BackendError`) or a model call
    through the service's proxy (`ProviderError`); None for anything else."""
    if isinstance(exc, BackendError):
        if exc.code != STUDY_PAUSED:
            return None
        fields, message = exc.extra, exc.message
    else:
        body = getattr(exc, "body", None)
        if not isinstance(body, dict) or body.get("error") != STUDY_PAUSED:
            return None
        fields, message = body, str(body.get("message") or "")
    try:
        return Paused(
            study_id=str(fields.get("study_id") or ""),
            spent_cents=float(fields.get("spent_cents") or 0),
            cap_cents=float(fields.get("cap_cents") or 0),
            message=message,
        )
    except (TypeError, ValueError):
        return Paused("", 0.0, 0.0, message)


_JOB_PAUSE_AMOUNT = re.compile(
    r"has spent \$([\d,]+(?:\.\d{1,2})?) of (?:its |a )?\$([\d,]+(?:\.\d{1,2})?) ceiling",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class JobPauseResolution:
    """What handling one finished Felix job changed."""

    decision: bool | None = None
    """True when raised, False when stopped, None when no question was needed."""
    handled: bool = False
    """Whether this job's refusal is conclusively accounted for."""
    note: str = ""
    """A fact to append to the job report sent back to the model."""


_RAISED_JOB_NOTE = (
    "The study's ceiling was raised. This refused job is finished; no replacement "
    "solve was started."
)
_CLEARED_JOB_NOTE = (
    "The study is no longer paused at its spending ceiling. This refused job is "
    "finished; no replacement solve was started."
)
_CHANGED_JOB_NOTE = (
    "The study's ceiling changed after this job was refused. This old refusal is no "
    "longer current; the study is paused again under its current ceiling. No "
    "replacement solve was started."
)


def _amount_cents(value: str) -> int | None:
    """A printed dollar amount in exact cents, without binary-float rounding."""
    try:
        cents = Decimal(value.replace(",", "")) * 100
    except (InvalidOperation, ValueError):
        return None
    return int(cents) if cents == cents.to_integral_value() else None


def _row_cents(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def job_pause(ctx: Any, facts: Any) -> tuple[Paused | None, bool, str]:
    """A structured Felix rejection against the service's current study row.

    `felix run` is itself a workspace job. A GPU solve refused by foamd therefore
    exits 1 and prints a final JSON status instead of raising the `BackendError` that
    `tools._handled` sees. The status proves this was a rejected Felix solve, rather
    than arbitrary output containing `study_paused`. Its ERROR line identifies the
    spend and ceiling that refusal saw; matching both against the live row prevents
    an old refusal from being mistaken for a later pause.

    The bool says the job is conclusively accounted for. False leaves it eligible for
    a later read because the log or study row was incomplete. The final string tells
    the model when a real refusal is stale because the ceiling changed elsewhere.
    """
    status = getattr(facts, "status", None)
    if not isinstance(status, dict) or status.get("state") != "rejected":
        return None, False, ""
    if str(status.get("error") or "").strip() != "409 study_paused":
        return None, False, ""
    message = "\n".join(getattr(facts, "errors", []) or [])
    amounts = _JOB_PAUSE_AMOUNT.findall(message)
    if not amounts:
        return None, False, ""
    refused_spent = _amount_cents(amounts[-1][0])
    refused_cap = _amount_cents(amounts[-1][1])
    if refused_spent is None or refused_cap is None:
        return None, False, ""
    service = getattr(ctx, "ceiling", None)
    row = service.study() if service is not None else None
    if row is None:
        return None, False, ""
    if not isinstance(row, dict):
        return None, False, ""
    spent = _row_cents(row.get("spent_cents"))
    cap = _row_cents(row.get("cap_cents"))
    paused_now = row.get("paused")
    # A transient ledger failure reports all three as None. It is not evidence that
    # the refusal is stale, so leave the job eligible for the next read.
    if (spent is None or cap is None
            or (paused_now is not True and paused_now is not False)):
        return None, False, ""
    # Spend is monotonic while the workspace is up. The row may therefore be above
    # the amount printed at refusal time, but it cannot be below it and still prove
    # this is the same event.
    if spent < refused_spent:
        return None, False, ""
    if paused_now is False:
        return None, True, _CLEARED_JOB_NOTE
    if cap != refused_cap:
        return None, True, _CHANGED_JOB_NOTE
    return Paused(
        str(row.get("id") or row.get("study_id") or service.study_id),
        float(spent),
        float(cap),
        message,
    ), True, ""


def resolve_job_pause(ctx: Any, facts: Any) -> JobPauseResolution:
    """Resolve a pause carried in Felix job output.

    A current pause is put to the person. A stale, externally cleared refusal is
    returned as a note for the model. Incomplete evidence remains unhandled so a
    later read can try again.
    """
    info, handled, note = job_pause(ctx, facts)
    if info is None:
        return JobPauseResolution(handled=handled, note=note)
    decision = resolve(ctx, info)
    return JobPauseResolution(
        decision=decision,
        handled=handled,
        note=_RAISED_JOB_NOTE if decision else "",
    )


def handle_finished_felix_job(
    ctx: Any, record: Any, status: Any, facts: Any
) -> JobPauseResolution:
    """Account for a failed Felix solve once, for both tool and watch paths."""
    from .felix_progress import is_felix_cmd

    if (record is None
            or getattr(status, "running", False)
            or getattr(status, "exit_code", None) in (None, 0)
            or getattr(record, "ceiling_pause_handled", False)
            or not is_felix_cmd(getattr(record, "cmd", ""))):
        return JobPauseResolution()
    result = resolve_job_pause(ctx, facts)
    if result.handled:
        ctx.store.update_job(record.job_id, ceiling_pause_handled=True)
    return result


class Ceiling:
    """The service's view of one study's ceiling: read it, raise it.

    Holds a `FoamdClient`, or a way to make one the first time it is needed -- a
    session choosing its solver has no client of its own until it has chosen, and
    most sessions never meet their ceiling at all."""

    def __init__(self, client: Any = None, *, connect: Callable[[], Any] | None = None,
                 study_id: str | Callable[[], str | None] = ""):
        self._client = client
        self._connect = connect
        self._owned = False
        self._study_id = study_id

    @property
    def study_id(self) -> str:
        value = self._study_id() if callable(self._study_id) else self._study_id
        return str(value or "")

    def client(self) -> Any:
        if self._client is None and self._connect is not None:
            self._client = self._connect()
            self._owned = True
        return self._client

    def study(self) -> dict[str, Any] | None:
        """The study's record, or None when it cannot be read. Never raises: what it
        feeds is a message, and a message is not worth failing a solve over."""
        if not self.study_id:
            return None
        try:
            client = self.client()
            row = client.get_study(self.study_id) if client is not None else None
        except Exception:  # noqa: BLE001 - see docstring
            return None
        return row if isinstance(row, dict) else None

    def raise_cap(self, study_id: str, cap_cents: int) -> dict[str, Any]:
        return self.client().raise_study_cap(study_id or self.study_id, int(cap_cents))

    def close(self) -> None:
        if self._owned and self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001 - closing down
                pass


# -- before the first solve -------------------------------------------------------------


def estimate_cents(tool_input: dict[str, Any]) -> float | None:
    """The `estimate_usd` a job_start carried, in cents; None when there was none."""
    raw = tool_input.get("estimate_usd")
    try:
        value = float(raw) if raw is not None and raw != "" else None
    except (TypeError, ValueError):
        return None
    return round(value * 100, 2) if value is not None and value >= 0 else None


def preamble(ctx: Any, tool_input: dict[str, Any]) -> str | None:
    """The line said before this session's first solve: the estimate and the ceiling.

    None when there is no ceiling to speak of (no hosted service behind the session)
    or when it has been said already. Records the estimate on `ctx` either way, so a
    later pause can show it."""
    cents = estimate_cents(tool_input)
    if cents is not None:
        ctx.estimate_cents = cents
    ceiling = getattr(ctx, "ceiling", None)
    if ceiling is None or getattr(ctx, "estimated", False):
        return None
    ctx.estimated = True
    if cents is not None:
        said = f"Estimate before the first solve: about {dollars(cents)}."
    else:
        said = "Before the first solve: the agent gave no estimate of its cost."
    row = ceiling.study() or {}
    cap = row.get("cap_cents")
    spent = row.get("spent_cents")
    if not isinstance(cap, (int, float)):
        if "style" not in row:
            return f"{said} This study's ceiling could not be read from the service."
        cap = default_cap_cents(row.get("style"))
    said += (f" This study's ceiling is {dollars(cap)}; it pauses at "
             f"{dollars(cap * PAUSE_AT)}")
    if isinstance(spent, (int, float)):
        said += f" (spent so far {dollars(spent)})"
    if row.get("paused") is True:
        # foamd's `paused` (None when it could not read the ledgers): the solve about
        # to start will meet the pause, so it is said here first.
        said += "; it is paused now, so the solve will ask to raise the ceiling"
    return said + "."


# -- at a pause ---------------------------------------------------------------------------


_AMOUNT = re.compile(r"\$?\s*(\d+(?:\.\d{1,2})?)")
_REFUSING = re.compile(r"^\s*(n|no|nope|stop|don't|do not|cancel)\b", re.IGNORECASE)


def typed_cap_cents(note: str) -> int | None:
    """A new ceiling the person typed, in dollars ("$35", "35", "make it 40"), as
    cents; None when what they said is not one, or says no."""
    if not note or _REFUSING.match(note):
        return None
    found = _AMOUNT.search(note)
    if not found:
        return None
    return int(round(float(found.group(1)) * 100))


def least_useful_cents(spent_cents: float) -> int:
    """The smallest ceiling, in whole dollars, that would not pause again at once:
    80% of it has to be above what is spent."""
    dollars_needed = int(spent_cents / PAUSE_AT // 100) + 1
    return dollars_needed * 100


def _question(info: Paused, ctx: Any, proposed: int, note: str = "") -> tuple[str, str]:
    title = "Study paused at its spending ceiling"
    lines = [
        f"Spent {dollars(info.spent_cents)} of a {dollars(info.cap_cents)} ceiling "
        f"(it pauses at {int(PAUSE_AT * 100)}%).",
    ]
    estimate = getattr(ctx, "estimate_cents", None)
    if estimate is not None:
        lines.append(f"The estimate before the first solve was {dollars(estimate)}.")
    if note:
        lines.append(note)
    lines.append(
        f"yes raises the ceiling to {dollars(proposed)} (it would pause again at "
        f"{dollars(proposed * PAUSE_AT)}); a dollar amount sets a different ceiling; "
        "no stops the study here."
    )
    return title, "\n".join(lines)


def stop_message(info: Paused, *, asked: bool, study_id: str = "", error: str = "") -> str:
    head = (f"This study is paused at its spending ceiling: {dollars(info.spent_cents)} "
            f"spent of {dollars(info.cap_cents)}.")
    if error:
        why = f" The ceiling could not be raised ({error}), so the study has stopped here."
    elif asked:
        why = " The ceiling was not raised, so the study has stopped here."
    else:
        target = study_id or "<study>"
        why = (" Nobody was here to raise the ceiling, so the study has stopped here. "
               f"Raise it from the study page or with POST /v1/studies/{target}/cap, "
               "then resume the study.")
    return (head + why + " Nothing more will run or be sent to the model until you "
            "say something.")


def resolve(ctx: Any, exc: BaseException | Paused) -> bool:
    """Put a pause to the person. True when the ceiling was raised and the paused call
    may go again; False when the study stops, with the reason left in `ctx.stopped`.

    Never raises: a failure to raise the ceiling is one more reason to stop, said."""
    info = exc if isinstance(exc, Paused) else paused(exc)
    if info is None:
        return False
    ceiling = getattr(ctx, "ceiling", None)
    approver = getattr(ctx, "approver", None)
    study_id = info.study_id or (ceiling.study_id if ceiling is not None else "")
    if approver is None or ceiling is None:
        ctx.stopped = stop_message(info, asked=False, study_id=study_id)
        return False
    note = ""
    while True:
        least = least_useful_cents(info.spent_cents)
        proposed = max(int(info.cap_cents * 2), least)
        while True:
            title, detail = _question(info, ctx, proposed, note)
            decision = approver.ask("ceiling", title, detail)
            if decision.approved:
                cents = proposed
            else:
                cents = typed_cap_cents(decision.note)
                if cents is None:
                    ctx.stopped = stop_message(info, asked=True)
                    return False
            if cents >= least:
                break
            note = (f"{dollars(cents)} would pause again at once: the ceiling has to be "
                    f"at least {dollars(least)} for 80% of it to be above what is spent.")
        try:
            answer = ceiling.raise_cap(study_id, cents)
        except Exception as error:  # noqa: BLE001 - see docstring
            ctx.stopped = stop_message(info, asked=True, error=str(error))
            return False
        if not (isinstance(answer, dict) and answer.get("paused") is True):
            break
        # The service takes any ceiling above the spend and says `paused: true` when
        # its own count -- open sandboxes priced to now, calls this side has not seen
        # -- is still past 80% of it. Asked again from that count, rather than sending
        # the paused call into a pause already known to be there.
        spent = answer.get("spent_cents")
        info = Paused(study_id, float(spent) if isinstance(spent, (int, float))
                      else info.spent_cents, float(cents), info.message)
        note = (f"The service took the {dollars(cents)} ceiling but counts "
                f"{dollars(info.spent_cents)} spent, so the study is still paused.")
    view = getattr(ctx, "view", None)
    if view is not None:
        view.notice(f"The study's ceiling is now {dollars(cents)}; carrying on.")
    raised = getattr(ctx, "on_ceiling_raised", None)
    if callable(raised):
        # A workspace whose start met the pause is started again here, so the call
        # that goes again waits for it instead of meeting the stored refusal.
        try:
            raised()
        except Exception:  # noqa: BLE001 - see docstring
            pass
    return True
