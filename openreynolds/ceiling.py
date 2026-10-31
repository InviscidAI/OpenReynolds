"""A study's spending ceiling, from the agent's side.

The workspace service gives every study a ceiling and pauses it at 80%: from then on
the study's new sandboxes, model calls and GPU solves answer `409 study_paused` with
`{error, message, study_id, spent_cents, cap_cents, times_normal, style}`, and `POST
/v1/studies/{id}/cap {cap_cents}` raises it (owner only; it must exceed what is spent,
and one still under spend/0.8 is taken but answered `paused: true`). Default ceilings
are the service's: 500 cents for a quick study, 1250 for a full one, so a default study
pauses at twice a normal one.

The ceiling is kept in cents, and so is everything the service reports about it, but
none of it is shown to the person as money. Plans are sold as studies, and what a study
has used is said as a multiple of a normal study: the study example in the service's
plans.json at list price, 200 cents for a quick study and 500 for a full one
(`NORMAL_CENTS`). The service sends that multiple itself (`times_normal`,
`pause_times_normal`). An older service that does not is answered from its cents and
the same normals here, never by falling back to dollars.

Two things are done with that here, and nothing else:

- **The pause is the person's question.** Using more than they planned for is not the
  model's call, so a pause is put to them through the same `Approver` every other
  question goes through, whatever mode the session is in: this study has used 2x the
  usage of a normal study, do you want to continue? Yes doubles the ceiling (or raises
  it to what would not pause again at once, when doubling falls short, without saying
  so) and the call that met the pause goes again; no, or nobody there to answer, stops
  the study: the turn ends after the calls in flight, a plain final message is shown
  and kept in the transcript, and nothing more is sent to the model until the person
  speaks (`Loop.blocked_reason`).
- **The first solve is announced.** Before the session's first `job_start` the person
  is told how far the study may go before it asks, as a multiple of a normal study, as
  a message, or as part of the job's question when they chose to approve compute
  (`modes.PARTIAL`). A study foamd reads as `exempt` (a staff owner, whom no ceiling
  holds) is told nothing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

from .backend.base import BackendError, without_money

STUDY_PAUSED = "study_paused"
PAUSE_AT = 0.8
"""The fraction of its ceiling at which the service pauses a study."""

DEFAULT_CAP_CENTS = {"quick": 500, "full": 1250}
"""The service's default ceilings by study style; a study with no style is full."""

NORMAL_CENTS = {"quick": 200, "full": 500}
"""What a normal study uses, by style: plans.json's study example at list price. Any
style that is not quick, an empty one included, is full. Every figure the person sees
about a study's usage is a multiple of one of these."""

MAX_RAISES_PER_CALL = 3
"""How many times one call may meet a pause and have it lifted before it is given up
on. A raise that does not lift the pause (the service counting spend this side has not
seen) would otherwise ask the person forever about the same call."""


def style_of(style: str | None) -> str:
    """The style as the normals and the defaults know it: quick, or else full."""
    return "quick" if (style or "").strip().lower() == "quick" else "full"


def default_cap_cents(style: str | None) -> int:
    return DEFAULT_CAP_CENTS[style_of(style)]


def normal_cents(style: str | None) -> int:
    return NORMAL_CENTS[style_of(style)]


def times_normal(cents: float, style: str | None) -> float:
    """`cents` as a multiple of a normal study of `style`, to one decimal, as the
    service rounds its own `times_normal`."""
    return round(float(cents) / normal_cents(style), 1)


def times(multiple: float) -> str:
    """A multiple as the person reads it: 2x, 2.5x."""
    return f"{round(float(multiple), 1):g}x"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(frozen=True)
class Paused:
    """What a `409 study_paused` said. `times_normal` is None and `style` empty from a
    service that predates them; `described` fills them in."""

    study_id: str
    spent_cents: float
    cap_cents: float
    message: str = ""
    times_normal: float | None = None
    style: str = ""


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
            times_normal=_number(fields.get("times_normal")),
            style=str(fields.get("style") or ""),
        )
    except (TypeError, ValueError):
        return Paused("", 0.0, 0.0, message)


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
    "The study is no longer paused at its ceiling. This refused job is "
    "finished; no replacement solve was started."
)


def job_pause(ctx: Any, facts: Any) -> tuple[Paused | None, bool, str]:
    """A foamd refusal from Felix against the service's current study row.

    `felix run` is itself a workspace job. A GPU solve refused by foamd therefore
    exits 1 and prints a final JSON status instead of raising the `BackendError` that
    `tools._handled` sees. The raw `refused` status proves foamd returned the 409,
    rather than a felixd solve rejection or arbitrary output mentioning the error.
    The current study row is authoritative after that: a live pause is the person's
    question, and a cleared pause is a fact for the model.

    The bool says the job is conclusively accounted for. False leaves it eligible for
    a later read because the final status or study row was incomplete.
    """
    status = getattr(facts, "status", None)
    if not isinstance(status, dict) or status.get("state") != "refused":
        return None, False, ""
    if status.get("status") != 409 or status.get("error") != STUDY_PAUSED:
        return None, False, ""
    service = getattr(ctx, "ceiling", None)
    row = service.study() if service is not None else None
    if not isinstance(row, dict):
        return None, False, ""
    paused_now = row.get("paused")
    if paused_now is False:
        return None, True, _CLEARED_JOB_NOTE
    spent = row.get("spent_cents")
    cap = row.get("cap_cents")
    if (paused_now is not True
            or isinstance(spent, bool) or not isinstance(spent, (int, float))
            or isinstance(cap, bool) or not isinstance(cap, (int, float))):
        return None, False, ""
    info = Paused(
        str(row.get("id") or row.get("study_id") or service.study_id),
        float(spent),
        float(cap),
        str(status.get("message") or ""),
        times_normal=_number(row.get("times_normal")),
        style=str(row.get("style") or ""),
    )
    return described(info), True, ""


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


def described(info: Paused, ceiling: Any = None) -> Paused:
    """`info` with its style and its multiple of a normal study: the service's own
    fields when it sent them, else the study's record and the cents. A study whose
    style cannot be read is full, as the service counts it."""
    style = info.style
    if not style and ceiling is not None:
        style = str((ceiling.study() or {}).get("style") or "")
    style = style_of(style)
    multiple = info.times_normal
    if multiple is None:
        multiple = times_normal(info.spent_cents, style)
    return dataclasses.replace(info, style=style, times_normal=multiple)


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


def preamble(ctx: Any) -> str | None:
    """The line said before this session's first solve: how far the study may go before
    it asks, as a multiple of a normal study.

    None when there is no ceiling to speak of (no hosted service behind the session),
    when it has been said already, or for an `exempt` study (a staff owner, foamd D11),
    which no ceiling holds. Nothing in it is money: the service's `pause_times_normal`
    when it sends one, else the ceiling in force (or the style's default) at 80% over
    a normal study of that style."""
    ceiling = getattr(ctx, "ceiling", None)
    if ceiling is None or getattr(ctx, "estimated", False):
        return None
    ctx.estimated = True
    row = ceiling.study() or {}
    if row.get("exempt") is True:
        # A staff owner (foamd D11): no ceiling ever holds the study, so none is named.
        return None
    if "style" not in row and _number(row.get("cap_cents")) is None:
        return "Before the first solve: this study's ceiling could not be read from the service."
    style = style_of(row.get("style"))
    until = _number(row.get("pause_times_normal"))
    if until is None:
        cap = _number(row.get("cap_cents"))
        cap = cap if cap is not None else default_cap_cents(style)
        until = cap * PAUSE_AT / normal_cents(style)
    said = (f"This study will ask before it uses more than {times(until)} a normal "
            f"{style} study.")
    used = _number(row.get("times_normal"))
    if used is None and _number(row.get("spent_cents")) is not None:
        used = times_normal(row["spent_cents"], style)
    if used:
        said += f" It has used {times(used)} so far."
    if row.get("paused") is True:
        # foamd's `paused` (None when it could not read the ledgers): the solve about
        # to start will meet the pause, so it is said here first.
        said += " It is paused now, so the solve will ask whether to continue."
    return said


# -- at a pause ---------------------------------------------------------------------------


def least_useful_cents(spent_cents: float) -> int:
    """The smallest ceiling, rounded up to a hundred cents, that would not pause again
    at once: 80% of it has to be above what is spent. Never shown. A raise that
    doubling would leave short of it goes to this instead."""
    return (int(spent_cents / PAUSE_AT // 100) + 1) * 100


def _question(info: Paused, proposed: int, note: str = "") -> tuple[str, str]:
    used = times(info.times_normal or 0)
    title = f"This study has used {used} a normal study"
    lines = [
        f"This study has used {used} the usage of a normal {info.style} study. "
        "Do you want to continue?",
    ]
    if note:
        lines.append(note)
    after = proposed * PAUSE_AT / normal_cents(info.style)
    lines.append(f"Yes lets it run to {times(after)} before asking again; no stops it here.")
    return title, "\n".join(lines)


def stop_message(info: Paused, *, asked: bool, study_id: str = "", error: str = "") -> str:
    if info.times_normal is not None and info.style:
        head = (f"This study is paused: it has used {times(info.times_normal)} the usage "
                f"of a normal {info.style} study.")
    else:
        head = "This study is paused at its ceiling."
    if error:
        why = (f" The ceiling could not be raised ({without_money(error)}), so the study "
               "has stopped here.")
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
    info = described(info, ceiling)
    study_id = info.study_id or (ceiling.study_id if ceiling is not None else "")
    if approver is None or ceiling is None:
        ctx.stopped = stop_message(info, asked=False, study_id=study_id)
        return False
    note = ""
    while True:
        # Double the ceiling, or what would not pause again at once when doubling falls
        # short of it (a service that counts more than the ceiling it last took).
        cents = max(int(info.cap_cents * 2), least_useful_cents(info.spent_cents))
        title, detail = _question(info, cents, note)
        decision = approver.ask("ceiling", title, detail)
        if not decision.approved:
            ctx.stopped = stop_message(info, asked=True)
            return False
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
        spent = _number(answer.get("spent_cents"))
        spent = spent if spent is not None else info.spent_cents
        multiple = _number(answer.get("times_normal"))
        info = dataclasses.replace(
            info, spent_cents=spent, cap_cents=float(cents),
            times_normal=multiple if multiple is not None else times_normal(spent, info.style),
        )
        note = (f"The service took the raise but counts {times(info.times_normal)} the "
                "usage of a normal study, so the study is still paused.")
    view = getattr(ctx, "view", None)
    if view is not None:
        until = _number(answer.get("pause_times_normal")) if isinstance(answer, dict) else None
        if until is None:
            until = cents * PAUSE_AT / normal_cents(info.style)
        view.notice(f"The study can now run to {times(until)} a normal {info.style} study "
                    "before asking again; carrying on.")
    raised = getattr(ctx, "on_ceiling_raised", None)
    if callable(raised):
        # A workspace whose start met the pause is started again here, so the call
        # that goes again waits for it instead of meeting the stored refusal.
        try:
            raised()
        except Exception:  # noqa: BLE001 - see docstring
            pass
    return True
