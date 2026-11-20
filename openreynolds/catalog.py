"""What a study says about itself once it is over: grouping, filtering, what it cost
in time and tokens, and starting another study from it.

Five things wanted the same missing piece. A project to group studies under, tags to
find them by, a dashboard of what is running and what finished, how long each took and
what it spent, and a way to run one again -- none of which could be built, because the
record on disk (`store.Session`) carried no project, no tag, no end time and no token
total, and nothing could be asked about a study except its id and its title.

The reading is `store.list_stores` and `compare.identity`, not a second copy of them.
`identity` already answers "what is this study" out of `session.json` and the
transcript, for `openreynolds compare`, and a listing that answered it differently
would be a listing that disagrees with the comparison about the same study.

THE MONEY LINE, which is a policy and not a detail. A hosted study's cost is never
shown as money: plans are sold as studies, the service's own spending cap is rendered
as a multiple of a normal study (`ceiling.py`), a money clause in a service message is
stripped out before it is shown (`backend.base.without_money`), and `/status` reports
tokens. So `row()` reports wall-clock time and token counts for every study, and a
dollar figure only where the person's own key paid their vendor directly -- a provider
recorded on the study, which is not `reynolds`, and whose model and every token
class it counted have a known rate (`llm.presets.priced`, #191). Everything else gets
`cost_usd: None`, which is the absence of a figure and not a figure of zero: an
unpriced run reported as $0.00 is the bug `prices()` returns None to prevent.

STATE IS NOT GUESSED. A job row says `running` because that is how the session that
started it left it; whether the process is alive is a question for the workspace
service, and asking it means starting a workspace -- which is not something a listing
may do. So a study with a job row still reading `running` is reported `unknown`, with
what is known said beside it, and never as `running`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from .backend.base import WORKSPACE_ROOT
from .compare import identity
from .llm.presets import REYNOLDS, priced
from .store import REVISED, Session, Store, list_stores, new_study_id, tag_name
from .watch import conversation_rows

NEW = "new"
FINISHED = "finished"
UNKNOWN = "unknown"
STATES = (NEW, FINISHED, UNKNOWN)
"""What a study's state can be read as from this machine alone. Not `running`: see
the module docstring."""


# -- state ---------------------------------------------------------------------


def state(session: Session) -> tuple[str, str]:
    """This study's state as the record can honestly say it, and why that is the answer.

    `unknown` is a real answer and the common one for a study whose session did not
    close cleanly: the job rows were last written as running, and the only thing that
    knows whether they still are is the workspace -- which this will not start to find
    out.
    """
    jobs = list(session.jobs.values())
    live = [job for job in jobs if job.status == "running"]
    if live:
        names = ", ".join(job.name or job.job_id[:8] for job in live)
        return UNKNOWN, (
            f"{len(live)} job(s) last recorded running ({names}); whether they still "
            "are is the workspace's to say, and asking would start one"
        )
    if jobs:
        last = jobs[-1]
        return FINISHED, f"last job {last.name or last.job_id[:8]}: {last.end_reason or last.status}"
    if session.ended_at:
        return FINISHED, "no jobs ran"
    return NEW, "nothing recorded as run"


# -- the money line ------------------------------------------------------------


def own_key(session: Session) -> bool:
    """Whether this study's model was the person's own key, billed to them by their
    vendor. A provider of `reynolds` is the hosted model metered to the plan, and a
    study with no provider recorded predates the record and is not assumed to be
    either."""
    who = (session.provider or "").strip().lower()
    return bool(who) and who != REYNOLDS


def cost_usd(session: Session) -> float | None:
    """What this study's tokens cost at list, in dollars -- or None, which means no
    figure rather than a figure of zero.

    None for a hosted study whatever it spent (the money line), for a study whose
    provider was never recorded, for a model with no rate in `PRICE_PER_MTOK` or a
    token class its rates do not list (#191), and for a study whose sessions all
    predate the token record.
    """
    if not own_key(session) or not session.tokens:
        return None
    return priced(session.tokens, session.model, session.provider)


# -- rows ----------------------------------------------------------------------


def row(studies_dir: Path, store: Store) -> dict[str, Any]:
    """One study as both the listing and the dashboard report it.

    Built on `compare.identity`, so the two commands cannot disagree about what a
    study is, with the grouping, the accounting and the state added.
    """
    session = store.session
    how, why = state(session)
    tokens = dict(session.tokens)
    return {
        **identity(studies_dir, session.study_id, store=store),
        "project": session.project,
        "tags": list(session.tags),
        "state": how,
        "state_detail": why,
        "running_jobs": sum(1 for job in session.jobs.values() if job.status == "running"),
        "finished_jobs": sum(1 for job in session.jobs.values() if job.status != "running"),
        "cloned_from": session.cloned_from,
        "ended_at": session.ended_at,
        "runtime_s": session.runtime_s,
        "tokens": tokens,
        "total_tokens": sum(tokens.values()),
        "own_key": own_key(session),
        "cost_usd": cost_usd(session),
    }


def rows(studies_dir: Path, torn: list[str] | None = None) -> list[dict[str, Any]]:
    """Every local study, newest first, as rows.

    `torn` collects the ids of studies whose record will not parse, which are left
    out rather than shown with invented facts (`store.list_stores`)."""
    return [row(studies_dir, store) for store in list_stores(studies_dir, torn)]


# -- filters -------------------------------------------------------------------


def matches(
    item: dict[str, Any],
    project: str = "",
    tags: Iterable[str] = (),
    text: str = "",
    want: str = "",
) -> bool:
    """Whether a row passes the filters asked for. Every filter given must pass.

    `tags` is an and, not an or: somebody who names two tags is narrowing, and a
    listing that widened instead would be useless on the one directory with enough
    studies in it to need a filter.
    """
    if project and (item["project"] or "").strip().lower() != project.strip().lower():
        return False
    for tag in tags:
        if tag_name(tag) not in item["tags"]:
            return False
    if text:
        needle = text.strip().lower()
        if needle not in f"{item['title']}\n{item['project']}".lower():
            return False
    if want and item["state"] != want:
        return False
    return True


def filtered(
    studies_dir: Path,
    project: str = "",
    tags: Iterable[str] = (),
    text: str = "",
    want: str = "",
    torn: list[str] | None = None,
) -> list[dict[str, Any]]:
    tags = tuple(tags)
    return [item for item in rows(studies_dir, torn)
            if matches(item, project, tags, text, want)]


# -- how long, in words --------------------------------------------------------


def elapsed(seconds: float) -> str:
    """Wall-clock time as a person reads it, or `-` for a study that never recorded any.

    `-` and not `0s`: a session that crashed, or one from before the record existed,
    spent time nobody counted, and a zero there reads as a study that took no time.
    """
    total = int(round(float(seconds or 0.0)))
    if total <= 0:
        return "-"
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{(total % 3600) // 60:02d}m"


# -- running one again ---------------------------------------------------------


class NoOpeningRequest(Exception):
    """A study whose transcript holds nothing the person asked for, so there is
    nothing to re-issue. Refused by name rather than reconstructed: a plausible
    paraphrase of a request would be a new study pretending to be a repeat of an old
    one, which is the one thing a rerun must not be."""


def opening_request(studies_dir: Path, study_id: str) -> str:
    """The first thing the person said in a study, verbatim.

    `watch.conversation_rows` decides what counts as something said -- the person's
    and the agent's text messages, no tool calls, no harness notes -- so a rerun and a
    resume briefing agree about what the conversation was.
    """
    path = Path(studies_dir) / study_id / "messages.jsonl"
    if not path.is_file():
        return ""
    said: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            said.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    first = next((item for item in conversation_rows(said) if item["role"] == "user"), None)
    return str(first["content"]).strip() if first else ""


CARRIED = ("project", "tags", "title", "solver", "model", "provider", "base_url", "mode")
"""What a clone takes from the study it came from: how the person filed it, and how
the session was set up. Not `instance_id`, `remote_study_id`, `home`, `jobs`,
`checkpoint` or the accounting -- those are facts about the run that happened, and a
new study has not happened yet. A clone's `home` is its own, as a new study's is."""


def duplicate(studies_dir: Path, study_id: str, rerun: bool = False) -> tuple[str, str]:
    """Make a new study from an existing one. Hands back the new id and, for a rerun,
    the request it will open on.

    A clone carries the record and the study's local files and nothing of the
    conversation: the point of one is to run the same setup again, and a transcript
    copied into it would be a new study that claims to remember a session it was not
    in.

    A rerun is a clone plus the opening request, written as the person's own first
    message and left unanswered -- which is exactly the state a resume already knows
    how to pick up (`watch.earlier_conversation`, whose unanswered messages a session
    answers first). Nothing new runs the request; the ordinary resume does.
    """
    studies_dir = Path(studies_dir)
    source = studies_dir / study_id
    if not (source / "session.json").is_file():
        raise FileNotFoundError(str(source))

    request = ""
    if rerun:
        request = opening_request(studies_dir, study_id)
        if not request:
            raise NoOpeningRequest(
                f"{study_id} has no opening request in its transcript, so there is "
                "nothing to re-issue"
            )

    old = Store(studies_dir, study_id)
    fresh = Store(studies_dir, new_study_id())
    for name in CARRIED:
        setattr(fresh.session, name, getattr(old.session, name))
    fresh.session.tags = list(old.session.tags)
    fresh.session.cloned_from = study_id
    # Its own directory, as any new study gets. Left empty, a resume read the clone
    # as a study from before homes and opened it in the shared workspace root (#170).
    fresh.session.home = f"{WORKSPACE_ROOT}/{fresh.session.study_id}"
    fresh.save()

    if old.files_dir.is_dir():
        shutil.copytree(old.files_dir, fresh.fetch_dir(), dirs_exist_ok=True)
    if request:
        fresh.append_message("user", request)
    return fresh.session.study_id, request


# -- what became of one study, in order ----------------------------------------


def lineage(studies_dir: Path, study_id: str) -> list[str]:
    """This study and the studies it was cloned from, oldest first.

    `cloned_from` has been recorded on every clone and rerun since they existed
    (`store.Session.cloned_from`) and nothing read it, so a study three reruns deep
    could not say what it was a rerun of. The walk stops at a parent this machine does
    not have -- a clone of a study since deleted, or one made on another machine --
    rather than reporting a gap as an origin, and a `cloned_from` cycle cannot wedge
    it: a study already seen ends the walk.
    """
    studies_dir = Path(studies_dir)
    chain, seen = [study_id], {study_id}
    while True:
        record = studies_dir / chain[0] / "session.json"
        if not record.is_file():
            break
        try:
            parent = str(json.loads(record.read_text(encoding="utf-8")).get("cloned_from") or "")
        except (OSError, json.JSONDecodeError):
            break
        if not parent or parent in seen or not (studies_dir / parent / "session.json").is_file():
            break
        chain.insert(0, parent)
        seen.add(parent)
    return chain


def history(studies_dir: Path, study_id: str) -> dict[str, Any]:
    """One study's own record of itself over time: where it came from, what was
    changed about its setup and when, and how much conversation is on disk.

    Three sources, none of them new: `revisions.jsonl`, which `Store.save` appends to
    whenever a setup field changes; `cloned_from`, walked by `lineage`; and
    `messages.jsonl`, counted rather than quoted. What this does NOT do is diff two
    studies' results -- `openreynolds compare` does that, and doing it twice in two
    voices is how two tools come to disagree about one pair of runs.
    """
    studies_dir = Path(studies_dir)
    store = Store(studies_dir, study_id)
    said: dict[str, int] = {}
    transcript = studies_dir / study_id / "messages.jsonl"
    if transcript.is_file():
        for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                role = str(json.loads(line).get("role") or "?")
            except json.JSONDecodeError:
                continue
            said[role] = said.get(role, 0) + 1
    return {
        "study_id": study_id,
        "created_at": store.session.created_at,
        "cloned_from": store.session.cloned_from,
        "lineage": lineage(studies_dir, study_id),
        "revisions": store.revisions(),
        "messages": said,
        "now": {field: getattr(store.session, field) for field in REVISED},
    }
