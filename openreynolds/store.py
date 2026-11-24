"""The local mirror: `./studies/<id>/`.

Holds session metadata, the message log, and everything fetched back. Job records are
not bookkeeping niceties — the service has no list-jobs endpoint, so without them a
resumed session cannot tell the model what is still running.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class TornRecord(Exception):
    """A `session.json` that is there but will not parse.

    Raised rather than answered with a default `Session`, because every caller of
    `Store` either shows the record or writes it, and both of those turn a silent
    default into a lie: the dashboard printed a `created_at` of now for an
    eight-week-old study, and the next `save()` made the blank permanent (#167).
    """


def new_study_id() -> str:
    """A short, sortable, human-typable id."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


@dataclass
class JobRecord:
    job_id: str
    name: str | None = None
    cmd: str = ""
    launched_at: str = ""
    status: str = "running"
    end_reason: str | None = None
    exit_code: int | None = None
    log_offset: int = 0
    cwd: str = ""
    """Where the command ran. The progress line finds the case's controlDict from it;
    records written before it existed load with it empty and are shown without an end."""
    """How far the model has already read this job's log."""
    ceiling_pause_handled: bool = True
    """Whether this job's output has already been looked at for a ceiling refusal
    (`ceiling.handle_finished_felix_job`).

    Kept with the job rather than in the live loop so reading another log page or
    resuming the study cannot ask the same spending question again. True by default
    and set False by `record_job`, so that records written before this field existed
    load as looked at: the first turn after resuming an older study would otherwise
    re-examine every failed Felix solve in it and could ask about a refusal the person
    dealt with days ago."""


@dataclass
class Session:
    study_id: str
    instance_id: str = ""
    remote_study_id: str = ""
    """Id assigned by the capture plane, when capture is on."""
    model: str = ""
    provider: str = ""
    """Whose model that was (`llm/presets.py`). Empty on studies made before the
    provider was recorded, and a resume reads that as a model it cannot place: an id
    does not name a provider on its own -- `claude-opus-5` was valid on `anthropic` and
    on `reynolds` both -- so those studies carry on on whatever the configuration says."""
    base_url: str = ""
    """Where that model was served from: `llm_base_url` as configured, empty for the
    preset's own endpoint.

    The pair alone is not enough. Two bring-your-own keys of one family -- a vendor's
    own and a gateway or router in front of it -- list different model ids, so a pair
    restored across endpoints asks a vendor for a model it has never heard of and fails
    a turn later with its 400. A study written before this was recorded has no
    `base_url` key at all, which a resume reads as "not recorded" rather than guessing
    an endpoint for it."""
    mode: str = ""
    """How much the person chose to be consulted (`modes.py`). Empty on studies made
    before modes existed, which a resume reads as whatever the configuration says."""
    solver: str | None = None
    """`openfoam` or `felix`, fixed once for the whole study (`solvers.py`), and
    written the moment it is known. None (`null` on disk) while the agent has yet to
    choose. A record from before solvers existed has no `solver` key at all
    (`Store.recorded_solver`), and is an OpenFOAM study."""
    home: str = ""
    """This study's own directory in the workspace.

    Empty on studies made before studies had one, which means the whole workspace.
    """
    created_at: str = ""
    title: str = ""
    project: str = ""
    """What this study is part of, as the person groups their own work. Free text, set
    by `openreynolds label --project` or `/project`, and empty on a study nobody has
    grouped -- which is every study made before this field existed."""
    tags: list[str] = field(default_factory=list)
    """This study's own labels, lower-case and in the order they were added
    (`tag_name`). Empty on a study nobody has tagged."""
    cloned_from: str = ""
    """The study this one was made from (`openreynolds clone`, `openreynolds rerun`).
    Empty on a study started from nothing, which is most of them."""
    ended_at: str = ""
    """When a session on this study last closed down. Empty while no session has
    finished one, and on studies made before it was recorded -- which is why a
    duration is read from `runtime_s` and not from this minus `created_at`."""
    runtime_s: float = 0.0
    """Wall-clock seconds spent in sessions on this study, added up across resumes.

    Not `ended_at - created_at`: a study resumed three days later spent minutes in
    sessions and three days on a disk, and the second number is not what anybody
    means by how long the study took. A session that crashed records nothing, so this
    is a floor rather than an estimate."""
    tokens: dict[str, int] = field(default_factory=dict)
    """Tokens this study has sent, by class -- `input`, `cache_read`, `cache_write`,
    `output` -- added up across sessions (`Loop.token_totals`, recorded at each
    session's close). Empty on a study whose sessions all predate the record."""
    capture_seq: int = 0
    checkpoint: dict[str, Any] | None = None
    """What the last End established about the workspace's durable copy
    (`cli._checkpoint_the_study`): `asked_at`, and then `taken_at`, `manifest_key` and
    `files` for a copy that landed, `stopped` for a workspace already down, or `error`
    for one that could not be made. None until a session has ended."""
    jobs: dict[str, JobRecord] = field(default_factory=dict)


REVISED = (
    "model", "provider", "base_url", "mode", "solver", "title", "project", "tags",
    "instance_id", "remote_study_id", "home", "cloned_from",
)
"""The record's fields whose earlier values are worth keeping: what the study was set
up as, and nothing that is accounting. `tokens`, `runtime_s`, `capture_seq`, `jobs`
and `checkpoint` change on nearly every save, so logging them would be a second and
worse copy of the transcript. `ended_at` is out for the same reason once measured:
`record_accounting` rewrites it every time it is called, and a row reading
`ended_at: one timestamp -> another` says nothing a session count does not
(`tests/test_store_revisions.py::test_the_accounting_fields_do_not_fill_the_log`)."""


class Store:
    """One study directory on the user's machine."""

    def __init__(self, root: Path, study_id: str):
        self.dir = root / study_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files_dir = self.dir / "files"
        self.renders_dir = self.dir / "renders"
        """The flat, obvious home for pictures. The mirror copies every render and
        assembled animation here, newest-first, so "where is the image?" has a
        one-word answer that does not involve the nested `files/<id>/.../renders`
        path the workspace happens to use."""
        self.session = Session(study_id=study_id, created_at=_now())
        self._solver_on_record = False
        """Whether the record on disk has a `solver` key, null or not: what tells a
        study still choosing from one recorded before solvers existed."""
        self._on_disk: dict[str, Any] = {}
        """The record as `save()` last saw it. Empty for a study with no record yet,
        which is what makes the first `save()` a plain write."""
        self._load()

    # -- persistence -----------------------------------------------------------

    @property
    def _session_path(self) -> Path:
        return self.dir / "session.json"

    @property
    def _messages_path(self) -> Path:
        return self.dir / "messages.jsonl"

    @property
    def _revisions_path(self) -> Path:
        return self.dir / "revisions.jsonl"

    def _read(self) -> dict[str, Any]:
        """The record as it is on disk this instant, or `{}` when there is no record.

        A record that exists and will not parse raises. It used to be swallowed, and
        the default `Session` that came back then -- `created_at` of now, no model, no
        home, no jobs -- was written over the real one by the very next `save()`, so
        one `openreynolds label` on a study whose file had been torn made the loss
        permanent and printed nothing (#167). There is nothing a reader can honestly
        do with a torn record except say so, and a writer must not touch it.
        """
        if not self._session_path.exists():
            return {}
        # Windows refuses the open for a moment while another writer's `os.replace`
        # lands, so a sound record is not called torn for that.
        for attempt in range(20):
            try:
                data = self._session_path.read_bytes()
                break
            except PermissionError as exc:
                if attempt == 19:
                    raise TornRecord(f"{self._session_path} cannot be read: {exc}") from exc
                time.sleep(0.02)
            except OSError as exc:
                raise TornRecord(f"{self._session_path} cannot be read: {exc}") from exc
        try:
            # `utf-8-sig` so a record saved by an editor that adds a byte-order mark
            # still loads. ValueError covers a file cut inside a UTF-8 character too.
            raw = json.loads(data.decode("utf-8-sig"))
            if not isinstance(raw, dict):
                raise ValueError(f"a JSON {type(raw).__name__}, not an object")
            for rec in (raw.get("jobs") or {}).values():
                JobRecord(**rec)
        except (ValueError, TypeError, AttributeError) as exc:
            raise TornRecord(
                f"{self._session_path} will not parse ({exc}); it has not been written "
                "to. Repair it or move it aside -- the transcript beside it is intact."
            ) from exc
        return raw

    def _load(self) -> None:
        raw = self._read()
        if not raw:
            return
        jobs = {jid: JobRecord(**rec) for jid, rec in (raw.pop("jobs", {}) or {}).items()}
        self._solver_on_record = "solver" in raw
        known = {f.name for f in Session.__dataclass_fields__.values()}
        # A null on disk is dropped as an unknown key is, so the field's own default
        # stands. Every field that means something by None defaults to None anyway,
        # and the ones that do not -- `tags`, `tokens` -- are iterated by the dashboard
        # and the filters, where a None from a hand-edited record would raise.
        self.session = Session(
            **{k: v for k, v in raw.items() if k in known and v is not None}, jobs=jobs)
        self._mark()

    def _mark(self) -> None:
        """Record the state `save()` will diff against, as a copy of its own.

        `asdict` and not the loaded dict: `tags` is edited in place (`commands.tag_lines`
        appends and removes) and `tokens` is added to in place, so a baseline sharing
        those objects with `self.session` changes when they do -- and a field that
        changed would look unchanged and never be written. `openreynolds label --untag`
        dropped nothing at all.
        """
        self._on_disk = asdict(self.session)

    @property
    def recorded_solver(self) -> str | None:
        """What the record says of the solver: `openfoam` or `felix` once chosen,
        `auto` for a study recorded still choosing, None for no record of one -- a
        study from before solvers existed, or one this machine has never held."""
        if self.session.solver in ("openfoam", "felix"):
            return self.session.solver
        return "auto" if self._solver_on_record else None

    def record_accounting(self, seconds: float, tokens: dict[str, int] | None = None) -> None:
        """Add one session's wall-clock time and token usage to the study's totals.

        Called once as a session closes, so what the study cost in time and tokens
        survives the session that spent it. Added rather than replaced: a study is
        resumed, and three sessions of twenty minutes is an hour of work.
        """
        self.session.runtime_s = round(self.session.runtime_s + max(float(seconds), 0.0), 1)
        for name, count in (tokens or {}).items():
            self.session.tokens[name] = self.session.tokens.get(name, 0) + int(count or 0)
        self.session.ended_at = _now()
        self.save()

    def save(self) -> None:
        """Write this study's record, atomically, without reverting anyone else's field.

        Two things used to go wrong here and they need two separate answers.

        The write truncated the file in place, so a reader -- the dashboard, a second
        command, this process after a crash -- could see half a record, and the loader
        then turned that into a blank study (#167). It now goes through a temporary
        file in the same directory and `os.replace`, which is atomic, the way
        `toolbox/study_state.py` and `config.py` already do it.

        And the whole record was written from whatever this Store loaded, which for a
        session is however long ago the session started. So `openreynolds label
        --project X` from a second terminal survived until the session's next message,
        which put the project the session had loaded at startup back (#168). The
        decision: **nobody owns the file, each writer owns the fields it changed.**
        `save()` re-reads the record and writes its own changes onto it -- a field this
        Store has not touched since it loaded keeps whatever is on disk now. The live
        session goes on owning `capture_seq`, the jobs and the accounting; `label` owns
        the project and the tags; neither reverts the other, and neither has to know
        the other exists.

        This is last-writer-wins per field rather than a lock. Two processes changing
        the *same* field within the microseconds between this read and its `os.replace`
        is still a lost update -- but it is one field's update, not a destroyed record,
        and a lock file held across a whole session would mean `label` refusing for
        hours, which is the thing #168 says should not be necessary.
        """
        mine = asdict(self.session)
        was = self._on_disk
        try:
            disk = self._read()
        except TornRecord:
            # Nothing to merge with, so this Store's own record is written whole. That
            # is a repair and not the loss #167 is about: `_load` refuses, so a Store
            # that exists either read a real record or found no file, and either way
            # what is in memory here is complete. A session whose file somebody tore
            # while it ran mends it on its next message rather than dying on one.
            disk = {}
        # Every field of the record, this Store's own unless it has not touched it
        # since it loaded -- in which case whatever is on disk now, which is the other
        # writer's. Starting from `mine` and not from `disk` keeps the record complete:
        # a study recorded before `tags` and `tokens` existed gets them written with
        # their defaults, as it always did.
        payload = dict(mine)
        for key, value in disk.items():
            if key not in mine or was.get(key, value) == mine[key]:
                payload[key] = value
        # The transcript's counter only goes forward. Per-field last-writer-wins let a
        # writer holding an older record put a lower `capture_seq` on disk, and this
        # Store, not having touched the field since, took it back: the next row was
        # numbered like the last one, and a reader keyed on `seq` dropped it as seen.
        try:
            payload["capture_seq"] = max(int(mine.get("capture_seq") or 0),
                                         int(disk.get("capture_seq") or 0))
        except (TypeError, ValueError):
            payload["capture_seq"] = mine.get("capture_seq") or 0
        # A temp name of this writer's own: with one shared name, a second save
        # truncates the file the first is about to move into place.
        tmp = self._session_path.with_name(
            f"{self._session_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            _replace(tmp, self._session_path)
        finally:
            tmp.unlink(missing_ok=True)
        self._record_revision(was, mine)
        self._adopt(payload, mine)
        self._mark()

    def _record_revision(self, was: dict[str, Any], mine: dict[str, Any]) -> None:
        """Append what THIS writer just changed about the record, with both values.

        `session.json` is rewritten on every save, so until this existed the record
        held only its latest state and every earlier one was gone: a study that ran
        forty turns on one model and a hundred on another looked, afterwards, like a
        study that had only ever used the second, and `tokens` is one total with no
        way to split it back. The transcript is append-only (`messages.jsonl`) and so
        is this, and between them nothing about a study is now written over in place.

        Only `REVISED` fields, which is the setup rather than the accounting. Tokens,
        the job table and the capture sequence change on almost every save, and their
        history is the transcript's rather than a second copy of it here.

        Only what this Store itself changed, which is `was != mine` rather than
        `was != payload`. The merge in `save()` means a field another process changed
        comes back in `payload` to a Store that never touched it -- recording that
        would write the same change twice, once from each process, and attribute it to
        both. The writer that made the change is the one that has it in `mine`.

        Best-effort on purpose: a record of what changed is worth less than the change
        itself, so an unwritable log is dropped rather than failing the save that
        carries the real update. The same reasoning as `_read`'s refusal to swallow a
        torn record, pointed the other way -- this file is a convenience, that one is
        the study.
        """
        if not was:
            return  # the study's first save is its creation, not a revision of it
        changes = {
            field: {"from": was.get(field), "to": mine[field]}
            for field in REVISED
            if field in mine and was.get(field) != mine[field]
        }
        if not changes:
            return
        row = {"at": _now(), "changes": changes}
        try:
            with self._revisions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass

    def revisions(self) -> list[dict[str, Any]]:
        """Every recorded change to this study's record, oldest first.

        Empty for a study nobody changed and for every study made before this was
        recorded, which is not distinguished: a study with no log is a study whose
        earlier states were never kept, and inventing one from `created_at` would be
        inventing history. A half-written final line is dropped, as the transcript
        reader does, since this is read while a session is appending to it.
        """
        rows: list[dict[str, Any]] = []
        try:
            text = self._revisions_path.read_text(encoding="utf-8")
        except OSError:
            return rows
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _adopt(self, payload: dict[str, Any], mine: dict[str, Any]) -> None:
        """Take on the fields somebody else changed, in the record just written.

        Without this the revert of #168 only moves one save later: the merge keeps
        the other writer's project this time, and the next `save()` sees this Store's
        own unchanged copy differ from what is now on disk and calls it a change.
        A field this Store does not own is a field it reads, so it reads it.
        """
        known = {f.name for f in Session.__dataclass_fields__.values()}
        for name, value in payload.items():
            if name not in known or name == "jobs" or value is None or value == mine.get(name):
                continue
            setattr(self.session, name, value)
        rows = payload.get("jobs")
        if isinstance(rows, dict) and rows != mine.get("jobs"):
            self.session.jobs = {jid: JobRecord(**rec) for jid, rec in rows.items()}

    # -- messages --------------------------------------------------------------

    def append_message(self, role: str, content: Any) -> int:
        """Mirror one message locally and hand back its sequence number.

        The capture plane does not assign sequence numbers, so this counter is the
        single source of ordering for both the local log and the upload.
        """
        seq = self.session.capture_seq
        self.session.capture_seq += 1
        row = {"seq": seq, "role": role, "at": _now(), "content": content}
        with self._messages_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        self.save()
        return seq

    # -- jobs ------------------------------------------------------------------

    def recent_messages(self, limit: int = 30,
                        roles: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """The last few transcript rows, oldest-first.

        Read from other threads than the one writing. Append-only, so reading the tail never races an appending writer:
        a half-written final line is dropped rather than raised on.

        `roles` keeps the last `limit` rows **of those roles** rather than the last
        `limit` rows of everything. The CAD desk reads the person's own words this way
        (`cad.agent.CadDesk._said`), and since the desk's own steps are transcript
        rows too -- up to thirty a run, plus the reviews -- a second call to the desk
        in one session would otherwise find its forty-row window full of the first
        call's progress and none of what the person said."""
        path = self._messages_path
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        rows: list[dict[str, Any]] = []
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if roles is not None and row.get("role") not in roles:
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
        rows.reverse()
        return rows

    def record_job(self, job_id: str, cmd: str, name: str | None, cwd: str = "") -> JobRecord:
        record = JobRecord(job_id=job_id, name=name, cmd=cmd, launched_at=_now(), cwd=cwd,
                           ceiling_pause_handled=False)
        self.session.jobs[job_id] = record
        self.save()
        return record

    def update_job(self, job_id: str, **changes: Any) -> JobRecord | None:
        record = self.session.jobs.get(job_id)
        if record is None:
            return None
        for key, value in changes.items():
            if hasattr(record, key):
                setattr(record, key, value)
        self.save()
        return record

    def live_jobs(self) -> list[JobRecord]:
        return [job for job in self.session.jobs.values() if job.status == "running"]

    # -- fetched files ---------------------------------------------------------

    def fetch_dir(self) -> Path:
        self.files_dir.mkdir(parents=True, exist_ok=True)
        return self.files_dir


def list_stores(root: Path, torn: list[str] | None = None) -> list[Store]:
    """Every local study, newest first, as the Store each was read from.

    Ids are timestamp-prefixed, so reverse name order is newest first. The Store
    rather than the record, because a caller that then wants the study's own
    directory, its `recorded_solver` or its transcript would otherwise build a second
    one for each study it has just been handed.

    A study whose record will not parse is left out and its id appended to `torn`.
    Listing is the one thing that goes on in spite of one, since the alternatives are
    both worse: it used to be listed with a `created_at` of now, no model and no jobs
    (#167), and raising here would mean one unreadable file emptying a dashboard of
    fifty-one good studies. A caller that is about to *write* builds the Store itself
    and so gets the `TornRecord`.
    """
    if not root.exists():
        return []
    found: list[Store] = []
    for child in sorted(root.iterdir(), reverse=True):
        if not (child / "session.json").is_file():
            continue
        try:
            found.append(Store(root, child.name))
        except TornRecord:
            if torn is not None:
                torn.append(child.name)
    return found


def list_studies(root: Path, torn: list[str] | None = None) -> list[Session]:
    """Every local study, newest first."""
    return [store.session for store in list_stores(root, torn)]


TAG_MAX = 60
"""The longest a tag is recorded at. A tag is a label somebody filters by, and a 500
character one was stored verbatim and read back into every listing that prints it
(#194). Long enough for a sentence, short enough to sit in a column."""


def tag_name(word: str) -> str:
    """One tag as it is recorded: trimmed, lower-case, spaces as hyphens, capped.

    `Mesh Study`, `mesh-study` and `mesh study ` are one tag, because somebody
    filtering by a tag they typed last week should not have to remember how they
    capitalised it.
    """
    return "-".join(word.strip().lower().split())[:TAG_MAX]


def _replace(tmp: Path, path: Path, tries: int = 20) -> None:
    """`os.replace`, which on Windows a concurrent reader can refuse.

    CPython opens a file for reading without `FILE_SHARE_DELETE`, so on Windows the
    rename over the destination fails with `PermissionError` while anybody has it
    open -- and `session.json` is read by the dashboard, the TUI and a second command
    while a session writes it. The window is a single `read_text`, so a bounded retry
    closes it; what must not happen is giving up and writing in place, which is the
    truncation #167 is about. A reader that never lets go raises, rather than quietly
    dropping the save.
    """
    for attempt in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(0.05)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
