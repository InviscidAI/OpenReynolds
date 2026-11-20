"""The meshes already built on this workspace, kept so the next study can see them.

A production workspace built the same mesh twice, twenty-four minutes apart: the text
"Inviscid AI" in a 2D channel, 728 s the first time and 402 s the second, while a
finished, checkMesh-clean copy of it sat in the first study's directory (#35). Neither
run was compute-bound; the time is the desk's round trips writing and revising gmsh
scripts. Nothing told the second study the first mesh was there: the session-start note
said only that `/work` held "4 other directories".

So every mesh the desk finishes adds one line to `<root>/.reynolds/meshes.jsonl`, made
of facts the finish check has already measured, and the index is offered back -- as
facts, not as an instruction -- in the workspace-ready note and in the desk's own task
message. Whether an entry is the mesh wanted is the model's call.

Local workspaces only, both recording and offering. The hosted service gives each study
a workspace of its own and never shares it (#92), so a hosted `/work` holds the current
study and nothing else: there the index could never answer #35, and would offer a study
its own earlier meshes as though somebody else had built them. Sharing meshes between
hosted studies needs the index kept on the service and the files copied between
workspaces, which is a separate decision. `Backend.shared_workspace` is the gate, and
it is read here rather than at the two call sites so neither can forget it.

Every function here swallows workspace errors. The index is a convenience: a mesh that
cannot be recorded is still a finished mesh, and a session that cannot read the index
is a session that builds, which is what it did before this file existed.
"""

from __future__ import annotations

import json
import shlex
import time
from typing import Any

from ..backend.base import WORKSPACE_ROOT

INDEX_REL = ".reynolds/meshes.jsonl"
"""Where the index is, relative to the workspace root -- which is also how every
command here names it, with `cwd` at the root. See `_rel`."""
INDEX_DIR = ".reynolds"
REQUEST_CHARS = 200
"""Enough of the request to recognise the geometry; the full text is in the study."""
SHOWN = 5
"""Entries offered at once, newest first. A workspace used for months accumulates
meshes, and the offer is a line each in a note the model reads every session."""
READ_LINES = 200
"""Lines read back from the end of the index; older entries are never offered."""


def _posix(path: str) -> str:
    r"""A path with forward slashes, which is how every path in the index is written.

    A native local workspace's root is this machine's own directory, and on Windows
    that is a backslash path (`LocalBackend.workspace_root`). JSON escapes each
    backslash, so the line on disk carried `C:\\Users\\...` -- which nothing downstream
    wants: not the shell on the workspace, which reads the path back out of the line,
    and not a comparison against the study's home. Git bash takes `C:/Users/...` as
    readily, and a hosted or Docker workspace is POSIX already, so this is a no-op
    there."""
    return path.replace("\\", "/")


def root_of(backend: Any) -> str:
    """The workspace root this backend's index lives under. Both call sites had the
    same `getattr` fallback written out; a backend that does not declare a root is
    addressed as `/work`, which is what the protocol advertises."""
    return getattr(backend, "workspace_root", "") or WORKSPACE_ROOT


def shared(backend: Any) -> bool:
    """Whether studies share this workspace, which is the whole premise of the index
    (see the module docstring)."""
    return bool(getattr(backend, "shared_workspace", False))


def index_path(backend_or_root: Any = WORKSPACE_ROOT) -> str:
    """The index's path, from a root or from the backend that has one."""
    root = backend_or_root if isinstance(backend_or_root, str) else root_of(backend_or_root)
    return f"{_posix(root).rstrip('/')}/{INDEX_REL}"


def entry(check: Any, case_dir: str, request: str, study_id: str) -> dict[str, Any]:
    """One line of the index, from a passing `Check`.

    `regions` is `Check.regions` -- `[""]` for a single-region case, the names for a
    multi-region one -- because the mesh of a CHT case is `constant/<region>/polyMesh`
    and both reading the entry back and copying it have to know which layout it is.
    """
    return {
        "case_dir": _posix(case_dir),
        "request": " ".join(request.split())[:REQUEST_CHARS],
        "cells": int(getattr(check, "cells", 0) or 0),
        "two_d": bool(getattr(check, "two_d", False)),
        "bounds": [float(b) for b in (getattr(check, "bounds", None) or [])],
        "patches": [str(p.get("name", "")) for p in (getattr(check, "patches", None) or [])
                    if isinstance(p, dict) and p.get("name")],
        "regions": [str(r) for r in (getattr(check, "regions", None) or [""])],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "study_id": study_id,
    }


def _rel(case_dir: str, root: str) -> str:
    r"""A case directory as the index stores it: relative to the workspace root.

    No absolute path goes into a line, and no absolute path goes into a command. Every
    command here runs with `cwd` at the root, which the backend resolves in Python
    (`LocalBackend.exec` hands it to `subprocess`), so the shell only ever sees relative
    POSIX paths.

    A Windows workspace root is why. `C:\Users\...` carries a drive colon, which POSIX
    `PATH` uses as a separator, and backslashes, which the shell and `re.sub` both read
    as escapes -- and whether an absolute Windows path resolves at all depends on which
    bash is on PATH. It did on this developer's machine and did not on the Windows CI
    runner, where `mkdir -p C:/Users/.../.reynolds` left Python unable to find the
    directory it had just been told was made: six tests, every one of them "expected a
    path, got nothing". Relative POSIX paths have none of those traps.

    A case directory outside the root is left absolute. Nothing produces one -- a
    study's home is inside the root by construction (`cli._home_path`) -- and if
    something did, it simply never matches the relative existence test and so is never
    offered, which is the safe direction.
    """
    case = _posix(case_dir)
    base = _posix(root).rstrip("/")
    if base and (case == base or case.startswith(base + "/")):
        return case[len(base):].lstrip("/")
    return case


def _abs(case_dir: str, root: str) -> str:
    """A stored case directory as the session refers to it. The model is handed a path
    it can copy, so what is read back out of the line is joined onto the root again."""
    case = _posix(case_dir)
    if case.startswith("/") or ":" in case.split("/", 1)[0]:
        return case
    return f"{_posix(root).rstrip('/')}/{case}"


def record(backend: Any, line: dict[str, Any], root: str | None = None) -> bool:
    """Append one entry. One short `>>` append, so two sessions on one workspace
    cannot lose each other's lines the way a read-modify-write would.

    False when nothing was written, which includes a workspace studies do not share
    and an append that failed: a `>>` into a read-only or full volume comes back with
    a non-zero exit code and no exception, and reporting that as recorded would put a
    line in the index that nobody could read back.
    """
    if not shared(backend):
        return False
    where = root if root is not None else root_of(backend)
    stored = {**line, "case_dir": _rel(str(line.get("case_dir") or ""), where)}
    text = json.dumps(stored, separators=(",", ":"))
    try:
        res = backend.exec(
            f"mkdir -p {INDEX_DIR} && printf '%s\\n' {shlex.quote(text)} >> {INDEX_REL}",
            cwd=where, timeout_s=60,
        )
    except Exception:  # noqa: BLE001 - see the module docstring
        return False
    return int(getattr(res, "exit_code", 0) or 0) == 0


def retire(backend: Any, case_dir: str, root: str | None = None) -> bool:
    """Say that `case_dir`'s recorded mesh is no longer the mesh that was recorded.

    Called when the desk starts on a case directory. Without it an entry is never
    invalidated: a study that re-runs the desk on a case it has already meshed
    overwrites `constant/polyMesh`, and if that run then fails its check or runs out
    of steps, nothing new is recorded and the old "checkMesh-clean" line still points
    at the mesh that is now half-written. The directory still exists, so the offer
    passed it and the next study copied it.

    A tombstone rather than an edit, because the index is append-only on purpose: two
    sessions on one workspace cannot lose each other's lines, and `read` already takes
    the newest line per case directory, so a tombstone written after an entry retires
    it and an entry written after a tombstone revives it. Nothing is created to retire
    into: with no index there is nothing to invalidate.
    """
    if not shared(backend):
        return False
    where = root if root is not None else root_of(backend)
    text = json.dumps({"case_dir": _rel(case_dir, where), "retired": True},
                      separators=(",", ":"))
    try:
        backend.exec(
            f"test -f {INDEX_REL} && printf '%s\\n' {shlex.quote(text)} "
            f">> {INDEX_REL} || true",
            cwd=where, timeout_s=60,
        )
    except Exception:  # noqa: BLE001 - see the module docstring
        return False
    return True


_READ_CMD = """\
test -f {path} || exit 0
tail -n {lines} {path} | while IFS= read -r line; do
  d=${{line#*:\\"}}
  d=${{d%%\\"*}}
  [ -n "$d" ] || continue
  for f in "$d"/constant/polyMesh/faces "$d"/constant/*/polyMesh/faces; do
    [ -f "$f" ] && printf '%s\\n' "$line" && break
  done
done
true
"""
r"""Tail the index and test each entry's mesh in one round trip, printing only the lines
whose mesh is still on disk.

Two commands before this, and both are charged to the desk's step budget ahead of its
first model call. The case directory is taken off the front of the line by the shell's
own prefix stripping rather than parsed, which is sound because this file writes the
line: `json.dumps` with compact separators and `case_dir` first. Not `sed`:
`s|\\\\|/|g` is an unterminated expression to the sed in Git for Windows.

Every path in it is relative, and it runs with `cwd` at the workspace root, for the
reason `_rel` gives: an absolute Windows path in command text is resolved by whichever
bash is on PATH, and the two disagreed.

Either mesh layout counts as present: `constant/polyMesh/faces` for a single-region
case and `constant/<region>/polyMesh/faces` for the multi-region one that
`cad.check.mesh_regions` accepts. Testing only the first dropped every CHT mesh on
every read, so they were recorded and never offered."""


def read(backend: Any, root: str | None = None, exclude: str = "") -> list[dict[str, Any]]:
    """The entries still on disk, newest first, one per case directory, leaving out
    `exclude` and anything under it (a case directory, or a study's own home).

    An entry whose mesh is gone (the study was deleted, the mesh was rebuilt elsewhere)
    is dropped by the command itself, so what is offered is what can be copied. A
    retired entry drops out here, and takes the case directory with it so no older
    entry for the same case is offered in its place. A line that does not parse is
    skipped rather than failing the read.
    """
    if not shared(backend):
        return []
    where = root if root is not None else root_of(backend)
    try:
        res = backend.exec(_READ_CMD.format(path=INDEX_REL, lines=READ_LINES),
                           cwd=where, timeout_s=60)
    except Exception:  # noqa: BLE001 - see the module docstring
        return []
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    left_out = _posix(exclude).rstrip("/")
    for raw in reversed((res.output or "").splitlines()):
        try:
            item = json.loads(raw)
        except ValueError:
            continue
        stored = item.get("case_dir") if isinstance(item, dict) else None
        if not isinstance(stored, str) or not stored or stored in seen:
            continue
        seen.add(stored)
        if item.get("retired"):
            continue
        # Stored relative, handed back absolute: the model is given a path it can copy.
        case_dir = _abs(stored, where)
        item["case_dir"] = case_dir
        if left_out and (case_dir == left_out or case_dir.startswith(left_out + "/")):
            continue
        out.append(item)
    return out[:SHOWN]


def mesh_paths(item: dict[str, Any]) -> list[str]:
    """What to copy out of this entry: the mesh, not the case directory.

    `constant/polyMesh` for a single-region case, one path per region otherwise. The
    case directory is the wrong answer twice over -- the desk's own case directory
    already exists (`CadDesk.run` makes it before anything else), so `cp -r A/mesh
    B/mesh` lands the mesh at `B/mesh/mesh/constant/polyMesh` where the finish check
    cannot see it, and it would bring any solved time directories and `processor*`
    across with it.
    """
    case = str(item.get("case_dir") or "").rstrip("/")
    regions = [str(r) for r in (item.get("regions") or [""])]
    return [f"{case}/constant/{r}/polyMesh" if r else f"{case}/constant/polyMesh"
            for r in regions]


def describe(item: dict[str, Any]) -> str:
    """One entry as one line of prose: what was asked for, its size, and what to copy."""
    request = item.get("request") or "a mesh"
    if len(request) > 80:
        request = request[:79] + "…"
    dims = "2D" if item.get("two_d") else "3D"
    patches = ", ".join(item.get("patches") or [])
    built = item.get("built_at", "")
    return (
        f'"{request}" -- {int(item.get("cells") or 0):,} cells, {dims}'
        + (f", patches {patches}" if patches else "")
        + f", mesh at {' '.join(mesh_paths(item))}"
        + (f", built {built}" if built else "")
    )


def offer(items: list[dict[str, Any]], root: str = WORKSPACE_ROOT) -> str:
    """The index as a note, or "" when there is nothing on it."""
    if not items:
        return ""
    lines = "\n".join(f"  - {describe(i)}" for i in items)
    return (
        "Meshes already built on this workspace, each one checkMesh-clean when it was "
        f"recorded (newest first; the index is {index_path(root)}):\n{lines}\n"
        "Copying one of these is a mesh in seconds where building from scratch takes "
        "the desk minutes. Copy the mesh directory named above into your own case's "
        "`constant/` -- not the case directory, which would nest the mesh where the "
        "finish check cannot find it and bring any solved times and `processor*` "
        "directories with it."
    )
