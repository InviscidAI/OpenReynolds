"""The meshes already built on this workspace, kept so the next study can see them.

A production workspace built the same mesh twice, twenty-four minutes apart: the text
"Inviscid AI" in a 2D channel, 728 s the first time and 402 s the second, while a
finished, checkMesh-clean copy of it sat in the first study's directory (#35). Neither
run was compute-bound; the time is the desk's round trips writing and revising gmsh
scripts. Nothing told the second study the first mesh was there: the session-start note
said only that `/work` held "4 other directories".

So every mesh the desk finishes adds one line to `/work/.reynolds/meshes.jsonl`, made of
facts the finish check has already measured, and the index is offered back -- as facts,
not as an instruction -- in the workspace-ready note and in the desk's own task message.
Whether an entry is the mesh wanted is the model's call.

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
REQUEST_CHARS = 200
"""Enough of the request to recognise the geometry; the full text is in the study."""
SHOWN = 5
"""Entries offered at once, newest first. A workspace used for months accumulates
meshes, and the offer is a line each in a note the model reads every session."""
READ_LINES = 200
"""Lines read back from the end of the index; older entries are never offered."""


def index_path(root: str = WORKSPACE_ROOT) -> str:
    return f"{root.rstrip('/')}/{INDEX_REL}"


def entry(check: Any, case_dir: str, request: str, study_id: str) -> dict[str, Any]:
    """One line of the index, from a passing `Check`."""
    return {
        "case_dir": case_dir,
        "request": " ".join(request.split())[:REQUEST_CHARS],
        "cells": int(getattr(check, "cells", 0) or 0),
        "two_d": bool(getattr(check, "two_d", False)),
        "bounds": [float(b) for b in (getattr(check, "bounds", None) or [])],
        "patches": [str(p.get("name", "")) for p in (getattr(check, "patches", None) or [])
                    if isinstance(p, dict) and p.get("name")],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "study_id": study_id,
    }


def record(backend: Any, line: dict[str, Any], root: str = WORKSPACE_ROOT) -> bool:
    """Append one entry. One short `>>` append, so two sessions on one workspace
    cannot lose each other's lines the way a read-modify-write would."""
    path = index_path(root)
    folder = path.rsplit("/", 1)[0]
    text = json.dumps(line, separators=(",", ":"))
    try:
        backend.exec(
            f"mkdir -p {shlex.quote(folder)} && printf '%s\\n' {shlex.quote(text)} >> {shlex.quote(path)}",
            timeout_s=60,
        )
    except Exception:  # noqa: BLE001 - see the module docstring
        return False
    return True


def read(backend: Any, root: str = WORKSPACE_ROOT, exclude: str = "") -> list[dict[str, Any]]:
    """The entries still on disk, newest first, one per case directory, leaving out
    `exclude` and anything under it (a case directory, or a study's own home).

    An entry whose `constant/polyMesh` is gone (the study was deleted, the mesh was
    rebuilt elsewhere) is dropped, so what is offered is what can be copied. A line
    that does not parse is skipped rather than failing the read.
    """
    path = shlex.quote(index_path(root))
    try:
        res = backend.exec(f"test -f {path} && tail -n {READ_LINES} {path} || true", timeout_s=60)
    except Exception:  # noqa: BLE001 - see the module docstring
        return []
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for raw in reversed((res.output or "").splitlines()):
        try:
            item = json.loads(raw)
        except ValueError:
            continue
        case_dir = item.get("case_dir") if isinstance(item, dict) else None
        if not isinstance(case_dir, str) or not case_dir or case_dir in seen:
            continue
        if exclude and (case_dir == exclude or case_dir.startswith(exclude.rstrip("/") + "/")):
            continue
        seen.add(case_dir)
        out.append(item)
    return _still_there(backend, out)[:SHOWN]


def _still_there(backend: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return items
    dirs = " ".join(shlex.quote(i["case_dir"]) for i in items)
    try:
        res = backend.exec(
            f"for d in {dirs}; do test -f \"$d/constant/polyMesh/faces\" && echo \"$d\"; done; true",
            timeout_s=60,
        )
    except Exception:  # noqa: BLE001 - see the module docstring
        return []
    present = set((res.output or "").splitlines())
    return [i for i in items if i["case_dir"] in present]


def describe(item: dict[str, Any]) -> str:
    """One entry as one line of prose: what was asked for, its size, and where it is."""
    request = item.get("request") or "a mesh"
    if len(request) > 80:
        request = request[:79] + "…"
    dims = "2D" if item.get("two_d") else "3D"
    patches = ", ".join(item.get("patches") or [])
    built = item.get("built_at", "")
    return (
        f'"{request}" -- {int(item.get("cells") or 0):,} cells, {dims}'
        + (f", patches {patches}" if patches else "")
        + f", in {item['case_dir']}"
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
        "A copy of one of these (`cp -r`) is a mesh in seconds; building from scratch "
        "takes the desk minutes."
    )
