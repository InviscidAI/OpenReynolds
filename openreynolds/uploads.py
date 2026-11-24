"""A file someone hands a study, put where its agent looks for it.

Every door a file comes through -- an MCP caller's upload, a page's upload on the
hosted service -- lands it here, under `<home>/uploads/`, so what the agent finds is
the same whichever door it came through.

An archive (zip, tar, tar.gz and the like) is also unpacked beside itself, into
`uploads/<stem>/`, by `unpack_script.py` run on the workspace. People zip a STEP file
with its parts, or a whole case, and a file the agent's tools cannot open is a file it
asks the person to paste instead (#216).
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any, Callable

from .backend.base import BackendError

Derive = Callable[[str, bytes], "tuple[list[tuple[str, bytes]], str]"]
"""(the placed path, its bytes) -> (derived files as (path under the original's stem,
bytes), a line saying what was derived). How an embedder unpacks a file the agent's
tools cannot read (a PDF into page images and text, say)."""


def safe_filename(name: str) -> str:
    """The basename, in the charset a workspace path can carry anywhere: no
    directories, no shell-significant characters, never empty."""
    base = Path(name.replace("\\", "/")).name
    # The extension is kept whole and apart: a name all in characters this charset
    # lacks (`机翼.zip`) or a long one would otherwise lose it, and an archive whose
    # suffix is gone is never unpacked (#216).
    lower = base.lower()
    ext = next((base[-len(s):] for s in ARCHIVE_SUFFIXES if lower.endswith(s) and len(base) > len(s)), None)
    if ext is None:
        stem, dot, tail = base.rpartition(".")
        ext = f".{tail}" if dot and stem.strip("._") else ""
    stem = base[: len(base) - len(ext)] if ext else base
    ext = re.sub(r"[^A-Za-z0-9.]+", "", ext)[:16].rstrip(".")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "upload"
    return stem[: 120 - len(ext)] + ext


# -- archives ---------------------------------------------------------------------------

ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".tar", ".zip")
UNPACK_EXEC_S = 115
"""The exec's own timeout: past the script's time limit (`unpack_script.TIME_LIMIT_S`),
under the window after which a backend promotes a command to a job and runs it again."""


def archive_stem(name: str) -> str | None:
    """The name without its archive suffix, or None when it is not an archive's name."""
    lower = name.lower()
    for suffix in ARCHIVE_SUFFIXES:
        if lower.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return None


def _unpack_command(target: str, dest: str, stem: str) -> str:
    source = (Path(__file__).with_name("unpack_script.py")).read_text(encoding="utf-8")
    # Hex, not base64: a local backend rewrites every `/work` in a command to its own
    # root (`LocalBackend._wrap`), and base64 can spell one by chance. Hex has no `/`.
    code = source.encode().hex()
    program = f"exec(bytes.fromhex('{code}').decode())"
    return " ".join(["python3", "-I", "-c", shlex.quote(program),
                     shlex.quote(target), shlex.quote(dest), shlex.quote(stem)])


def _place_archive(backend: Any, target: str) -> tuple[list[str], str]:
    """Unpack the archive already placed at `target` into a folder beside it, on the
    workspace itself (`unpack_script.py` says why there). Returns the workspace paths
    written, or the first of them when there are many, and a line saying what
    happened, which is said to the agent either way: an archive left packed is still
    there for its own shell. Never raises: the archive itself is already placed, and a
    caller that retried on an error would place it again."""
    folder, _, base = target.rpartition("/")
    stem = archive_stem(base) or base
    dest = f"{folder}/{stem}"
    try:
        result = backend.exec(_unpack_command(target, dest, stem), timeout_s=UNPACK_EXEC_S)
    except Exception as exc:  # noqa: BLE001 - the workspace's refusal, whatever its type
        return [], f"not unpacked: the workspace could not run the unpacking ({exc})"
    if getattr(result, "job_id", "") or getattr(result, "promoted", False):
        return [], "not unpacked: it took too long"
    answer = None
    for line in reversed((result.output or "").splitlines()):
        kind, _, body = line.partition(" ")
        if kind in ("UNPACKED", "UNPACK_REFUSED"):
            try:
                answer = (kind, json.loads(body))
            except ValueError:
                pass
            break
    if answer is None:
        last = next((ln for ln in reversed((result.output or "").splitlines()) if ln.strip()), "")
        return [], f"not unpacked: the workspace could not unpack it ({last[:200] or f'exit {result.exit_code}'})"
    kind, body = answer
    if kind == "UNPACK_REFUSED":
        why = body.get("why") if isinstance(body, dict) else None
        return [], f"not unpacked: {str(why or 'it could not be read')[:300]}"
    try:
        where = f"{dest}{str(body.get('suffix') or '')}"
        count = int(body.get("count") or 0)
        files = [f"{where}/{rel}" for rel in body.get("files") or [] if isinstance(rel, str)]
    except (AttributeError, TypeError, ValueError):
        return [], "not unpacked: the workspace answered in a form this does not read"
    return files, f"unpacked {count} file{'s' if count != 1 else ''} into {where}/"


def place_upload(backend: Any, home: str, name: str, data: bytes,
                 derive: Derive | None = None) -> tuple[str, list[str], str]:
    """Put a file under `<home>/uploads/` on the workspace. Returns the workspace path,
    the derived files and a line saying what was derived. Raises `BackendError` when
    the workspace refuses the file."""
    base = safe_filename(name)
    target = f"{home}/uploads/{base}"
    # A second upload of the same name sits beside the first, not over it: the first
    # may already be what a running job is reading.
    stem, dot, ext = base.partition(".")
    for n in range(2, 21):
        try:
            backend.stat(target)
        except BackendError:
            break
        target = f"{home}/uploads/{stem}-{n}{dot}{ext}" if dot else f"{home}/uploads/{stem}-{n}"
    backend.put_file(target, data)
    extras: list[str] = []
    made = ""
    if archive_stem(Path(target).name) is not None:
        extras, made = _place_archive(backend, target)
    elif derive is not None:
        derived, made = derive(target, data)
        root = target.rsplit(".", 1)[0] if "." in Path(target).name else target
        for rel, blob in derived:
            try:
                backend.put_file(f"{root}/{rel}", blob)
                extras.append(f"{root}/{rel}")
            except BackendError:
                break          # the original is up; partial derivatives are fine
        # The derive's line names what it made, not where: say where, and claim only
        # what landed.
        if derived and len(extras) == len(derived):
            made = f"unpacked to {root}/: {made}" if made else f"{len(extras)} files made in {root}/"
        elif extras:
            made = f"{len(extras)} of {len(derived)} files made in {root}/; the rest could not be written"
        elif derived:
            made = "nothing could be made of it on the workspace"
    return target, extras, made


def placed_line(target: str, extras: list[str], made: str) -> str:
    """How one placed file is named to the agent: its path and what came of it."""
    if made:
        return f"{target} ({made})"
    return target
