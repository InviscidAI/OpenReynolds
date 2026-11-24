"""Unpack one uploaded archive beside itself, on the workspace (#216).

`openreynolds.uploads` sends this file's source to the workspace and runs it there, as
`python3 -c <this> <archive> <dest> <stem>`, after the archive itself has been placed.
It must stay standard-library only: it runs on whatever Python the workspace has.

It runs on the workspace and not in the server that took the upload because an archive
is a program for a decompressor. A few kilobytes can ask for gigabytes of memory or
hours of CPU through corners of `zipfile` and `tarfile` nobody wrote for hostile input
(sparse maps, global headers, entries that overlap). Four rounds of review found a new
one each time while this ran in the shared server, where it could take every other
customer's requests down with it. On the study's own workspace, the worst an archive
can do is slow the study it was uploaded to, the same as the agent running `unzip`
itself; and a time limit and a memory limit keep even that short.

What it still refuses, because it is about where files land and not about cost: names
that climb out of `dest` or are absolute, links of any kind, and devices. It writes
into `<dest>.partial-unpack` and renames that into place only when it has finished, so
a refusal or a crash leaves no half-unpacked folder behind.

The last line it prints is the answer, one of:

    UNPACKED {"suffix": "-2", "count": 12, "files": ["part.step", ...]}
    UNPACK_REFUSED {"why": "it is password-protected"}

`suffix` is what was added to `dest` because something already had that name; `files`
are relative to the folder written and are only the first of them when there are many.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import stat
import sys
import tarfile
import zipfile

MAX_BYTES = 1 << 30
"""What one archive may unpack to: more than a case needs, less than a workspace has."""
MAX_FILES = 20_000
"""A decomposed case has a few thousand files; this is an order of magnitude more."""
MAX_MEMORY = 2 << 30
TIME_LIMIT_S = 100
"""Under the backends' synchronous exec window (120 s), so a slow archive is refused
rather than promoted to a job, which would run it again from the start."""
MAX_DEPTH = 64
MAX_NAME = 1024
LISTED = 16_000
"""How many characters of file names the answer carries: the output is capped."""
READ = 1 << 20
JUNK = re.compile(r"(^|/)(__MACOSX|\.DS_Store|\._[^/]*|Thumbs\.db)(/|$)")
"""What a Mac or Windows zip adds that is nobody's file."""


class Refused(Exception):
    """Why the archive was left as it came, in words for the agent."""


def safe_part(part: str) -> str:
    return (re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip(" ") or "_")[:120]


def member_path(name: str) -> str | None:
    """A member's name as a relative path made of safe parts, or None for one that
    climbs out (`..`), is absolute, is absurdly deep or long, or is junk an archiver
    added."""
    if len(name) > MAX_NAME:
        return None
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or len(parts) > MAX_DEPTH or ".." in parts \
            or re.match(r"^[A-Za-z]:", parts[0]) or name.startswith(("/", "\\")):
        return None
    if JUNK.search("/".join(parts)):
        return None
    return "/".join(safe_part(p) for p in parts)


def folders(rel: str) -> list[str]:
    parts = rel.split("/")[:-1]
    return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


def unique(rel: str, taken: set[str], dirs: set[str], after: dict[str, int]) -> str:
    """`rel`, or `stem-2.ext`... when another member already took that name: names that
    differ only in characters `safe_part` replaces (two CJK names, `a b` and `a_b`)
    come out the same, and neither may be lost."""
    low = rel.lower()
    if low not in taken and low not in dirs:
        return rel
    folder, _, base = rel.rpartition("/")
    stem, dot, ext = base.partition(".")
    n = after.get(low, 2)
    while True:
        candidate = f"{folder + '/' if folder else ''}{stem}-{n}{dot}{ext}"
        n += 1
        if candidate.lower() not in taken and candidate.lower() not in dirs:
            after[low] = n
            return candidate


def members(path: str):
    """(member name, opener) for each regular file, one at a time. Links, devices and
    directories are passed over: a link could point anywhere on the workspace."""
    too_many = Refused(f"it holds more than {MAX_FILES} files")
    if path.lower().endswith(".zip"):
        if not zipfile.is_zipfile(path):
            raise Refused("it is not a readable zip")
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES:
                raise too_many
            for info in infos:
                kind = stat.S_IFMT(info.external_attr >> 16)   # 0 when a Windows zipper left it out
                if info.is_dir() or (kind and kind != stat.S_IFREG):
                    continue
                if info.flag_bits & 0x1:
                    raise Refused("it is password-protected")
                yield info.filename, (lambda info=info: archive.open(info))
        return
    try:
        archive = tarfile.open(path, mode="r|*")
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise Refused("it is not a readable tar") from exc
    with archive:
        for seen, info in enumerate(archive, 1):
            if seen > MAX_FILES:
                raise too_many
            if info.isreg():
                yield info.name, (lambda info=info: archive.extractfile(info))
            archive.members = []                    # a stream never looks back


def unpack(path: str, into: str) -> list[str]:
    """Write the archive's files under `into`. Returns them, relative to it. Sizes are
    counted as the bytes come out, not read off the headers, which can lie."""
    written: list[str] = []
    taken: set[str] = set()      # files written, lower-cased
    dirs: set[str] = set()       # folders made, lower-cased
    after: dict[str, int] = {}
    total = 0
    for member, opener in members(path):
        rel = member_path(member)
        if rel is None:
            continue
        parents = folders(rel)
        if any(d.lower() in taken for d in parents):
            continue            # a file where a folder must go: a malformed archive
        rel = unique(rel, taken, dirs, after)
        taken.add(rel.lower())
        dirs.update(d.lower() for d in parents)
        target = os.path.join(into, *rel.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with opener() as source, open(target, "wb") as sink:
            while chunk := source.read(READ):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise Refused(f"it unpacks to more than {MAX_BYTES >> 20} MB")
                sink.write(chunk)
        written.append(rel)
    return written


def why(exc: BaseException) -> str:
    """An error in a few words, without the paths an OSError carries."""
    if isinstance(exc, MemoryError):
        return "it needs more memory than unpacking is allowed"
    if isinstance(exc, OSError) and exc.strerror:
        return exc.strerror
    return (str(exc) or type(exc).__name__)[:200]


def limit() -> None:
    """At most `MAX_MEMORY` of address space and `TIME_LIMIT_S` of time, where the
    platform has the means."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY, MAX_MEMORY))
    except (ImportError, ValueError, OSError):
        pass
    if hasattr(signal, "SIGALRM"):
        def expired(_signum, _frame):
            raise Refused(f"it took more than {TIME_LIMIT_S} s to unpack")
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(TIME_LIMIT_S)
    else:
        # Windows: no alarm, and the backend's own timeout kills the shell and not this.
        # Exit at the limit rather than finish later and rename a folder into place
        # after the caller was told it was not unpacked.
        import threading

        def expired_here() -> None:
            print("UNPACK_REFUSED " + json.dumps({"why": f"it took more than {TIME_LIMIT_S} s to unpack"}),
                  flush=True)
            os._exit(3)
        timer = threading.Timer(TIME_LIMIT_S, expired_here)
        timer.daemon = True
        timer.start()


def main(archive: str, dest: str, stem: str) -> dict:
    work = f"{dest}.partial-unpack"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        written = unpack(archive, work)
        if not written:
            raise Refused("it holds no files")
        root = work
        tops = {rel.split("/", 1)[0] for rel in written}
        # A top folder named like the archive is taken off, so `wing.zip` holding
        # `wing/part.step` gives `uploads/wing/part.step`. Any other is kept: `polyMesh/`
        # and `system/` are names tools look for.
        if len(tops) == 1 and all("/" in rel for rel in written) \
                and next(iter(tops)).lower() == safe_part(stem).lower():
            root = os.path.join(work, tops.pop())
            written = [rel.split("/", 1)[1] for rel in written]
        final, n = dest, 2
        while os.path.lexists(final):
            final, n = f"{dest}-{n}", n + 1
        os.rename(root, final)
        listed, size = [], 0
        for rel in written:
            size += len(rel) + 4
            if size > LISTED:
                break
            listed.append(rel)
        return {"suffix": final[len(dest):], "count": len(written), "files": listed}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__" or (len(sys.argv) == 4 and sys.argv[0] == "-c"):
    try:
        limit()
        answer = main(*sys.argv[1:4])
        print("UNPACKED " + json.dumps(answer))
    except Refused as exc:
        print("UNPACK_REFUSED " + json.dumps({"why": str(exc)}))
    except BaseException as exc:  # noqa: BLE001 - every failure is an answer, never a trace
        print("UNPACK_REFUSED " + json.dumps({"why": f"it could not be read ({why(exc)})"}))
