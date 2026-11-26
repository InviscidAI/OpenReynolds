"""Files too big for a sync, fetched straight from the workspace's copy in S3.

A sync copies files through the workspace service in archives held in memory, so it
leaves any file over its one-file limit (and anything past its budget) on the
instance, and says so. `openreynolds pull` asked for exactly those files and still
could not have them: a 9.55 GB result stayed out of reach (InviscidAI/ui#137).

The workspace service hands out short-lived links to its copy of a study's files
(`GET /v1/studies/{id}/workspace/links`); each file is streamed from S3 to disk here,
into a `.part` beside its place, and renamed into it only once every byte has come.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx2 as httpx

from .browse import human
from .mirror import OutsideMirror, Skip, local_for

CHUNK = 4 * 1024 * 1024
HEADROOM = 1024 * 1024 * 1024
"""Disk kept free after a fetch: a download that fills the disk breaks everything else."""


@dataclass
class Fetched:
    pulled: list[Path] = field(default_factory=list)
    bytes_pulled: int = 0
    current: list[Path] = field(default_factory=list)
    """Already here as the copy has it: same size and the copy's own write time."""
    failed: list[str] = field(default_factory=list)

    def done(self) -> set[str]:
        """Local paths that are now as the copy has them."""
        return {str(p) for p in self.pulled + self.current}


def fetch(foamd_url: str, api_key: str, study_id: str, skips: list[Skip], local_dir: Path, *,
          http: httpx.Client | None = None, say: Callable[[str], None] = print) -> Fetched:
    """Fetch every `too_big` file in `skips` from the workspace's copy into `local_dir`,
    where a sync would have put it. A file the copy does not have yet (a running study
    copies its workspace every few minutes), or that does not fit on this disk, is
    reported, not raised."""
    out = Fetched()
    wanted = [s for s in skips if s.too_big]
    if not wanted:
        return out
    client = http or httpx.Client(timeout=httpx.Timeout(60.0, read=600.0), follow_redirects=True)
    try:
        for skip in wanted:
            try:
                target = local_for(local_dir, skip.path)
            except OutsideMirror:
                out.failed.append(f"{skip.path}: would land outside the study")
                continue
            link, problem = _link(client, foamd_url, api_key, study_id, skip.path)
            if link is None:
                out.failed.append(f"{skip.path}: {problem}")
                continue
            size, mtime = int(link.get("size") or 0), int(link.get("mtime") or 0)
            # Size alone would keep a stale field: on a fixed mesh every write of it is
            # the same size. The copy's write time is put on the file when it lands.
            if target.is_file() and target.stat().st_size == size and mtime and int(target.stat().st_mtime) == mtime:
                out.current.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(target.parent).free
            if free < size + HEADROOM:
                out.failed.append(f"{skip.path}: {human(size)}, and this disk has {human(free)} free")
                continue
            as_of = f", as of {link['as_of']}" if link.get("as_of") else ""
            say(f"fetching {skip.path} ({human(size)}) from the workspace's copy{as_of}")
            try:
                _stream(client, str(link["url"]), target, size, mtime)
            except (httpx.HTTPError, OSError) as exc:
                out.failed.append(f"{skip.path}: {type(exc).__name__}: {exc}")
                continue
            out.pulled.append(target)
            out.bytes_pulled += size
    finally:
        if http is None:
            client.close()
    return out


def _link(client: httpx.Client, foamd_url: str, api_key: str, study_id: str,
          path: str) -> tuple[dict | None, str]:
    """The file's link (with the copy's `as_of`), or None and why not."""
    try:
        r = client.get(f"{foamd_url.rstrip('/')}/v1/studies/{study_id}/workspace/links",
                       params={"path": path}, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError as exc:
        return None, f"the workspace service could not be reached ({type(exc).__name__})"
    if r.status_code == 404:
        return None, "not in the workspace's copy yet; a running study is copied every few minutes"
    if r.status_code != 200:
        return None, f"the workspace service answered {r.status_code}"
    body = r.json()
    one = next((f for f in body.get("files") or [] if f.get("path") == path), None)
    if one is None:
        return None, "not in the workspace's copy"
    return {**one, "as_of": body.get("as_of")}, ""


def _stream(client: httpx.Client, url: str, target: Path, size: int, mtime: int = 0) -> None:
    part = target.with_name(target.name + ".part")
    got = 0
    try:
        with client.stream("GET", url) as r, part.open("wb") as f:
            r.raise_for_status()
            for chunk in r.iter_bytes(CHUNK):
                got += len(chunk)
                f.write(chunk)
        if got != size:
            raise OSError(f"{got} bytes came of {size}")
        if mtime:
            os.utime(part, (mtime, mtime))
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
