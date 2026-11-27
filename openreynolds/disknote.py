"""The service's word that the disk under a running job is filling (#133 on the service).

A solve runs detached, so the agent is often not running a command when its
workspace disk fills: it is waiting in `job_check`, or its turn is over and the
harness is watching the job. The service puts a note on a running job's status
(`JobStatus.disk`) past a share of what the solver can write. This decides when that
note is news, so the agent is woken with it once, and again only when the disk has
changed size (the service grows it once): `watch` wakes the agent with it, and a held
`job_check` ends early to deliver it.
"""
from __future__ import annotations

import threading
from typing import Any

_told: set[tuple[str, int]] = set()
_lock = threading.Lock()


def _key(status: Any) -> tuple[str, int]:
    # The filesystem's size, which changes when a grow reaches the files; the volume's
    # changes a step earlier, while the files still have the old room.
    disk = status.disk or {}
    return (str(status.job_id), int(disk.get("disk_gb") or disk.get("volume_gb") or 0))


def is_news(status: Any) -> bool:
    """Whether this status carries a disk note the agent has not been given yet."""
    if not getattr(status, "disk_note", None):
        return False
    with _lock:
        return _key(status) not in _told


def take(status: Any) -> str | None:
    """The note, if it is news; it is not news again for this job and disk size."""
    if not getattr(status, "disk_note", None):
        return None
    key = _key(status)
    with _lock:
        if key in _told:
            return None
        _told.add(key)
    return status.disk_note


def mark(status: Any) -> None:
    """The agent has seen this note (in a `job_check`), so `watch` need not wake for it."""
    if getattr(status, "disk_note", None):
        with _lock:
            _told.add(_key(status))
