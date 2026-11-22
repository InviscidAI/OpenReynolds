"""A kernel whose directory is gone has to stop, because nothing can reach it.

Every request, control and reply between a session and its kernel travels through a
file in the kernel's own directory. `LocalBackend.close()` stops the kernel on a normal
exit, but a session that is killed rather than closed never runs it, and the directory
is cleaned up without the process. The kernel then polls a path that does not exist,
forever. Ninety-eight had accumulated on one laptop by 2026-11-22, the oldest from the
previous day, holding about 2.5 GB between them, and they leave no trace on disk to
find them by -- which is why this is the driver's own job rather than a reaper's.

The driver is written into the workspace and run there, so this test runs it the same
way: as a subprocess, from a real directory, which is then taken away.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from openreynolds.backend import kernel as kernel_mod


def _driver_in(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "driver.py"
    path.write_text(kernel_mod._DRIVER, encoding="utf-8")
    return path


def _wait_ready(folder: Path, proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (folder / "ready").exists():
            return
        if proc.poll() is not None:
            pytest.fail(f"the driver exited before it was ready: {proc.returncode}")
        time.sleep(0.05)
    pytest.fail("the driver never wrote its ready file")


def test_the_kernel_stops_when_its_directory_is_taken_away(tmp_path):
    work = tmp_path / "case"
    work.mkdir()
    folder = tmp_path / ".kernel" / "deadbeef"
    _driver_in(folder)

    proc = subprocess.Popen([sys.executable, str(folder / "driver.py"), str(work)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_ready(folder, proc)
        assert proc.poll() is None, "it should still be serving while its directory is there"

        # What a killed session leaves behind: the directory goes, the process does not.
        for item in folder.iterdir():
            item.unlink()
        folder.rmdir()

        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.1)
        assert proc.poll() is not None, (
            "the kernel outlived its own directory, which is the leak this prevents")
        assert proc.returncode == 0, f"it should stop cleanly, got {proc.returncode}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def test_it_keeps_serving_while_the_directory_is_there(tmp_path):
    """The other half: a guard that exits early would be worse than the leak."""
    work = tmp_path / "case"
    work.mkdir()
    folder = tmp_path / ".kernel" / "feedface"
    _driver_in(folder)

    proc = subprocess.Popen([sys.executable, str(folder / "driver.py"), str(work)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_ready(folder, proc)
        time.sleep(3.0)          # several times the one-second check interval
        assert proc.poll() is None, "it stopped while its directory was still there"
    finally:
        proc.kill()
        proc.wait(timeout=10)
