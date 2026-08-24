#!/usr/bin/env python3
"""Live end-to-end check of the workspace client, with no model in the loop.

Exercises the real `Backend` against the real service, so it verifies the contract
this package was written against rather than a fake of it. Needs credentials:

    FOAMD_URL=... FOAMD_API_KEY=... python scripts/smoke.py
    python scripts/smoke.py --instance <id>     # reuse a specific workspace
    python scripts/smoke.py --keep              # leave the study directory behind

It never deletes an instance, so the persistent volume is safe.
"""

from __future__ import annotations

import argparse
import base64
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openreynolds.backend import hosted  # noqa: E402
from openreynolds.backend.base import BackendError  # noqa: E402
from openreynolds.browse import Browser  # noqa: E402
from openreynolds.config import Config  # noqa: E402
from openreynolds.stopping import running_solvers, stop_everything  # noqa: E402
from openreynolds.store import Store, new_study_id  # noqa: E402
from openreynolds.tools import ToolContext, dispatch  # noqa: E402

PASSED = 0
FAILED = 0


def check(condition: bool, label: str, detail: str = "") -> bool:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {label}")
        return True
    FAILED += 1
    print(f"  FAIL  {label}")
    if detail:
        print(f"        {detail.strip()}")
    return False


def section(title: str) -> None:
    print(f"\n== {title}")


SURFACE_SCRIPT = '''\
import pyvista as pv

pv.OFF_SCREEN = True
pv.Box(bounds=(0, 0.06, 0, 0.02, 0, 0.01)).triangulate().save("/work/.smoke/box.stl")
pv.Plane(center=(0.05, 0.01, 0.005), i_size=0.01, j_size=0.01).triangulate().save(
    "/work/.smoke/lid.stl"
)
print("SURFACES_OK")
'''


PLOT_SCRIPT = """\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 200)
plt.plot(x, np.sin(x))
plt.title("openreynolds smoke")
plt.savefig("/work/.smoke/plot.png", dpi=80)
print("PLOT_OK")
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default=None)
    parser.add_argument("--keep", action="store_true", help="keep the local study directory")
    args = parser.parse_args()

    cfg = Config.load()
    gaps = [g for g in cfg.missing() if g != "ANTHROPIC_API_KEY"]
    if gaps:
        print(f"missing configuration: {', '.join(gaps)}")
        return 2

    study_id = "smoke-" + new_study_id()
    store = Store(cfg.studies_dir, study_id)

    section("acquire a workspace")
    started = time.monotonic()
    backend, client, instance_id = hosted.acquire(cfg.foamd_url, cfg.foamd_api_key, args.instance)
    check(bool(instance_id), f"instance {instance_id} ready in {time.monotonic() - started:.1f}s")

    ctx = ToolContext(backend=backend, store=store, max_output=cfg.max_tool_output)

    try:
        section("bash")
        out, err = dispatch(ctx, "bash", {"cmd": "echo hello; echo $WM_PROJECT_VERSION"})
        check(not err and "hello" in out, "a command runs with the OpenFOAM env sourced", out[:200])
        check("2512" in out, "OpenFOAM v2512 is on PATH", out[:200])

        out, err = dispatch(ctx, "bash", {"cmd": "exit 7"})
        check("exit_code: 7" in out, "a non-zero exit code comes back intact")

        section("truncation")
        out, _ = dispatch(ctx, "bash", {"cmd": "python3 -c \"print('B' * 200000)\""})
        check("[truncated" in out, "long output is marked, not silently cut")
        check("/work/" in out and "read_file" in out, "the marker points at the full log")

        section("files")
        dispatch(ctx, "bash", {"cmd": "mkdir -p /work/.smoke"})
        body = "".join(f"line {i}\n" for i in range(1, 501))
        _, err = dispatch(ctx, "write_file", {"path": "/work/.smoke/lines.txt", "content": body})
        check(not err, "write_file")

        out, err = dispatch(ctx, "read_file", {"path": "/work/.smoke/lines.txt", "offset": 0,
                                               "limit": 20})
        check(not err and "line 1" in out, "read_file windows by byte offset", out[:200])
        check(f"of {len(body)}" in out, "the window states the full size")

        out, _ = dispatch(ctx, "read_file", {"path": "/work/.smoke"})
        check("directory" in out and "lines.txt" in out, "read_file lists a directory")

        section("toolbox sync")
        backend.put_tree(Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox",
                         "/work/.toolbox")
        info = backend.stat("/work/.toolbox")
        check("log_digest.py" in info.entries, "the toolbox landed", str(info.entries)[:200])
        out, _ = dispatch(ctx, "bash", {"cmd": "python3 /work/.toolbox/cells_estimate.py --help"})
        check("exit_code: 0" in out, "a toolbox script runs in the container")

        section("jobs")
        out, err = dispatch(ctx, "job_start", {"cmd": "for i in $(seq 1 5); do echo tick $i; "
                                                      "sleep 1; done; echo done",
                                               "name": "ticker"})
        job_id = out.split()[2] if not err else ""
        check(bool(job_id), "job_start returns an id", out)

        seen_partial = False
        for _ in range(30):
            out, _ = dispatch(ctx, "job_check", {"job_id": job_id})
            if "tick" in out:
                seen_partial = True
            if "status=exited" in out:
                break
            time.sleep(1)
        check(seen_partial, "job_check streams the log while the job runs")
        check("status=exited" in out and "exit_code=0" in out, "the job finished cleanly", out[:300])

        section("kill_on")
        out, _ = dispatch(ctx, "job_start", {
            "cmd": "echo starting; sleep 1; echo '--> FOAM FATAL ERROR: smoke'; sleep 30",
            "name": "fatal", "kill_on": ["FOAM FATAL"]})
        fatal_id = out.split()[2]
        for _ in range(30):
            out, _ = dispatch(ctx, "job_check", {"job_id": fatal_id})
            if "status=" in out and "status=running" not in out:
                break
            time.sleep(1)
        check("kill_on_match" in out, "a kill_on regex the caller chose fired", out[:300])
        check("FOAM FATAL ERROR: smoke" in out, "the matched line is reported back")

        section("fetch")
        dispatch(ctx, "write_file", {"path": "/work/.smoke/plot.py", "content": PLOT_SCRIPT})
        out, _ = dispatch(ctx, "bash", {"cmd": "python3 /work/.smoke/plot.py", "timeout_s": 180})
        check("PLOT_OK" in out, "matplotlib renders headlessly in the container", out[:300])

        out, err = dispatch(ctx, "fetch", {"paths": ["/work/.smoke/plot.png"]})
        # Members come back relative to the workspace root, so the shape is kept.
        local = store.fetch_dir() / ".smoke" / "plot.png"
        check(not err and local.exists() and local.stat().st_size > 1000,
              f"fetch pulled the PNG to {local}", out[:300])


        section("seeing")
        # The picture has to come back as a picture. Everything else about a render is
        # indistinguishable from a render of nothing.
        content, err = dispatch(ctx, "read_file", {"path": "/work/.smoke/plot.png"})
        blocks = content if isinstance(content, list) else []
        image = next((b for b in blocks if b.get("type") == "image"), None)
        check(not err and image is not None, "read_file on a PNG returns an image block",
              str(content)[:300])
        if image:
            raw = base64.b64decode(image["source"]["data"])
            check(raw[:8] == b"\x89PNG\r\n\x1a\n",
                  "and the bytes are the PNG the container wrote")
            caption = next((b["text"] for b in blocks if b.get("type") == "text"), "")
            check("x" in caption and "image/png" in caption,
                  "with its size and shape stated alongside", caption)

        content, _ = dispatch(ctx, "read_file", {"path": "/work/.smoke/plot.py"})
        check(isinstance(content, str), "an ordinary file still comes back as text")

        section("looking at the workspace")
        browser = Browser(backend, store)
        entries = browser.tree("/work/.smoke")
        names = {entry.name for entry in entries}
        check("plot.png" in names and "plot.py" in names,
              "the tree lists what is there", ", ".join(sorted(names)))
        check(any(e.size > 1000 for e in entries if e.name == "plot.png"),
              "with real sizes on it")

        text, is_text = browser.read("/work/.smoke/plot.py")
        check(is_text and "matplotlib" in text, "a text file reads back")
        _, is_text = browser.read("/work/.smoke/plot.png")
        check(not is_text, "and a binary one is described rather than dumped")

        section("geometry")
        dispatch(ctx, "write_file", {"path": "/work/.smoke/geo.py", "content": SURFACE_SCRIPT})
        out, _ = dispatch(ctx, "bash", {"cmd": "python3 /work/.smoke/geo.py", "timeout_s": 280})
        check("SURFACES_OK" in out, "pyvista writes an STL headlessly", out[-400:])

        out, _ = dispatch(ctx, "bash", {
            "cmd": "cd /work/.smoke && python3 /work/.toolbox/geometry_view.py . --out r",
            "timeout_s": 280,
        })
        check("open edges 0" in out, "geometry_view reports a closed surface as closed",
              out[-600:])
        check("not closed, so no volume" in out,
              "and an open one as open, with no volume quoted for it")
        check("OpenFOAM reads these as metres" in out,
              "extents are stated in the units OpenFOAM will read them in")

        content, err = dispatch(ctx, "read_file", {"path": "/work/.smoke/r/geometry.png"})
        check(not err and isinstance(content, list),
              "and the drawing it produced can be looked at", str(content)[:200])

        section("stopping what outlived its job")
        backend.exec("cp /bin/sleep /tmp/simpleFoam", timeout_s=60)
        loose = backend.job_start("/tmp/simpleFoam 600", name="smoke-loose")
        store.record_job(loose, cmd="/tmp/simpleFoam 600", name="smoke-loose")
        time.sleep(4)
        check("simpleFoam" in running_solvers(backend),
              "a solver outside the job's process group is seen")

        report = stop_everything(backend, store, force=True)
        check(report.clean, "stop --force actually stops it", "; ".join(report.lines()))
        check("simpleFoam" not in running_solvers(backend), "and it is gone afterwards")
        backend.exec("rm -f /tmp/simpleFoam", timeout_s=60)

        section("job records survive a restart")
        reopened = Store(cfg.studies_dir, study_id)
        check(job_id in reopened.session.jobs, "job ids persist locally (no list-jobs endpoint)")

    finally:
        try:
            backend.exec("rm -rf /work/.smoke")
        except BackendError:
            pass
        backend.close()
        if not args.keep:
            shutil.rmtree(store.dir, ignore_errors=True)

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
