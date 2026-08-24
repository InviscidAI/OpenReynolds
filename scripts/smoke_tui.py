#!/usr/bin/env python3
"""The interface, driven headlessly against the real workspace.

`tests/test_tui.py` drives every pane against a fake that answers instantly and never
fails. That is the right way to test the panes and the wrong way to find out whether
the interface works, because everything it papers over -- a listing that takes two
seconds, a file that is bigger than expected, a path with a space in it, an image that
has to be copied out before it can be seen -- only exists on the far side of a network.

So this is the same panes, the same view, the same browser, pointed at the real
service. No model in the loop and nothing typed: it fills the tree, opens a file, opens
an image, and reports what came back.

    FOAMD_URL=... FOAMD_API_KEY=... python scripts/smoke_tui.py

It never deletes an instance and writes only under a directory of its own.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openreynolds.backend import hosted  # noqa: E402
from openreynolds.backend.base import BackendError  # noqa: E402
from openreynolds.browse import Browser  # noqa: E402
from openreynolds.config import Config  # noqa: E402
from openreynolds.store import Store, new_study_id  # noqa: E402
from openreynolds.terminal import tolerant_stdout  # noqa: E402

tolerant_stdout()

PASSED = 0
FAILED = 0

PLOT = """\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.plot([0, 1, 2], [0, 1, 4])
plt.savefig("{home}/renders/curve.png", dpi=70)
print("PLOT_OK")
"""


def check(condition: bool, label: str, detail: str = "") -> bool:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {label}")
        return True
    FAILED += 1
    print(f"  FAIL  {label}")
    if detail:
        print(f"        {detail.strip()[:400]}")
    return False


async def drive(browser: Browser, home: str) -> None:
    from openreynolds.tui import FilesTree, OpenReynoldsApp, TuiView

    released = asyncio.Event()
    app = OpenReynoldsApp(lambda running: None)

    async with app.run_test() as pilot:
        view = TuiView(app)

        print("\n== the interface finds the workspace")
        await asyncio.to_thread(view.workspace, browser)
        for _ in range(60):
            await pilot.pause()
            tree = app.query_one("#filestree", FilesTree)
            if tree.root.children:
                break
            await asyncio.sleep(0.25)

        tree = app.query_one("#filestree", FilesTree)
        labels = [str(node.label) for node in tree.root.children]
        check(bool(labels), "the files pane filled from the live workspace", str(labels))
        check(tree.root_path == home, f"rooted at this study's own directory ({home})")
        check(any("renders" in label for label in labels), "and shows what was written")

        print("\n== opening a text file")
        app.open_path(f"{home}/notes.md")
        for _ in range(40):
            await pilot.pause()
            if getattr(app.screen, "body", None):
                break
            await asyncio.sleep(0.25)
        body = getattr(app.screen, "body", "")
        check("a note from the smoke run" in body, "the file opened with its contents", body)
        app.pop_screen()
        await pilot.pause()

        print("\n== opening an image")
        app.open_path(f"{home}/renders/curve.png")
        for _ in range(40):
            await pilot.pause()
            if "curve.png" in getattr(app.screen, "body", ""):
                break
            await asyncio.sleep(0.25)
        body = getattr(app.screen, "body", "")
        check("Copied to your machine" in body, "an image is copied out, since it cannot draw", body)
        pulled = [p for p in browser.local() if p.name == "curve.png"]
        check(bool(pulled) and pulled[0].exists(), "and it really landed on this machine",
              str(pulled))

        released.set()


def main() -> int:
    cfg = Config.load()
    if cfg.missing():
        print(f"missing configuration: {', '.join(cfg.missing())}")
        return 2
    try:
        import textual  # noqa: F401
    except ImportError:
        print("textual is not installed; nothing to drive")
        return 2

    backend, _client, instance = hosted.acquire(cfg.foamd_url, cfg.foamd_api_key, None)
    store = Store(cfg.studies_dir, f"smoke-tui-{new_study_id()}")
    home = f"/work/{store.session.study_id}"
    store.session.instance_id = instance
    store.session.home = home
    store.save()
    print(f"instance {instance[:8]}   home {home}")

    try:
        backend.exec(f"mkdir -p {home}/renders", timeout_s=60)
        backend.put_file(f"{home}/notes.md", b"# a note from the smoke run\n")
        backend.put_file(f"{home}/plot.py", PLOT.format(home=home).encode())
        result = backend.exec(f"python3 {home}/plot.py", timeout_s=180)
        check("PLOT_OK" in result.output, "the workspace drew something to look at",
              result.output)

        asyncio.run(drive(Browser(backend, store, home=home), home))
    finally:
        try:
            backend.exec(f"rm -rf {home}", timeout_s=60)
        except BackendError:
            pass
        backend.close()
        shutil.rmtree(store.dir, ignore_errors=True)

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
