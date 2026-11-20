"""`openreynolds view`: a pulled study's pictures and reports as one local page.

A finished study already contains the pictures that answer "what shape is the
geometry, what does the mesh look like, what did the field do": `geometry_view.py`
drew the surface from four fixed angles, `mesh_look.py` coloured the boundary one
colour per patch, `first_look.py` assembled the contact sheet, `animate.py` left a
directory of frames. They are on this machine after `openreynolds pull`, scattered
across `renders/`, `files/<case>/renders/`, `figs/`, `*_frames/`. What was missing
was a way to *look* at them: `openreynolds renders` lists one flat folder and
`--open` hands you a file manager.

So this writes one HTML page next to the study, grouped by the directory each
picture came from, newest first, and opens it. No ParaView, no server, no network.

**It links the pictures, it does not embed them.** A page of relative `<img src>`
is a few kilobytes whatever the study weighs, it opens instantly, and
`loading="lazy"` -- the browser's own, not a script's -- means a `*_frames/`
directory of 32 PNGs costs nothing until it is scrolled to. The page that travels
is a different page and already exists: `toolbox/gallery.py` embeds every image as
a base64 data URI precisely so it can be attached to a message and opened on a
laptop that has never seen `/work`. This one is for the laptop the study was pulled
to, so it reads the files where they lie.

There is no JavaScript in the page at all -- no CDN, no vendored library, no
`<script>`. That is not an economy, it is the whole reason it works offline.

What it deliberately does not do is stream the mesh to the browser for
interactive rotation. See `NOT_INTERACTIVE`.
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
"""What goes on the page as a picture. The same set `toolbox/gallery.py` embeds."""

REPORT_SUFFIXES = (".md", ".json", ".csv", ".html")
"""What goes on the page as a link. A browser shows Markdown and JSON as text,
which is all a report wants here; `.html` catches a `gallery.html` the study
already carries, so the embedded page is reachable from this one."""

PAGE_NAME = "view.html"

SECTION_LIMIT = 60
"""Pictures listed per directory. A `*_frames/` directory is a flipbook, not a
gallery -- 32 of them in one real study, and `animate.py` will write hundreds --
and a page that lists every frame buries the one render that matters. The rest are
counted, not hidden: the line says how many and where they are."""

SKIP_DIRS = frozenset({".reynolds", ".git", "__pycache__", "node_modules"})
"""Directories not walked by name. `.reynolds` is the study's own bookkeeping."""

PROCESSOR = re.compile(r"processor\d+$")
"""`processor0`, `processor1`, ... -- one rank's copy of what the reconstructed case
already holds. Matched rather than listed, because a decomposed case has as many of
them as it had cores and a literal `processor0` in the skip list would walk the
other seven. Nothing in them can reach the page anyway (an OpenFOAM field file has
no suffix), so this is what it costs to walk them, not what it would show."""

NOT_INTERACTIVE = (
    "This page shows the pictures the study rendered. It does not stream the mesh "
    "to the browser for rotation: that needs a surface extracted from polyMesh, and "
    "pyvista/VTK is not part of a stock install (pyproject.toml declares it under "
    "the toolbox and dev extras only)."
)


def _is_hidden(part: str) -> bool:
    return part.startswith(".") and part not in (".", "..")


def walk(root: Path) -> tuple[list[Path], list[Path]]:
    """Every picture and every report under `root`, as two sorted lists.

    One pass, because a study is a deep tree on a laptop's disk and two walks of
    666 MB to answer one page is a second of nothing.
    """
    pictures: list[Path] = []
    reports: list[Path] = []
    for here, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                         and not _is_hidden(d) and not PROCESSOR.match(d))
        for name in names:
            if _is_hidden(name) or name == PAGE_NAME:
                continue
            suffix = Path(name).suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                pictures.append(Path(here) / name)
            elif suffix in REPORT_SUFFIXES:
                reports.append(Path(here) / name)
    return pictures, reports


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        # A file that went away between the walk and the stat sorts oldest rather
        # than taking the page down with it.
        return 0.0


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def sections(pictures: list[Path], root: Path) -> list[tuple[str, list[Path], int]]:
    """The pictures grouped by the directory they are in, newest directory first.

    The directory *is* the grouping. Without the manifest there is nothing else to
    group by, and the directory is not a poor substitute for it: `renders_zoom/`,
    `figs/viv_cycle_frames/` and `cylRe100/renders_wall/` were each named by
    whatever wrote them, so the folder already says what the pictures are of.

    Each entry is (label, the pictures to show, how many were left off).
    """
    groups: dict[Path, list[Path]] = {}
    for picture in pictures:
        groups.setdefault(picture.parent, []).append(picture)
    out = []
    for directory, found in groups.items():
        found.sort(key=_mtime, reverse=True)
        shown = found[:SECTION_LIMIT]
        out.append((_label(directory, root), shown, len(found) - len(shown)))
    out.sort(key=lambda row: _mtime(row[1][0]), reverse=True)
    return out


def _label(directory: Path, root: Path) -> str:
    try:
        rel = directory.relative_to(root)
    except ValueError:
        return str(directory)
    return "." if rel == Path(".") else rel.as_posix()


def _link(target: Path, page: Path) -> str:
    """`target` as the page at `page` must spell it.

    Relative, so the page and the pictures move together and nothing depends on
    where the study sits. `os.path.relpath` across drives raises on Windows -- a
    page written to a different drive from the study it describes -- and the
    absolute path is right there to fall back on.
    """
    try:
        return Path(os.path.relpath(target, page.parent)).as_posix()
    except ValueError:
        return target.as_posix()


def human(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    for unit, scale in (("KB", 1024), ("MB", 1024**2), ("GB", 1024**3)):
        if count < scale * 1024 or unit == "GB":
            return f"{count / scale:.1f} {unit}"
    return f"{count} B"


STYLE = """
:root { color-scheme: light dark;
        --bg: #fbfbfa; --fg: #1a1a18; --dim: #6b6b66; --line: #e2e2dd; --card: #fff; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #16161a; --fg: #e9e9e4; --dim: #9a9a93; --line: #2d2d33; --card: #1e1e23; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 2rem 1rem 4rem; background: var(--bg); color: var(--fg);
       font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: .95rem; font-weight: 600; margin: 2.5rem 0 .75rem;
     padding-bottom: .4rem; border-bottom: 1px solid var(--line);
     font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
p.sub, p.note { color: var(--dim); margin: .25rem 0 0; font-size: .85rem; }
p.note { margin-top: 1.5rem; }
.grid { display: grid; gap: 1rem; align-items: start;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
figure { margin: 0; background: var(--card); border: 1px solid var(--line);
         border-radius: 8px; overflow: hidden; }
figure img { display: block; width: 100%; height: auto; background: #fff; }
figcaption { padding: .5rem .6rem; font-size: .78rem; color: var(--dim);
             font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
             word-break: break-all; }
figcaption b { color: var(--fg); font-weight: 600; }
a { color: inherit; text-decoration: none; }
figure a:hover { outline: 2px solid var(--dim); }
ul.reports { list-style: none; padding: 0; margin: 0;
             display: grid; gap: .35rem;
             grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }
ul.reports a { display: block; padding: .5rem .7rem; background: var(--card);
               border: 1px solid var(--line); border-radius: 6px; font-size: .82rem;
               font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
               word-break: break-all; }
ul.reports a:hover { border-color: var(--dim); }
ul.reports span { color: var(--dim); }
.more { color: var(--dim); font-size: .82rem; margin: .75rem 0 0; }
"""


def _figure(picture: Path, page: Path) -> str:
    href = html.escape(_link(picture, page))
    return (
        f'<figure><a href="{href}"><img src="{href}" loading="lazy" '
        f'alt="{html.escape(picture.name)}"></a>'
        f"<figcaption><b>{html.escape(picture.name)}</b><br>"
        f"{human(_size(picture))}</figcaption></figure>"
    )


def render(root: Path, page: Path, pictures: list[Path], reports: list[Path],
           title: str = "") -> str:
    """The whole page, as a string. No script, no font, no stylesheet, no network."""
    groups = sections(pictures, root)
    total = sum(_size(p) for p in pictures)
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(title or root.name)}</title>",
        f"<style>{STYLE}</style></head><body><main>",
        f"<h1>{html.escape(title or root.name)}</h1>",
        f'<p class="sub">{len(pictures)} picture(s), {human(total)}, in '
        f"{len(groups)} folder(s) under {html.escape(str(root))}</p>",
    ]
    if not pictures and not reports:
        parts.append(
            '<p class="note">Nothing to show: no pictures and no reports under this '
            "study. If the study ran on an instance, its files may not be here yet "
            "-- <code>openreynolds pull</code> brings them down.</p>"
        )
    for label, shown, left in groups:
        parts.append(f"<h2>{html.escape(label)}</h2>")
        parts.append('<div class="grid">')
        parts.extend(_figure(p, page) for p in shown)
        parts.append("</div>")
        if left:
            parts.append(
                f'<p class="more">{left} more picture(s) in this folder, not listed '
                f"here. They are in {html.escape(label)}.</p>"
            )
    if reports:
        parts.append("<h2>reports and data</h2><ul class=\"reports\">")
        for report in sorted(reports, key=_mtime, reverse=True):
            parts.append(
                f'<li><a href="{html.escape(_link(report, page))}">'
                f"{html.escape(_label(report, root))}"
                f" <span>{human(_size(report))}</span></a></li>"
            )
        parts.append("</ul>")
    parts.append(f'<p class="note">{html.escape(NOT_INTERACTIVE)}</p>')
    parts.append("</main></body></html>")
    return "\n".join(parts)


def build(root: Path, out: Path | None = None, title: str = "") -> tuple[Path, int, int]:
    """Write the page for the study at `root`. Returns (page, pictures, reports)."""
    root = Path(root)
    page = Path(out) if out else root / PAGE_NAME
    pictures, reports = walk(root)
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(render(root, page, pictures, reports, title=title), encoding="utf-8")
    return page, len(pictures), len(reports)


def demo() -> None:
    """The checks that fail if the grouping, the cap or the linking breaks."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "study"
        (root / "renders").mkdir(parents=True)
        (root / "files" / "case" / "figs").mkdir(parents=True)
        (root / ".reynolds").mkdir()
        (root / "files" / "case" / "figs" / "U.png").write_bytes(b"\x89PNG" + b"0" * 10)
        (root / "files" / "case" / "REPORT.md").write_text("# it ran")
        (root / ".reynolds" / "hidden.png").write_bytes(b"\x89PNG")
        for n in range(SECTION_LIMIT + 5):
            frame = root / "renders" / f"frame_{n:03d}.png"
            frame.write_bytes(b"\x89PNG")
            os.utime(frame, (1_700_000_000, 1_700_000_000))
        # Newest in its folder and newest overall, so the cap keeps it and `renders`
        # sorts first. Stamped rather than ordered, because 65 writes on a fast disk
        # can share an mtime and a tie would make this test flap.
        newest = root / "renders" / "mesh.png"
        newest.write_bytes(b"\x89PNG" + b"0" * 2000)
        os.utime(newest, (1_800_000_000, 1_800_000_000))

        pictures, reports = walk(root)
        names = {p.name for p in pictures}
        assert "hidden.png" not in names, "a .reynolds picture must not be walked"
        assert "mesh.png" in names and "U.png" in names
        assert [Path(r).name for r in reports] == ["REPORT.md"], reports

        groups = sections(pictures, root)
        labels = {label for label, _, _ in groups}
        assert labels == {"renders", "files/case/figs"}, labels
        capped = next(row for row in groups if row[0] == "renders")
        assert len(capped[1]) == SECTION_LIMIT, len(capped[1])
        assert capped[2] == 6, capped[2]  # 65 frames + mesh.png - 60 shown

        page, n_pictures, n_reports = build(root)
        assert page == root / PAGE_NAME and page.is_file()
        assert (n_pictures, n_reports) == (len(pictures), 1)
        text = page.read_text(encoding="utf-8")
        assert "<script" not in text and "http://" not in text and "https://" not in text
        assert 'loading="lazy"' in text
        assert "renders/mesh.png" in text, "pictures are linked relative to the page"
        assert "files/case/REPORT.md" in text
        assert human(2000) == "2.0 KB", human(2000)

        far = Path(tmp) / "elsewhere" / "p.html"
        build(root, out=far)
        assert "../study/renders/mesh.png" in far.read_text(encoding="utf-8")
    print("viewer: ok")


if __name__ == "__main__":
    demo()
