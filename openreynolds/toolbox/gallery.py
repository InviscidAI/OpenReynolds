#!/usr/bin/env python3
"""Everything a study has produced, as one page, one picture and one list.

The manifest under `<study>/.reynolds/` already knows what every artifact is for.
What it does not do is hand the artifacts over: "show the user the final results"
still means reading a manifest, resolving a dozen paths and opening a dozen files,
and "show me the latest mesh picture" means knowing which of four renders is the
latest. This turns the manifest into three answers:

- `gallery.html` -- one file, every image embedded as a base64 data URI, grouped
  by what the picture is for. Embedded and not linked because of where a page like
  this ends up: attached to a message, dropped in a Downloads folder, opened on a
  laptop that has never seen `/work`. A page that referenced `renders/mesh.png`
  arrives as a column of broken frames. This one has no external CSS, no script,
  no font and no network of any kind in it, so it is the same page wherever it is
  opened, and it is legible whether the browser is in light or dark mode.
- a contact-sheet PNG of the newest artifact of each kind -- the whole study as
  one `read_file` rather than one per picture.
- the same index as text, for when the answer is wanted in the terminal.

    python3 gallery.py /work/study                  # writes both, prints their paths
    python3 gallery.py /work/study --list           # the index, grouped by kind
    python3 gallery.py /work/study --final          # the paths of the latest of each kind
    python3 gallery.py /work/study --html /tmp/g.html --sheet /tmp/g.png
    python3 gallery.py /work/study --case cylinder  # one case out of several

A row whose file has since been deleted is shown as missing rather than dropped or
raised on. A manifest saying a picture once existed is worth reading -- it is often
the only surviving evidence that a step ran -- and a gallery that dies on the one
study somebody tidied up is a gallery nobody trusts.

This reads state and writes pictures. It draws no conclusions about what any of
them show, and picks nothing for you beyond "newest of its kind".
"""

from __future__ import annotations

import argparse
import base64
import html
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402 - after the backend is fixed
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_state  # noqa: E402


HTML_NAME = "gallery.html"
SHEET_NAME = "contact_sheet.png"

SOURCE_TAG = "gallery"
"""Written into the meta of the rows this script registers, and skipped when the
next run reads the manifest back. Without it every run would add two rows and the
gallery after that would be mostly pictures of previous galleries."""

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
"""What can go into the page as a picture."""

RASTER_SUFFIXES = (".png", ".jpg", ".jpeg")
"""What matplotlib will read back for the contact sheet. It does not read SVG, and
an animated GIF would land on the sheet as whichever frame came first, so both are
listed on the page and left off the sheet."""

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}

EMBED_LIMIT = 12_000_000
"""Bytes per image, above which the page carries the path instead of the picture.
Base64 costs a third on top of the file, and a browser given a few hundred megabytes
of data URI stops being a browser. Anything skipped says so where the picture would
have been, rather than silently going missing."""


# -- one manifest row, resolved ----------------------------------------------------


class Entry:
    """A manifest row plus the three things a page needs and the manifest cannot
    know: whether the file is still there, how big it is, and what to show for it.

    Written out by hand rather than as a dataclass for the same reason the other
    toolbox scripts do it -- these modules are loaded by path, without a
    `sys.modules` entry, and `@dataclass` resolves annotations through `sys.modules`.
    """

    def __init__(self, kind, path, rel="", label="", case="", at="", meta=None):
        self.kind = str(kind or "other")
        self.path = Path(path)
        self.rel = rel or str(path)
        self.label = str(label or "")
        self.case = str(case or "")
        self.at = str(at or "")
        self.meta = meta or {}
        self.exists = False
        self.is_dir = False
        self.size = 0
        self.frames = 0
        self.refresh()

    def refresh(self) -> "Entry":
        try:
            self.exists = self.path.exists()
            self.is_dir = self.path.is_dir()
            if self.is_dir:
                # An animation is registered as its `*_frames/` directory, so its
                # size is the frames in it, not a file size.
                frames = [p for p in self.path.iterdir() if p.suffix.lower() in RASTER_SUFFIXES]
                self.frames = len(frames)
                self.size = sum(p.stat().st_size for p in frames)
            elif self.exists:
                self.size = self.path.stat().st_size
        except OSError:
            # A path on a mount that has gone away reads as missing, which is what
            # it is from here.
            self.exists = False
        return self

    def __repr__(self) -> str:
        return f"Entry({self.kind!r}, {self.rel!r}, exists={self.exists!r})"


def entry_from_row(row: dict[str, Any]) -> Entry:
    return Entry(
        kind=row.get("kind", "other"),
        path=row.get("abspath") or row.get("path", ""),
        rel=str(row.get("path", "")),
        label=row.get("label", ""),
        case=row.get("case", ""),
        at=row.get("at", ""),
        meta=row.get("meta") if isinstance(row.get("meta"), dict) else {},
    )


def is_own_output(row: dict[str, Any]) -> bool:
    """Whether a manifest row was written by this script on an earlier run."""
    meta = row.get("meta")
    return isinstance(meta, dict) and meta.get("source") == SOURCE_TAG


def collect(root: Path | str = ".", case: str = "") -> list[Entry]:
    """Every registered artifact, oldest first, own output excluded.

    `exists=False` on purpose: a row whose file is gone is part of the answer here,
    and dropping it would make a deleted picture look like one that was never made.
    """
    rows = study_state.artifacts(root=root, case=case, exists=False)
    return [entry_from_row(row) for row in rows if not is_own_output(row)]


# -- grouping and picking ----------------------------------------------------------


def kind_order(kind: str) -> int:
    """Where a kind sits on the page. Kinds the manifest does not know about sort
    after the ones it does, rather than being refused a place."""
    try:
        return study_state.KINDS.index(kind)
    except ValueError:
        return len(study_state.KINDS)


def group_by_kind(entries: Iterable[Entry]) -> list[tuple[str, list[Entry]]]:
    """`[(kind, entries)]` in pipeline order -- geometry, mesh, fields, reports --
    which is also the order somebody reads a study in.

    Inside a kind the manifest order is kept, oldest first, so the last one is the
    newest and the sequence of four attempts at a mesh reads as a sequence.
    """
    buckets: dict[str, list[Entry]] = {}
    for entry in entries:
        buckets.setdefault(entry.kind, []).append(entry)
    names = sorted(buckets, key=lambda name: (kind_order(name), name))
    return [(name, buckets[name]) for name in names]


def newest(entries: Iterable[Entry]) -> Entry | None:
    """The last entry whose file is still on disk, or None.

    Last rather than latest-timestamp: `at` is stamped to the second and two
    artifacts written in the same second would tie, while the manifest is
    append-only and its order is never ambiguous.
    """
    present = [entry for entry in entries if entry.exists]
    return present[-1] if present else None


def final_entries(entries: Iterable[Entry]) -> list[tuple[str, Entry]]:
    """The newest surviving artifact of each kind -- the "give the user every final
    result" answer. A kind whose files have all been deleted drops out rather than
    contributing a path that would not open."""
    finals: list[tuple[str, Entry]] = []
    for kind, group in group_by_kind(entries):
        pick = newest(group)
        if pick is not None:
            finals.append((kind, pick))
    return finals


def image_for(entry: Entry, suffixes: Iterable[str] = IMAGE_SUFFIXES) -> Path | None:
    """The picture that stands for an entry, or None if it has none.

    A directory stands for itself through its first frame: an animation is
    registered as its `*_frames/` directory, and its first frame is the only thing
    about it that can be shown in a still page.
    """
    if not entry.exists:
        return None
    wanted = tuple(s.lower() for s in suffixes)
    if entry.is_dir:
        try:
            frames = sorted(p for p in entry.path.iterdir() if p.suffix.lower() in wanted)
        except OSError:
            return None
        return frames[0] if frames else None
    if entry.path.suffix.lower() in wanted:
        return entry.path
    return None


# -- the text index ----------------------------------------------------------------


def human_size(count: int) -> str:
    if count <= 0:
        return "-"
    units = ("B", "kB", "MB", "GB")
    size = float(count)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def describe(entry: Entry) -> str:
    """The right-hand column of one index line: label, case, time, state."""
    bits = []
    if entry.label:
        bits.append(entry.label)
    if entry.case:
        bits.append(f"case {entry.case}")
    if entry.at:
        bits.append(entry.at)
    if not entry.exists:
        bits.append("MISSING")
    elif entry.is_dir:
        bits.append(f"{entry.frames} frames")
    return "  ".join(bits)


def text_index(groups: list[tuple[str, list[Entry]]]) -> str:
    """The manifest as a page of text, grouped by purpose."""
    lines: list[str] = []
    for kind, group in groups:
        gone = sum(1 for entry in group if not entry.exists)
        heading = f"{kind}  ({len(group)})"
        if gone:
            heading += f"  {gone} missing"
        lines.append(heading)
        width = max(len(entry.rel) for entry in group)
        for entry in group:
            lines.append(f"  {entry.rel:<{width}}  {describe(entry)}".rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def final_paths(finals: list[tuple[str, Entry]]) -> list[str]:
    """Absolute paths, one per line, nothing else on the line -- so the output can
    be read by a person or fed straight to something that opens files."""
    return [str(entry.path) for _kind, entry in finals]


# -- the page ----------------------------------------------------------------------


def data_uri(path: Path, limit: int = EMBED_LIMIT) -> str:
    """A file as a `data:` URI, or an empty string if it is too big or unreadable.

    Callers treat the empty string as "show the path instead", which keeps the
    decision in one place: a page that half-embeds is still a page.
    """
    path = Path(path)
    try:
        if path.stat().st_size > limit:
            return ""
        payload = path.read_bytes()
    except OSError:
        return ""
    mime = MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


STYLE = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #16202a; --muted: #5c6f7d; --card: #f4f7f9;
  --line: #d3dde4; --edge: #b7c5cf; --tag: #e3ebf1;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12171c; --fg: #e6edf3; --muted: #9aabb8; --card: #1b2229;
    --line: #2b353e; --edge: #3a4650; --tag: #253039;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px; line-height: 1.5;
}
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; font-weight: 600; }
h2 {
  font-size: 1.05rem; margin: 2.5rem 0 .75rem; font-weight: 600;
  border-bottom: 1px solid var(--line); padding-bottom: .35rem;
}
h2 .count { color: var(--muted); font-weight: 400; font-size: .85rem; margin-left: .5rem; }
.sub { color: var(--muted); font-size: .85rem; margin: 0 0 .5rem; }
.grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }
.card {
  border: 1px solid var(--line); border-radius: 8px; background: var(--card);
  overflow: hidden; display: flex; flex-direction: column;
}
.card.gone { border-style: dashed; }
.shot {
  /* Renders are drawn on white and many carry transparency, so the frame stays
     white in dark mode too -- otherwise the axes and captions in the picture
     disappear into the page. */
  background: #ffffff; border-bottom: 1px solid var(--line);
  display: flex; align-items: center; justify-content: center; min-height: 60px;
}
.shot img { display: block; width: 100%; height: auto; }
.placeholder {
  padding: 1.75rem 1rem; text-align: center; color: var(--muted);
  font-size: .82rem; background: var(--card); width: 100%;
}
.body { padding: .7rem .85rem .85rem; }
.label { font-weight: 600; font-size: .95rem; word-break: break-word; }
.meta { color: var(--muted); font-size: .8rem; margin-top: .3rem; word-break: break-all; }
.path { font-family: ui-monospace, SFMono-Regular, Consolas, "Courier New", monospace;
        font-size: .78rem; color: var(--muted); word-break: break-all; margin-top: .35rem; }
.tag {
  display: inline-block; background: var(--tag); border: 1px solid var(--edge);
  border-radius: 999px; padding: 0 .5rem; font-size: .7rem; color: var(--muted);
  margin-right: .35rem;
}
table.final { border-collapse: collapse; width: 100%; font-size: .85rem; }
table.final td { padding: .3rem .5rem .3rem 0; border-bottom: 1px solid var(--line);
                 vertical-align: top; }
table.final td.k { color: var(--muted); white-space: nowrap; }
table.final td.p { font-family: ui-monospace, SFMono-Regular, Consolas, "Courier New", monospace;
                   word-break: break-all; }
footer { color: var(--muted); font-size: .78rem; margin-top: 3rem;
         border-top: 1px solid var(--line); padding-top: .75rem; }
"""


def _escape(text: Any) -> str:
    return html.escape(str(text), quote=True)


def card_html(entry: Entry, *, limit: int = EMBED_LIMIT, latest: bool = False) -> str:
    """One artifact as one card: the picture if there is one, then what it is."""
    picture = image_for(entry)
    if not entry.exists:
        shot = '<div class="placeholder">file missing -- recorded, not on disk now</div>'
    elif picture is None:
        shot = '<div class="placeholder">not a picture -- open the path below</div>'
    else:
        uri = data_uri(picture, limit=limit)
        if uri:
            alt = _escape(entry.label or entry.kind)
            shot = f'<img src="{uri}" alt="{alt}" loading="lazy">'
        else:
            shot = (
                '<div class="placeholder">too large to embed ('
                + _escape(human_size(entry.size))
                + ") -- open the path below</div>"
            )

    tags = []
    if latest:
        tags.append("latest")
    if entry.case:
        tags.append(f"case {entry.case}")
    if entry.is_dir and entry.frames:
        tags.append(f"{entry.frames} frames")
    if entry.exists and entry.size:
        tags.append(human_size(entry.size))
    if not entry.exists:
        tags.append("missing")

    parts = [
        f'<div class="card{"" if entry.exists else " gone"}">',
        f'<div class="shot">{shot}</div>',
        '<div class="body">',
        f'<div class="label">{_escape(entry.label or entry.kind)}</div>',
        '<div class="meta">' + "".join(f'<span class="tag">{_escape(tag)}</span>' for tag in tags),
        _escape(entry.at),
        "</div>",
        f'<div class="path">{_escape(entry.path)}</div>',
        "</div></div>",
    ]
    return "\n".join(parts)


def html_document(
    groups: list[tuple[str, list[Entry]]],
    *,
    title: str = "study gallery",
    subtitle: str = "",
    finals: list[tuple[str, Entry]] | None = None,
    limit: int = EMBED_LIMIT,
    generated: str = "",
) -> str:
    """The whole page as one string: doctype, style, cards, nothing fetched.

    Everything is inline on purpose. There is no stylesheet link, no script tag and
    no remote image in here, so the page renders identically on a machine with no
    network -- which is where it is written, and often where it is read.
    """
    total = sum(len(group) for _kind, group in groups)
    gone = sum(1 for _kind, group in groups for entry in group if not entry.exists)
    finals = finals if finals is not None else []

    body: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head>',
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escape(title)}</title>",
        f"<style>{STYLE}</style>",
        "</head><body><main>",
        f"<h1>{_escape(title)}</h1>",
    ]
    summary = f"{total} artifact(s) in {len(groups)} kind(s)"
    if gone:
        summary += f", {gone} recorded but no longer on disk"
    if generated:
        summary += f" -- written {generated}"
    body.append(f'<p class="sub">{_escape(summary)}</p>')
    if subtitle:
        body.append(f'<p class="sub">{_escape(subtitle)}</p>')

    if finals:
        body.append('<h2>latest of each kind<span class="count">the final results</span></h2>')
        body.append('<table class="final">')
        for kind, entry in finals:
            body.append(
                f'<tr><td class="k">{_escape(kind)}</td>'
                f'<td class="p">{_escape(entry.path)}</td></tr>'
            )
        body.append("</table>")

    for kind, group in groups:
        pick = newest(group)
        body.append(f'<h2>{_escape(kind)}<span class="count">{len(group)}</span></h2>')
        body.append('<div class="grid">')
        for entry in group:
            body.append(card_html(entry, limit=limit, latest=entry is pick and len(group) > 1))
        body.append("</div>")

    body.append(
        "<footer>Every picture here is embedded in this file; it needs no other file "
        "and no network to open. Paths are where the artifacts were when it was "
        "written.</footer>"
    )
    body.append("</main></body></html>")
    return "\n".join(body) + "\n"


def write_html(
    groups: list[tuple[str, list[Entry]]],
    out: Path,
    *,
    title: str = "study gallery",
    subtitle: str = "",
    finals: list[tuple[str, Entry]] | None = None,
    limit: int = EMBED_LIMIT,
) -> Path:
    out = Path(out)
    document = html_document(
        groups, title=title, subtitle=subtitle, finals=finals, limit=limit,
        generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    return out


# -- the contact sheet -------------------------------------------------------------


SHEET_DPI = 110
PANEL_INCHES = (4.4, 3.6)


def grid_shape(count: int, max_cols: int = 3) -> tuple[int, int]:
    """Rows and columns for `count` panels, as square as it gets.

    Three columns at most: the sheet is read by opening one image, and past three
    across the captions are unreadable at the size a viewer picks for it.
    """
    if count <= 0:
        return (0, 0)
    cols = min(max_cols, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / cols)
    return (rows, cols)


def sheet_caption(kind: str, entry: Entry) -> str:
    """Two lines under one panel: what it is for, then which one it is."""
    second = " ".join(bit for bit in (entry.label, f"({entry.case})" if entry.case else "") if bit)
    return f"{kind}\n{second or entry.rel}"


def sheet_panels(
    finals: list[tuple[str, Entry]]
) -> list[tuple[str, Entry, Path]]:
    """The finals that can actually be drawn, with the image to draw for each."""
    panels = []
    for kind, entry in finals:
        picture = image_for(entry, RASTER_SUFFIXES)
        if picture is not None:
            panels.append((kind, entry, picture))
    return panels


def render_sheet(
    finals: list[tuple[str, Entry]], out: Path, title: str = ""
) -> Path | None:
    """The newest picture of each kind on one PNG, or None if there is none.

    None rather than a blank sheet: an empty PNG registered as a contact sheet is
    worse than no contact sheet, because the next reader spends a `read_file` on it
    before finding out.
    """
    panels = sheet_panels(finals)
    if not panels:
        return None
    rows, cols = grid_shape(len(panels))
    figure, grid = plt.subplots(
        rows, cols,
        figsize=(PANEL_INCHES[0] * cols, PANEL_INCHES[1] * rows + 0.4),
        dpi=SHEET_DPI, squeeze=False,
    )
    cells = [cell for row in grid for cell in row]
    for cell in cells[len(panels):]:
        cell.set_axis_off()
    for cell, (kind, entry, picture) in zip(cells, panels):
        cell.set_xticks([])
        cell.set_yticks([])
        for spine in cell.spines.values():
            spine.set_color("#b0bec5")
            spine.set_linewidth(0.8)
        cell.set_title(sheet_caption(kind, entry), fontsize=8.5, loc="left", color="black")
        try:
            cell.imshow(mpimg.imread(str(picture)), aspect="auto")
        except Exception as exc:  # noqa: BLE001 - one lost panel, not a lost sheet
            cell.set_facecolor("#fafafa")
            cell.text(
                0.5, 0.5, f"{picture.name}\ncould not be read back\n{type(exc).__name__}: {exc}",
                transform=cell.transAxes, family="monospace", fontsize=7,
                va="center", ha="center", color="#78909c",
            )
    if title:
        figure.suptitle(title, fontsize=11, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.97 if title else 1.0))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(out), facecolor="white")
    plt.close(figure)
    return out


# -- the command line --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", nargs="?", default=Path("."), type=Path,
                        help="a path inside the study (default: here)")
    parser.add_argument("--html", type=Path, default=None,
                        help=f"where the page goes (default: <study>/{HTML_NAME})")
    parser.add_argument("--sheet", type=Path, default=None,
                        help=f"where the contact sheet goes (default: <study>/{SHEET_NAME})")
    parser.add_argument("--list", dest="listing", action="store_true",
                        help="print the index grouped by kind; writes nothing unless"
                             " --html or --sheet names a file")
    parser.add_argument("--final", action="store_true",
                        help="print the path of the latest artifact of each kind, one per"
                             " line; writes nothing unless --html or --sheet names a file")
    parser.add_argument("--case", default="", help="only artifacts registered for this case")
    parser.add_argument("--title", default="", help="heading for the page and the sheet")
    parser.add_argument("--max-embed", type=float, default=EMBED_LIMIT / 1e6,
                        help="megabytes per image to embed in the page (default: 12)")
    args = parser.parse_args(argv)

    root = study_state.find_root(args.root)
    entries = collect(root=root, case=args.case)
    if not entries:
        where = root / study_state.STATE_DIR / study_state.MANIFEST
        which = f" for case {args.case}" if args.case else ""
        print(f"no artifacts registered{which} in {root}")
        print(f"the manifest is {where}; nothing has been recorded into it yet")
        return 0

    groups = group_by_kind(entries)
    finals = final_entries(entries)

    if args.listing:
        print(text_index(groups), end="")
    if args.final:
        paths = final_paths(finals)
        if paths:
            print("\n".join(paths))
        else:
            print("every registered artifact has been deleted; nothing is on disk")

    # Asking for the text answers is not asking for files to be written. Naming a
    # --html or a --sheet is -- and it asks for that file, not for the other one as
    # well: `--list --html /tmp/p.html` should not leave an unasked-for PNG and its
    # manifest row behind in the study.
    text_only = args.listing or args.final
    want_page = args.html is not None if text_only else True
    want_sheet = args.sheet is not None if text_only else True
    if not want_page and not want_sheet:
        return 0

    title = args.title or f"{root.name or 'study'} -- artifacts"
    subtitle = f"study {root}"
    limit = int(max(0.0, args.max_embed) * 1e6)

    if want_page:
        page = write_html(
            groups, args.html or root / HTML_NAME,
            title=title, subtitle=subtitle, finals=finals, limit=limit,
        )
        study_state.record(
            "gallery", page, root=root, case=args.case,
            label=f"gallery of {len(entries)} artifact(s)", source=SOURCE_TAG,
        )
        print(page)

    if want_sheet:
        sheet = render_sheet(finals, args.sheet or root / SHEET_NAME, title=title)
        if sheet is None:
            print("no picture among the latest artifacts, so no contact sheet was drawn")
        else:
            study_state.record(
                "contact-sheet", sheet, root=root, case=args.case,
                label=f"latest of each kind ({len(sheet_panels(finals))} panels)",
                source=SOURCE_TAG,
            )
            print(sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
