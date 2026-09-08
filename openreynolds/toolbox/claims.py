#!/usr/bin/env python3
"""The number a figure was drawn from, carried by the figure, so it can be put
next to the number the answer quotes.

Why this exists. In the backward-facing-step study the agent measured the
reattachment length twice with two of its own scripts. `reattach.py` found the
wall-shear sign changes at 0.21, 6.32, 7.86 and 9.97 and read x_r/h = 6.32, which
is right -- 0.21 is the corner eddy and 6.32 matches Gartling's 6.1. `analyze.py`,
which is the one that *drew the picture*, read 7.5667, which is wrong. The written
answer quoted 6.4, pointed the reader at the figure, and the figure was annotated
"reattachment x/h=7.57" with a green line across it. Nothing reconciled the two, so
a reader who trusts the picture -- which is what a picture is for -- came away with
a number 24% above the benchmark instead of 4% above it. Both numbers existed in
the same session's tool output, minutes apart.

The same shape has happened twice more. The ONERA M6 notes said the upper-surface
contour "shows the classic lambda-shock herringbone pattern clearly" when the
structure in the image was a trailing-edge artefact on the wrong part of the chord,
which turned a clean negative result into a false positive. And a Tesla-valve
geometry passed its island count while violating every geometric clause it was
given, because the report measured none of the three properties, so there was
nothing for a reviewer to disagree with.

The mechanism is the same every time: a claim and the artefact that is supposed to
support it are produced by different code, and nothing ever puts the two numbers
side by side. This does that, and it is deliberately not an instruction to be
careful.

how a figure carries its number

A claim is stamped into the PNG itself, as a `tEXt` chunk under the keyword
`reynolds-claims`. It travels with the file as bytes: fetched to a laptop, copied,
renamed, mailed to somebody, or embedded in `gallery.html`, whose images are the
file's own bytes in base64 (`gallery.py:embed`) -- the number the picture was drawn
from is still in it, and `claims.py read <png>` prints it. It does NOT survive
anything that redraws the picture, which includes `gallery.py`'s contact sheet:
that reads each panel back through `imread` and saves a new figure, so the sheet
carries pixels and no chunk. The same claim also gets a line in the study manifest
when the study has one, so `claims.py list` answers "what has this study asserted"
without opening any images.

The property that makes it worth anything: the caption drawn on the figure and the
value stamped into it come from one call, `caption(claim)`, so the annotation and
the record cannot drift apart the way `analyze.py`'s green line drifted from its
own printed output.

what `check` compares

Two things, and they are different.

- Claim against claim. Two claims with the same name and different values are a
  contradiction on their face -- 6.32 from one script and 7.5667 from another, both
  recorded, is exactly F-36 and needs no prose to catch. Two claims whose unit
  labels differ are not compared, because guessing at a conversion is how a
  reconciliation tool starts inventing its own errors -- but they are REPORTED as
  not compared. Skipping them quietly and then printing the all-clear is a false
  pass in the one direction this file must not fail in, and it is what the first
  version of it did, on the F-36 pair itself.
- Claim against the written answer. For each claim it finds every number the answer
  quotes near a mention of that claim, and reports the claim as disagreeing only
  when *none* of them is within tolerance. That is deliberately the weak form: a
  check that fires on a benchmark value quoted next to the result is a check that
  gets turned off, and the failure worth catching is the one where the answer and
  the figure share no number at all.

It exits non-zero when it finds a disagreement -- like `preflight.py`, and unlike
the scripts here that only ever report -- because the thing it reports is arithmetic
rather than judgement: two numbers either differ by more than the tolerance or they
do not. `--exit-zero` when you want the report without the status.

what this does not do

Nothing calls it. No agent code imports any script in this directory -- the toolbox is
synced to the workspace and indexed in `README.md`, and it is offered rather than
wired in -- so it is not an oversight peculiar to this file. There is one precedent
for running one of these where the deliverable is assembled, and it is not an import:
the mesher shells out to `mesh_look.py` itself on every mesh it builds and reads its
`--json` back (`openreynolds/mesher/check.py:31`, `:142`, `:292`), with no agent
choosing to. So "nothing runs the toolbox" is not true of the toolbox; it is true of
this file. That bounds what this can fix.
`contradictions` compares claims that were stamped, and F-36's figure was drawn
by a hand-written script that stamped nothing; run against the delivered material as
it stands, `check` finds one claim or none and says so. What it closes is the gap for
a figure somebody stamped -- which `reattach.py --plot` does without being asked, and
`attach` does to a picture any other script drew. Making the comparison unavoidable
means running it where the deliverable is assembled, which is not in this directory.

    python3 claims.py attach fig.png --name reattachment-length --symbol x_r/h --value 6.32
    python3 claims.py read fig.png
    python3 claims.py list                       # every claim this study has recorded
    python3 claims.py check                      # claim against claim
    python3 claims.py check --answer answer.md   # and against what the answer says
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_state  # noqa: E402  (sibling script, not a package)

KEYWORD = "reynolds-claims"
"""The PNG `tEXt` keyword the claims live under. PNG keywords are Latin-1, 1-79
characters, no leading or trailing space; this one is ASCII and 15."""

CLAIM_KIND = "claim"
"""The manifest kind. `study_state.KINDS` does not list it, and that is fine by
that module's own rule -- "a kind that is not on this list is recorded as it is
given rather than refused" -- and the query side takes a kind string, so nothing
needs to know about this one in advance."""

DEFAULT_TOL = 0.05
"""Relative agreement. A figure and an answer that disagree by 5% disagree about
something. On F-36's own numbers, and measured with `apart()` below -- which
normalises on the larger magnitude, so these are the percentages this tool actually
prints, not the ones the study's own write-up quoted: the figure's 7.5667 against
the answer's 6.4 is 15%, against the other script's 6.32 is 16%, and against
Gartling's 6.1 is 19%."""

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# -- what a claim is ----------------------------------------------------------------


def claim(
    name: str,
    value: float,
    *,
    symbol: str = "",
    units: str = "",
    source: str = "",
    note: str = "",
    aliases: Iterable[str] = (),
) -> dict[str, Any]:
    """One asserted number, with enough around it to be compared later.

    `name` is the identity -- two claims are about the same thing when their names
    match -- and `symbol` is how the answer is likely to write it (`x_r`), which is
    what the prose search looks for, along with `aliases`. Keep the normalisation
    out of the symbol when `units` already carries it; `caption` explains why.
    `source` says which code produced it, and
    exists because the whole incident was two scripts producing one quantity: a
    claim that cannot say where it came from cannot be adjudicated.
    """
    row: dict[str, Any] = {
        "name": name,
        "value": float(value),
        "symbol": symbol,
        "units": units,
        "source": source,
    }
    if note:
        row["note"] = note
    extra = [str(item) for item in aliases if str(item).strip()]
    if extra:
        row["aliases"] = extra
    row["caption"] = caption(row)
    return row


def caption(row: dict[str, Any]) -> str:
    """The one string the figure is annotated with and the stamp records.

    Both come from here on purpose. `analyze.py` drew "reattachment x/h=7.57" from
    one variable and printed 7.5667 from another; when the drawing and the record
    are the same call there is no gap for them to drift through.

    `symbol` and `units` are both printed, in that order, so they must not both
    carry the same normalisation: `symbol="x_r/h"` with `units="h"` renders
    `x_r/h = 6.32 h`, a ratio that has already divided by the step height with the
    step height re-attached as its unit. That shipped once. The symbol names the
    quantity; the unit label says what the number is in, and it is the label
    `contradictions` compares on. A ratio spelling belongs in `aliases`, where the
    prose search still finds it (`reattach.py:CLAIM_SYMBOL` is that arrangement).
    """
    label = row.get("symbol") or row.get("name") or "value"
    text = f"{label} = {float(row['value']):.6g}"
    units = str(row.get("units") or "")
    return f"{text} {units}".strip()


def searchable(row: dict[str, Any]) -> list[str]:
    """Every spelling of this claim worth looking for in prose, longest first.

    Longest first because "reattachment length" and "reattachment" would otherwise
    both match at the same place and the shorter one would decide the window.
    """
    words = {str(row.get("symbol") or ""), str(row.get("name") or "")}
    words.add(str(row.get("name") or "").replace("-", " "))
    words.add(str(row.get("name") or "").replace("_", " "))
    words.update(str(item) for item in row.get("aliases", ()))
    return sorted((word for word in words if word.strip()), key=len, reverse=True)


# -- the stamp in the PNG -----------------------------------------------------------


class StampError(Exception):
    """The file is not a PNG we can read or write. A fact about the file."""


def _chunks(data: bytes) -> list[tuple[str, bytes]]:
    """A PNG's chunks as `(type, payload)`, in file order.

    Written out rather than reached for in a library because none of the imaging
    libraries here offers "add a text chunk to a file somebody else wrote without
    touching its pixels": matplotlib and imageio both go through a decode and a
    re-encode. Pillow is present (it comes in under matplotlib) and its `PngInfo`
    could do it, but only by rewriting the image, and `struct` plus `zlib` copies
    every other chunk byte for byte. An earlier version of this comment said there
    was no Pillow on the image, which was read off `ENVIRONMENT.md` not listing it;
    absence from a list is not a measurement, and `import PIL` answers in a second.
    """
    if not data.startswith(PNG_SIGNATURE):
        raise StampError("not a PNG (bad signature)")
    out: list[tuple[str, bytes]] = []
    at = len(PNG_SIGNATURE)
    while at + 8 <= len(data):
        (length,) = struct.unpack(">I", data[at : at + 4])
        kind = data[at + 4 : at + 8].decode("latin-1")
        payload = data[at + 8 : at + 8 + length]
        if len(payload) < length:
            raise StampError(f"truncated {kind} chunk")
        out.append((kind, payload))
        at += 12 + length  # length, type, payload, crc
        if kind == "IEND":
            break
    if not out or out[-1][0] != "IEND":
        raise StampError("no IEND chunk -- the file is truncated")
    return out


def _encode(kind: str, payload: bytes) -> bytes:
    raw = kind.encode("latin-1") + payload
    return struct.pack(">I", len(payload)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)


def stamp_png(path: Path | str, claims: list[dict[str, Any]]) -> Path:
    """Write `claims` into the PNG at `path` as a `tEXt` chunk, replacing any
    stamp already there.

    Replacing rather than appending: a figure redrawn with a corrected number must
    not end up carrying both, which is the very confusion this is here to stop. The
    JSON is written `ensure_ascii=True` because a PNG `tEXt` payload is Latin-1 and
    a degree sign in a unit string is not worth a `zTXt`.
    """
    path = Path(path)
    data = path.read_bytes()
    payload = KEYWORD.encode("latin-1") + b"\x00" + json.dumps(claims, ensure_ascii=True).encode("latin-1")
    rebuilt = bytearray(PNG_SIGNATURE)
    inserted = False
    for kind, chunk in _chunks(data):
        if kind == "tEXt" and chunk.split(b"\x00", 1)[0] == KEYWORD.encode("latin-1"):
            continue  # the old stamp; the new one goes in its place below
        if kind == "IEND" and not inserted:
            rebuilt += _encode("tEXt", payload)
            inserted = True
        rebuilt += _encode(kind, chunk)
    path.write_bytes(bytes(rebuilt))
    return path


def read_png(path: Path | str) -> list[dict[str, Any]]:
    """The claims a PNG carries, or `[]`. Never raises on a file that simply has
    no stamp -- most PNGs do not, and "this one says nothing" is an answer."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError:
        return []
    try:
        chunks = _chunks(data)
    except StampError:
        return []
    want = KEYWORD.encode("latin-1")
    for kind, payload in chunks:
        if kind != "tEXt":
            continue
        key, _, text = payload.partition(b"\x00")
        if key != want:
            continue
        try:
            rows = json.loads(text.decode("latin-1"))
        except json.JSONDecodeError:
            return []
        return [row for row in rows if isinstance(row, dict) and "value" in row]
    return []


# -- attaching, and the manifest line -----------------------------------------------


def attach(
    figure: Path | str,
    claims: list[dict[str, Any]],
    *,
    root: Path | str = ".",
    case: str = "",
    record: bool | None = None,
) -> dict[str, Any]:
    """Stamp a figure and, when the study keeps state, give it a manifest line.

    `record` defaults to "only if this study already has a `.reynolds/`". A tool
    asked to draw a picture in a scratch directory should not conjure study state
    around it; a tool run inside a study should not need to be told to record.
    """
    figure = Path(figure)
    stamp_png(figure, claims)
    study = study_state.find_root(root if root != "." else figure.parent)
    if record is None:
        record = (study / study_state.STATE_DIR).is_dir()
    row: dict[str, Any] = {"figure": str(figure), "claims": claims, "recorded": False}
    if record:
        label = "; ".join(caption(one) for one in claims)
        study_state.record(
            CLAIM_KIND, figure, root=study, case=case, label=label, claims=claims
        )
        row["recorded"] = True
        row["study"] = str(study)
    return row


def collect(root: Path | str = ".", *, case: str = "") -> list[dict[str, Any]]:
    """Every claim this study has, from the manifest and from the PNGs themselves.

    Both sources, because they fail differently. The manifest misses a figure drawn
    before the study had state, or one carried in from another directory; the
    images miss nothing but have to be found. A claim present in both is counted
    once, keyed by (figure, name).
    """
    study = study_state.find_root(root)
    found: dict[tuple[str, str], dict[str, Any]] = {}

    def take(figure: str, rows: Iterable[dict[str, Any]]) -> None:
        for one in rows:
            if not isinstance(one, dict) or "value" not in one:
                continue
            entry = dict(one)
            entry["figure"] = figure
            found[(figure, str(entry.get("name", "")))] = entry

    for artifact in study_state.artifacts(root=study, kind=CLAIM_KIND, case=case, exists=False):
        meta = artifact.get("meta") or {}
        take(str(artifact.get("abspath") or artifact.get("path")), meta.get("claims") or [])
    for png in sorted(study.rglob("*.png")):
        if study_state.STATE_DIR in png.parts:
            continue
        take(str(png), read_png(png))
    return [found[key] for key in sorted(found)]


# -- the two comparisons ------------------------------------------------------------


def apart(a: float, b: float) -> float:
    """Relative difference, on the larger magnitude, so it is symmetric and does
    not divide by a value that happens to be near zero."""
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


def unit_key(row: dict[str, Any]) -> str:
    """The unit label two claims have to share before their values mean the same
    thing. Folded for case and surrounding space, because `H` and ` h ` typed on two
    command lines are not two units, and every difference here costs a comparison."""
    return str(row.get("units", "")).strip().casefold()


def contradictions(claims: list[dict[str, Any]], tol: float = DEFAULT_TOL) -> list[dict[str, Any]]:
    """Claims of the same name whose values do not sit together, in two kinds.

    Every row carries `comparable`, and the caller has to read it:

    - `comparable: True` -- the two carry the same unit label and their values
      differ by more than `tol`. That is the F-36 pair, and it is what makes `check`
      exit non-zero.
    - `comparable: False` -- the labels differ, so the values were NOT converted and
      `apart` is None. 6.32 step heights and 0.0594 metres are one measurement, and
      guessing at a conversion is how a reconciliation tool starts inventing its own
      errors.

    The second kind exists because the first version of this function did the right
    arithmetic and then dropped those pairs on the floor, and `report` printed "no
    two claims of the same name disagree" over the silence. On the F-36 replay that
    was the actual outcome: `reattach.py` labelled its 6.32 "m" and a hand-attached
    7.5667 carried the default empty label, the pair was skipped, and the tool built
    to catch that exact contradiction printed the all-clear. A pair that cannot be
    compared is a thing the reader has to be told, not a pass.

    The tolerance gate applies to comparable pairs ONLY, and the order of those two
    tests is the whole of it. An earlier version asked "are these within tol?" first
    and skipped the pair when they were, before it had looked at the labels -- so a
    pair whose RAW numbers happened to sit close under two different units was
    dropped silently and `report` printed the all-clear over it. Measured: 6.32
    labelled `h` and 6.30 labelled `m` came back `[]`, and the last line of the
    report was "no two claims of the same name disagree" -- over a pair that was
    never compared, and that nothing here has any reason to read as one size. On the
    step's own 9.4 mm they are 59 mm and 6.3 m. Numbers
    under two labels being numerically close is a coincidence of two scales, not
    evidence of agreement, so there is no tolerance to apply to them: every pair of
    one name whose labels differ is reported, near or far.

    The cost is that a study which labels one quantity two ways gets a line every
    run -- and a check that prints something every run is a check that gets turned
    off. That is accepted here because the line is not noise: one name under two unit
    labels is a thing nothing in this file has checked, which is the state it exists
    to refuse to be quiet about.
    """
    by_name: dict[str, list[dict[str, Any]]] = {}
    for one in claims:
        by_name.setdefault(str(one.get("name", "")), []).append(one)
    out: list[dict[str, Any]] = []
    for name, rows in sorted(by_name.items()):
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                left, right = rows[i], rows[j]
                same = unit_key(left) == unit_key(right)
                gap = apart(float(left["value"]), float(right["value"]))
                # Comparability before tolerance. `gap` on two different unit labels
                # is arithmetic on two scales and means nothing, so it cannot be the
                # thing that decides to say nothing.
                if same and gap <= tol:
                    continue
                out.append({
                    "name": name, "left": left, "right": right,
                    "comparable": same,
                    "apart": gap if same else None,
                })
    return out


_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\d.]*\s*%)")
"""A number the answer quotes for this quantity, excluding percentages.

The trailing lookahead is not decoration. "within 5% of Gartling's benchmark 6.1" is
an ordinary way to write a comparison, and without it that sentence offers 5.0 as a
candidate value for the reattachment length -- so a figure stamped 5.1, a perfectly
ordinary answer, agreed with an answer that said 6.4. A percentage written as digits
is how far apart two things are, never how long the bubble is.

The `[\\d.]*` inside the lookahead is what makes that hold past one digit, and it was
missing. With a bare `(?!\\s*%)` the engine simply backtracks and matches the leading
digits instead: measured, "within 15% of the benchmark" yielded 15.0 rejected and
then 1.0 ACCEPTED, "some 24% above Gartling" yielded 2.0, "12.5%" yielded 12.0. Those
truncations are candidates that did not exist before the guard, and 1.0-2.0 is an
ordinary reattachment length, so the guard was manufacturing the failure it was added
to stop. Blocking the digits and dots that follow leaves the number no shorter match
to fall back on.

What it does not cover, and no lookahead here will: a percentage spelled out ("5 per
cent", "5 percent"). That number is still offered as a candidate."""

_SENTENCE_END = re.compile(r"[.;:!?]\s|\n")
LOOK_BACK = 200
"""How far back a sentence boundary is looked for, and how far back the window
reaches when there is none.

Both, and that is the point: it is the budget, not just the search. The first two
versions searched this far back and then fell back to character 0 of the document
when the search found nothing, so a mention inside a long unpunctuated stretch --
one 256-character sentence is enough -- got a window that was the whole file, and
every number in it became a candidate."""

LOOK_AHEAD = 60
"""How far past a mention a number still counts as being about it, before the
sentence boundary is even reached.

Measured off the sentence this was written for -- "The reattachment length is
x_r/h = 6.4, in good agreement with Gartling's 6.1" -- where, from the end of the
symbol, 6.4 sits 3 characters on and the benchmark 42. Wide enough to catch both."""


def _window(text: str, at: int, end: int) -> str:
    """The text a mention at `[at, end)` could plausibly be quoting a number in.

    Backward: to the start of its own sentence, or `LOOK_BACK` characters, whichever
    is nearer. Forward: to the end of its own sentence, or `LOOK_AHEAD` characters,
    whichever is nearer. Neither edge can run away: a mention with no sentence
    boundary behind it inside `LOOK_BACK` gets `LOOK_BACK`, not the start of the
    document, which is what the previous two versions gave it -- `back` was
    initialised to 0 and only ever written inside the search loop, so on a long
    unpunctuated stretch the window was the whole file and a benchmark 250
    characters upstream rescued a claim that disagreed. Measured on ordinary prose,
    no contrivance: one 256-character sentence naming 7.6 and then the reattachment
    length turned a 7.5667 claim into `agrees`.

    BOTH edges are clipped at the sentence. The first version clipped only the
    backward one, on the argument that the two directions fail differently -- and
    the argument is right, it just does not favour either direction: a stray number
    swept in from the NEXT sentence rescues a real disagreement exactly as well as
    one from the previous sentence does. Measured, on shapes that are not contrived:
    a claim of 7.5667 against "The reattachment length is x_r/h = 6.4. The mesh has
    7.5 M cells." came back `agrees` on the cell count, and against "The
    reattachment length is plotted in the figure. Gartling reports 7.6 for this
    case." it came back `agrees` on the benchmark of a sentence that was not about
    it. Both are ordinary ways to write the sentence that points a reader at a
    picture, which is the sentence this whole file is about.

    The cost of clipping is the other failure, and it is the acceptable one: a
    number that belongs to the mention but sits past a boundary -- across a hard
    line wrap, or after "Fig. 3" -- is missed, and the claim comes back
    `unmentioned`, which is a verdict that gets reported and looked at rather than a
    pass that gets believed.
    """
    floor = max(0, at - LOOK_BACK)
    back = floor  # the budget IS the fallback; see LOOK_BACK
    for boundary in _SENTENCE_END.finditer(text[floor:at]):
        back = floor + boundary.end()
    ahead = text[end : end + LOOK_AHEAD]
    stop = _SENTENCE_END.search(ahead)
    if stop is not None:
        ahead = ahead[: stop.start()]
    return text[back:end] + ahead


def mentions(text: str, row: dict[str, Any]) -> list[float]:
    """Every number the text quotes near a mention of this claim, in order.

    A crude proximity search on purpose. Parsing what a paragraph means is a
    research problem; noticing that the paragraph nowhere writes the figure's
    number is arithmetic, and it is the whole of what went wrong.
    """
    seen: list[float] = []
    for word in searchable(row):
        for hit in re.finditer(
            rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", text, re.IGNORECASE
        ):
            window = _window(text, hit.start(), hit.end())
            for number in _NUMBER.finditer(window):
                try:
                    value = float(number.group())
                except ValueError:
                    continue
                if value not in seen:
                    seen.append(value)
        if seen:
            break  # the longest spelling that matched is the one that meant it
    return seen


def against_answer(
    claims: list[dict[str, Any]], text: str, tol: float = DEFAULT_TOL
) -> list[dict[str, Any]]:
    """Each claim judged against the prose: `agrees`, `disagrees`, or `unmentioned`.

    `unmentioned` is a third verdict rather than a pass, because "the answer never
    quotes this number" and "the answer quotes it correctly" are different states
    and only one of them is fine.
    """
    out: list[dict[str, Any]] = []
    for one in claims:
        numbers = mentions(text, one)
        if not numbers:
            verdict = "unmentioned"
        elif any(apart(float(one["value"]), number) <= tol for number in numbers):
            verdict = "agrees"
        else:
            verdict = "disagrees"
        out.append({"claim": one, "verdict": verdict, "answer_numbers": numbers})
    return out


def report(
    claims: list[dict[str, Any]],
    clashes: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    answer: Path | None,
    tol: float,
) -> str:
    lines = [f"{len(claims)} claim(s) found; agreement within {tol:.0%}"]
    for one in claims:
        where = Path(str(one.get("figure", ""))).name
        source = f"  ({one['source']})" if one.get("source") else ""
        lines.append(f"  {caption(one):<28} {where}{source}")

    def whose(row: dict[str, Any]) -> str:
        return str(row.get("source") or Path(str(row.get("figure", ""))).name)

    def written(row: dict[str, Any]) -> str:
        return f"{float(row['value']):.6g} {row.get('units') or ''}".strip()

    compared = [clash for clash in clashes if clash["comparable"]]
    skipped = [clash for clash in clashes if not clash["comparable"]]

    if compared or not skipped:
        lines.append("")
    if compared:
        lines.append("claims that contradict each other:")
        for clash in compared:
            left, right = clash["left"], clash["right"]
            lines.append(
                f"  {clash['name']}: {float(left['value']):.6g} ({whose(left)}) vs "
                f"{float(right['value']):.6g} ({whose(right)}) "
                f"— {clash['apart']:.0%} apart"
            )
    elif not skipped:
        lines.append("no two claims of the same name disagree")
    # Never the all-clear over a skipped pair. The line above used to print whether
    # or not anything had been left uncompared, and on the F-36 replay that is the
    # line the tool printed about the pair it was written to catch. This holds only
    # because `contradictions` tests comparability BEFORE tolerance: while it tested
    # tolerance first, an uncompared pair that happened to be numerically close never
    # reached `skipped` at all and this line printed over it anyway.
    if skipped:
        lines.append("")
        lines.append("claims of the same name that were NOT compared, because their units differ:")
        for clash in skipped:
            left, right = clash["left"], clash["right"]
            lines.append(
                f"  {clash['name']}: {written(left)} ({whose(left)}) vs "
                f"{written(right)} ({whose(right)}) — no conversion is guessed here. If "
                "these are two units for one measurement, fine; if they are the same "
                "unit under two labels, one of these numbers is wrong and nothing "
                "above checked it"
            )

    if answer is not None:
        lines.append("")
        lines.append(f"against {answer.name}:")
        for entry in verdicts:
            one = entry["claim"]
            numbers = ", ".join(f"{value:.6g}" for value in entry["answer_numbers"]) or "-"
            lines.append(
                f"  {entry['verdict']:<12} {caption(one):<28} "
                f"answer quotes near it: {numbers}"
            )
        bad = [entry for entry in verdicts if entry["verdict"] == "disagrees"]
        if bad:
            lines.append("")
            for entry in bad:
                one = entry["claim"]
                closest = min(
                    entry["answer_numbers"], key=lambda v: apart(float(one["value"]), v)
                )
                lines.append(
                    f"  {Path(str(one.get('figure',''))).name} carries "
                    f"{float(one['value']):.6g}; the nearest number the answer quotes for "
                    f"it is {closest:.6g} — {apart(float(one['value']), closest):.0%} apart. "
                    "One of the two is wrong."
                )
    return "\n".join(lines)


# -- the command line ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("attach", help="Stamp a claim into a figure that already exists.")
    add.add_argument("figure", type=Path)
    add.add_argument("--name", required=True)
    add.add_argument("--value", type=float, required=True)
    add.add_argument("--symbol", default="")
    add.add_argument("--units", default="")
    add.add_argument("--source", default="", help="what produced the number")
    add.add_argument("--note", default="")
    add.add_argument("--alias", action="append", default=[],
                     help="another spelling the answer may use; repeatable")

    show = sub.add_parser("read", help="The claims a figure carries.")
    show.add_argument("figure", type=Path)
    show.add_argument("--json", action="store_true")

    listing = sub.add_parser("list", help="Every claim in this study.")
    listing.add_argument("--root", type=Path, default=Path("."))
    listing.add_argument("--json", action="store_true")

    check = sub.add_parser("check", help="Put the claims next to each other and to the answer.")
    check.add_argument("--root", type=Path, default=Path("."))
    check.add_argument("--answer", type=Path, default=None,
                       help="the written answer, so its numbers can be compared too")
    check.add_argument("--tol", type=float, default=DEFAULT_TOL)
    check.add_argument("--json", action="store_true")
    check.add_argument("--exit-zero", action="store_true",
                       help="report a disagreement without returning non-zero")

    args = parser.parse_args(argv)

    if args.command == "attach":
        row = claim(args.name, args.value, symbol=args.symbol, units=args.units,
                    source=args.source, note=args.note, aliases=args.alias)
        try:
            result = attach(args.figure, [row])
        except (StampError, OSError) as exc:
            print(f"could not stamp {args.figure}: {exc}", file=sys.stderr)
            return 1
        where = " and recorded in the manifest" if result["recorded"] else ""
        print(f"{args.figure} now carries {caption(row)}{where}")
        return 0

    if args.command == "read":
        rows = read_png(args.figure)
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        if not rows:
            print(f"{args.figure} carries no claim")
            return 0
        for row in rows:
            source = f"   ({row['source']})" if row.get("source") else ""
            print(f"{caption(row)}{source}")
        return 0

    if args.command == "list":
        rows = collect(args.root)
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        if not rows:
            print("no claims recorded or stamped in this study")
            return 0
        for row in rows:
            print(f"{caption(row):<28} {Path(str(row.get('figure',''))).name}")
        return 0

    if args.command == "check":
        rows = collect(args.root)
        clashes = contradictions(rows, args.tol)
        verdicts: list[dict[str, Any]] = []
        text = ""
        if args.answer is not None:
            try:
                text = args.answer.read_text(errors="replace")
            except OSError as exc:
                print(f"could not read {args.answer}: {exc}", file=sys.stderr)
                return 1
            verdicts = against_answer(rows, text, args.tol)
        if args.json:
            print(json.dumps(
                {"claims": rows, "contradictions": clashes, "answer": verdicts}, indent=2
            ))
        else:
            print(report(rows, clashes, verdicts, args.answer, args.tol))
        # A pair that was not compared does not fail the run: nothing here knows
        # whether metres and step heights on one wall are a contradiction or two
        # honest readings, and a status that goes red on the honest case is a status
        # that gets suppressed. It is printed instead, every time, which is the part
        # that was missing. Only arithmetic this tool actually did sets the code.
        disagreed = any(clash["comparable"] for clash in clashes) or any(
            v["verdict"] == "disagrees" for v in verdicts
        )
        return 0 if args.exit_zero or not disagreed else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
