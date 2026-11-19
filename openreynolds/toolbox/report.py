#!/usr/bin/env python3
"""A finished study as one PDF: the numbers, the figures, and the settings behind them.

`results.py` already writes `results.md` -- what was produced, what was skipped and
why, the headline numbers -- and that stays the working record. What it is not is a
document somebody outside the run can be handed: a markdown file plus a directory of
PNGs arrives as a directory of PNGs, and the settings that decide whether any of those
numbers mean anything are still spread over five dictionaries nobody opens. This
assembles the one file: cover, figures, methods, and what was asked for and not
produced. It complements `results.md` rather than replacing it, and it is written only
when it is asked for.

the writer is matplotlib

There is no PDF library on this instance and none is coming -- the network is sealed
(`ENVIRONMENT.md`). `matplotlib.backends.backend_pdf.PdfPages` is a multi-page PDF
writer that is already here, so every page is a matplotlib figure: text pages are text
drawn in a monospaced font at a fixed line pitch, and figure pages are the PNGs read
back through `imread`. That has one consequence worth knowing: a figure on a page of
this document is pixels, so the `reynolds-claims` chunk stamped in the original PNG
does NOT survive onto the page, the same way `gallery.py`'s contact sheet does not
carry it. The stamp is read from the original file on disk, before the page is drawn,
and what the page prints is the reading.

why the key numbers are not simply printed

An executive summary that prints a number beside a figure is the exact failure
`claims.py` was written for: in the backward-facing-step study the answer quoted 6.4,
the figure beside it was annotated 7.57, and nothing put the two side by side. So the
numbers on the cover of this document are not collected from whatever produced them
and set next to whatever picture looked relevant. Each one is tied to a figure or it is
not:

- **verified** -- the value is stamped into a PNG that is still on disk, under the
  `reynolds-claims` keyword, by the same call that drew the annotation on it
  (`claims.caption`). The cover names the figure.
- **disagrees** -- a measured value and a stamped claim of the same quantity differ by
  more than `claims.DEFAULT_TOL`. Both numbers are printed, with the gap. This is F-36,
  and it is on the cover rather than in a log.
- **unverified** -- a number read out of the study's own files (a coefficient's tail
  mean, a final residual) with no stamped figure behind it. It is printed and labelled
  as having no figure behind it, because a number nobody drew is still the answer
  somebody needs; what it is not is evidence.

A number whose figure has been deleted since is unverified too, and says so.

what it picks, and what it says about picking

The hero figure is chosen by a stated rule, printed on stdout and on the cover, and
`--hero` overrides it. Beyond that this picks nothing: the figures are grouped the way
`gallery.py` groups them, newest last, and a manifest row whose file is gone appears as
MISSING rather than quietly vanishing -- a study somebody tidied up should read as a
study somebody tidied up.

the template

`--template` takes a JSON file: the cover text, the fixed front matter, and the section
order. One is shipped beside this script as `report_template.json`. A `--template` that
is not there, or is not readable JSON, is refused by name rather than falling back to
the built-in default: a document that silently came out in the house default when a
company template was asked for is a document that gets sent.

    python3 report.py /work/study
    python3 report.py /work/study --out ~/acme-report.pdf --json went-in.json
    python3 report.py /work/study --template report_template.json --logo acme.png
    python3 report.py /work/study --hero renders/vorticity.png --case cylinder
    python3 report.py /work/study --no-layers        # skip the wall-layer measurement
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402 - after the backend is fixed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claims  # noqa: E402  (sibling scripts, not a package)
import gallery  # noqa: E402
import log_digest  # noqa: E402
import mesh_digest  # noqa: E402
import preflight  # noqa: E402
import results  # noqa: E402
import study_state  # noqa: E402

PDF_NAME = "report.pdf"
TEMPLATE_NAME = "report_template.json"

SOURCE_TAG = "report-pack"
"""Written into the meta of the row this script registers, and skipped when the next
run reads the manifest back -- `gallery.py`'s arrangement, for its reason: without it
the second report would carry a page about the first one."""

REPORT_KIND = "report"
"""The manifest kind. `study_state.KINDS` already has it, and `results.py` registers
`results.md` under it: both are "the study written up", and a reader asking for the
report of a study wants whichever is newer, not a vocabulary lesson. No new kind."""

PAGE_INCHES = (8.27, 11.69)  # A4 portrait, because this gets printed
PAGE_DPI = 110

BODY_LINES = 56
"""Monospaced lines that fit on a text page at `BODY_SIZE`. Measured by filling one."""
BODY_SIZE = 7.6
LINE_PITCH = 0.0152

FIGURES_PER_PAGE = 4

HERO_KINDS = (
    "velocity", "pressure", "vorticity", "streamlines",
    "forces", "residuals", "mesh-full", "geometry-preview",
)
"""The fallback order for a hero when no figure carries a claim: the picture of the
answer before the picture of the method. It is a preference, not a judgement about any
particular study, which is why the rule that was used is printed."""

DEFAULT_TEMPLATE: dict[str, Any] = {
    "title": "",
    "subtitle": "",
    "front_matter": [],
    "sections": ["summary", "figures", "methods", "gaps"],
    "footer": "Numbers as produced. Whether they are converged, mesh independent or "
              "right is not decided here.",
}

SECTIONS = ("summary", "figures", "methods", "gaps")


# -- the template ------------------------------------------------------------------


class Refused(Exception):
    """An argument that would have produced a plausible-looking wrong document."""


def load_template(path: Path | str | None) -> tuple[dict[str, Any], list[str]]:
    """The template, and any notes about it worth printing.

    Returns the built-in default when nothing was asked for, and raises when something
    was asked for and is not there. The asymmetry is the whole point: the default is
    what you get when you did not ask, never what you get instead of what you asked
    for.
    """
    if path is None:
        return dict(DEFAULT_TEMPLATE), []
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refused(
            f"refused: no template at {path}.\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            "A --template names the cover text, the fixed front matter and the section\n"
            "order. Falling back to the built-in default here would write a document in\n"
            "the house style under the name of the one you asked for, and nothing on the\n"
            "page would say so. There is an example beside this script at "
            f"{TEMPLATE_NAME}."
        ) from exc
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Refused(
            f"refused: the template at {path} is not readable JSON.\n\n"
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(loaded, dict):
        raise Refused(
            f"refused: the template at {path} is a {type(loaded).__name__}, not an "
            "object.\n\nIt wants keys: title, subtitle, front_matter, sections, footer."
        )
    template = dict(DEFAULT_TEMPLATE)
    template.update(loaded)
    notes: list[str] = []
    wanted = [str(name) for name in template.get("sections") or []]
    unknown = [name for name in wanted if name not in SECTIONS]
    if unknown:
        notes.append(
            f"template section(s) {', '.join(unknown)} are not ones this writes; "
            f"known: {', '.join(SECTIONS)}"
        )
    kept = [name for name in wanted if name in SECTIONS]
    if not kept:
        kept = list(DEFAULT_TEMPLATE["sections"])
        notes.append("the template named no section this writes, so all four are written")
    template["sections"] = kept
    return template, notes


# -- the case the methods section is about -------------------------------------------


def choose_case(study: Path, wanted: str = "") -> tuple[Path | None, str, list[str]]:
    """The case directory the methods are read from, the rule, and the ones not used.

    A study can hold several cases, and a methods section built from two of them at
    once would describe neither. So one is chosen and the rest are named on the page:
    "this is the mesh and the settings of `cylinder`" is a document, "these are some
    settings from somewhere in the study" is not.
    """
    # `discover_cases` already puts the study itself first when the study home is the
    # case (the legacy `/work` layout). Prepending it again counted it twice, so a
    # one-case study was offered as "the first of 2 cases ()" -- the empty name being
    # the home directory's own.
    cases = study_state.discover_cases(study)
    if wanted:
        for case in cases:
            if case.name == wanted:
                return case, f"--case {wanted}", [c.name for c in cases if c != case]
        if (study / wanted).is_dir():
            return (study / wanted), f"--case {wanted}", [c.name for c in cases]
        return None, f"--case {wanted} names no directory under the study", [c.name for c in cases]
    if not cases:
        return None, "no case directory under the study home", []
    if len(cases) == 1:
        return cases[0], f"the study's only case ({cases[0].name})", []
    return cases[0], (f"the first of {len(cases)} cases ({cases[0].name}); "
                      "name another with --case"), [c.name for c in cases[1:]]


# -- the numbers, each one tied to a figure or not ------------------------------------


def stamped_claims(study: Path, case: str = "") -> list[dict[str, Any]]:
    """Every claim this study has, with whether its figure is still readable.

    `claims.collect` reads the manifest and the PNGs; this adds the one thing a cover
    page has to know and neither source states -- whether the number can still be read
    out of the picture's own bytes right now. A claim whose figure has been deleted, or
    redrawn by something that did not stamp it, is a claim with nothing behind it any
    more, and that is a different thing from a claim.
    """
    out: list[dict[str, Any]] = []
    for row in claims.collect(study, case=case):
        figure = Path(str(row.get("figure", "")))
        entry = dict(row)
        entry["figure_exists"] = figure.is_file()
        entry["in_png"] = bool(claims.read_png(figure)) if entry["figure_exists"] else False
        out.append(entry)
    return out


def measured_numbers(case: Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    """The numbers the case's own files carry: the coefficients and the reference
    values behind them, and where the residuals ended up.

    These come from the files rather than from a claim, so none of them arrives with a
    figure attached. `tie_numbers` is what decides whether any of them has one.
    """
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    if case is None:
        return rows, ["no case directory, so no coefficients or residuals were read"]

    try:
        grouped = results.find_force_files(case)
        if grouped:
            key = results.choose_force_series(grouped)
            history = results.read_history(grouped[key])
            columns = results.plot_columns(history["columns"])
            for label, value in results.force_headline(history, columns).items():
                rows.append({"label": label, "text": str(value), "source": f"{key}"})
            for line in history.get("superseded", []) + history.get("ignored", []):
                notes.append(f"force record: {line}")
        else:
            notes.append("no force or coefficient .dat under postProcessing/")
    except Exception as exc:  # noqa: BLE001 - one missing number, not a missing document
        notes.append(f"coefficients not read: {type(exc).__name__}: {exc}")

    log = results.find_solver_log(case)
    if log is None:
        notes.append("no solver log in the case, so no final residuals")
    else:
        try:
            data = log_digest.digest(log)
            shape = log_digest.residual_shape(data.get("residuals") or {})
            for field, last in sorted(shape.get("last", {}).items()):
                rows.append({
                    "label": f"final residual {field}",
                    "text": f"{last:.3g} ({shape['fields'].get(field, '?')})",
                    "source": log.name,
                })
        except Exception as exc:  # noqa: BLE001
            notes.append(f"residuals not read: {type(exc).__name__}: {exc}")
    return rows, notes


def _number_in(text: str) -> float | None:
    """The one number in a formatted value, when there is exactly one to read.

    `claims._NUMBER` rather than a fresh regex, deliberately: it is the pattern that
    refuses a percentage, and the reasoning for that -- "within 5% of Gartling's 6.1"
    offering 5.0 as the value -- is written out beside it. A second pattern here would
    be a second chance to get that wrong.
    """
    found = claims._NUMBER.findall(str(text))
    if len(found) != 1:
        return None
    try:
        return float(found[0])
    except ValueError:
        return None


def tie_numbers(
    figure_claims: list[dict[str, Any]],
    measured: list[dict[str, Any]],
    tol: float = claims.DEFAULT_TOL,
) -> list[dict[str, Any]]:
    """Every number for the cover, each with the figure behind it or the reason there
    is none.

    This is the function the whole script exists for. `claims.py` says why: a claim and
    the artefact that is supposed to support it are produced by different code, and
    nothing ever puts the two numbers side by side. Here they are side by side before
    either reaches a page, and the three outcomes are kept apart on purpose --
    `verified` carries a figure, `disagrees` carries two numbers and the gap between
    them, `unverified` carries a number and the admission that nothing drew it.

    Matching is by spelling, through `claims.searchable`: a measured label contains one
    of the claim's own spellings, folded for case. Deliberately weak in the same
    direction `claims.check` is weak -- a claim nobody can match reads as unverified,
    which is true, rather than as agreement.
    """
    out: list[dict[str, Any]] = []
    for row in figure_claims:
        figure = Path(str(row.get("figure", "")))
        entry = {
            "label": str(row.get("name") or row.get("symbol") or "claim"),
            "text": claims.caption(row),
            "value": float(row["value"]),
            "units": str(row.get("units") or ""),
            "figure": str(figure),
            "source": str(row.get("source") or ""),
        }
        if row.get("in_png"):
            entry["state"] = "verified"
            entry["why"] = f"stamped in {figure.name}"
        elif row.get("figure_exists"):
            entry["state"] = "unverified"
            entry["why"] = (f"{figure.name} is on disk and carries no claim stamp "
                            "(redrawn by something that did not stamp it)")
        else:
            entry["state"] = "unverified"
            entry["why"] = f"the figure it was drawn from is gone ({figure})"
        out.append(entry)

    for row in measured:
        label = str(row["label"])
        folded = label.casefold()
        value = _number_in(row["text"])
        entry = {
            "label": label,
            "text": str(row["text"]),
            "value": value,
            "units": "",
            "figure": "",
            "source": str(row.get("source") or ""),
            "state": "unverified",
            "why": "no figure is stamped with this number",
        }
        for claimed in figure_claims:
            if not claimed.get("in_png"):
                continue
            if not any(word.casefold() in folded for word in claims.searchable(claimed)):
                continue
            figure = Path(str(claimed.get("figure", "")))
            if value is None:
                entry["why"] = (f"{figure.name} claims this quantity; this value is not "
                                "a single number, so nothing was compared")
                break
            gap = claims.apart(value, float(claimed["value"]))
            entry["figure"] = str(figure)
            if gap <= tol:
                entry["state"] = "verified"
                entry["why"] = f"agrees with {figure.name} ({claims.caption(claimed)})"
            else:
                entry["state"] = "disagrees"
                entry["why"] = (f"{figure.name} was drawn from "
                                f"{float(claimed['value']):.6g}, this reads {value:.6g} "
                                f"-- {gap * 100:.0f}% apart")
            break
        out.append(entry)
    return out


# -- the hero ------------------------------------------------------------------------


def pick_hero(
    entries: list[gallery.Entry],
    figure_claims: list[dict[str, Any]],
    given: Path | None = None,
) -> tuple[Path | None, str]:
    """The one figure on the cover, and the rule that chose it.

    Returned together because the rule is part of the answer. A document that puts one
    picture at the front has made an editorial decision, and a reader who cannot see
    which decision cannot disagree with it. `--hero` is the override, and it is honoured
    whether or not the file is registered in the manifest.
    """
    if given is not None:
        return Path(given), "given on the command line with --hero"

    stamped = [row for row in figure_claims if row.get("in_png")]
    if stamped:
        best = max(stamped, key=lambda row: len(claims.read_png(Path(str(row["figure"])))))
        return (Path(str(best["figure"])),
                "the figure carrying the most stamped claims -- a picture with the "
                "answer written into it")

    present = [entry for entry in entries if gallery.image_for(entry) is not None]
    for kind in HERO_KINDS:
        of_kind = [entry for entry in present if entry.kind == kind]
        if of_kind:
            return (gallery.image_for(of_kind[-1]),
                    f"no figure carries a claim, so the newest '{kind}' artifact, "
                    f"first of {', '.join(HERO_KINDS)} to appear in this study")
    if present:
        return (gallery.image_for(present[-1]),
                f"no figure carries a claim and none is of a preferred kind, so the "
                f"newest registered picture ({present[-1].kind})")
    return None, "the study has registered no picture that is still on disk"


# -- the methods material -------------------------------------------------------------


def read_dict(case: Path | None, relative: str) -> str:
    if case is None:
        return ""
    try:
        return (case / relative).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def as_written(body: str) -> list[str]:
    """A dictionary block's own lines, comments gone, blanks gone, nothing parsed.

    Used where the keys are not words. `preflight.entry_values` reads `key value;`
    pairs, and a `divSchemes` block's keys are `div(phi,U)` -- so the pattern skips
    past the key it cannot match and reports the first bare word it finds inside the
    value, which came out as `bounded  Gauss limitedLinear 1`. The divergence scheme is
    one of the few entries in a case that changes the answer on its own, so it is
    quoted rather than parsed: the block as the case writes it is exactly right here,
    and is what somebody reproducing the run would copy.
    """
    return [line.strip() for line in preflight.strip_comments(body).splitlines()
            if line.strip()]


def turbulence(case: Path | None) -> tuple[str, str, str]:
    """`simulationType`, the model, and the file it was read from.

    Two files, because OpenFOAM moved the entry: `constant/momentumTransport` in the
    .com line, `constant/turbulenceProperties` in the ESI one that is on this image.
    `ladder.py` reads both for the same reason.
    """
    for name in ("constant/turbulenceProperties", "constant/momentumTransport"):
        text = read_dict(case, name)
        if not text:
            continue
        values = preflight.entry_values(text)
        kind = values.get("simulationType", "")
        model = ""
        for keyword, entry in (("RAS", "RASModel"), ("LES", "LESModel")):
            body = preflight.block_body(text, keyword)
            if body:
                inner = preflight.entry_values(body)
                model = inner.get(entry) or inner.get("model") or ""
                if model:
                    break
        if kind or model:
            return kind, model, name
    return "", "", ""


def boundary_types(case: Path | None, field: str) -> list[tuple[str, str]]:
    """`(patch, type)` out of one `0/` field's `boundaryField`.

    The boundary conditions are the case, far more than the solver name is: the same
    mesh with `fixedValue` and with `zeroGradient` on the outlet are two different
    problems with two different answers, and nothing else in the document would say
    which was solved.
    """
    text = read_dict(case, f"0/{field}")
    if not text:
        return []
    body = preflight.block_body(text, "boundaryField")
    rows = []
    for name, inner in preflight.block_entries(body):
        if not name:
            continue
        rows.append((name, preflight.entry_values(inner).get("type", "?")))
    return rows


def find_check_log(case: Path | None) -> Path | None:
    """The newest `checkMesh` output in the case, whatever it was called.

    `log.checkMesh` is the convention and `checkMesh.log` is what half the scripts in
    the world write, so both are looked for rather than one being declared correct.
    """
    if case is None:
        return None
    found = [p for p in case.glob("*") if p.is_file() and "checkmesh" in p.name.casefold()]
    if not found:
        found = [p for p in case.glob("log*") if p.is_file()
                 and "Checking geometry" in p.read_text(errors="replace")[:20000]]
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def methods_lines(
    case: Path | None,
    case_rule: str,
    others: list[str],
    *,
    layers: bool = True,
) -> tuple[list[str], list[str]]:
    """The methods section as lines, and the files it was read from.

    Only the settings that change an answer. A methods section that reproduced every
    entry of every dictionary would be a copy of the case, which the case already is;
    what a reader needs is the solver, the schemes, the turbulence model, the time
    controls, the convergence criteria, the boundary conditions, the mesh it ran on,
    and the reference values any coefficient was divided by.
    """
    lines: list[str] = []
    read: list[str] = []
    lines.append(f"case: {case if case else 'none'}")
    lines.append(f"chosen by: {case_rule}")
    if others:
        lines.append(f"other cases in this study, not described here: {', '.join(others)}")
    lines.append("")

    control = preflight.parse_control(read_dict(case, "system/controlDict"))
    if control:
        read.append("system/controlDict")
        lines.append("solver and time controls (system/controlDict, top level only)")
        for key in ("application", "startFrom", "startTime", "stopAt", "endTime",
                    "deltaT", "writeControl", "writeInterval", "purgeWrite",
                    "adjustTimeStep", "maxCo", "maxDeltaT", "writeFormat"):
            if key in control:
                lines.append(f"  {key:<16} {control[key]}")
    else:
        lines.append("solver and time controls: system/controlDict was not readable")
    lines.append("")

    kind, model, where = turbulence(case)
    if where:
        read.append(where)
        lines.append(f"turbulence ({where})")
        lines.append(f"  simulationType   {kind or 'not stated'}")
        lines.append(f"  model            {model or 'none named'}")
    else:
        lines.append("turbulence: no turbulenceProperties or momentumTransport in the case")
    transport = preflight.entry_values(read_dict(case, "constant/transportProperties"))
    if transport:
        read.append("constant/transportProperties")
        lines.append("")
        lines.append("transport (constant/transportProperties, as written)")
        for key, value in sorted(transport.items()):
            lines.append(f"  {key:<16} {value}")
    lines.append("")

    schemes_text = read_dict(case, "system/fvSchemes")
    if schemes_text:
        read.append("system/fvSchemes")
        lines.append("discretisation (system/fvSchemes)")
        for block in ("ddtSchemes", "gradSchemes", "divSchemes", "laplacianSchemes",
                      "interpolationSchemes", "snGradSchemes"):
            body = preflight.block_body(schemes_text, block)
            if not body.strip():
                continue
            lines.append(f"  {block}")
            lines.extend(f"    {line}" for line in as_written(body))
    else:
        lines.append("discretisation: system/fvSchemes was not readable")
    lines.append("")

    solution_text = read_dict(case, "system/fvSolution")
    if solution_text:
        read.append("system/fvSolution")
        lines.append("linear solvers and convergence (system/fvSolution)")
        solvers_body = preflight.block_body(solution_text, "solvers")
        for field, body in preflight.block_entries(solvers_body):
            values = preflight.entry_values(body)
            bits = [f"{key} {values[key]}" for key in ("solver", "smoother", "tolerance",
                                                       "relTol", "preconditioner")
                    if key in values]
            lines.append(f"  {field:<12} " + ", ".join(bits))
        for block in ("SIMPLE", "PIMPLE", "PISO"):
            body = preflight.block_body(solution_text, block)
            if not body:
                continue
            lines.append(f"  {block}")
            for key, value in preflight.entry_values(preflight.outer_text(body)).items():
                if value.strip():
                    lines.append(f"    {key:<24} {value}")
            control_body = preflight.block_body(body, "residualControl")
            if control_body.strip():
                lines.append("    residualControl")
                lines.extend(f"      {line}" for line in as_written(control_body))
        relax = preflight.block_body(solution_text, "relaxationFactors")
        if relax.strip():
            lines.append("  relaxationFactors")
            lines.extend(f"    {line}" for line in as_written(relax))
    else:
        lines.append("linear solvers and convergence: system/fvSolution was not readable")
    lines.append("")

    zero = (case / "0") if case is not None else None
    fields = sorted(p.name for p in zero.iterdir() if p.is_file()) \
        if zero is not None and zero.is_dir() else []
    if fields:
        read.append("0/")
        lines.append(f"initial and boundary conditions (0/: {', '.join(fields)})")
        for field in fields:
            rows = boundary_types(case, field)
            initial = preflight.entry_values(
                preflight.outer_text(read_dict(case, f"0/{field}"))
            ).get("internalField", "")
            lines.append(f"  {field}  internalField {initial or 'not stated'}")
            for patch, patch_type in rows:
                lines.append(f"    {patch:<22} {patch_type}")
    else:
        lines.append("initial and boundary conditions: no 0/ directory in the case")
    lines.append("")

    check = find_check_log(case)
    if check is not None:
        read.append(check.name)
        data = mesh_digest.parse(check.read_text(errors="replace"))
        lines.append(f"mesh ({check.name})")
        for key, value in sorted(data.get("counts", {}).items()):
            lines.append(f"  {key:<16} {value:,}")
        for key in ("non_ortho", "skewness", "aspect_ratio", "openness",
                    "face_determinant", "cell_volume", "face_area", "bounding_box"):
            if key in data:
                value = data[key]
                lines.append(f"  {key:<16} "
                             + (" to ".join(str(v) for v in value)
                                if isinstance(value, tuple) else str(value)))
        if data.get("patches"):
            lines.append("  patches")
            for row in data["patches"]:
                lines.append(f"    {row[0]:<22} {row[1]} faces, {row[2]} points")
        for failure in data.get("failures", []):
            lines.append(f"  FAILED: {failure}")
        for warning in data.get("warnings", []):
            lines.append(f"  warning: {warning}")
        if data.get("mesh_ok_line"):
            lines.append(f"  {data['mesh_ok_line']}")
    else:
        lines.append("mesh: no checkMesh output in the case")
    lines.append("")

    log = results.find_solver_log(case) if case is not None else None
    if log is not None:
        read.append(log.name)
        data = log_digest.digest(log)
        lines.append(f"what the solver did ({log.name})")
        for line in log_digest.how_it_ended(data, log_digest.requested_end_time(log)):
            lines.extend(f"  {wrapped}" for wrapped in _wrap(line, 96, "    "))
        times = data.get("times") or []
        if times:
            lines.append(f"  time steps parsed {len(times)}, last t = {times[-1]:g}")
    else:
        lines.append("what the solver did: no solver log in the case")
    lines.append("")

    if layers and case is not None and (case / "constant" / "polyMesh" / "points").exists():
        import layer_report  # noqa: PLC0415 - reads the whole mesh; only when asked

        measured = layer_report.measure(case)
        lines.append("wall layers (layer_report.py, from constant/polyMesh)")
        for patch in measured.get("patches", []):
            first = patch.get("first_cell_m") or {}
            median = first.get("median")
            lines.append(f"  {str(patch.get('patch', '?')):<22} "
                         f"type {patch.get('type', '?')}, "
                         f"faces {patch.get('faces', '?')}, "
                         "first cell height median "
                         + (f"{median:.4g} m" if isinstance(median, float) else "not measured")
                         + (f"  {patch['note']}" if patch.get("note") else ""))
        for problem in measured.get("problems", []):
            lines.append(f"  not measured: {problem}")
        read.append("constant/polyMesh")
    elif layers:
        lines.append("wall layers: no constant/polyMesh/points to measure")
    else:
        lines.append("wall layers: not measured (--no-layers)")

    return lines, read


# -- what was asked for and not produced ----------------------------------------------


def gaps(study: Path, case: str = "") -> list[str]:
    """What the study itself recorded as skipped or failed, in its own words.

    Read rather than inferred. `phases.json` distinguishes `skipped` from `pending` on
    purpose -- "not wanted" and "not done" do not read the same -- and `results.py`
    carries each skipped output's reason into `results.md` because "there is no forces
    plot" and "there is no forces plot because nothing logged any" are different facts.
    Both are quoted here; nothing new is concluded.
    """
    lines: list[str] = []
    phases = study_state.load_phases(study)
    rows = [row for row in phases.get("phases", [])
            if isinstance(row, dict) and row.get("status") in ("skipped", "failed")]
    if rows:
        lines.append("phases the study did not complete (.reynolds/phases.json)")
        for row in rows:
            note = str(row.get("note") or "")
            lines.append(f"  {str(row.get('name')):<12} {row.get('status')}"
                         + (f"  {note}" if note else ""))
        lines.append("")

    latest = study_state.latest(REPORT_KIND, root=study, case=case)
    if latest and str(latest.get("path", "")).endswith(".md"):
        path = Path(str(latest["abspath"]))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        for heading in ("## skipped", "## failed"):
            if heading not in text:
                continue
            block = text.split(heading, 1)[1].split("\n## ", 1)[0]
            quoted = [line.strip() for line in block.splitlines() if line.strip().startswith("-")]
            if quoted:
                lines.append(f"{heading.strip('# ')}, as {path.name} recorded it")
                lines.extend(f"  {line}" for line in quoted)
                lines.append("")
    if not lines:
        lines.append("The study recorded nothing as skipped or failed. That is the "
                     "record, not a guarantee that nothing was left out.")
    return lines


# -- drawing the pages ----------------------------------------------------------------


def _wrap(text: str, width: int, indent: str = "  ") -> list[str]:
    """Hand-rolled because every line here is already laid out for a monospaced page,
    and `textwrap` would re-flow the alignment out of the tables above.

    A word longer than the width is cut rather than left to run off the edge of the
    page, because the words that do that here are absolute paths and a path that runs
    into the margin is a path nobody can read.
    """
    width = max(width, len(indent) + 8)
    out: list[str] = []
    line = ""
    for word in str(text).split():
        while len(indent) + len(word) > width:
            if line.strip() and width - len(line) - 1 < 8:
                # Not worth starting a path on the tail of a line.
                out.append(line)
                line = ""
                continue
            room = (width - len(line) - 1) if line.strip() else width
            out.append(f"{line} {word[:room]}".rstrip() if line.strip() else word[:room])
            word = word[room:]
            line = indent
        candidate = f"{line} {word}".rstrip() if line.strip() else f"{line}{word}"
        if len(candidate) > width and line.strip():
            out.append(line)
            line = indent + word
        else:
            line = candidate
    if line.strip():
        out.append(line)
    return out or [""]


def _blank_page() -> tuple[Any, Any]:
    figure = plt.figure(figsize=PAGE_INCHES, dpi=PAGE_DPI)
    figure.patch.set_facecolor("white")
    return figure, figure.add_axes((0, 0, 1, 1))


def _frame(axes) -> None:
    axes.set_axis_off()
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)


def place_logo(figure, logo: Path | None, notes: list[str]) -> None:
    """The logo on the cover, at whatever aspect ratio it came in at.

    The box is fixed and the image keeps its own shape inside it, which is the one
    thing a logo must not lose: a wordmark stretched to fill a square is worse than no
    logo. A file that will not read back is a note, not an exception -- the document is
    the deliverable and the logo is decoration on it.
    """
    if logo is None:
        return
    try:
        image = mpimg.imread(str(logo))
    except Exception as exc:  # noqa: BLE001
        notes.append(f"logo not drawn: {type(exc).__name__}: {exc}")
        return
    height, width = image.shape[0], image.shape[1]
    box_width, box_height = 0.20, 0.07
    scale = min(box_width / max(width, 1), box_height / max(height, 1))
    draw_width, draw_height = width * scale, height * scale
    # Right-aligned in the box, so a wide wordmark and a square mark both end at the
    # same margin. The box is in figure coordinates; the image keeps its own aspect.
    left = 0.93 - draw_width
    axes = figure.add_axes((left, 0.935 - draw_height, draw_width, draw_height))
    axes.imshow(image)
    axes.set_axis_off()


def cover_page(
    pdf: PdfPages,
    *,
    template: dict[str, Any],
    study: Path,
    digest: dict[str, Any],
    numbers: list[dict[str, Any]],
    hero: Path | None,
    hero_rule: str,
    logo: Path | None,
    notes: list[str],
) -> None:
    """The one page somebody who reads nothing else will read: who this is, what the
    numbers are, whether each has a figure behind it, and one picture."""
    figure, axes = _blank_page()
    _frame(axes)
    place_logo(figure, logo, notes)

    title = str(template.get("title") or f"{digest['study']} -- study report")
    axes.text(0.07, 0.945, title, fontsize=19, va="top", ha="left", weight="bold")
    subtitle = str(template.get("subtitle") or "")
    y = 0.905
    if subtitle:
        axes.text(0.07, y, subtitle, fontsize=10.5, va="top", ha="left", color="#44555f")
        y -= 0.028

    identity = [
        f"study        {digest['study']}  ({study})",
        f"case         {digest.get('case') or '-'}",
        "phases       " + "  ".join(f"{name}:{status}" for name, status in digest["phases"]),
        f"artifacts    {digest['artifact_count']} registered and on disk",
        f"written      {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
    ]
    for line in identity:
        for wrapped in _wrap(line, 100, "             "):
            axes.text(0.07, y, wrapped, fontsize=7.4, va="top", ha="left",
                      family="monospace", color="#2b3b45")
            y -= LINE_PITCH
    for line in template.get("front_matter") or []:
        for wrapped in _wrap(str(line), 100, "  "):
            axes.text(0.07, y, wrapped, fontsize=7.6, va="top", ha="left", color="#2b3b45")
            y -= LINE_PITCH
    y -= 0.012

    axes.text(0.07, y, "key numbers", fontsize=11.5, va="top", ha="left", weight="bold")
    y -= 0.026
    axes.text(0.07, y,
              "Each number is shown with the figure it was drawn from, or with the "
              "reason there is none.",
              fontsize=7.6, va="top", ha="left", color="#55666f")
    y -= 0.022

    colour = {"verified": "#1b5e20", "disagrees": "#b71c1c", "unverified": "#5c6f7d"}
    if not numbers:
        axes.text(0.07, y, "none: this study has recorded no claim and its files carry "
                           "no coefficient or residual",
                  fontsize=8, va="top", ha="left", family="monospace", color="#5c6f7d")
        y -= LINE_PITCH
    for row in numbers:
        state = row["state"]
        axes.text(0.07, y, f"{state:<11}", fontsize=7.6, va="top", ha="left",
                  family="monospace", color=colour.get(state, "#5c6f7d"), weight="bold")
        axes.text(0.185, y, f"{row['label']}: {row['text']}", fontsize=7.6, va="top",
                  ha="left", family="monospace")
        y -= LINE_PITCH
        for wrapped in _wrap(row["why"], 82, ""):
            axes.text(0.195, y, wrapped, fontsize=6.9, va="top", ha="left",
                      family="monospace", color="#55666f")
            y -= LINE_PITCH * 0.92
        if y < 0.44:
            axes.text(0.07, y, "... the rest are in the sidecar and on stdout",
                      fontsize=7, va="top", ha="left", family="monospace", color="#5c6f7d")
            break

    hero_box = figure.add_axes((0.07, 0.07, 0.86, 0.30))
    hero_box.set_xticks([])
    hero_box.set_yticks([])
    drawn = False
    if hero is not None:
        try:
            hero_box.imshow(mpimg.imread(str(hero)))
            drawn = True
        except Exception as exc:  # noqa: BLE001
            hero_box.text(0.5, 0.5, f"hero figure could not be read back\n{hero}\n"
                                    f"{type(exc).__name__}: {exc}",
                          transform=hero_box.transAxes, ha="center", va="center",
                          family="monospace", fontsize=7.5, color="#78909c")
            notes.append(f"hero not drawn: {type(exc).__name__}: {exc}")
    else:
        hero_box.text(0.5, 0.5, "no hero figure", transform=hero_box.transAxes,
                      ha="center", va="center", family="monospace", fontsize=8,
                      color="#78909c")
    # The name and the rule, not the whole path: the path is in the sidecar and on
    # stdout, and a caption that runs off into the margin is a caption nobody reads.
    caption = [f"hero: {hero.name if hero else 'none'}"]
    caption += _wrap(f"chosen because: {hero_rule}", 112, "")
    hero_box.set_title("\n".join(caption), fontsize=6.8, loc="left", color="#44555f")
    if not drawn:
        hero_box.set_facecolor("#fafafa")

    footer = str(template.get("footer") or "")
    if footer:
        for index, wrapped in enumerate(_wrap(footer, 110, "")):
            axes.text(0.07, 0.045 - index * 0.016, wrapped, fontsize=6.6, va="top",
                      ha="left", color="#55666f")
    pdf.savefig(figure)
    plt.close(figure)


BODY_WIDTH = 100
"""Monospaced characters that fit across a text page at `BODY_SIZE`. Lines are folded
to it here rather than where they are built, so one page-geometry number governs every
section and no section has to know how wide the page is."""


def text_pages(pdf: PdfPages, title: str, lines: Iterable[str]) -> int:
    """A block of already-laid-out monospaced lines, paginated. Returns the pages."""
    folded: list[str] = []
    for line in lines:
        if len(line) <= BODY_WIDTH:
            folded.append(line)
            continue
        # The leading space is the table column the line sits in, so it is kept and
        # the fold happens inside it.
        pad = line[:len(line) - len(line.lstrip())]
        folded.extend(pad + piece
                      for piece in _wrap(line.strip(), BODY_WIDTH - len(pad), "    "))
    lines = folded or ["(nothing)"]
    pages = 0
    for start in range(0, len(lines), BODY_LINES):
        chunk = lines[start:start + BODY_LINES]
        figure, axes = _blank_page()
        _frame(axes)
        heading = title if start == 0 else f"{title} (continued)"
        axes.text(0.07, 0.955, heading, fontsize=13, va="top", ha="left", weight="bold")
        y = 0.915
        for line in chunk:
            axes.text(0.07, y, line, fontsize=BODY_SIZE, va="top", ha="left",
                      family="monospace")
            y -= LINE_PITCH
        pdf.savefig(figure)
        plt.close(figure)
        pages += 1
    return pages


def figure_pages(pdf: PdfPages, groups: list[tuple[str, list[gallery.Entry]]]) -> tuple[int, list[dict]]:
    """The figures, grouped as the gallery groups them, four to a page.

    A row whose file is gone gets its panel anyway, labelled MISSING. `gallery.py` shows
    it that way for a reason worth repeating here: a manifest saying a picture once
    existed is often the only surviving evidence that a step ran, and a document that
    drops it makes a deleted figure look like one that was never made.
    """
    pages = 0
    placed: list[dict] = []
    for kind, group in groups:
        pick = gallery.newest(group)
        for start in range(0, len(group), FIGURES_PER_PAGE):
            chunk = group[start:start + FIGURES_PER_PAGE]
            figure = plt.figure(figsize=PAGE_INCHES, dpi=PAGE_DPI)
            figure.patch.set_facecolor("white")
            gone = sum(1 for entry in group if not entry.exists)
            heading = f"{kind}  ({len(group)}" + (f", {gone} missing" if gone else "") + ")"
            figure.suptitle(heading, fontsize=12, x=0.07, ha="left", weight="bold")
            # Sized to what is on the page, through the gallery's own rule, so a kind
            # with one picture gets a page with one picture on it rather than a
            # quarter-page panel and three empty frames.
            rows, cols = gallery.grid_shape(len(chunk), max_cols=2)
            grid = figure.subplots(rows, cols, squeeze=False)
            cells = [cell for row in grid for cell in row]
            for cell in cells[len(chunk):]:
                cell.set_axis_off()
            for cell, entry in zip(cells, chunk):
                cell.set_xticks([])
                cell.set_yticks([])
                for spine in cell.spines.values():
                    spine.set_color("#b0bec5")
                    spine.set_linewidth(0.8)
                state = "MISSING" if not entry.exists else ""
                picture = gallery.image_for(entry)
                if not entry.exists:
                    cell.set_facecolor("#fbfbfb")
                    cell.text(0.5, 0.5, "MISSING\nrecorded, not on disk now\n"
                                        f"{entry.rel}",
                              transform=cell.transAxes, ha="center", va="center",
                              family="monospace", fontsize=7, color="#90a4ae")
                elif picture is None:
                    cell.set_facecolor("#fbfbfb")
                    cell.text(0.5, 0.5, f"not a picture\n{entry.rel}",
                              transform=cell.transAxes, ha="center", va="center",
                              family="monospace", fontsize=7, color="#90a4ae")
                    state = "not a picture"
                else:
                    try:
                        cell.imshow(mpimg.imread(str(picture)))
                    except Exception as exc:  # noqa: BLE001 - one panel, not one page
                        cell.set_facecolor("#fbfbfb")
                        cell.text(0.5, 0.5, f"{picture.name}\ncould not be read back\n"
                                            f"{type(exc).__name__}: {exc}",
                                  transform=cell.transAxes, ha="center", va="center",
                                  family="monospace", fontsize=6.5, color="#90a4ae")
                        state = "unreadable"
                label = gallery.describe(entry) or entry.rel
                mark = "  [latest]" if entry is pick and len(group) > 1 else ""
                cell.set_title("\n".join(_wrap(f"{entry.rel}{mark}", 52, "  ")
                                         + _wrap(label, 52, "  ")),
                               fontsize=6.4, loc="left", color="#44555f")
                placed.append({
                    "kind": kind, "path": str(entry.path), "rel": entry.rel,
                    "exists": entry.exists, "state": state or "drawn",
                    "page": pages + 1,
                })
            figure.tight_layout(rect=(0.02, 0.01, 0.98, 0.95))
            pdf.savefig(figure)
            plt.close(figure)
            pages += 1
    return pages, placed


# -- the whole document ----------------------------------------------------------------


def write_report(
    study: Path,
    out: Path,
    *,
    template: dict[str, Any],
    case: str = "",
    hero: Path | None = None,
    logo: Path | None = None,
    layers: bool = True,
) -> dict[str, Any]:
    """The document, and a record of what went into it and what was missing."""
    entries = [entry for entry in gallery.collect(root=study, case=case)
               if (entry.meta or {}).get("source") != SOURCE_TAG]
    groups = gallery.group_by_kind(entries)
    figure_claims = stamped_claims(study, case=case)
    case_dir, case_rule, others = choose_case(study, case)
    measured, measure_notes = measured_numbers(case_dir)
    numbers = tie_numbers(figure_claims, measured)
    hero_path, hero_rule = pick_hero(entries, figure_claims, hero)
    notes = list(measure_notes)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pages: list[str] = []
    placed: list[dict] = []
    methods: list[str] = []
    sources: list[str] = []
    gap_lines: list[str] = []

    with PdfPages(str(out)) as pdf:
        for section in template["sections"]:
            if section == "summary":
                cover_page(
                    pdf, template=template, study=study,
                    digest=study_state.digest(study), numbers=numbers,
                    hero=hero_path, hero_rule=hero_rule, logo=logo, notes=notes,
                )
                pages.append("summary")
            elif section == "figures":
                count, placed = figure_pages(pdf, groups)
                if count == 0:
                    count = text_pages(pdf, "figures", [
                        "No artifact is registered in this study's manifest.",
                        f"The manifest is {study / study_state.STATE_DIR / study_state.MANIFEST}.",
                    ])
                pages.extend(["figures"] * count)
            elif section == "methods":
                methods, sources = methods_lines(case_dir, case_rule, others, layers=layers)
                pages.extend(["methods"] * text_pages(pdf, "methods", methods))
            elif section == "gaps":
                gap_lines = gaps(study, case=case)
                pages.extend(["gaps"] * text_pages(
                    pdf, "asked for and not produced", gap_lines))

    return {
        "study": str(study),
        "out": str(out.resolve()),
        "pages": len(pages),
        "page_sections": pages,
        "case": str(case_dir) if case_dir else "",
        "case_rule": case_rule,
        "other_cases": others,
        "hero": str(hero_path) if hero_path else "",
        "hero_rule": hero_rule,
        "logo": str(logo) if logo else "",
        "numbers": numbers,
        "figures": placed,
        "missing": [row for row in placed if not row["exists"]],
        "methods_read": sources,
        "gaps": gap_lines,
        "notes": notes,
        "sections": list(template["sections"]),
    }


def render_summary(record: dict[str, Any]) -> str:
    """The same answer in the terminal, because most of the time that is enough."""
    lines = [
        f"{record['out']}",
        f"{record['pages']} page(s): " + ", ".join(record["page_sections"]),
        f"hero: {record['hero'] or 'none'}",
        f"  rule: {record['hero_rule']}",
        f"case for the methods section: {record['case'] or 'none'}  ({record['case_rule']})",
    ]
    counts: dict[str, int] = {}
    for row in record["numbers"]:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    lines.append("key numbers: " + (", ".join(f"{state} x{count}"
                                              for state, count in sorted(counts.items()))
                                    or "none"))
    for row in record["numbers"]:
        if row["state"] != "verified":
            lines.append(f"  {row['state']}: {row['label']}: {row['text']} -- {row['why']}")
    if record["missing"]:
        lines.append(f"{len(record['missing'])} registered artifact(s) are MISSING and are "
                     "on the pages as MISSING:")
        for row in record["missing"]:
            lines.append(f"  {row['rel']}")
    if record["methods_read"]:
        lines.append("methods read: " + ", ".join(record["methods_read"]))
    for note in record["notes"]:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", nargs="?", default=Path("."), type=Path,
                        help="a path inside the study (default: here)")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"where the PDF goes (default: <study>/{PDF_NAME}); a "
                             "relative path is relative to the directory you run this "
                             "from, not to the case")
    parser.add_argument("--json", dest="sidecar", type=Path, default=None,
                        help="write what went into the document, and what was missing,"
                             " as JSON")
    parser.add_argument("--template", type=Path, default=None,
                        help=f"a JSON template: title, subtitle, front_matter, sections,"
                             f" footer (example: {TEMPLATE_NAME})")
    parser.add_argument("--logo", type=Path, default=None,
                        help="an image for the top right of the cover")
    parser.add_argument("--hero", type=Path, default=None,
                        help="the figure for the cover, instead of the chosen one")
    parser.add_argument("--case", default="",
                        help="the case the methods section describes, and the only"
                             " case's artifacts to include")
    parser.add_argument("--no-layers", dest="layers", action="store_false",
                        help="skip the wall-layer measurement, which reads the whole"
                             " polyMesh")
    args = parser.parse_args(argv)

    try:
        template, template_notes = load_template(args.template)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 2

    study = study_state.find_root(args.root)
    record = write_report(
        study, args.out or study / PDF_NAME,
        template=template, case=args.case, hero=args.hero, logo=args.logo,
        layers=args.layers,
    )
    record["template"] = str(args.template) if args.template else "(built in)"
    record["notes"] = template_notes + record["notes"]

    study_state.record(
        REPORT_KIND, record["out"], root=study, case=args.case,
        label=f"report pack, {record['pages']} pages", source=SOURCE_TAG,
    )
    record["registered_as"] = REPORT_KIND

    print(render_summary(record))
    print(f"registered in the manifest under kind '{REPORT_KIND}'")
    if args.sidecar:
        Path(args.sidecar).parent.mkdir(parents=True, exist_ok=True)
        Path(args.sidecar).write_text(json.dumps(record, indent=2, default=str),
                                      encoding="utf-8")
        print(Path(args.sidecar).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
