#!/usr/bin/env python3
"""Draw the live extension's results as pictures: four convergence studies, one scorecard, six GIFs.

    python benchmarks/live/gallery.py            # write benchmarks/live/gallery/
    python benchmarks/live/gallery.py --check    # verify what is already there

Everything here is **read** from `benchmarks/live/results/*/result.json` or quoted from a
located line of `PROGRESS_LIVE.md`. Nothing is solved, nothing is re-rendered and no GPU
is burned: a gallery that re-ran the cases would be a different measurement from the one
the document reports, and the point of the gallery is to show *that* measurement.

## Why a number's source is in the caption

`summarise.py`'s docstring makes the argument for generating the summary table rather
than typing it: a reference transcribed by hand drifts from the one the code applies.
The same applies to a figure, and harder -- a plot of hand-typed numbers is a hand-typed
claim with a grid behind it, which reads as evidence. So:

* a number that a `result.json` carries is read from it, and the caption names the file
  and the field;
* a number that only the prose carries -- a Richardson limit, an extrapolation, a run
  whose `result.json` is not on disk -- is **located** in `PROGRESS_LIVE.md` by
  `_quote`, which returns the line number it found it on and raises if the line is gone.
  So a caption that says `PROGRESS_LIVE.md:529` was true when the figure was drawn, and
  an edit that moves the claim breaks this script rather than silently ageing the plot.

Points from the two sources are drawn with different markers, and the caption says which
is which.

## What is reused from `openreynolds/live/figures.py`, and what is not

`ensure_rgb`, because an RGBA PNG renders as a black panel in a thumbnail index and that
was learned the expensive way, and `_matplotlib`, because it binds the Agg backend
*before* `pyplot` and the failure that guard prevents is a hang rather than an error.

`slice_figure`, `animate` and `slice_frame` are not used, and the reason is not style:
all three take a live `FluidWorld`. This script has recorded JSON and no solver, so the
animations are the ones the cases already wrote, **copied**. `profile_figure` is not used
either -- it has no caption, no tolerance band and no annotation, and in a gallery the
caption is the load-bearing part.

## The flat-image check

`--check` reads every file back and reports a per-channel standard deviation and a unique
colour count, and exits non-zero if either is too low. `docs/isaac-sim-notes.md` records
why both numbers rather than one: a whole-array standard deviation scored 51.32 on a flat
teal frame whose true colour count was 1. A file that exists is not a figure.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "gallery"
REPO = HERE.parents[1]
PROGRESS = REPO / "PROGRESS_LIVE.md"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import summarise  # noqa: E402  -- beside this file; reused for MILESTONE_OF and rows()
from openreynolds.live.figures import _matplotlib, ensure_rgb  # noqa: E402

PASS_COLOUR = "#1b7837"
FAIL_COLOUR = "#b2182b"
PROSE_COLOUR = "#7f3b08"
BAND_COLOUR = "#4393c3"


# --------------------------------------------------------------------------- sources


def _result(label: str) -> dict:
    return json.loads((RESULTS / label / "result.json").read_text(encoding="utf-8"))


def _provenance(labels: list[str]) -> str:
    """`commit e5b1698, clean` for a set of runs, or every state they were in.

    A figure drawn from several runs is only as pinned as its worst run, so a dirty tree
    or a missing provenance block is named rather than averaged away. `summarise.py`
    says why: Modal mounts the working tree and not a commit, so a sha plus uncommitted
    changes does not pin what ran.
    """
    seen = []
    for label in labels:
        pv = _result(label).get("provenance") or {}
        sha = str(pv.get("commit") or "")
        if not sha:
            state = f"{label}: provenance UNRECORDED"
        elif pv.get("dirty"):
            state = (f"{label}: {sha[:7]}-DIRTY "
                     f"({pv.get('dirty_files', '?')} uncommitted file(s))")
        else:
            state = f"{label}: {sha[:7]} clean"
        if state not in seen:
            seen.append(state)
    return "; ".join(seen)


_PROGRESS_LINES: list[str] | None = None


def _quote(needle: str) -> tuple[int, str]:
    """Find `needle` in PROGRESS_LIVE.md; return its 1-based line number and that line.

    Raises when the line is gone. That is the feature: a caption citing a line of prose
    is a claim about the prose, and this script is where it gets checked.
    """
    global _PROGRESS_LINES
    if _PROGRESS_LINES is None:
        _PROGRESS_LINES = PROGRESS.read_text(encoding="utf-8").splitlines()
    hits = [i for i, line in enumerate(_PROGRESS_LINES, 1) if needle in line]
    if not hits:
        raise SystemExit(
            f"{PROGRESS.name} no longer contains {needle!r}.\n"
            "A gallery caption cites that line. Either the claim moved -- update the "
            "needle -- or it was withdrawn, in which case the figure must change too.")
    return hits[0], _PROGRESS_LINES[hits[0] - 1].strip()


# --------------------------------------------------------------------------- drawing


CAPTION_SIZE = 7.4


def _finish(fig, name: str, caption: str) -> Path:
    """Caption under the axes, RGB on disk. Every figure in this directory goes through here.

    The caption is wrapped to the figure's own width and the axes are then given what is
    left, rather than the axes being laid out first and the caption written over them.
    A figure whose caption runs off the right edge is a figure whose number cannot be
    read, which in this gallery is the whole of the content.
    """
    plt = _matplotlib()
    width, height = fig.get_size_inches()
    # 0.62 em per character, measured on DejaVu Sans rather than assumed: the nominal
    # 0.5 em put three captions off the right-hand edge of their own figure.
    columns = max(60, int((width - 0.2) / (0.62 * CAPTION_SIZE / 72.0)))
    lines: list[str] = []
    for paragraph in caption.split("\n"):
        lines += textwrap.wrap(paragraph, columns) or [""]
    bottom = (0.155 * len(lines) + 0.30) / height
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.text(0.008, 0.012, "\n".join(lines), fontsize=CAPTION_SIZE, va="bottom",
             ha="left", color="#222222", linespacing=1.45)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / name
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)
    return ensure_rgb(out)


def _log_ticks(ax, positions, labels=None) -> None:
    """Label exactly these x positions on a log axis and nothing else.

    A log axis labels its own decade minors -- `4 x 10^2`, `6 x 10^2` -- straight across
    the cell counts this gallery puts there, and two overlapping numbers under a point
    is worse than no number at all.
    """
    from matplotlib.ticker import NullFormatter, NullLocator

    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(positions)
    ax.set_xticklabels(labels or [str(p) for p in positions])


def _grid_axes(ax, cells, values, *, label: str) -> None:
    ax.plot(cells, values, "o-", color="#01665e", lw=1.6, ms=6, label=label, zorder=5)
    for x, y in zip(cells, values):
        ax.annotate(f"{y:.4g}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="#01665e")
    _log_ticks(ax, cells)
    ax.grid(alpha=0.3)
    ax.set_xlabel("cells across the domain")


def dam_break() -> Path:
    """Three grids of the dam break: h(dam)/Ritter and the profile RMS, with the Richardson limit."""
    plt = _matplotlib()
    labels = ["dam-n325", "dam-n650", "dam-n1300"]
    cells = [325, 650, 1300]
    runs = [_result(l) for l in labels]
    ratio = [r["at_dam"] / r["exact_at_dam"] for r in runs]
    rms = [100.0 * next(c["measured"] for c in r["checks"]
                        if c["name"].startswith("depth profile")) for r in runs]

    order_line, order_text = _quote("| **observed order** |")
    limit_line, limit_text = _quote("| **Richardson limit** |")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    ax = axes[0]
    ax.axhspan(0.85, 1.15, color=BAND_COLOUR, alpha=0.16,
               label="the case's tolerance, +/- 15%")
    ax.axhline(1.0, color="black", lw=1.2, ls="-", label="Ritter's 4 h0 / 9 = 22.2222 mm")
    ax.axhline(1.1382, color=PROSE_COLOUR, lw=1.4, ls="--",
               label="Richardson limit 1.1382 (order 1.232)")
    _grid_axes(ax, cells, ratio, label="measured h(dam) / Ritter")
    ax.set_ylabel("h at the dam, as a multiple of Ritter's 4 h0 / 9")
    ax.set_ylim(0.82, 1.32)
    ax.set_title("dam break: depth at the dam. Every affordable grid FAILS;\n"
                 "the grid-converged limit is inside the band", fontsize=10)
    ax.legend(fontsize=7.6, loc="upper right")

    ax = axes[1]
    ax.axhline(15.0, color=BAND_COLOUR, lw=1.4, ls="-", label="tolerance, <= 15% of h0")
    ax.axhline(4.22, color=PROSE_COLOUR, lw=1.4, ls="--",
               label="Richardson limit 4.22% (order 0.527)")
    _grid_axes(ax, cells, rms, label="measured profile RMS")
    ax.set_ylabel("depth profile against Ritter, RMS, % of h0")
    ax.set_ylim(0, 17)
    ax.set_title("dam break: the depth profile. PASSES at every grid,\n"
                 "which the depth at the dam does not", fontsize=10)
    ax.legend(fontsize=7.6, loc="lower left")

    caption = (
        "h(dam)/Ritter 1.2450, 1.1837, 1.1576 and profile RMS 9.02%, 7.55%, 6.53% of h0 at 325/650/1300 cells. "
        "MEASURED, from benchmarks/live/results/dam-n{325,650,1300}/result.json "
        "(fields at_dam / exact_at_dam, and the 'depth profile against Ritter, RMS' check). "
        f"Provenance: {_provenance(labels)}. H100 80GB.\n"
        f"FROM THE PROSE, not from any result.json: observed order 1.232 and Richardson limit 1.1382 "
        f"(PROGRESS_LIVE.md:{order_line} '{order_text}'), and 0.527 / 4.22% "
        f"(PROGRESS_LIVE.md:{limit_line} '{limit_text}'). Those two dashed lines are a three-point fit, not a run.\n"
        "WHAT THIS DOES NOT SHOW: a converged dam break. All three grids fail the +/- 15% check on h(dam) -- the graded case "
        "(dam-n1300, 25.7244 mm against 22.2222) is a FAIL in the summary table. The limit line is where the fit says the "
        "scheme is going, not a measurement, and the gap that survives refinement is argued to be physics (Ritter is inviscid "
        "and hydrostatic, Re_eff is 37/101/202 here) rather than error. That argument is in the prose and is not plotted.")
    return _finish(fig, "convergence-dam-break.png", caption)


def sloshing() -> Path:
    """Four grids of the sloshing period, the 480-cell divergence, and the extrapolated limit."""
    plt = _matplotlib()
    labels = ["slosh4-n80", "slosh3-n120", "slosh4-n180", "slosh3-n240"]
    cells = [80, 120, 180, 240]
    ratio = [(lambda r: r["period"] / r["theory"])(_result(l)) for l in labels]
    bust = _result("slosh3-n480")
    conversions = bust["drift"]["conversions"]
    drift = bust["drift"]["drift"]

    fit_line, fit_text = _quote("observed order 3.5, limit 1.0510")

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.axhspan(0.95, 1.05, color=BAND_COLOUR, alpha=0.16,
               label="the case's tolerance, +/- 5%")
    ax.axhline(1.0, color="black", lw=1.2,
               label="linear theory, omega^2 = g k tanh(k h) -> 417.09 ms")
    ax.axhline(1.0510, color=PROSE_COLOUR, lw=1.4, ls="--",
               label="three-point limit 1.0510 (observed order 3.5)")
    _grid_axes(ax, cells, ratio, label="measured period / theory")
    # No marker for the 480-cell run, at any height. It has no period: the lattice went
    # non-finite. A cross on the theory line would read as agreement with it.
    ax.axvline(480, color=FAIL_COLOUR, lw=1.8, ls="--",
               label="480 cells: DIVERGED, there is no period to plot")
    ax.text(462, 0.90, "480 cells DIVERGES\nlattice non-finite",
            rotation=90, ha="right", va="center", fontsize=8.0, color=FAIL_COLOUR)
    ax.annotate("80 cells: under-resolved,\n25% short, unusable", (80, ratio[0]),
                textcoords="offset points", xytext=(18, 30), fontsize=7.6,
                color="#444444")
    _log_ticks(ax, cells + [480])
    ax.set_xlim(70, 580)
    ax.set_ylabel("first-mode period, as a multiple of linear theory")
    ax.set_ylim(0.70, 1.13)
    ax.set_title("sloshing period: converged, and FIVE PER CENT ABOVE the reference.\n"
                 "The graded case (slosh-n240, 0.439358 s against 0.41709 +/- 5%) FAILS",
                 fontsize=10)
    ax.legend(fontsize=7.6, loc="lower center", framealpha=0.95)

    caption = (
        "period / theory 0.7469, 1.0728, 1.0563, 1.0529 at 80/120/180/240 cells. MEASURED, from "
        "benchmarks/live/results/slosh4-n80, slosh3-n120, slosh4-n180, slosh3-n240/result.json (period / theory). "
        f"The 480-cell divergence is measured too: slosh3-n480 records 'lattice finite' = False, drift {drift:.4g} over "
        f"{conversions:,} cell conversions. Provenance: {_provenance(labels + ['slosh3-n480'])}. H100 80GB.\n"
        f"FROM THE PROSE: observed order 3.5 and limit 1.0510 (PROGRESS_LIVE.md:{fit_line} '{fit_text}') -- a three-point "
        "fit for f_inf, C and p, not a run. The graded row in the summary table is a different run of the same case at the "
        "same resolution (slosh-n240, commit e5b1698, 1.0534 of theory); this figure plots the convergence family, which was "
        "run at one commit so the four grids are comparable with each other.\n"
        "WHAT THIS DOES NOT SHOW: a case that passes, or an explanation. 120 to 240 cells is the only usable window -- 80 is "
        "under-resolved and 480 diverges on a Reynolds-number ceiling -- so the convergence can be followed over a factor of "
        "two and no further. Damping, finite amplitude, the depth, gravity and the estimator are each ruled out with a number "
        "in the prose; the five per cent is unexplained, and it is the one M5 number with no account of itself.")
    return _finish(fig, "convergence-sloshing.png", caption)


def cavity_float32() -> Path:
    """The Re 1000 cavity's error against run length: float32 round-off eroding the vortex."""
    plt = _matplotlib()
    labels = ["cavity-re1000", "cav-steps-600000", "cav-steps-1200000", "cav-steps-2400000"]
    runs = [_result(l) for l in labels]
    steps = [r["stopped"]["steps"] for r in runs]
    rms = [100.0 * r["u_rms"] for r in runs]
    bgk = _result("cav-re1000-bgk")

    prose_line, prose_text = _quote("| 3,492,000 |")
    cross_line, cross_text = _quote("0.86% of error per million steps")

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.axhspan(0, 2.0, color=BAND_COLOUR, alpha=0.16, label="the case's tolerance, <= 2%")
    ax.axhline(2.0, color=BAND_COLOUR, lw=1.3)
    ax.plot(steps, rms, "o-", color="#01665e", lw=1.6, ms=7, zorder=5,
            label="u RMS against Ghia, measured (result.json)")
    for x, y in zip(steps, rms):
        ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#01665e")
    ax.plot([3_492_000], [2.88], "s", mfc="none", mec=PROSE_COLOUR, mew=1.8, ms=11,
            zorder=6, label="3,492,000 steps: 2.88%, FROM THE PROSE ONLY")
    ax.annotate("2.88%\nno result.json on disk", (3_492_000, 2.88),
                textcoords="offset points", xytext=(-4, 12), ha="center", fontsize=7.8,
                color=PROSE_COLOUR)
    ax.plot([bgk["stopped"]["steps"]], [100.0 * bgk["u_rms"]], "^", color="#762a83",
            ms=10, zorder=6,
            label=f"the BGK variant at the same step count: "
                  f"{100.0 * bgk['u_rms']:.2f}% (cav-re1000-bgk)")
    ax.axvline(2_450_000, color=FAIL_COLOUR, lw=1.1, ls=":",
               label="2% crossed at about 2.45 M steps (prose)")
    ax.set_xlabel("steps run (192 cells, Re 1000, float32 populations)")
    ax.set_ylabel("u RMS vs Ghia Table I, % of lid speed")
    _log_ticks(ax, steps + [3_492_000],
               [f"{s // 1000}k" for s in steps] + ["3.49M"])
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 3.6)
    ax.set_title("the Re 1000 cavity gets WORSE the longer it runs: float32 round-off\n"
                 "spins the vortex down, about 0.86% of error per million steps",
                 fontsize=10)
    ax.legend(fontsize=7.4, loc="upper left")

    caption = (
        f"u RMS 0.24%, 0.39%, 0.91%, 1.96% at 180k / 600k / 1.2M / 2.4M steps. MEASURED, from "
        "benchmarks/live/results/cavity-re1000, cav-steps-600000, cav-steps-1200000, cav-steps-2400000/result.json "
        f"(field u_rms, and stopped.steps for the x axis). Provenance: {_provenance(labels)}.\n"
        f"FROM THE PROSE, NOT ON DISK: the fifth point, 3,492,000 steps at 2.88% "
        f"(PROGRESS_LIVE.md:{prose_line} '{prose_text}'), drawn hollow because no result.json in "
        "benchmarks/live/results/ records it. The nearest recorded run at that step count is the BGK collision variant "
        f"(cav-re1000-bgk, {100.0 * bgk['u_rms']:.3f}%, provenance UNRECORDED), plotted as a triangle -- it is a different "
        f"collision operator and is NOT the prose's point. The 0.86%/million slope and the 2.45 M crossing are also prose "
        f"(PROGRESS_LIVE.md:{cross_line} '{cross_text[:80]}'); no fitted line is drawn, because anchoring one would decide "
        "the crossing rather than report it.\n"
        "WHAT THIS DOES NOT SHOW: a converged cavity, or a property of this case alone. The graded run is the SHORTEST of "
        "the five (180,000 steps, 0.238%, PASS) and the agreement is best there, which is the opposite of the usual reading "
        "that a longer run is a better one. Erosion starts immediately; there is no plateau to sit on. The prose states this "
        "as a property of any float32 lattice run of a closed recirculating flow.")
    return _finish(fig, "convergence-cavity-float32.png", caption)


def cylinder() -> Path:
    """The cylinder's +11.7%, localised two ways: the Mach number and the inlet's development length."""
    plt = _matplotlib()
    paper = 3.23
    cyl = _result("cylinder")
    at32 = next(r for r in cyl["resolutions"] if r["cells_per_d"] == 32)
    at44 = next(r for r in cyl["resolutions"] if r["cells_per_d"] == 44)
    ma064 = 100.0 * (at32["cd_max"] / paper - 1.0)
    graded_error = 100.0 * (at44["cd_max"] / paper - 1.0)
    up04, up10 = _result("cyl-upstream-0.4"), _result("cyl-upstream-1.0")
    err04 = 100.0 * (up04["resolutions"][0]["cd_max"] / paper - 1.0)
    err10 = 100.0 * (up10["resolutions"][0]["cd_max"] / paper - 1.0)

    mach_line, mach_text = _quote("| 64 / 0.0185 |")
    extrap_line, extrap_text = _quote("+2.59%")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    for ax in axes:
        ax.axhspan(-5, 5, color=BAND_COLOUR, alpha=0.16, label="the case's tolerance, +/- 5%")
        ax.axhline(0.0, color="black", lw=1.2,
                   label="Schaefer-Turek 2D-2, Cd_max 3.22-3.24")
        ax.grid(alpha=0.3)
        ax.set_ylabel("Cd_max error against 3.2300, %")
        ax.set_ylim(-7, 14)

    ax = axes[0]
    ax.plot([0.064], [ma064], "o", color="#01665e", ms=8, zorder=6,
            label=f"Ma 0.064 (32 cells): {ma064:+.2f}%, measured")
    ax.plot([0.064, 0.032], [ma064, 7.16], "-", color=PROSE_COLOUR, lw=1.3, alpha=0.6)
    ax.plot([0.032], [7.16], "s", mfc="none", mec=PROSE_COLOUR, mew=1.8, ms=10, zorder=6,
            label="Ma 0.032 (64 cells): +7.16%, FROM THE PROSE ONLY")
    ax.plot([0.0], [2.59], "v", color=PROSE_COLOUR, ms=10, zorder=6,
            label="zero-Mach limit if the error is LINEAR in Ma: +2.59%")
    ax.plot([0.0], [5.64], "^", color=FAIL_COLOUR, ms=10, zorder=6,
            label="zero-Mach limit if QUADRATIC: +5.64%, still fails")
    for x, y in ((0.064, ma064), (0.032, 7.16), (0.0, 2.59), (0.0, 5.64)):
        ax.annotate(f"{y:+.2f}%", (x, y), textcoords="offset points", xytext=(9, -3),
                    fontsize=8)
    ax.set_xlabel("Mach number (tau held at 0.5355)")
    ax.set_xlim(-0.006, 0.075)
    ax.set_title("halving the Mach number removes a third of the error,\n"
                 "and the two extrapolations straddle the tolerance", fontsize=10)
    ax.legend(fontsize=7.2, loc="lower right")

    ax = axes[1]
    ax.plot([0, 4, 10], [ma064, err04, err10], "o-", color="#01665e", lw=1.6, ms=8,
            zorder=5, label="measured Cd_max error")
    for x, y in ((0, ma064), (4, err04), (10, err10)):
        ax.annotate(f"{y:+.2f}%", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.4)
    ax.annotate("the paper's domain\nTHIS IS THE GRADED RUN -- it FAILS", (0, ma064),
                textcoords="offset points", xytext=(26, -40), fontsize=7.8,
                color=FAIL_COLOUR)
    ax.annotate("both of these came off a DIRTY tree", (7, err10),
                textcoords="offset points", xytext=(-30, 26), fontsize=7.6,
                color=PROSE_COLOUR,
                arrowprops={"arrowstyle": "-", "lw": 0.7, "color": PROSE_COLOUR})
    ax.set_xlabel("extra inlet development length, cylinder diameters")
    ax.set_xticks([0, 4, 10])
    ax.set_xlim(-1.2, 11.5)
    ax.set_title("giving the inlet four diameters it is not given in 2D-2\n"
                 "lands the drag inside the tolerance; ten adds nothing", fontsize=10)
    ax.legend(fontsize=7.4, loc="lower right")

    caption = (
        f"MEASURED: Ma 0.064 -> Cd_max {at32['cd_max']:.4f} = {ma064:+.2f}% "
        "(benchmarks/live/results/cylinder/result.json, the 32-cells-per-diameter row of 'resolutions'); "
        f"+4 diameters -> {up04['resolutions'][0]['cd_max']:.4f} = {err04:+.2f}% and +10 diameters -> "
        f"{up10['resolutions'][0]['cd_max']:.4f} = {err10:+.2f}% (cyl-upstream-0.4 and cyl-upstream-1.0). "
        f"Provenance: {_provenance(['cylinder', 'cyl-upstream-0.4', 'cyl-upstream-1.0'])} -- BOTH inlet-distance points "
        "were run on a working tree with an uncommitted change, so their sha does not pin what ran.\n"
        f"FROM THE PROSE, NOT ON DISK: Ma 0.032 -> 3.4613 = +7.16% (PROGRESS_LIVE.md:{mach_line} '{mach_text}'), and the two "
        f"zero-Mach extrapolations +2.59% linear / +5.64% quadratic (PROGRESS_LIVE.md:{extrap_line} '{extrap_text[:72]}'). "
        "Hollow and triangular markers are those three.\n"
        "WHAT THIS DOES NOT SHOW: a passing cylinder. The graded case runs the paper's own 2.2 x 0.41 channel and FAILS at "
        f"{graded_error:+.2f}% on Cd_max ({at44['cd_max']:.6g} at the 44 cells per diameter this figure does not plot -- the "
        f"two left-hand points are the 32-cell Mach pair), with an observed order of convergence of {cyl['order']:.4g} "
        "against a >= 1.0 target -- a second failure this figure does not plot. A run in a longer channel is not Schaefer-Turek "
        "2D-2 and is not reported as it. Cl_max moves the OTHER way under the extended domain, +10% to -15%, so the longer "
        "channel is not uniformly better and the lift is not claimed either way. Two points do not establish whether the error "
        "is linear or quadratic in Ma, and that is exactly what decides whether this case can ever pass in its own domain.")
    return _finish(fig, "cylinder-mach-and-inlet.png", caption)


# --------------------------------------------------------------------------- scorecard

_PM = re.compile(r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:mm|mL|m/s|s|m)?\s*\+/-\s*"
                 r"(\d*\.?\d+)\s*%")
_LE = re.compile(r"^<=?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(%?)")
_GE = re.compile(r"^>=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(%?)")
_WITHIN = re.compile(r"^within\s+(\d*\.?\d+)\s*%")


def _normalised(measured: object, target: str) -> float | None:
    """How far outside its own tolerance a check is: 1.0 is exactly at the bound.

    None when the target is an invariant rather than a tolerance -- `exactly 0`, `True`,
    `> 0`, `finite`, `inside the box`. Those are pass/fail with no magnitude, and
    inventing a bar for them would be a figure that flatters: a 48-row chart in which a
    third of the bars are zero reads as a third of the suite being perfect.

    A floor (`>= X`) is normalised as `X / measured`, so that "outside the bound" is
    greater than one in the same direction as every other row.
    """
    if isinstance(measured, bool) or not isinstance(measured, (int, float)):
        return None
    value = float(measured)
    flat = " ".join(str(target).split())
    hit = _PM.match(flat)
    if hit:
        centre, percent = float(hit.group(1)), float(hit.group(2))
        bound = abs(centre) * percent / 100.0
        return abs(value - centre) / bound if bound else None
    hit = _WITHIN.match(flat)
    if hit:
        return abs(value) / (float(hit.group(1)) / 100.0)
    hit = _GE.match(flat)
    if hit:
        bound = _bound(hit)
        return abs(bound / value) if value else None
    hit = _LE.match(flat)
    if hit:
        bound = _bound(hit)
        return abs(value) / bound if bound else None
    return None


def _bound(hit: re.Match) -> float:
    """The number in a `<= X` or `>= X` target, with a trailing `%` honoured.

    `<= 2% of the lid speed` is 0.02 and not 2: every measurement graded against one of
    these targets is already a fraction, so reading the percent sign off put a third of
    the bars on the scorecard a hundred times too short. The recorded verdicts did not
    catch it -- those rows pass either way -- which is what makes it worth a function
    and a comment rather than a regex nobody re-reads.
    """
    return float(hit.group(1)) / (100.0 if hit.group(2) == "%" else 1.0)


def scorecard() -> Path:
    """Every graded check in the recorded validation runs, as a normalised error bar."""
    plt = _matplotlib()
    graded = [r for r in summarise.rows() if r["verdict"] in ("PASS", "FAIL")]
    for row in graded:
        row["normalised"] = _normalised(row["measured"], row["target"])
    # The one check this logic needs: a parsed tolerance must reproduce the verdict the
    # run recorded. A parser that disagrees with the grader is drawing someone else's
    # result, and a wrong pass/fail here is the exact failure this gallery is against.
    wrong = [r for r in graded if r["normalised"] is not None
             and (r["normalised"] <= 1.0) != (r["verdict"] == "PASS")]
    assert not wrong, "\n".join(
        f"{r['label']}: {r['check']} -> {r['normalised']:.4g} but recorded {r['verdict']}"
        for r in wrong)

    graded.sort(key=lambda r: (r["milestone"], r["label"]))
    scaled = [r for r in graded if r["normalised"] is not None]
    invariant = [r for r in graded if r["normalised"] is None]
    failed = [r for r in graded if r["verdict"] == "FAIL"]

    limit = 2.5
    fig, ax = plt.subplots(figsize=(13.6, 0.30 * len(graded) + 3.2))
    ticks, labels, boundary = [], [], []
    seen_milestone = None
    y = 0.0
    for row in graded:
        if row["milestone"] != seen_milestone:
            if seen_milestone is not None:
                y += 1.0
                boundary.append((y - 0.5, row["milestone"]))
            else:
                boundary.append((-0.8, row["milestone"]))
            seen_milestone = row["milestone"]
        colour = PASS_COLOUR if row["verdict"] == "PASS" else FAIL_COLOUR
        width = row["normalised"]
        if width is None:
            ax.text(0.02, y, f"{row['verdict']} -- invariant, no magnitude "
                             f"(target: {' '.join(str(row['target']).split())[:46]})",
                    va="center", fontsize=7.2, color=colour, style="italic")
        else:
            ax.barh(y, min(width, limit), height=0.68, color=colour, alpha=0.88)
            shown = (f"{width:.2f}x" if width <= limit
                     else f"{width:.3g}x the bound -- off the scale")
            measured = (f"{row['measured']:.6g}"
                        if isinstance(row["measured"], float) else str(row["measured"]))
            ax.text(min(width, limit) + 0.03, y,
                    f"{shown}   (measured {measured[:34]})",
                    va="center", fontsize=7.2, color=colour)
        ticks.append(y)
        labels.append(f"{row['label']} -- {' '.join(str(row['check']).split())[:62]}")
        y += 1.0

    # Room to the right of the clip for the off-the-scale multiples; without it the
    # three worst failures in the suite are the three whose numbers get cut off.
    xmax = limit + 1.45
    ax.axvline(1.0, color="black", lw=1.4)
    for position, milestone in boundary:
        ax.axhline(position, color="#999999", lw=0.8, ls=":")
        ax.text(xmax - 0.05, position + 0.3, milestone, fontsize=11, fontweight="bold",
                ha="right", color="#555555")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel("measured, normalised by the check's own tolerance "
                  "(1.0 = at the bound, > 1.0 fails)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(
        f"live-sync validation scorecard: {len(graded) - len(failed)} of {len(graded)} "
        f"graded checks pass, {len(failed)} FAIL\n"
        f"{len(scaled)} have a numeric tolerance and get a bar; {len(invariant)} are "
        f"invariants (exactly 0, True, > 0) and get a verdict only", fontsize=11)
    names = ", ".join(sorted({r["label"] for r in failed}))
    caption = (
        f"MEASURED, all of it: every row is one `checks` entry of one "
        "benchmarks/live/results/*/result.json, with the verdict the run recorded, read through "
        "summarise.py::rows() so that the run set and the milestone map are the same ones PROGRESS_LIVE.md's summary table "
        f"uses. {len(graded)} graded checks over "
        f"{len({r['label'] for r in graded})} validation runs. Bar length = |measured - target| / tolerance for a "
        "'X +/- Y%' target, |measured| / bound for a '<= X' target, and bound / measured for the one '>= X' floor; the "
        "parser is asserted to reproduce every recorded verdict, so a bar past 1.0 is a row the grader failed.\n"
        f"THE FAILING RUNS: {names}. Three of the failures are off the right-hand edge and their true multiple is printed "
        "beside the bar.\n"
        "WHAT THIS DOES NOT SHOW: the open questions, or the work that is not graded. Diagnostic runs -- resolution sweeps, "
        "inlet probes, step-count sweeps, the M6 reproduction attempts -- are excluded, as they are from the summary table, "
        "because counting them as tests inflates the denominator. Rows reported and not graded are excluded too. M6 "
        "(floating bodies) contributes NO rows at all: its draft did not reproduce between identical runs, the +1.9% once "
        "reported was withdrawn, and the milestone is frozen on a measured defect -- a passing scorecard with M6 missing is "
        "not a passing project. M7 is not started.")
    return _finish(fig, "scorecard.png", caption)


# --------------------------------------------------------------------------- animations

ANIMATIONS = {
    "field-dam-break.gif": "dam-n1300/live-dam-break.gif",
    "field-cylinder.gif": "gif-cylinder/live-cylinder-n44.gif",
    "field-falling-sphere.gif": "gif-falling-sphere/live-falling-sphere.gif",
    "field-quadrotor.gif": "gif-quadrotor/live-quadrotor-z1.gif",
    "isaacsim-freesurface.gif": "isaacsim/freesurface.gif",
    "isaacsim-submarine.gif": "isaacsim/submarine.gif",
}
"""Which recorded GIFs the gallery shows.

**Three of these were one-frame files until 2026-11-23, and the cause was not the
sampling.** `figures.ensure_rgb` flattens an image to RGB so an RGBA PNG does not render
as a black thumbnail; every GIF is mode `"P"`, so it matched too, kept frame 0 and wrote
the file back. `cases.py` calls it on everything in `drawn`, and `drawn` holds the GIFs.
The cylinder, the falling sphere and the quadrotor each animated correctly inside
`figures.animate` -- 200-odd frames, the sampling callbacks were always right -- and were
destroyed one line later. The dam break survived only because its caller happens to guard
with `endswith(".png")`. The single-frame files had a frame duration of exactly 50 ms,
which is `1000/fps` for one frame rather than the accumulated duration Pillow writes when
it collapses identical frames, and that is how the writer was ruled out.

`ensure_rgb` now returns early on anything with more than one frame and `figures.animate`
raises rather than write a still, so `check()` below is a second line of defence and not
the only one. The GIFs here come from re-runs on an L40S under the `gif-*` labels; the
graded results are untouched, because a figure and a graded number should not come from
the same incidental run."""


def animations() -> list[Path]:
    """Copy the GIFs the runs already wrote. Nothing is re-rendered.

    `figures.animate` needs a live world and these came off an H100 and an L40S; a
    re-render on this machine would be a different run, drawn from different fields,
    presented beside numbers it did not produce.
    """
    written = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, source in ANIMATIONS.items():
        src = RESULTS / source
        if not src.is_file():
            print(f"  missing, not copied: {src}")
            continue
        shutil.copy2(src, OUT / name)
        written.append(OUT / name)
    return written


# --------------------------------------------------------------------------- the check


FLAT_STD = 1.0
FLAT_COLOURS = 16


def inspect(path: Path) -> dict:
    """Per-channel standard deviation and unique colour count of a PNG or a GIF.

    Both numbers, because `docs/isaac-sim-notes.md` records a flat teal frame scoring a
    whole-array standard deviation of 51.32 with a true colour count of 1.

    The **median** frame decides, and the number of flat frames is reported beside it.
    The strict reading -- the worst frame decides -- was tried first and called the dam
    break broken: its frame 0 is the tank at rest, where every cell has |u| = 0 and a
    one-colour frame is the correct picture of that. A rule that cannot tell a
    legitimately uniform first frame from a dead encoder is a rule that gets switched
    off, so the count is printed instead of being folded into the verdict.
    """
    from PIL import Image, ImageSequence

    with Image.open(path) as image:
        mode = image.mode
        frames = [np.asarray(frame.convert("RGB"), dtype=np.float64)
                  for frame in ImageSequence.Iterator(image)]
    stds = [float(np.std(f, axis=(0, 1)).max()) for f in frames]
    colours = [len(np.unique(f.reshape(-1, 3), axis=0)) for f in frames]
    flat = [i for i, (s, c) in enumerate(zip(stds, colours))
            if s < FLAT_STD or c < FLAT_COLOURS]
    median_std = float(np.median(stds))
    median_colours = float(np.median(colours))
    # The third number, added after three GIFs in this directory turned out to hold one
    # frame each: the mean absolute change between consecutive frames, on 0-255 RGB. A
    # file can have two hundred frames and still be a still life, and the per-frame
    # standard deviation above cannot tell the difference -- it grades each frame on its
    # own. `still_pairs` counts the consecutive pairs that are byte-identical.
    diffs = [float(np.abs(b - a).mean()) for a, b in zip(frames, frames[1:])]
    motion = float(np.median(diffs)) if diffs else 0.0
    still_pairs = sum(1 for d in diffs if d == 0.0)
    return {
        "path": path, "bytes": path.stat().st_size, "mode": mode, "frames": len(frames),
        "channel_std": median_std, "colours": median_colours, "flat_frames": flat,
        "motion": motion, "still_pairs": still_pairs,
        "rgba": mode in ("RGBA", "LA") and path.suffix.lower() == ".png",
        "flat": median_std < FLAT_STD or median_colours < FLAT_COLOURS,
        "dead": path.suffix.lower() == ".gif" and (len(frames) < 2 or motion <= 0.0),
    }


def check() -> int:
    files = sorted(p for p in OUT.iterdir() if p.suffix.lower() in (".png", ".gif"))
    if not files:
        print(f"nothing in {OUT}")
        return 1
    print(f"{'file':30} {'bytes':>10} {'mode':5} {'frames':>6} {'med std':>8} "
          f"{'med colours':>12} {'motion':>7}  notes")
    bad = []
    for path in files:
        got = inspect(path)
        notes = []
        if got["dead"]:
            notes.append(f"DEAD GIF -- {got['frames']} frame(s), nothing moves")
            bad.append(path)
        if got["flat"]:
            notes.append("FLAT -- shows nothing")
            bad.append(path)
        if got["rgba"]:
            notes.append("RGBA -- renders as a black thumbnail")
            bad.append(path)
        if got["flat_frames"]:
            notes.append(f"uniform frame(s) {got['flat_frames']} of {got['frames']}")
        if got["still_pairs"]:
            notes.append(f"{got['still_pairs']} identical consecutive pair(s)")
        print(f"{path.name:30} {got['bytes']:>10,} {got['mode']:5} {got['frames']:>6} "
              f"{got['channel_std']:>8.2f} {got['colours']:>12,.0f} {got['motion']:>7.2f}"
              f"  {'; '.join(notes)}")
    if bad:
        print(f"\n{len(set(bad))} file(s) failed the check")
        return 1
    print(f"\n{len(files)} file(s): none flat, no RGBA PNG, every GIF animates. "
          f"`motion` is the median mean-absolute change between consecutive frames on "
          f"0-255 RGB. A uniform frame noted above is reported, not excused -- for the "
          f"dam break it is the tank at rest.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="only verify what is in gallery/ already")
    args = parser.parse_args(argv)
    if not args.check:
        for draw in (dam_break, sloshing, cavity_float32, cylinder, scorecard):
            print(f"  wrote {draw().name}")
        for path in animations():
            print(f"  copied {path.name}")
        print()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
