#!/usr/bin/env python3
"""How fast a signal oscillates, and whether it has stopped moving yet.

Two questions a transient run raises every time and nothing here could answer.

on the frequency

There was no FFT anywhere in `openreynolds/`. The only frequency ever measured was
a zero-crossing count inside `benchmarks/moving_mesh/grade.py`
(`upward_crossings`), which is offline grading code a study cannot reach: it needs
`Cl` already oscillating about zero, it needs whole cycles, and it reports nothing
at all when the record holds fewer than two crossings. So a study that wanted a
shedding frequency wrote its own -- or, more often, quoted a Strouhal number from a
paper and never measured one. The spectrum here agrees with those crossings on a
clean signal (that is a test), and unlike them it survives a mean offset, a drift
and a record that is not periodic at all.

The spectrum is Welch's: the record cut into overlapping segments, each linearly
detrended and Hann windowed, the periodograms averaged. The detrend matters because
a Cd still settling has a ramp in it and an un-detrended ramp puts all its power in
the lowest bins, where it outbids the tone. The averaging matters because a single
periodogram of white noise has a tallest bin some nine times its own median -- so a
peak alone is not evidence of a tone, and what is reported beside the peak is its
prominence, peak over median, which averaging makes meaningful. A flat spectrum
comes back `aperiodic` with that number attached rather than as a frequency.

What the detrend does not remove is a curve. A settling transient is an exponential
and an exponential is not a line, so a Cd of `1.2 + 3 exp(-t/2)` with a small tone on
it reports its tallest bin at 0.084 Hz -- 1.67 cycles of a 20 s record, which is the
transient and not the flow -- and comes back `too-short`. That is why the frequency,
the cycle count and the prominence are all printed rather than only a verdict, and why
`--from` exists: the settling section says where the transient ends, and the second
call takes the spectrum of what is left.

`numpy.fft` only: scipy is not on the image (`ENVIRONMENT.md`), so there is no
`scipy.signal.welch` to call and this is the thirty lines that replace it.

on the time axis, which is the incident this script was rewritten by

The first real case it met was a cylinder at Re = 100 whose `system/controlDict`
says `application simpleFoam`, `deltaT 1`, `endTime 500`. That is a STEADY run: the
`Time` column counts SIMPLE iterations, not seconds. This script read it as seconds
and reported `shedding: periodic` at 0.0133 Hz, a period of 75, 6.66 cycles in the
record and `St = 0.00133` -- against the 0.164 a cylinder at that Reynolds number
sheds at, and which `benchmarks/moving_mesh/grade.py` holds a transient run to
within 2%. Four numbers, all of them meaningless on an iteration axis, printed to
six figures.

So the axis is now read rather than assumed. A case's `system/controlDict` gives
`application`, and a steady name is recognised by the predicate copied from
`openreynolds/convergence.py:99-104`; `system/fvSchemes` gives `ddtSchemes default`,
and `steadyState` there is the second signal, the one that catches a `foamRun` whose
name says nothing. On an iteration axis the frequency, the period, the Strouhal
number and the cycle count are refused BY NAME -- the spectrum of the iteration
history is still printed, because the shape of it is informative, but no number that
needs a clock is. Handed a bare `.dat` with no case anywhere above it, the script
says that seconds are an assumption rather than quietly making one.

The other line that case wanted: the Nyquist limit. A step of 1 resolves nothing
above 0.5 per unit, while St = 0.164 at L = 0.1 m and U = 1 m/s needs 1.64 per
second. Nothing at the shedding frequency could have been in that record at any
amplitude, and `--strouhal` is how the expectation gets checked against the sampling
instead of against a spectrum that cannot hold it.

on the Strouhal number

St = f L / U, and only when `--length` and `--speed` are both given. A diameter is
not guessed from the mesh and a free-stream speed is not guessed from anything:
`grade.py:_need` refuses rather than assumes, and the same rule holds here. Without
them the frequency is reported in Hz and St is absent, not defaulted. On an
iteration axis it is refused whatever is given, because there is no f to put in it.
It is also refused whenever the shedding verdict above it rejected the peak: a
`too-short` Cd printed `St = 0.0113328` where the same record cropped past its
transient gives 0.332192, and that is one number 29x out and not two readings.

on settling

What existed was `results.py:tail_mean` -- the mean of the last 25% -- reported next
to the last value and deliberately not called converged, because a series still
climbing has a tail mean too and it means nothing. This answers the question that
makes it mean something: does the second half's mean sit inside the first half's
scatter, and from which sample onwards is that true. The window it finds is the
window to average over, and the tail mean of a `still-moving` series is still not a
number to quote.

The confidence interval is on the effective sample size, not the sample count. A
force history is written every time step and consecutive steps are nearly the same
number; treating 20,000 correlated samples as 20,000 independent ones understates
the error bar by the square root of the integral time scale over the step, which on
a shedding case is a factor of tens. The autocorrelation is summed to its first zero
crossing (`numpy`, no scipy) and the count divided by `1 + 2 tau/dt`.

The same record is why the halves test grew two more checks. It rose from +0.0078 at
iteration 1 to +0.018 by 250 and fell to -0.107 by 500, so it is not MONOTONE: the
two halves have nearly the same mean, the test scored 0.58 and called the record
stationary from sample zero -- averaging in a startup transient whose Cd begins at
7.74 and falls to 1.17, and reporting a mean of 0.00178 for a series ending at
-0.107, wrong by two and a half times its own quoted scatter. A two-halves
comparison has a null space and any drift symmetric about the midpoint sits in it.
What is asked now is a first third against a last third as well, and whether the
fluctuation is still GROWING, which is what that record was really doing.

Judgements are `no-variation`/`periodic`/`aperiodic`/`trend-dominated`/`too-short` and
`settled`/`still-moving`/`too-short`, and each is printed next to the number it came
from -- the prominence, the bin index, the period count, the half-mean and
third-mean differences over the scatter -- so a reader can disagree with the
threshold without having to rerun anything. Exit code is 0 whatever it finds.

    python3 spectra.py /work/case
    python3 spectra.py /work/case --column Cl --length 0.01 --speed 1.0
    python3 spectra.py /work/case --column Cl --length 0.1 --speed 1 --strouhal 0.164
    python3 spectra.py /work/case --column Cd --from 6.8
    python3 spectra.py postProcessing/forceCoeffs/0/coefficient.dat --column Cd --png fft.png
    python3 spectra.py history.csv --column y --json spectra.json
    python3 spectra.py /work/case --list
    python3 spectra.py /work/case --column torqueOuter/moment.dat:viscous_z
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results  # noqa: E402

PROMINENCE_PERIODIC = 10.0
"""Peak over median of the averaged spectrum, above which a tone is called real.

A single periodogram of white noise has a tallest bin around nine times its median
over a few hundred bins -- the maximum of that many exponential draws -- so on one
segment this number cannot separate anything. Averaging K segments divides the
scatter by sqrt(K), and at the default four segments white noise measured 2 to 4
here while a sine with 30% noise on it measured in the hundreds. Ten sits in the
empty space between, and the measured prominence is always printed, so a case that
lands near it is visibly near it rather than silently classified.

What it does not separate is red noise: a strongly autocorrelated series with no
tone in it at all has a spectrum rising towards zero frequency, and its lowest bin
beats the median of the rest by a wide margin (an AR(1) at 0.98 measured 2530x
here). A median over the whole spectrum is a global floor and cannot tell that
shape from a peak. What catches it is the two numbers printed beside the verdict --
a frequency at the first or second bin, with the cycle count to match -- and the
figure, which is why the spectrum panel is drawn on a log axis with the floor on
it. A local background would be the fix and it is not written: that is one more
window length to choose and nothing here has yet been misled by it."""

MIN_PERIODS = 2.0
"""Below two cycles of the frequency found, nothing is called periodic or settled.

One cycle fits any sinusoid, a drift and a step equally well, and the lowest
resolvable bin of a record is 1/T -- a "frequency" of one period per record is the
record's own length coming back. Both verdicts become `too-short`, which is a
statement about the record and not about the flow."""

MIN_SAMPLES = 16
"""Fewer samples than this and the halves are too small to have a scatter."""

SETTLED_RATIO = 1.0
"""|mean(second half) - mean(first half)| over the halves' scatter, under which the
record is called settled: the later mean sits inside the record's own spread. For a
clean limit cycle this is near zero (the halves have the same mean and a large
spread); for a linear ramp it is 3.5 at every scale, since the half means differ by
half the range while a half's own std is only a twelfth of it -- which is the point,
a ramp is never stationary over any window. See `drift_ratio` for which scatter.

The same number is applied to the first third against the last third, which is the
check the halves test needed: a ramp scores 3.5 on the halves and 6.9 on the thirds,
so the threshold is if anything stricter there, and the Re = 100 record that scored
0.58 on its halves -- rising to iteration 250 and falling after -- scores 1.32 on
its thirds and is refused."""

TREND_BINS = 3
"""A peak in the lowest this many bins is a trend, not a tone.

The constant above says a strongly autocorrelated series with no tone in it measures
2530x prominence and that only the printed frequency catches it. Nothing caught it:
the Re = 100 record measured 2.552e+10x, was called `periodic`, and its peak sat at
0.0133 on a 0.005 bin grid -- bin 3, the bottom of the spectrum. A settling
transient is an exponential and the per-segment detrend removes a line, so what is
left of it lands here, outbids everything, and is a straight answer to the wrong
question. Three bins is where a four-segment Welch's leakage from DC dies out; the
bin index is printed with the verdict so a peak at bin 4 is visibly at bin 4."""

SCATTER_GROWTH = 2.0
"""The last third's scatter over the first third's, above which a record is still
growing and no window of it is stationary.

Stationarity is a stable mean AND a stable variance, and only the mean was ever
asked about. The Re = 100 Cl is the case that showed why: its halves agree, its
thirds agree from iteration 180 on, and its oscillation grows from 0.011 to 0.072 in
scatter across the record -- a steady solver losing stability, which a mean over any
window of it describes badly. Measured on the RECORD and not on a candidate window,
deliberately: trimming the front of a series that is still growing at the end does
not make it stationary, and the sub-windows measure less growth than the record only
because they are shorter. The one-sidedness is also deliberate: a SHRINKING scatter
is a decaying transient, whose bias the halves and thirds tests already judge.

Two is the threshold because a factor of two in scatter is a factor of four in
variance, far outside sampling noise, while the hardest honest fixture here -- an
AR(1) at 0.98, correlated with no trend in it -- measures 1.2. The record measured
6.3. A startup that grows and then saturates (a shedding case begun from a symmetric
state) is refused by this and its route is the one `--from` already exists for: crop
past the growth and ask again."""

FLAT_RELATIVE = 1e-12
"""An RMS fluctuation under this fraction of the column's own magnitude is nothing.

`acoustics.py:FLAT_RELATIVE`, the same number for the same reason, and the reason it
is now here too: every other test in this file is a ratio within the spectrum, and a
ratio cannot tell a signal from its own round-off. A column of one repeated number is
a real thing to find -- a probe inside a wall, a force component that is identically
zero by symmetry -- and the linear detrend of an exactly constant column does not
come back exactly zero; `polyfit` leaves residue around 1e-16 of the magnitude. The
spectrum of that residue has a tallest bin like any other, and 2978 rows of exactly
1.35 reported a confident `St = 0.167926`, inside the published Re = 100 band of
0.164 to 0.167."""

NEGLIGIBLE_FRACTION = 1e-9
"""Below this much of the record's own largest value, a column is zero.

`report.py:NEGLIGIBLE_FRACTION` and `zero_the_negligible`, the same number against
the same scale: the record's own largest value, so nothing here depends on a unit or
on what sort of quantity the column holds. A column can be pure round-off without
being flat. `pressure_z` of a 2D extruded case decays smoothly from 5.7346e-25 to
5.7343e-25 -- a real variation of 6e-6 of itself, so the test above passes it -- while
the same file's `total_y` runs to 7.3e-13. The z-force of that case is zero by
construction and what the solver wrote is the round-off of the summation; its
spectrum stood 1.232e+07x over its own median and gave `St = 0.837456`.

The time column is left out of the scale. It is not a quantity the other columns are
small against: a residual of 1e-9 beside a Time that runs to 500 would be called
negligible on an axis, which is the one thing this must not do."""


def is_steady_solver(executable: str) -> bool:
    """Whether a solver is a steady one, from its name: `simpleFoam`, `rhoSimpleFoam`,
    `buoyantSimpleFoam`, `SRFSimpleFoam`, `porousSimpleFoam` and their kin all carry
    SIMPLE in the name and iterate towards a steady state. `foamRun` with a
    `steadyState` ddt scheme cannot be told from its name, and is not claimed.

    Copied from `openreynolds/convergence.py:99-104`, which is where it belongs and
    where the docstring above it was written. The copy is here because a toolbox
    script runs as a standalone file from `/work/.toolbox/` with the `openreynolds`
    package neither installed nor on the path -- `import openreynolds.convergence`
    there raises, and `tests/test_toolbox.py` enforces that the only non-stdlib
    imports a toolbox script may make are the image's libraries and its own
    siblings. Six words of predicate duplicated beats a script that cannot run. If
    the list of steady solvers ever changes, it changes in both places; the second
    signal (`ddtSchemes default steadyState`) is what makes that less load-bearing
    than it looks."""
    return "simple" in (executable or "").lower()


def foam_entry(text: str, key: str) -> str:
    """One `key value;` out of an OpenFOAM dictionary, by regex.

    `application`, `deltaT` and `endTime` are one-word entries on their own line and
    that is all this reads. A dictionary parser is not written here: the toolbox has
    no OpenFOAM bindings, and a wrong answer from this is a missing axis note rather
    than a wrong axis -- an unreadable `controlDict` falls through to the same place
    a missing one does, which says that seconds are assumed."""
    found = re.search(rf"^\s*{key}\s+([^;\s]+)\s*;", text, re.MULTILINE)
    return found.group(1) if found else ""


def scheme_default(text: str, block: str) -> str:
    """The `default` entry of one scheme block, e.g. `ddtSchemes { default steadyState; }`."""
    found = re.search(block + r"\s*\{(.*?)\}", text, re.DOTALL)
    return foam_entry(found.group(1), "default") if found else ""


def find_case(source: Path) -> Path | None:
    """The case directory a series belongs to, or None if there is none to read.

    A case is a directory with `system/controlDict` in it. The parents are walked
    because the path handed over is as often the file as the case --
    `<case>/postProcessing/forceCoeffs/0/coefficient.dat` is three levels down -- and
    the axis a `.dat` sits on is the axis of the run that wrote it, whichever path
    was typed."""
    source = Path(source)
    for base in ([source] if source.is_dir() else []) + list(source.parents)[:4]:
        if (base / "system" / "controlDict").is_file():
            return base
    return None


def time_axis(case: Path | None) -> dict[str, Any]:
    """What the first column counts: seconds, iterations, or seconds by assumption.

    This is the check whose absence made the script's first real answer wrong in four
    places at once. A steady run's `Time` column is an iteration counter, and a
    frequency, a period, a Strouhal number and a cycle count are all nonsense on it
    -- not imprecise, nonsense -- so they are refused rather than scaled.

    Two signals, because neither alone covers the cases. The solver name is read from
    `application` and judged by the predicate copied from `convergence.py`, which
    claims the SIMPLE family and nothing else. `ddtSchemes default` is read from
    `system/fvSchemes`, and `steadyState` there says it outright: there is no time
    derivative in the equations being solved, whatever the executable is called."""
    if case is None:
        return {"kind": "seconds", "assumed": True, "unit": "Hz", "per": "s",
                "basis": "no case directory above this file (nothing with "
                         "system/controlDict in it), so nothing here says what the Time "
                         "column counts. Seconds are ASSUMED -- that is an assumption and "
                         "not a reading: on a steady run the column counts iterations and "
                         "every frequency below is meaningless"}
    control = (case / "system" / "controlDict").read_text(encoding="utf-8", errors="replace")
    schemes = ""
    if (case / "system" / "fvSchemes").is_file():
        schemes = (case / "system" / "fvSchemes").read_text(encoding="utf-8", errors="replace")
    application = foam_entry(control, "application")
    ddt = scheme_default(schemes, "ddtSchemes")
    signals = []
    if is_steady_solver(application):
        signals.append(f"system/controlDict says `application {application}`, which carries "
                       "SIMPLE in its name and iterates towards a steady state")
    if ddt.startswith("steadyState"):
        signals.append("system/fvSchemes says `ddtSchemes default steadyState`, so there is "
                       "no time derivative in the equations being solved")
    axis = {"application": application, "ddt": ddt, "case": str(case),
            "deltaT": foam_entry(control, "deltaT"), "endTime": foam_entry(control, "endTime")}
    if signals:
        axis.update({"kind": "iterations", "assumed": False, "unit": "per iteration",
                     "per": "iteration",
                     "basis": " and ".join(signals) + ". The Time column counts ITERATIONS, "
                              f"not seconds (deltaT {axis['deltaT'] or '?'}, endTime "
                              f"{axis['endTime'] or '?'})"})
        return axis
    axis.update({"kind": "seconds", "assumed": False, "unit": "Hz", "per": "s",
                 "basis": (f"system/controlDict says `application {application}`" if application
                           else "system/controlDict names no steady application")
                          + " and nothing says `ddtSchemes default steadyState`, so the Time "
                            "column is read as seconds"})
    return axis


def read_file_series(path: Path) -> dict[str, Any]:
    """One `.dat` or CSV file through `results.parse_dat`.

    `parse_dat` reads OpenFOAM's `#`-commented header and treats commas as column
    separators already, so a CSV's data rows need nothing. Its header row does: a
    bare `Time,Cl` line is not a comment and `parse_dat` drops any line it cannot
    read as numbers, which would leave the columns named `c1`, `c2`. Marking that
    one line as a comment is the whole adapter -- the alternative was a second
    parser beside the one that already handles restarts, vector columns and
    half-written rows.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if not line.lstrip().startswith("#") and not results.data_values(line):
            # `split_header_names` splits a header on whitespace and parentheses, since
            # that is what OpenFOAM writes; the commas go too, or a CSV's whole header
            # arrives as one column name and `--column y` finds nothing called `y`.
            lines[index] = "# " + line.replace(",", " ")
            text = "\n".join(lines)
        break
    data = results.parse_dat(text)
    data["sources"] = [str(path)]
    data["superseded"] = []
    return data


def case_histories(case: Path) -> dict[str, dict[str, Any]]:
    """Every series the case logged, read and stitched, keyed `function/name`.

    Through `results.find_series` and `results.read_history`, which is the part nobody
    should write twice: a restart writes a second time directory holding a file of the
    same name, and an abandoned run leaves rows past the start of the run that replaced
    it. Reading only the newest directory loses half the history and merging all of them
    blindly makes the record end in the run that was thrown away.

    Every series and not the force ones alone. `find_force_files` keeps only the files
    whose stem starts `force` or `coefficient`, and `choose_force_series` then picks one
    of those by rank and then alphabetically with no tie-break reported: on
    `20260912-190302-ae39/grid/conf_L3` that reached `forces/force.dat` out of five
    series, and `--list` printed it as though it were the case (#189).
    """
    grouped = results.find_series(case)
    if not grouped:
        raise results.Unavailable(
            f"no .dat series under {(case / 'postProcessing').as_posix()}")
    return {key: results.read_history(grouped[key]) for key in sorted(grouped)}


def holds_column(history: dict[str, Any], column: str) -> bool:
    """Whether a series has that column, by exact then case-insensitive name.

    Nothing looser, on `grid._column_index`'s rule: a substring match would take `Cd`
    out of `Cd(f)` or `Cdf`, which are different numbers.
    """
    columns = list(history["columns"])
    return column in columns or column.lower() in [name.lower() for name in columns]


def choose_history(histories: dict[str, dict[str, Any]], column: str,
                   where: str) -> tuple[str, dict[str, Any]]:
    """Which series the column asked for lives in, refusing rather than picking.

    `grid.py` learned this in #146 and this script had not: an ambiguous name is
    answered by naming the candidates, and `SERIES:COLUMN` is how the answer gets used.
    Every column of a `forces` function object appears in both `force.dat` and
    `moment.dat`, so a bare column name is ambiguous on every case that logs forces,
    and picking the alphabetical first is picking which number the reader is looking at.
    """
    wanted, name = results.split_series_column(column) if column else (None, "")
    matched = {key: history for key, history in histories.items()
               if wanted is None or results.series_matches(key, wanted)}
    if not matched:
        raise results.Unavailable(
            f"no series under {where} matches {wanted!r}; --column takes SERIES:COLUMN "
            "and the series here are " + ", ".join(sorted(histories)))
    if not name:
        if len(matched) == 1:
            return next(iter(matched.items()))
        raise results.Unavailable(
            f"{len(matched)} series under {where} and no --column to say which: "
            + ", ".join(sorted(matched))
            + ". --list prints their columns; --column takes SERIES:COLUMN")
    holders = [key for key, history in matched.items() if holds_column(history, name)]
    if not holders:
        seen = sorted({c for history in matched.values() for c in history["columns"]})
        raise results.Unavailable(
            f"no column {name!r} under {where}; the columns there are "
            + (", ".join(seen) or "none that parsed"))
    if len(holders) > 1:
        candidates = [f"{key}:{name}" for key in holders]
        raise results.Unavailable(
            f"{name!r} is a column in {len(holders)} series here; name the one you mean "
            f"as SERIES:COLUMN -- --column {candidates[0]} -- since picking for you is "
            "picking which number this is about. The candidates are "
            + ", ".join(candidates))
    return holders[0], matched[holders[0]]


def read_source(source: Path, column: str = "") -> dict[str, Any]:
    """A history from either a case directory or a single series file.

    The column is read here and not only in `pick_column`, because on a case it is what
    says WHICH series is being read: the same name in two of them is a question and not
    a default (#189).
    """
    source = Path(source)
    if source.is_dir():
        where = (source / "postProcessing").as_posix()
        key, history = choose_history(case_histories(source), column, where)
        history["series"] = key
        return history
    if not source.is_file():
        raise results.Unavailable(f"no such case directory or series file: {source}")
    history = read_file_series(source)
    history["series"] = source.name
    return history


def pick_column(history: dict[str, Any], name: str) -> str:
    """Which column to analyse: the one asked for, else the first worth drawing.

    `results.plot_columns` already decides what is worth looking at in a force file
    -- whole-body coefficients before the front/rear splits, totals before the
    pressure part -- so the default here is its first choice rather than column 1,
    which on a `forces.dat` is the x component of the pressure force.
    """
    columns = list(history["columns"])
    # `read_source` has already spent the series half of a `SERIES:COLUMN` on choosing
    # which series this is; what is left here is the column name (#189).
    name = results.split_series_column(name)[1] if name else name
    if name:
        if name not in columns:
            raise results.Unavailable(f"no column {name!r} in {history['series']}; it has {columns}")
        return name
    drawable = results.plot_columns(columns)
    if not drawable:
        raise results.Unavailable(f"{history['series']} has no column besides {columns[:1]}")
    return drawable[0]


def uniform_samples(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """The series on an evenly spaced time axis, resampled only if it is not already.

    An FFT assumes a constant step and a solver with adjustable time stepping does
    not provide one -- a Courant-limited run's step changes by a factor of several
    over a startup. Interpolating onto the mean step is a distortion, but a smaller
    one than feeding ragged samples to `rfft` and reading the bin labels as
    frequencies, and the report says when it happened.
    """
    steps = np.diff(t)
    if steps.size == 0 or not np.all(steps > 0):
        raise results.Unavailable("the time column does not increase; nothing to transform")
    if (steps.max() - steps.min()) <= 1e-3 * steps.mean():
        return t, x, False
    grid = np.linspace(t[0], t[-1], t.size)
    return grid, np.interp(grid, t, x), True


def welch(t: np.ndarray, x: np.ndarray, segments: int = 4) -> dict[str, Any]:
    """Welch power spectral density of an evenly sampled series, `numpy.fft` only.

    `segments` is how many averages are wanted; with 50% overlap that makes each
    segment 2N/(K+1) long. The window is Hann and each segment is linearly
    detrended before it is windowed, in that order: detrending after windowing
    fits the ramp to a taper rather than to the signal.
    """
    dt = float(np.mean(np.diff(t)))
    count = x.size
    nperseg = int(max(8, min(count, round(2 * count / (segments + 1)))))
    window = np.hanning(nperseg)
    # One-sided density: the sum of the squared window and the step set the scale, and
    # the interior bins carry the power of their negative-frequency twins as well.
    #
    # The step multiplies, it does not divide. This read `1.0 / (dt * sum(w**2))` until
    # an absolute level was wanted from it, which made every density a factor of
    # `1/dt**2` -- the sampling rate squared -- too large: a 1 Pa RMS sine sampled at
    # 10 kHz integrated to 1e8 rather than to 1. Nothing this script reports could see
    # it, since a peak's frequency, its prominence over the median and the settling
    # verdict are all ratios, and a constant factor cancels in every one of them.
    scale = dt / float(np.sum(window ** 2))
    step = max(1, nperseg // 2)
    total = np.zeros(nperseg // 2 + 1)
    taken = 0
    for start in range(0, count - nperseg + 1, step):
        segment = detrend(x[start:start + nperseg])
        power = np.abs(np.fft.rfft(segment * window)) ** 2 * scale
        power[1:-1] *= 2.0
        total += power
        taken += 1
    freq = np.fft.rfftfreq(nperseg, dt)
    return {"frequency": freq, "power": total / max(1, taken),
            "nperseg": nperseg, "segments": taken, "dt": dt,
            "resolution": float(freq[1] - freq[0]) if freq.size > 1 else float("nan")}


def detrend(values: np.ndarray) -> np.ndarray:
    """The series with its best-fit straight line removed.

    A Cd that is still falling carries a ramp whose power lands in the lowest bins,
    which is where a slow shedding tone also lands; subtracting the mean alone
    leaves it there. This is why a record that has not settled can still have its
    frequency measured.
    """
    index = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(index, values, 1)
    return values - (slope * index + intercept)


def variation(history: dict[str, Any], name: str, values: np.ndarray) -> dict[str, Any]:
    """How far the column moves, against the two absolute scales that say whether that
    movement is a signal at all.

    This is the one reading in the file that is not a ratio inside the spectrum.
    `prominence`, `bin`, `periods` and the settling ratios all divide one part of the
    record by another, so each of them survives being handed round-off: a flat signal
    does not give a flat spectrum, it gives a round-off spectrum whose own median is
    1e-67, and the peak over that median explodes. Both thresholds are the ones this
    codebase already set against this exact failure -- `acoustics.FLAT_RELATIVE` and
    `report.NEGLIGIBLE_FRACTION` -- and both numbers measured are printed with the
    verdict, so a reader can disagree with either without rerunning anything.
    """
    magnitude = float(np.max(np.abs(values))) if values.size else 0.0
    rms = float(np.sqrt(np.mean(detrend(values) ** 2))) if values.size > 1 else 0.0
    # Every column but the time axis, whether or not it is one of the ones worth
    # plotting: the scale wanted is the largest number the solver wrote in this record.
    scale = magnitude
    for other in list(history.get("columns", []))[1:]:
        series = results.column(history, other)
        if series.size:
            scale = max(scale, float(np.max(np.abs(series))))
    flat = rms <= FLAT_RELATIVE * magnitude
    negligible = scale > 0.0 and magnitude <= NEGLIGIBLE_FRACTION * scale
    basis = ""
    if flat:
        basis = (f"its RMS about its own trend is {rms:.4g}, under {FLAT_RELATIVE:g} of its "
                 f"own magnitude {magnitude:.4g} -- that is the residue `polyfit` leaves on "
                 "a constant column, not a fluctuation")
    elif negligible:
        basis = (f"its largest value {magnitude:.4g} is under {NEGLIGIBLE_FRACTION:g} of the "
                 f"largest value in this record, {scale:.4g}, so the column is zero to within "
                 "the record's own scale and what varies in it is the round-off of a quantity "
                 "that is not there")
    return {"rms": rms, "magnitude": magnitude, "record_scale": scale,
            "flat": bool(flat), "negligible": bool(negligible),
            "varies": not (flat or negligible), "basis": basis}


def dominant(spectrum: dict[str, Any], span: float) -> dict[str, Any]:
    """The tallest bin above DC, refined, with how much it stands out and how many
    cycles of it the record actually holds.

    The refinement is a parabola through the log of the peak and its two
    neighbours, which is the standard correction for a windowed peak sitting
    between bins; without it the answer is quantised to the bin width and the bin
    width of a four-segment Welch over a 20 s record is 0.125 Hz. `prominence` is
    the peak over the median of the whole spectrum: a real tone is orders of
    magnitude above the floor, noise is a small multiple of it.
    """
    freq, power = spectrum["frequency"], spectrum["power"]
    if freq.size < 3:
        raise results.Unavailable("too few samples to form a spectrum")
    body = power[1:]
    peak = int(np.argmax(body)) + 1
    floor = float(np.median(power[1:]))
    estimate = float(freq[peak])
    if 0 < peak < power.size - 1:
        left, middle, right = (float(np.log(max(value, 1e-300)))
                               for value in power[peak - 1:peak + 2])
        denominator = left - 2.0 * middle + right
        if denominator != 0.0:
            offset = 0.5 * (left - right) / denominator
            if abs(offset) <= 0.5:
                estimate = float(freq[peak] + offset * spectrum["resolution"])
    return {"frequency": estimate,
            "bin_frequency": float(freq[peak]),
            # The bin INDEX, not only its frequency: a peak in the lowest few bins is a
            # trend the detrend left behind, and that is a statement about the index.
            "bin": int(peak),
            "bins": int(power.size),
            "nyquist": float(freq[-1]),
            "power": float(power[peak]),
            "floor": floor,
            "prominence": float(power[peak] / floor) if floor > 0 else float("inf"),
            "periods": float(estimate * span) if estimate > 0 else 0.0,
            "resolution": spectrum["resolution"]}


def shedding_verdict(peak: dict[str, Any], unit: str = "Hz",
                     varies: dict[str, Any] | None = None) -> tuple[str, str]:
    """`no-variation` / `periodic` / `trend-dominated` / `aperiodic` / `too-short`, and
    the number it came from.

    `trend-dominated` exists because the prominence alone cannot tell a trend from a
    tone and the docstring on `PROMINENCE_PERIODIC` already said so -- it recorded an
    AR(1) at 0.98 measuring 2530x -- and then the Re = 100 record measured 2.552e+10x
    and was called `periodic`. The tell was in the printed output all along: the peak
    sat at bin 3 of a 101-bin spectrum. So the bin index is now a verdict and not
    only a number beside one.

    The order is variation, then cycle count, then bin, then prominence. `no-variation`
    comes first because it is the only test here with an absolute scale in it, and
    every test after it is a ratio that round-off satisfies as well as a signal does:
    a `pressure_z` of 8e-23 stood 1.232e+07x over its own median and 2978 rows of
    exactly 1.35 gave 7.446e+05x, both at a confident frequency. `varies` is
    `variation()`; passed as None the test cannot be made and is not made, which is
    the bare-spectrum call in the tests and not a path `analyse` takes.

    A record under two cycles long is too short to say anything else about, which is
    the case a settling transient usually lands in, and that reading comes next."""
    if varies is not None and not varies["varies"]:
        return "no-variation", (f"the {'column does not vary over the record' if varies['flat'] else 'column is round-off'}: "
                                f"{varies['basis']}. There is no oscillation here, so the "
                                f"frequency, the period, the prominence and the cycle count "
                                f"above are readings of the round-off and not of a flow")
    if peak["periods"] < MIN_PERIODS:
        return "too-short", (f"the record spans {peak['periods']:.2f} cycles of "
                             f"{peak['frequency']:.4g} {unit}, under the {MIN_PERIODS:g} needed")
    if peak["prominence"] >= PROMINENCE_PERIODIC and peak["bin"] <= TREND_BINS:
        return "trend-dominated", (
            f"the tallest bin is bin {peak['bin']} of {peak['bins']}, inside the lowest "
            f"{TREND_BINS:g}, and stands {peak['prominence']:.4g}x over the spectrum's median: "
            "that is the shape of a trend the linear detrend did not remove -- a settling "
            "transient is an exponential, not a line -- and not the shape of a tone")
    if peak["prominence"] >= PROMINENCE_PERIODIC:
        return "periodic", (f"the peak stands {peak['prominence']:.4g}x over the spectrum's "
                            f"median, above {PROMINENCE_PERIODIC:g}")
    return "aperiodic", (f"the tallest bin stands only {peak['prominence']:.4g}x over the "
                         f"spectrum's median, under {PROMINENCE_PERIODIC:g}: a floor, not a tone")


def integral_time(values: np.ndarray, dt: float) -> dict[str, Any]:
    """The autocorrelation time of a series, summed to the first zero crossing.

    Past the first zero crossing the autocorrelation is noise about zero and
    summing it adds variance, not information -- the standard truncation, and the
    reason this is not a sum over the whole lag range.
    """
    centred = values - values.mean()
    if centred.size < 4 or not np.any(centred):
        return {"tau": 0.0, "lags": 0, "independent": float(centred.size)}
    full = np.correlate(centred, centred, mode="full")
    rho = full[centred.size - 1:]
    rho = rho / rho[0]
    negative = np.nonzero(rho <= 0)[0]
    cut = int(negative[0]) if negative.size else rho.size
    tau = float(dt * np.sum(rho[1:cut]))
    independent = centred.size / (1.0 + 2.0 * max(tau, 0.0) / dt)
    return {"tau": max(tau, 0.0), "lags": max(cut - 1, 0),
            "independent": float(max(independent, 1.0))}


def settle(t: np.ndarray, x: np.ndarray, periods: float | None = None) -> dict[str, Any]:
    """From which sample the series is stationary, and the mean over that window.

    The test on a window is that the second half's mean sits inside the first
    half's scatter, that the last THIRD's mean does too, and that the window's
    scatter is not still growing. The stationary index is the earliest start for
    which all of that is true, found by walking the candidate starts forwards -- the
    point being to report the longest usable window rather than to confirm that the
    last 25% of the record looks calm, which it does on a series that is still
    drifting slowly.

    The two extra checks are the Re = 100 incident. The halves test has a null
    space and any drift symmetric about the midpoint is in it: that record rose
    from +0.0078 at iteration 1 to +0.018 by 250 and fell to -0.107 by 500, its two
    halves therefore agreed, it scored 0.58, and it was reported `settled` from
    sample ZERO -- so the mean quoted, 0.00178, averaged in a startup transient
    whose Cd runs from 7.74 down to 1.17, on a series whose last value is -0.107.
    Wrong by two and a half times its own quoted scatter, which is not a near miss.
    A curved trend defeats the halves the same way. The thirds comparison catches
    both; `SCATTER_GROWTH` catches what that record was actually doing underneath,
    which was oscillating harder and harder.

    The interval on the mean is 1.96 sigma over the root of the EFFECTIVE sample
    size: a force history has one row per time step and consecutive rows are the
    same number to three figures, so the raw count is not a count of independent
    measurements and an interval built on it is wrong by a large factor rather than
    a small one.
    """
    if x.size < MIN_SAMPLES:
        return {"verdict": "too-short", "basis": f"{x.size} samples, under {MIN_SAMPLES}",
                "index": None, "mean": float(results.tail_mean(x)) if x.size else float("nan")}
    growth = scatter_growth(x)
    if growth > SCATTER_GROWTH:
        return {"verdict": "still-moving",
                "basis": f"the fluctuation is still GROWING: the last third of the record "
                         f"scatters {growth:.2f}x as much as the first third, over "
                         f"{SCATTER_GROWTH:g}. No start is offered, because trimming the front "
                         f"of a record that is still growing at the END does not make it "
                         f"stationary -- crop past the growth with --from and ask again",
                "index": None, "mean": float(results.tail_mean(x)),
                "scatter_growth": float(growth)}

    index, ratio, thirds = None, float("nan"), float("nan")
    best, best_thirds = float("inf"), float("inf")
    for candidate in range(0, int(x.size * 0.9), max(1, x.size // 50)):
        window = x[candidate:]
        if window.size < MIN_SAMPLES:
            break
        measure = drift_ratio(window)
        across = thirds_ratio(window)
        best, best_thirds = min(best, measure), min(best_thirds, across)
        if (measure <= SETTLED_RATIO and across <= SETTLED_RATIO
                and window_cycles(t[candidate:], window) >= MIN_PERIODS):
            index, ratio, thirds = candidate, measure, across
            break
    if index is None:
        # Only here does the cycle count get a say, and the order matters. It was
        # asked first, and a Cd of `1.2 + 3 exp(-t/2) + a small tone` -- a settling
        # transient, which is the ordinary shape of a drag history -- came back
        # `too-short` for settling: the per-segment detrend removes a line and an
        # exponential is not a line, so the spectrum's tallest bin was the transient
        # at 0.084 Hz, 1.67 cycles of the record, and that suppressed a settling
        # answer the record plainly contained. A window that passes the halves test
        # has demonstrated a stationary state whatever the spectrum thinks; it is
        # only when no window passes that the two readings cannot be told apart,
        # because under two cycles a drift and a phase of an oscillation look alike.
        if periods is not None and 0.0 < periods < MIN_PERIODS:
            return {"verdict": "too-short",
                    "basis": f"no start passes, and the record spans only {periods:.2f} "
                             f"cycles of the frequency found, under {MIN_PERIODS:g}: over "
                             "less than a cycle a drift and a phase look the same",
                    "index": None, "mean": float(results.tail_mean(x)),
                    "drift_ratio": float(best), "thirds_ratio": float(best_thirds),
                    "scatter_growth": float(growth)}
        return {"verdict": "still-moving",
                "basis": f"no start passes: the closest any tried window came was a "
                         f"second-half mean {best:.2f} half-record standard deviations "
                         f"away and a last-third mean {best_thirds:.2f} away, over "
                         f"{SETTLED_RATIO:g}",
                "index": None, "mean": float(results.tail_mean(x)),
                "drift_ratio": float(best), "thirds_ratio": float(best_thirds),
                "scatter_growth": float(growth)}

    window = x[index:]
    dt = float(np.mean(np.diff(t))) if t.size > 1 else 1.0
    correlation = integral_time(window, dt)
    sigma = float(np.std(window, ddof=1))
    half_width = 1.96 * sigma / np.sqrt(correlation["independent"])
    return {"verdict": "settled",
            "basis": f"from t = {t[index]:g} on, the second half's mean sits "
                     f"{ratio:.2f} half-record standard deviations away and the last "
                     f"third's {thirds:.2f}, both under {SETTLED_RATIO:g}, and the "
                     f"record's scatter grows {growth:.2f}x, under {SCATTER_GROWTH:g}",
            "index": int(index), "from_time": float(t[index]),
            "thirds_ratio": float(thirds), "scatter_growth": float(growth),
            "mean": float(np.mean(window)), "std": sigma,
            "ci95": float(half_width),
            "samples": int(window.size),
            "independent": correlation["independent"],
            "integral_time": correlation["tau"],
            "drift_ratio": float(ratio)}


def window_cycles(t: np.ndarray, values: np.ndarray) -> float:
    """How many cycles of its OWN dominant frequency a candidate window spans.

    A guard on the halves test, and it has to be measured on the window rather than
    taken from the whole record's spectrum, for a reason from each side.

    Why the guard exists: at exactly 1.5 cycles of a sine the two halves are 0.75
    cycles each, 270 degrees apart, and their means are equal -- not approximately,
    identically, since the integral of a sine over three quarters of a period is the
    same from either end. So a record a cycle and a half long passes the drift test
    at sample zero with a ratio of 0.000 and would be reported `settled` with a mean
    that is an artefact of where the write stopped. Two cycles in the window is the
    floor under that.

    Why the window and not the record: a drag history is `1.2 + 3 exp(-t/2) + a
    small tone`, and the per-segment detrend removes a line while an exponential is
    not a line, so the whole record's tallest bin is the transient -- 0.084 Hz, 1.67
    cycles of a 20 s record. Judged on that number the record is too short to settle,
    which is wrong: past the transient the window carries 25 cycles of the real tone,
    and that is what it is asked.
    """
    if values.size < MIN_SAMPLES or t.size < 2:
        return 0.0
    try:
        spectrum = welch(t, values, segments=4)
        return dominant(spectrum, float(t[-1] - t[0]))["periods"]
    except (results.Unavailable, ValueError, np.linalg.LinAlgError):
        return 0.0


def drift_ratio(values: np.ndarray) -> float:
    """|second-half mean - first-half mean| over the halves' scatter.

    The scatter is the SMALLER of the two halves' standard deviations, and that is
    the whole difficulty of this test. Taking the first half's alone -- the obvious
    reading of "does the later mean sit inside the earlier scatter" -- lets a
    startup transient hide itself: a decaying transient three times the size of the
    oscillation it decays to inflates the first half's own standard deviation until
    any later mean sits comfortably inside it, so the record is declared stationary
    from sample zero and the mean reported is the transient's. Taking the second
    half's alone has the mirror fault, which is worse, because a run that goes
    unstable at the end is exactly the one nobody should be quoting a mean from. The
    minimum fails on either, which is the answer wanted in both.
    """
    half = values.size // 2
    first, second = values[:half], values[half:]
    scatters = [float(np.std(part, ddof=1)) for part in (first, second) if part.size > 1]
    scatter = min(scatters) if scatters else 0.0
    gap = abs(float(np.mean(second) - np.mean(first)))
    if scatter == 0.0:
        return 0.0 if gap == 0.0 else float("inf")
    return gap / scatter


def thirds_ratio(values: np.ndarray) -> float:
    """The same comparison between the FIRST third and the LAST third, skipping the
    middle.

    The halves test's null space, in one line. Two halves of a record that rises and
    then falls have nearly the same mean -- the Re = 100 Cl rose to iteration 250 and
    fell to -0.107 by 500 and scored 0.58, passing from sample zero -- and so does
    any drift symmetric about the midpoint, and so does a curve. Dropping the middle
    third puts the two ends against each other where a non-monotone drift is visible:
    the same record scores 1.32 here and is refused.

    It is `drift_ratio` of the two ends concatenated rather than a second copy of the
    arithmetic, so the choice of scatter -- the smaller of the two blocks', which is
    what keeps a startup transient from hiding inside its own spread -- is made in
    one place and documented once."""
    third = values.size // 3
    if third < 2:
        return 0.0
    return drift_ratio(np.concatenate([values[:third], values[-third:]]))


def scatter_growth(values: np.ndarray) -> float:
    """The last third's standard deviation over the first third's.

    Whether the fluctuation is getting bigger, which is the half of stationarity
    nothing here asked about: a steady solver losing stability oscillates harder and
    harder about a mean that stays put, so every test on the MEAN passes while the
    record is plainly not converged. See `SCATTER_GROWTH` for the threshold, the
    measurements behind it and why this is asked of the record rather than of each
    candidate window."""
    third = values.size // 3
    if third < 4:
        return 1.0
    early = float(np.std(values[:third], ddof=1))
    late = float(np.std(values[-third:], ddof=1))
    if early == 0.0:
        return 1.0 if late == 0.0 else float("inf")
    return late / early


def analyse(source: Path, column: str = "", segments: int = 4,
            length: float | None = None, speed: float | None = None,
            start: float | None = None, expect: float | None = None) -> dict[str, Any]:
    """The whole measurement: spectrum, shedding verdict, Strouhal if asked, settling.

    `start` drops everything before a time, and it is here because the first call on a
    drag history often cannot answer the frequency question: a settling transient is
    not a straight line, so the per-segment detrend leaves it in, and its power sits
    in the lowest bins where it outbids a real tone (a `1.2 + 3 exp(-t/2)` Cd reported
    its tallest bin at 0.084 Hz, which is the transient and not the flow). The
    settling section finds where that transient ends; `--from` is how the number it
    found gets used, rather than being printed next to a spectrum it invalidates.
    """
    history = read_source(source, column)
    if not history["rows"]:
        raise results.Unavailable(f"{history['series']} has no data rows")
    name = pick_column(history, column)
    times = results.column(history, history["columns"][0])
    values = results.column(history, name)
    cropped = None
    if start is not None:
        keep = times >= start
        if keep.sum() < MIN_SAMPLES:
            raise results.Unavailable(
                f"--from {start:g} leaves {int(keep.sum())} rows of {times.size}, "
                f"under the {MIN_SAMPLES} a spectrum needs")
        cropped = {"from": float(start), "dropped": int(times.size - keep.sum())}
        times, values = times[keep], values[keep]
    if times.size < MIN_SAMPLES:
        raise results.Unavailable(f"{times.size} rows in {history['series']}, "
                                  f"under the {MIN_SAMPLES} a spectrum needs")

    axis = time_axis(find_case(source))
    grid, resampled, interpolated = uniform_samples(times, values)
    spectrum = welch(grid, resampled, segments=segments)
    span = float(grid[-1] - grid[0])
    peak = dominant(spectrum, span)
    varies = variation(history, name, values)
    verdict, basis = shedding_verdict(peak, unit=axis["unit"], varies=varies)
    nyquist = 0.5 / spectrum["dt"]

    # The expectation is checked against the SAMPLING, not against the spectrum. The
    # Re = 100 record was written every iteration of a deltaT 1 run, so it resolves
    # nothing above 0.5 per unit, while St = 0.164 at L = 0.1 and U = 1 needs 1.64 per
    # second: no amplitude of real shedding could have appeared in it, and that is the
    # most useful thing the script can say about it.
    request = None
    if expect is not None and length is not None and speed is not None and length != 0.0:
        wanted = expect * speed / length
        request = {"strouhal": expect, "frequency": wanted, "nyquist": nyquist,
                   "resolvable": bool(wanted <= nyquist)}

    strouhal = None
    if length is not None and speed is not None:
        if speed == 0.0:
            raise results.Unavailable("--speed 0 gives no Strouhal number")
        # Not on an iteration axis: St = f L / U needs an f per second, and the peak of
        # an iteration history is per iteration. The refusal is in `report`.
        if axis["kind"] != "iterations":
            # And not from a peak this script's own shedding test has just rejected.
            # `too-short` on a real Cd printed `St = 0.0113328` where cropping past the
            # transient gives 0.332192 -- 29x out -- and `--json` carried the two as
            # unlinked sibling keys, so a consumer reading `strouhal.value` got the
            # wrong number with nothing beside it to say so. `value` is now None
            # whenever the verdict rejects the peak: the frequency is still there to be
            # read, under a name that is not a Strouhal number.
            quotient = peak["frequency"] * length / speed
            strouhal = {"value": quotient if verdict == "periodic" else None,
                        "length": length, "speed": speed,
                        "verdict": verdict,
                        "quotient": quotient,
                        "refused": "" if verdict == "periodic" else basis}

    return {
        "source": str(Path(source)),
        "series": history["series"],
        "sources": history.get("sources", []),
        "superseded": history.get("superseded", []),
        "column": name,
        "columns": list(history["columns"]),
        "axis": axis,
        "record": {"samples": int(times.size), "t0": float(times[0]), "t1": float(times[-1]),
                   "span": span, "dt": spectrum["dt"], "resampled": interpolated,
                   "cropped": cropped, "nyquist": nyquist},
        "spectrum": {"nperseg": spectrum["nperseg"], "segments": spectrum["segments"],
                     "resolution": spectrum["resolution"]},
        "peak": peak,
        "variation": varies,
        "shedding": {"verdict": verdict, "basis": basis},
        "strouhal": strouhal,
        "requested": request,
        "settling": settle(grid, resampled, periods=peak["periods"]),
        "meta": history.get("meta", {}),
        "_arrays": {"t": grid, "x": resampled,
                    "frequency": spectrum["frequency"], "power": spectrum["power"]},
    }


def report(found: dict[str, Any]) -> str:
    """The measurement in words, every verdict beside the number behind it, and on an
    iteration axis the four numbers that are refused named one by one.

    Refused by name rather than omitted: a reader who asked for a Strouhal number and
    got a report with no Strouhal line in it concludes the script is broken, or worse,
    reads the frequency and does the division themselves. The line has to say which
    number is missing and why the axis cannot carry it."""
    record, peak, axis = found["record"], found["peak"], found.get("axis") or {}
    unit = axis.get("unit", "Hz")
    iterations = axis.get("kind") == "iterations"
    refused = "not reported -- the Time column counts iterations, not seconds"
    lines = [f"series: {found['series']}  column: {found['column']}",
             f"record: {record['samples']} samples, t = {record['t0']:g} to {record['t1']:g}"
             f" (span {record['span']:g}), mean step {record['dt']:.4g}"]
    if axis:
        lines.append(f"time axis: {axis['kind']}"
                     f"{' (ASSUMED)' if axis.get('assumed') else ''} -- {axis['basis']}")
    if record.get("cropped"):
        lines.append(f"  --from {record['cropped']['from']:g} dropped the first "
                     f"{record['cropped']['dropped']} rows")
    if record["resampled"]:
        lines.append("  the time step varies, so the series was interpolated onto the mean "
                     "step before the transform")
    if len(found.get("sources", [])) > 1:
        lines.append(f"  stitched from {len(found['sources'])} files "
                     f"(restarts): {', '.join(Path(p).parent.name + '/' + Path(p).name for p in found['sources'])}")
    if found.get("superseded"):
        lines.append(f"  {len(found['superseded'])} file(s) had rows past the start of a "
                     "later run; those rows are dropped")

    spectrum = found["spectrum"]
    lines += ["",
              f"spectrum: Welch, {spectrum['segments']} segment(s) of {spectrum['nperseg']} "
              f"samples, Hann window, linear detrend",
              f"  bin width         {spectrum['resolution']:.4g} {unit}",
              f"  tallest bin       bin {peak['bin']} of {peak['bins']} "
              f"({peak['bin_frequency']:.6g} {unit})",
              f"  nyquist limit     {record.get('nyquist', float('nan')):.6g} {unit} "
              f"-- a step of {record['dt']:.4g} resolves nothing above it"]
    if iterations:
        lines += [f"  frequency         {refused}",
                  f"  period            {refused}",
                  f"  prominence        {peak['prominence']:.4g}x the spectrum median",
                  # The shedding verdict below is still derived from an oscillation count,
                  # so the count is said rather than hidden -- as a fact about the
                  # sampling, which it is on any axis, and not as cycles of a flow.
                  f"  cycles in record  {refused} (the series holds "
                  f"{peak['periods']:.2f} oscillations of that bin, which is a fact about "
                  f"the sampling and not a count of shedding cycles)"]
    else:
        lines += [f"  dominant          {peak['frequency']:.6g} {unit} "
                  f"(bin {peak['bin_frequency']:.6g} {unit})",
                  f"  period            {1.0 / peak['frequency']:.6g}" if peak["frequency"] > 0
                  else "  period            -",
                  f"  prominence        {peak['prominence']:.4g}x the spectrum median",
                  f"  cycles in record  {peak['periods']:.2f}"]
    varies = found.get("variation")
    if varies:
        lines.append(f"  variation         rms {varies['rms']:.4g} about its own trend, "
                     f"magnitude {varies['magnitude']:.4g}, largest in the record "
                     f"{varies['record_scale']:.4g}")
    lines += ["",
              f"shedding: {found['shedding']['verdict']} -- {found['shedding']['basis']}"]

    st = found["strouhal"]
    if st is not None and st.get("value") is not None:
        lines += ["",
                  f"strouhal: St = {st['value']:.6g}  "
                  f"(f = {peak['frequency']:.6g}, L = {st['length']:g}, U = {st['speed']:g})"]
    elif st is not None:
        # The refusal names the number it did not print, because a reader who sees no St
        # line concludes the script is broken and does the division themselves.
        hint = ""
        settled = found.get("settling") or {}
        if st["verdict"] == "too-short" and settled.get("verdict") == "settled":
            hint = (f" The settling section below puts the transient's end at "
                    f"t = {settled['from_time']:g}; `--from {settled['from_time']:g}` is the "
                    f"second call that can answer this.")
        lines += ["",
                  f"strouhal: not reported -- the shedding verdict above is "
                  f"`{st['verdict']}`, which rejects this peak, and St = f L / U is a "
                  f"Strouhal number only when f is a shedding frequency. The quotient "
                  f"itself is {st['quotient']:.6g} for L = {st['length']:g} and U = "
                  f"{st['speed']:g}; it is a division, not a measurement.{hint}"]
    elif iterations:
        lines += ["",
                  "strouhal: not computed -- St = f L / U needs a frequency per second and "
                  "the Time column counts iterations, so there is no f to put in it. A "
                  "Strouhal number from a steady run is a number about the solver's "
                  "iteration count, not about the flow."]
    else:
        lines += ["",
                  "strouhal: not computed -- give both --length and --speed. No length scale "
                  "or free-stream speed is assumed here."]

    requested = found.get("requested")
    if requested is not None:
        verdict = "resolvable" if requested["resolvable"] else "UNRESOLVABLE"
        lines.append(f"  expected St = {requested['strouhal']:g} implies "
                     f"{requested['frequency']:.6g} per second, against a nyquist limit of "
                     f"{requested['nyquist']:.6g} per unit of the Time column: {verdict}")
        if not requested["resolvable"]:
            lines.append("  nothing at that frequency is in this record at any amplitude: the "
                         "sampling cannot hold it, so neither a peak nor the absence of one "
                         "says anything about the shedding")

    settling = found["settling"]
    lines += ["", f"settling: {settling['verdict']} -- {settling['basis']}"]
    if settling["verdict"] == "settled":
        lines += [f"  stationary from   index {settling['index']}, t = {settling['from_time']:g}",
                  f"  mean              {settling['mean']:.6g} +/- {settling['ci95']:.4g} (95%)",
                  f"  scatter           {settling['std']:.4g} over {settling['samples']} samples",
                  f"  integral time     {settling['integral_time']:.4g} "
                  f"-> {settling['independent']:.1f} independent samples"]
    else:
        lines.append(f"  mean of the last 25% is {settling['mean']:.6g}; on a series that is "
                     "not stationary that is a number, not a result")
    return "\n".join(lines)


def plot(found: dict[str, Any], target: Path) -> Path:
    """Two panels: the series with the stationary window marked, and the spectrum
    with the peak marked.

    The window and the peak are drawn because the two numbers that are judgements
    -- where the record settled and which bin is the tone -- are the two a reader
    most needs to disagree with, and a picture is how they do that without rerunning
    anything.
    """
    arrays = found["_arrays"]
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7))

    top.plot(arrays["t"], arrays["x"], lw=0.8, color="#1f77b4")
    settling = found["settling"]
    if settling["verdict"] == "settled":
        top.axvspan(settling["from_time"], arrays["t"][-1], color="#2ca02c", alpha=0.12)
        top.axhline(settling["mean"], color="#2ca02c", lw=1.0)
        top.axhspan(settling["mean"] - settling["ci95"], settling["mean"] + settling["ci95"],
                    color="#2ca02c", alpha=0.25)
        top.set_title(f"{found['column']}: stationary from t = {settling['from_time']:g}, "
                      f"mean {settling['mean']:.5g} +/- {settling['ci95']:.3g}")
    else:
        top.set_title(f"{found['column']}: {settling['verdict']} -- {settling['basis']}",
                      fontsize=9)
    top.set_xlabel((found.get("axis") or {}).get("kind", "time"))
    top.set_ylabel(found["column"])
    top.grid(alpha=0.3)

    peak = found["peak"]
    unit = (found.get("axis") or {}).get("unit", "Hz")
    bottom.semilogy(arrays["frequency"][1:], arrays["power"][1:], lw=0.8, color="#d62728")
    bottom.axvline(peak["frequency"], color="#333333", ls="--", lw=1.0)
    bottom.axhline(peak["floor"], color="#888888", ls=":", lw=1.0)
    # The figure carries the axis too, or it becomes the thing someone quotes a
    # frequency off after the report refused to give them one.
    label = (f"bin {peak['bin']} of {peak['bins']} ({peak['bin_frequency']:.5g} {unit}), "
             f"{peak['prominence']:.4g}x median")
    if unit == "Hz":
        label = f"f = {peak['frequency']:.5g} {unit}, {peak['prominence']:.4g}x median"
    if (found["strouhal"] or {}).get("value") is not None:
        label += f", St = {found['strouhal']['value']:.4g}"
    bottom.set_title(f"{found['shedding']['verdict']}: {label}", fontsize=10)
    bottom.set_xlabel(f"frequency ({unit})")
    bottom.set_ylabel("power")
    bottom.grid(alpha=0.3, which="both")

    figure.tight_layout()
    figure.savefig(target, dpi=120)
    plt.close(figure)
    return target


def payload(found: dict[str, Any]) -> dict[str, Any]:
    """The findings without the arrays, for a sidecar another script reads."""
    return {key: value for key, value in found.items() if key != "_arrays"}


def from_cwd(path: Path) -> Path:
    """A relative `--png` or `--json` resolved against the invocation directory, never
    against the case -- the one convention the toolbox keeps (`mesh_look.from_cwd`)."""
    return path if path.is_absolute() else Path.cwd() / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dominant frequency, Strouhal number and settling of a time series.",
        epilog="Exits 0 whatever it finds. St is reported only with --length and --speed.")
    parser.add_argument("source", type=Path,
                        help="a case directory (its force/coefficient history is found) "
                             "or a single .dat/CSV series file")
    parser.add_argument("--column", default="",
                        help="which column to analyse, or SERIES:COLUMN where the name "
                             "is in more than one series (default: the first worth "
                             "drawing, where the case logged only one series)")
    parser.add_argument("--list", action="store_true",
                        help="print the columns available and stop")
    parser.add_argument("--from", dest="start", type=float, default=None,
                        help="ignore rows before this time -- how the stationary time this "
                             "reports gets used, since a startup transient is not a straight "
                             "line and the detrend leaves its power in the lowest bins")
    parser.add_argument("--segments", type=int, default=4,
                        help="Welch segments to average; 1 is a plain periodogram "
                             "(default 4, which is what makes the prominence meaningful)")
    parser.add_argument("--length", type=float, default=None,
                        help="length scale L for St = f L / U; nothing is assumed without it")
    parser.add_argument("--speed", type=float, default=None,
                        help="free-stream speed U for St = f L / U")
    parser.add_argument("--strouhal", type=float, default=None,
                        help="the St you EXPECT (with --length and --speed): its frequency "
                             "U St / L is checked against the Nyquist limit of the sampling, "
                             "which is how a record that could not have held the shedding at "
                             "all says so")
    parser.add_argument("--png", type=Path, default=None,
                        help="write the two-panel figure here (relative to where you run)")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the findings here as JSON (relative to where you run)")
    args = parser.parse_args(argv)

    try:
        if args.list:
            # What is on disk, which is what a list is for. It printed one series of a
            # real case's five before #189, and the one the study was about was not it.
            source = Path(args.source)
            if source.is_dir():
                histories = case_histories(source)
            else:
                one = read_source(source)
                histories = {one["series"]: one}
            print(f"{len(histories)} series under {source.as_posix()}:")
            for key, history in histories.items():
                print(f"  {key}")
                print("    columns: "
                      + (", ".join(history["columns"]) or "none that parsed"))
            print("--column takes a column name, or SERIES:COLUMN where a name is in "
                  "more than one series")
            return 0
        found = analyse(args.source, column=args.column, segments=max(1, args.segments),
                        length=args.length, speed=args.speed, start=args.start,
                        expect=args.strouhal)
    except results.Unavailable as exc:
        # Not being able to measure something is a reading, and a reading is what this
        # offers; `reattach.py` and `layer_report.py` exit 0 on theirs for the same reason.
        print(f"not measured: {exc}")
        return 0

    print(report(found))
    if args.png is not None:
        written = plot(found, from_cwd(args.png))
        print(f"\nfigure: {written}")
    if args.json is not None:
        target = from_cwd(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload(found), indent=2, default=float), encoding="utf-8")
        print(f"json: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
