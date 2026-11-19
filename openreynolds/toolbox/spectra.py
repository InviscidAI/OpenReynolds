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

on the Strouhal number

St = f L / U, and only when `--length` and `--speed` are both given. A diameter is
not guessed from the mesh and a free-stream speed is not guessed from anything:
`grade.py:_need` refuses rather than assumes, and the same rule holds here. Without
them the frequency is reported in Hz and St is absent, not defaulted.

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

Judgements are `periodic`/`aperiodic`/`too-short` and `settled`/`still-moving`/
`too-short`, and each is printed next to the number it came from -- the prominence,
the period count, the half-mean difference over the scatter -- so a reader can
disagree with the threshold without having to rerun anything. Exit code is 0
whatever it finds.

    python3 spectra.py /work/case
    python3 spectra.py /work/case --column Cl --length 0.01 --speed 1.0
    python3 spectra.py /work/case --column Cd --from 6.8
    python3 spectra.py postProcessing/forceCoeffs/0/coefficient.dat --column Cd --png fft.png
    python3 spectra.py history.csv --column y --json spectra.json
    python3 spectra.py /work/case --list
"""

from __future__ import annotations

import argparse
import json
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
a ramp is never stationary over any window. See `drift_ratio` for which scatter."""


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


def read_source(source: Path) -> dict[str, Any]:
    """A history from either a case directory or a single series file.

    A case goes through `results.find_force_files` / `choose_force_series` /
    `read_history`, which is the part nobody should write twice: a restart writes a
    second time directory holding a file of the same name, and an abandoned run
    leaves rows past the start of the run that replaced it. Reading only the newest
    directory loses half the history and merging all of them blindly makes the
    record end in the run that was thrown away.
    """
    source = Path(source)
    if source.is_dir():
        grouped = results.find_force_files(source)
        key = results.choose_force_series(grouped)
        history = results.read_history(grouped[key])
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
    scale = 1.0 / (dt * float(np.sum(window ** 2)))
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
            "power": float(power[peak]),
            "floor": floor,
            "prominence": float(power[peak] / floor) if floor > 0 else float("inf"),
            "periods": float(estimate * span) if estimate > 0 else 0.0,
            "resolution": spectrum["resolution"]}


def shedding_verdict(peak: dict[str, Any]) -> tuple[str, str]:
    """`periodic` / `aperiodic` / `too-short`, and the number it came from."""
    if peak["periods"] < MIN_PERIODS:
        return "too-short", (f"the record spans {peak['periods']:.2f} cycles of "
                             f"{peak['frequency']:.4g} Hz, under the {MIN_PERIODS:g} needed")
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
    half's scatter. The stationary index is the earliest start for which that is
    true, found by walking the candidate starts forwards -- the point being to
    report the longest usable window rather than to confirm that the last 25% of
    the record looks calm, which it does on a series that is still drifting slowly.

    The interval on the mean is 1.96 sigma over the root of the EFFECTIVE sample
    size: a force history has one row per time step and consecutive rows are the
    same number to three figures, so the raw count is not a count of independent
    measurements and an interval built on it is wrong by a large factor rather than
    a small one.
    """
    if x.size < MIN_SAMPLES:
        return {"verdict": "too-short", "basis": f"{x.size} samples, under {MIN_SAMPLES}",
                "index": None, "mean": float(results.tail_mean(x)) if x.size else float("nan")}
    index, ratio, best = None, float("nan"), float("inf")
    for candidate in range(0, int(x.size * 0.9), max(1, x.size // 50)):
        window = x[candidate:]
        if window.size < MIN_SAMPLES:
            break
        measure = drift_ratio(window)
        best = min(best, measure)
        if measure <= SETTLED_RATIO and window_cycles(t[candidate:], window) >= MIN_PERIODS:
            index, ratio = candidate, measure
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
                    "drift_ratio": float(best)}
        return {"verdict": "still-moving",
                "basis": f"no start passes: the closest any tried window came was a "
                         f"second-half mean {best:.2f} half-record standard deviations "
                         f"away, over {SETTLED_RATIO:g}",
                "index": None, "mean": float(results.tail_mean(x)),
                "drift_ratio": float(best)}

    window = x[index:]
    dt = float(np.mean(np.diff(t))) if t.size > 1 else 1.0
    correlation = integral_time(window, dt)
    sigma = float(np.std(window, ddof=1))
    half_width = 1.96 * sigma / np.sqrt(correlation["independent"])
    return {"verdict": "settled",
            "basis": f"from t = {t[index]:g} on, the second half's mean sits "
                     f"{ratio:.2f} half-record standard deviations away, "
                     f"under {SETTLED_RATIO:g}",
            "index": int(index), "from_time": float(t[index]),
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


def analyse(source: Path, column: str = "", segments: int = 4,
            length: float | None = None, speed: float | None = None,
            start: float | None = None) -> dict[str, Any]:
    """The whole measurement: spectrum, shedding verdict, Strouhal if asked, settling.

    `start` drops everything before a time, and it is here because the first call on a
    drag history often cannot answer the frequency question: a settling transient is
    not a straight line, so the per-segment detrend leaves it in, and its power sits
    in the lowest bins where it outbids a real tone (a `1.2 + 3 exp(-t/2)` Cd reported
    its tallest bin at 0.084 Hz, which is the transient and not the flow). The
    settling section finds where that transient ends; `--from` is how the number it
    found gets used, rather than being printed next to a spectrum it invalidates.
    """
    history = read_source(source)
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

    grid, resampled, interpolated = uniform_samples(times, values)
    spectrum = welch(grid, resampled, segments=segments)
    span = float(grid[-1] - grid[0])
    peak = dominant(spectrum, span)
    verdict, basis = shedding_verdict(peak)

    strouhal = None
    if length is not None and speed is not None:
        if speed == 0.0:
            raise results.Unavailable("--speed 0 gives no Strouhal number")
        strouhal = {"value": peak["frequency"] * length / speed,
                    "length": length, "speed": speed}

    return {
        "source": str(Path(source)),
        "series": history["series"],
        "sources": history.get("sources", []),
        "superseded": history.get("superseded", []),
        "column": name,
        "columns": list(history["columns"]),
        "record": {"samples": int(times.size), "t0": float(times[0]), "t1": float(times[-1]),
                   "span": span, "dt": spectrum["dt"], "resampled": interpolated,
                   "cropped": cropped},
        "spectrum": {"nperseg": spectrum["nperseg"], "segments": spectrum["segments"],
                     "resolution": spectrum["resolution"]},
        "peak": peak,
        "shedding": {"verdict": verdict, "basis": basis},
        "strouhal": strouhal,
        "settling": settle(grid, resampled, periods=peak["periods"]),
        "meta": history.get("meta", {}),
        "_arrays": {"t": grid, "x": resampled,
                    "frequency": spectrum["frequency"], "power": spectrum["power"]},
    }


def report(found: dict[str, Any]) -> str:
    """The measurement in words, every verdict beside the number behind it."""
    record, peak = found["record"], found["peak"]
    lines = [f"series: {found['series']}  column: {found['column']}",
             f"record: {record['samples']} samples, t = {record['t0']:g} to {record['t1']:g}"
             f" (span {record['span']:g}), mean step {record['dt']:.4g}"]
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
              f"  bin width         {spectrum['resolution']:.4g} Hz",
              f"  dominant          {peak['frequency']:.6g} Hz "
              f"(bin {peak['bin_frequency']:.6g} Hz)",
              f"  period            {1.0 / peak['frequency']:.6g}" if peak["frequency"] > 0 else
              "  period            -",
              f"  prominence        {peak['prominence']:.4g}x the spectrum median",
              f"  cycles in record  {peak['periods']:.2f}",
              "",
              f"shedding: {found['shedding']['verdict']} -- {found['shedding']['basis']}"]

    if found["strouhal"] is not None:
        st = found["strouhal"]
        lines += ["",
                  f"strouhal: St = {st['value']:.6g}  "
                  f"(f = {peak['frequency']:.6g}, L = {st['length']:g}, U = {st['speed']:g})"]
    else:
        lines += ["",
                  "strouhal: not computed -- give both --length and --speed. No length scale "
                  "or free-stream speed is assumed here."]

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
    top.set_xlabel("time")
    top.set_ylabel(found["column"])
    top.grid(alpha=0.3)

    peak = found["peak"]
    bottom.semilogy(arrays["frequency"][1:], arrays["power"][1:], lw=0.8, color="#d62728")
    bottom.axvline(peak["frequency"], color="#333333", ls="--", lw=1.0)
    bottom.axhline(peak["floor"], color="#888888", ls=":", lw=1.0)
    label = f"f = {peak['frequency']:.5g} Hz, {peak['prominence']:.4g}x median"
    if found["strouhal"] is not None:
        label += f", St = {found['strouhal']['value']:.4g}"
    bottom.set_title(f"{found['shedding']['verdict']}: {label}", fontsize=10)
    bottom.set_xlabel("frequency")
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
                        help="which column to analyse (default: the first worth drawing)")
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
    parser.add_argument("--png", type=Path, default=None,
                        help="write the two-panel figure here (relative to where you run)")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the findings here as JSON (relative to where you run)")
    args = parser.parse_args(argv)

    try:
        if args.list:
            history = read_source(args.source)
            print(f"series: {history['series']}")
            print(f"columns: {', '.join(history['columns'])}")
            return 0
        found = analyse(args.source, column=args.column, segments=max(1, args.segments),
                        length=args.length, speed=args.speed, start=args.start)
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
