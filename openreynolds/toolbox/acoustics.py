#!/usr/bin/env python3
"""The noise in a pressure history, as dB against frequency, and a `.wav` of it.

The roadmap asked for aeroacoustics: noise spectra and a file you can listen to.
What a CFD case actually holds is a column of pressure against time, and the
distance from that column to a decibel is three traps, each of which produces a
plausible-looking number rather than an error.

on the units -- the trap that matters most

An incompressible OpenFOAM solver does not solve for pressure. It solves for
`p/rho`, in m2/s2, and writes that into `0/p` with `dimensions [0 2 -2 0 0 0 0]`.
An SPL computed from those numbers as though they were pascals is wrong by
`20*log10(rho)` -- 1.76 dB in air at 1.225 kg/m3, 60.0 dB in water at 1000, and in
every case a clean-looking figure that survives into a report. `preflight.py`
already reads that dimension set for the force-units check (`parse_dimensions`,
`KINEMATIC_PRESSURE`, `STATIC_PRESSURE`) and the same reading decides here which
convention the case used. A kinematic field with no density given is refused, not
multiplied by a guessed 1.225: `grade.py:_need`'s rule, and the rule `heat.py`
follows for every one of q, T_wall and k. The convention and the density are
printed beside the dB figures, because a level is meaningless without them.

on a steady run having no sound

`spectra.py` reported a Strouhal number a hundredfold too small on a steady
`simpleFoam` run, because that run's `Time` column counts iterations and not
seconds. The same column read here would give a spectrum in cycles per iteration
and a `.wav` whose pitch means nothing. A SIMPLE-family solver is detected from
`system/controlDict`'s `application` entry and refused outright -- no spectrum, no
file. The predicate is `openreynolds/convergence.py:is_steady_solver`, copied below
rather than imported: that module sits in the package above this directory and the
toolbox is shipped on its own to `/work/.toolbox/`, where the import would not
resolve.

on where the level comes from

The spectrum is `spectra.welch` -- Welch's method on `numpy.fft`, each segment
linearly detrended then Hann windowed -- because a second spectrum estimator in the
same directory is one too many. Its shape is what is used; the absolute level is
anchored to the record's own mean square (Parseval), so the band levels sum to the
overall SPL by construction and the answer does not depend on which density
constant the estimator carries. The detrend is wanted here as well as there: a mean
static pressure of 101325 Pa is not sound, and neither is a slow drift.

The bands are one-third-octave, exact centres `1000 * 10**(n/10)`, labelled with the
ISO 266 preferred values (the R10 Renard series, 1 / 1.25 / 1.6 / 2 / 2.5 / 3.15 /
4 / 5 / 6.3 / 8 per decade: 31.5 Hz, 63 Hz, 125 Hz, 1250 Hz and the rest come out of
that formula, they are not a hand-typed table). A band narrower than the spectrum's
own resolution holds no bins and is reported as unresolved with the record length
that would resolve it, since a 0 dB band and a band nobody could measure look
identical in a table.

on the `.wav` not being sound

It is a normalised, resampled pressure history written through the standard
library's `wave` module. The scale factor in counts per pascal is printed, so what
was done is recoverable; the file is not calibrated and no playback chain here
knows its own gain. Pitch is only right if the time column is in seconds and the
playback is real time -- `--stretch` slows it to bring a 40 Hz shedding tone into
hearing, and divides every frequency by the same factor, which the report states in
Hz rather than leaving as an exercise.

on what this is not

A near-field pressure history is not a far-field prediction. No Ffowcs
Williams-Hawkings surface integral is performed here, and none is implied: the
pressure at a probe 2 diameters from a cylinder is not the pressure at an observer
50 m away, and the difference is not a constant. Where the case ran OpenFOAM's own
`noise` utility or an acoustic-analogy function object, that result is the one to
quote and this prints where it is. What is on the instance is probed, not assumed
(`cfmesh.py:where`'s habit).

Exit code is 0 whatever it finds.

    python3 acoustics.py /work/case
    python3 acoustics.py /work/case --density 1.225
    python3 acoustics.py /work/case --column probe0 --density 1.225 --wav sound.wav
    python3 acoustics.py probes/0/p --density 1.225 --stretch 8 --wav sound.wav
    python3 acoustics.py history.csv --column p --pa --png spl.png --json acoustics.json
    python3 acoustics.py /work/case --list
    python3 acoustics.py --where
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import wave
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight  # noqa: E402  (sibling script, not a package)
import results  # noqa: E402
import spectra  # noqa: E402

P_REF = 20e-6
"""The reference pressure for airborne sound, 20 micropascal, by definition.

Stated as a constant and printed beside every level because dB is a ratio and a
ratio with an unnamed denominator is not a measurement. Underwater acoustics uses 1
micropascal and the same arithmetic gives numbers 26 dB apart, which is the reason
this is never left implicit."""

RENARD_10 = (1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0)
"""The R10 series ISO 266 rounds its preferred frequencies to, one decade of it.

Scaled by powers of ten this is the whole preferred series: x10 gives 10, 12.5, 16,
20, 25, 31.5, 40, 50, 63, 80 and x1000 gives 1000 ... 8000. So the nominal label of
band `n` is arithmetic on this tuple rather than a transcribed table, and the exact
centre stays `1000 * 10**(n/10)` -- 31.62 Hz labelled 31.5 Hz, which is what the
standard does."""

BAND_RATIO = 2.0 ** (1.0 / 6.0)
"""Half a third of an octave: the factor from a band's centre to either edge."""

AUDIO_RATE = 44100
"""Frames per second in the `.wav`. The CD rate, because every player has it."""

FULL_SCALE = 32767
HEADROOM = 0.9
"""Peak sample as a fraction of full scale. Short of 1.0 so a player's own
resampling, which overshoots on a transient, does not clip what it was handed."""

MIN_SAMPLES = 16
"""Fewer rows than this and there is no spectrum worth forming. `spectra.py`'s
number, kept the same so a record that is too short is too short in both."""

FLAT_RELATIVE = 1e-12
"""An RMS fluctuation under this fraction of the record's own magnitude is nothing.

A column of one repeated number is a real thing to find -- a probe inside a wall, a
field that never moved -- and the linear detrend of an exactly constant column does
not come back exactly zero: `polyfit` leaves rounding residue around 1e-16 of the
magnitude. The level of that residue is a finite, confident, meaningless -120 dB,
which is the kind of figure that gets quoted, so it is refused instead."""


# -- what the case says about itself -------------------------------------------


def is_steady_solver(application: str) -> bool:
    """Whether a solver iterates towards a steady state, from its name.

    Copied from `openreynolds/convergence.py:is_steady_solver` (and `foamRun` with a
    `steadyState` ddt scheme is not claimed there either): that module is in the
    package above this directory, and the toolbox is copied to `/work/.toolbox/`
    on its own, where `import openreynolds` resolves to nothing. A copy with its
    origin named is cheaper than a fragile import.
    """
    return "simple" in (application or "").lower()


def steady_reading(case: Path) -> dict[str, Any]:
    """`application` out of `system/controlDict`, and what it implies for a spectrum.

    A steady run's `Time` is an iteration counter. Treating it as seconds gives a
    frequency in cycles per iteration that prints as Hz, a Strouhal number out by
    whatever the time step would have been, and a `.wav` at an arbitrary pitch --
    which is exactly the defect `spectra.py` carried.
    """
    control = preflight.read_text(Path(case) / "system" / "controlDict")
    application = preflight.parse_control(control).get("application", "") if control else ""
    steady = is_steady_solver(application)
    return {
        "application": application,
        "steady": steady,
        "why": (
            f"{application} is a SIMPLE-family solver, so its Time column counts "
            "iterations and not seconds: there is no physical time axis in this "
            "history and therefore no frequency, no spectrum and no sound"
        ) if steady else "",
    }


def pressure_convention(case: Path | None, density: float | None,
                        already_pa: bool) -> dict[str, Any]:
    """Which pressure convention the numbers are in, and the density that converts it.

    Read from `0/p`'s `dimensions` entry through `preflight.parse_dimensions`, the
    same reading the force-units check makes. Four answers and only one of them is a
    guess-free pascal: static (`[1 -1 -2 ...]`) is already Pa; kinematic
    (`[0 2 -2 ...]`) needs a density; an unreadable or absent field needs to be told
    which it is; and anything else is reported as the exponents it found.

    The refusal is the point. An SPL from a kinematic field with rho left out is
    wrong by `20*log10(rho)` and looks entirely normal, so no default density is
    applied -- not 1.225, not the `rhoInf` of some other function object unless that
    is what the case itself declares, and then it is named as the source.
    """
    found: dict[str, Any] = {"dimensions": None, "field": "", "density": None,
                             "density_source": "", "kind": "", "basis": ""}
    text = ""
    if case is not None:
        reader = preflight.Case(Path(case))
        text = reader.field_texts.get("p", "")
        found["field"] = str(Path(case) / "0" / "p") if text else ""
        found["dimensions"] = preflight.parse_dimensions(text)

    dimensions = found["dimensions"]
    if already_pa:
        found["kind"] = "static"
        found["density_source"] = "--pa"
    elif dimensions == preflight.STATIC_PRESSURE:
        found["kind"] = "static"
        found["density_source"] = "0/p dimensions [1 -1 -2 0 0 0 0]"
    elif dimensions == preflight.KINEMATIC_PRESSURE:
        found["kind"] = "kinematic"
    elif dimensions is None:
        found["kind"] = "unstated"
    else:
        found["kind"] = "other"

    if found["kind"] == "static":
        found["basis"] = "static p in Pa as written, no density applied"
        return found

    if density is None and case is not None:
        declared = [entry for entry in preflight.find_force_objects(
            preflight.read_text(Path(case) / "system" / "controlDict")) if entry["rhoInf"]]
        if declared:
            found["density"] = float(declared[0]["rhoInf"])
            found["density_source"] = (
                f"rhoInf {found['density']:g} declared by the "
                f"{declared[0]['name']} function object")
    if density is not None:
        found["density"] = float(density)
        found["density_source"] = "--density"

    if found["density"] is None:
        shape = ("unreadable or absent" if found["kind"] == "unstated" else
                 "kinematic (m2/s2)" if found["kind"] == "kinematic" else
                 "dimensions " + " ".join(f"{value:g}" for value in (dimensions or ())))
        raise results.Unavailable(
            f"the pressure is {shape} and no density was given, so no level can be "
            f"stated: an SPL from p/rho read as Pa is wrong by 20*log10(rho), which is "
            f"1.76 dB in air at 1.225 kg/m3 and 60.0 dB in water -- pass --density "
            f"<kg/m3> to convert it, or --pa if the column is already in pascals"
        )
    if found["density"] <= 0:
        raise results.Unavailable(f"--density {found['density']:g} is not a density")
    found["basis"] = (f"kinematic p x rho = {found['density']:g} kg/m3 "
                      f"({found['density_source']})")
    return found


# -- the history ---------------------------------------------------------------


PROBE_POSITION = re.compile(r"^#\s*Probe\s+(\d+)\s*\(([^)]*)\)")
"""`# Probe 3 (0.1 0 0.02)`, the header a `probes` function object writes.

`results.parse_dat` keeps `# key : value` lines and column names and drops these,
so the columns of a probe file come out `c1`, `c2`. The coordinates are the whole
reason a near-field level means anything, so they are read back and the columns
renamed `probe0`, `probe1`."""

PRESSURE_SUFFIXES = ("", ".dat", ".csv", ".xy")


def looks_like_pressure(name: str) -> bool:
    """Whether a `postProcessing` file is likely to be a pressure history.

    Deliberately wide -- `p` from a `probes`, `pressure.dat`, an
    `areaAverage(p).dat` from a `surfaceFieldValue` -- and the series actually chosen
    is ranked and printed with the alternatives, so a wrong pick is visible rather
    than silent. `p_rgh` matches too and is ranked last: it is the pressure with the
    hydrostatic part removed, which is a legitimate thing to look at and not the
    first thing to assume somebody meant.
    """
    lowered = name.lower()
    return (lowered in ("p", "p.dat", "p.csv", "p.xy")
            or lowered.startswith(("p_", "p.", "pressure"))
            or "pressure" in lowered or "(p)" in lowered)


def find_pressure_files(case: Path | str) -> dict[str, list[Path]]:
    """Pressure histories under `postProcessing`, grouped into series.

    The grouping is `results.find_force_files`' -- a restart writes a second time
    directory holding a file of the same name and the two are one history -- with the
    force-file name filter swapped for a pressure one.
    """
    base = Path(case) / "postProcessing"
    if not base.is_dir():
        return {}
    grouped: dict[str, list[Path]] = {}
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in PRESSURE_SUFFIXES:
            continue
        if not looks_like_pressure(path.name):
            continue
        parts = path.relative_to(base).parts
        if len(parts) < 2:
            continue
        function = "/".join(parts[:-2]) if len(parts) >= 3 else parts[0]
        grouped.setdefault(f"{function}/{results.series_name(path)[0]}", []).append(path)
    for paths in grouped.values():
        paths.sort(key=results.written_order)
    return grouped


def choose_pressure_series(grouped: dict[str, list[Path]]) -> str:
    """Which pressure series to read when a case logged several.

    `p` itself before a derived or renamed one, and `p_rgh` last. The ones not
    chosen are printed, which is `choose_force_series`' arrangement: the ranking is
    a default and the report shows what it ranked below.
    """
    if not grouped:
        raise results.Unavailable(
            "no pressure history under postProcessing/: a `probes` function object on "
            "p, or a surface sample of it, is what this reads")

    def rank(key: str) -> tuple[int, str]:
        name = key.rsplit("/", 1)[-1].lower()
        if name in ("p", "p.dat", "p.csv", "p.xy"):
            return (0, key)
        if "(p)" in name:
            return (1, key)
        if name.startswith("p_rgh"):
            return (3, key)
        return (2, key)

    return sorted(grouped, key=rank)[0]


def name_probe_columns(history: dict[str, Any], text: str) -> dict[int, tuple[float, ...]]:
    """Columns renamed `probeN` where the header says where probe N was, in place.

    Only when the count matches: a vector field writes three columns per probe and
    renaming them one-to-one would label `Uy` as a probe that does not exist.
    """
    positions: dict[int, tuple[float, ...]] = {}
    for line in text.splitlines():
        match = PROBE_POSITION.match(line.strip())
        if match:
            values = [preflight.as_float(part) for part in match.group(2).split()]
            if all(value is not None for value in values):
                positions[int(match.group(1))] = tuple(float(v) for v in values)
    columns = list(history["columns"])
    if positions and len(columns) - 1 == len(positions):
        history["columns"] = [columns[0]] + [f"probe{index}" for index in range(len(positions))]
    return positions


def read_pressure(source: Path, column: str = "") -> dict[str, Any]:
    """A pressure history from a case directory or from one file, and its column.

    A case goes through the grouping above and `results.read_history`, which is the
    part worth not writing twice: it stitches restarts and drops the rows of a run
    that a later one superseded. A single file goes through
    `spectra.read_file_series`, which marks a bare CSV header line as a comment so
    `parse_dat` keeps the column names.
    """
    source = Path(source)
    if source.is_dir():
        grouped = find_pressure_files(source)
        key = choose_pressure_series(grouped)
        history = results.read_history(grouped[key])
        history["series"] = key
        history["alternatives"] = [other for other in sorted(grouped) if other != key]
        text = grouped[key][-1].read_text(encoding="utf-8", errors="replace")
    elif source.is_file():
        history = spectra.read_file_series(source)
        history["series"] = source.name
        history["alternatives"] = []
        text = source.read_text(encoding="utf-8", errors="replace")
    else:
        raise results.Unavailable(f"no such case directory or series file: {source}")

    if not history["rows"]:
        raise results.Unavailable(f"{history['series']} has no data rows")
    history["probes"] = name_probe_columns(history, text)
    history["column"] = pick_column(history, column)
    return history


def pick_column(history: dict[str, Any], name: str) -> str:
    """Which column carries the pressure: the one asked for, else the first after Time.

    Not `spectra.pick_column`, which prefers whole-body force coefficients over
    their front/rear splits -- a sensible default for a `forceCoeffs.dat` and the
    wrong question of a probe file, where the columns are positions and there is no
    ranking to make.
    """
    columns = list(history["columns"])
    if name:
        if name not in columns:
            raise results.Unavailable(
                f"no column {name!r} in {history['series']}; it has {columns}")
        return name
    if len(columns) < 2:
        raise results.Unavailable(
            f"{history['series']} has only {columns}, so there is no series to read")
    return columns[1]


# -- the spectrum --------------------------------------------------------------


def band_index_range(low: float, high: float) -> range:
    """The one-third-octave band numbers whose centres lie between two frequencies."""
    first = int(math.ceil(10.0 * math.log10(max(low, 1e-12) / 1000.0)))
    last = int(math.floor(10.0 * math.log10(max(high, 1e-12) / 1000.0)))
    return range(first, last + 1)


def band(index: int) -> dict[str, Any]:
    """One-third-octave band `index`: exact centre, ISO 266 nominal label, and edges.

    `1000 * 10**(n/10)` is the base-ten system the standard defines, and the nominal
    label is `RENARD_10` scaled by the decade -- so band 15 is 3162 Hz exactly and
    3150 Hz on the label, which is the number a report should carry.
    """
    exact = 1000.0 * 10.0 ** (index / 10.0)
    nominal = RENARD_10[index % 10] * 10.0 ** (3 + math.floor(index / 10.0))
    return {"index": index, "centre": exact, "nominal": nominal,
            "lower": exact / BAND_RATIO, "upper": exact * BAND_RATIO}


def level(mean_square: float) -> float:
    """dB re `P_REF` from a mean square pressure in Pa2. Silence is -inf, not 0 dB."""
    return 10.0 * math.log10(mean_square / P_REF ** 2) if mean_square > 0 else -math.inf


def spl_spectrum(t: np.ndarray, pressure: np.ndarray, segments: int = 4) -> dict[str, Any]:
    """The narrowband spectrum as levels, anchored to the record's own mean square.

    `spectra.welch` does the estimation -- segments, linear detrend, Hann window,
    averaged periodograms, `numpy.fft` only. What is taken from it is the shape; the
    density is then scaled so that its integral over the bins above DC equals the
    mean square of the detrended record. Two reasons. The overall level and the sum
    of the bands then agree by construction rather than to within a window's leakage,
    so a band table that does not add up is a real disagreement. And the level stops
    depending on the estimator's normalisation constant, which is a thing that can be
    corrected underneath this without the dB figures here moving.

    The detrend is why the static pressure does not appear: 101325 Pa is not sound,
    and a slow drift is not either. The level is of the fluctuation, which is what
    SPL is defined on.
    """
    spectrum = spectra.welch(t, pressure, segments=segments)
    frequency, power = spectrum["frequency"], spectrum["power"]
    if frequency.size < 3:
        raise results.Unavailable("too few samples to form a spectrum")
    fluctuation = spectra.detrend(pressure)
    mean_square = float(np.mean(fluctuation ** 2))
    if math.sqrt(mean_square) <= FLAT_RELATIVE * float(np.max(np.abs(pressure)) + 0.0):
        raise results.Unavailable(
            "the pressure does not vary over the record (its RMS about its own trend "
            "is rounding noise), so there is no sound in it")
    width = float(frequency[1] - frequency[0])
    density = np.array(power, dtype=float)
    density[0] = 0.0
    carried = float(np.sum(density) * width)
    if carried <= 0:
        raise results.Unavailable("the spectrum came out empty above DC")
    density *= mean_square / carried
    return {
        "frequency": frequency,
        "density": density,
        "bin_level": np.array([level(float(value) * width) for value in density]),
        "resolution": width,
        "nyquist": float(frequency[-1]),
        "mean_square": mean_square,
        "overall": level(mean_square),
        "nperseg": spectrum["nperseg"],
        "segments": spectrum["segments"],
        "dt": spectrum["dt"],
    }


def third_octave(spectrum: dict[str, Any], segments: int,
                 lowest: float | None = None) -> dict[str, Any]:
    """One-third-octave band levels, and which bands the record could not resolve.

    A band holds the bins between its edges. Its width is `0.2316 * centre`, so a
    band narrower than the spectrum's own resolution cannot be formed -- and a band
    reported as 0 dB because nothing landed in it, or as a number because one bin
    happened to land inside it, is indistinguishable from a measured one. Those are
    listed separately with the record length that would resolve them: the resolution
    is `1/(nperseg*dt)`, the segment is `2T/(segments+1)`, so the span needed is
    `(segments+1)/(2*band_width)`.

    The bands that are reported therefore sum to a little under the overall SPL when
    the low end is unresolved, and the report prints both numbers rather than
    reconciling them quietly.
    """
    frequency, density = spectrum["frequency"], spectrum["density"]
    width = spectrum["resolution"]
    bands: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    def span_for(band_width: float) -> float:
        return (segments + 1) / (2.0 * band_width)

    for index in band_index_range(max(width, 1e-12) / 4.0, spectrum["nyquist"]):
        entry = band(index)
        inside = (frequency >= entry["lower"]) & (frequency < entry["upper"]) & (frequency > 0)
        count = int(np.count_nonzero(inside))
        band_width = entry["upper"] - entry["lower"]
        if count == 0 or band_width < width:
            # A band narrower than one bin is not saved by a bin happening to land
            # inside it. At 25 Hz resolution the 100 Hz band (89 to 112 Hz) catches
            # the 100 Hz bin and the 125 Hz band catches the 125 Hz bin, and the two
            # "band levels" are then two labels on one resolution element -- a
            # partition the record cannot support, printed to two decimals.
            entry["needs_span"] = span_for(band_width)
            unresolved.append(entry)
            continue
        entry["bins"] = count
        entry["mean_square"] = float(np.sum(density[inside]) * width)
        entry["level"] = level(entry["mean_square"])
        bands.append(entry)
    asked = None
    if lowest is not None:
        if lowest <= 0:
            raise results.Unavailable(f"--min-band {lowest:g} is not a frequency")
        # The nearest band to the frequency asked for, so `--min-band 31.5` and
        # `--min-band 31.62` are the same request: the nominal label and the exact
        # centre are the same band and nobody should have to know which to type.
        wanted = band(int(round(10.0 * math.log10(lowest / 1000.0))))
        asked = {"nominal": wanted["nominal"], "centre": wanted["centre"],
                 "resolved": any(entry["index"] == wanted["index"] for entry in bands),
                 "needs_span": span_for(wanted["upper"] - wanted["lower"])}
    total = float(sum(entry["mean_square"] for entry in bands))
    return {"bands": bands, "unresolved": unresolved, "asked": asked,
            "mean_square": total, "level": level(total)}


# -- the sound -----------------------------------------------------------------


def write_wav(t: np.ndarray, pressure: np.ndarray, target: Path,
              rate: int = AUDIO_RATE, stretch: float = 1.0) -> dict[str, Any]:
    """The history resampled to an audio rate, normalised, and written with `wave`.

    `wave` is the standard library and writes a 16-bit mono PCM file any player
    reads, so no dependency is added for this (`ENVIRONMENT.md`: the network is
    sealed, and there would be nowhere to get one from).

    What is written is the detrended pressure -- the same fluctuation the spectrum is
    of, because a mean of 101325 Pa would use the whole dynamic range on a constant.
    The peak is scaled to `HEADROOM` of full scale and the factor is returned in
    counts per pascal, which is the only way back from the file to the physics. It is
    not calibrated sound and the report says so.

    `stretch` multiplies the duration: at 1.0 the record plays in real time and the
    pitch is the flow's own, at 8 it plays eight times slower and every frequency in
    it is divided by eight, which is how a 40 Hz shedding tone becomes something a
    speaker can reproduce. Resampling is linear interpolation; where the record's own
    rate exceeds the playback rate that aliases, and the report says when.
    """
    fluctuation = spectra.detrend(pressure)
    peak = float(np.max(np.abs(fluctuation)))
    if peak <= 0:
        raise results.Unavailable("the pressure does not vary, so there is nothing to hear")
    physical = float(t[-1] - t[0])
    audio_seconds = physical * stretch
    frames = int(round(audio_seconds * rate))
    if frames < 2:
        raise results.Unavailable(
            f"{physical:g} s at stretch {stretch:g} is {frames} audio frames at "
            f"{rate} Hz; --stretch is what makes a short record audible")
    grid = np.linspace(t[0], t[-1], frames)
    resampled = np.interp(grid, t, fluctuation)
    scale = HEADROOM * FULL_SCALE / peak
    samples = np.clip(np.rint(resampled * scale), -FULL_SCALE - 1, FULL_SCALE).astype("<i2")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    record_rate = (t.size - 1) / physical if physical > 0 else float("nan")
    return {
        "path": str(target),
        "rate": rate,
        "frames": frames,
        "seconds": frames / float(rate),
        "physical_seconds": physical,
        "stretch": float(stretch),
        "scale": scale,
        "peak_pa": peak,
        "record_rate": record_rate,
        "playback_rate": record_rate / stretch if stretch else float("nan"),
        "aliased": bool(record_rate / max(stretch, 1e-12) > rate),
        "calibrated": False,
    }


# -- what the case already has -------------------------------------------------


ACOUSTIC_TYPES = ("noise", "surfaceNoise", "pointNoise", "FfowcsWilliamsHawkings",
                  "FWH", "Curle", "acousticAnalogy")
HISTORY_TYPES = ("probes", "surfaceFieldValue", "pressure", "fieldValueDelta")


def function_objects(case: Path) -> list[dict[str, str]]:
    """Every function object in `system/`, as name and type.

    Three layouts, the same three `preflight.find_force_objects` had to learn: inside
    `controlDict`'s `functions` block, one level down inside an `#includeFunc`
    wrapper, and a whole file that is itself the function object.
    """
    found: list[dict[str, str]] = []
    system = Path(case) / "system"
    if not system.is_dir():
        return found
    for path in sorted(system.iterdir()):
        if not path.is_file():
            continue
        text = preflight.read_text(path)
        functions = preflight.block_body(text, "functions")
        body = functions or preflight.strip_comments(text)
        for name, inner in preflight.block_entries(body):
            values = preflight.entry_values(inner)
            if values.get("type"):
                found.append({"name": name.strip('"'), "type": values["type"],
                              "file": path.name})
            for sub_name, sub_inner in preflight.block_entries(inner):
                sub = preflight.entry_values(sub_inner)
                if sub.get("type"):
                    found.append({"name": sub_name.strip('"'), "type": sub["type"],
                                  "file": path.name})
        if not functions:
            top = preflight.entry_values(body)
            if top.get("type"):
                found.append({"name": path.name, "type": top["type"], "file": path.name})
    return found


def where(case: Path | None = None) -> dict[str, Any]:
    """What acoustic tooling is on this instance and in this case, measured.

    Probed rather than asserted, which is `cfmesh.py:where`'s habit and for its
    reason: a release note is not a filesystem. ESI v2512 is documented as shipping
    a `noise` utility, so `shutil.which` is asked; a case's own dictionaries and
    outputs are looked for on disk. A missing tool is an answer, not an error.
    """
    found: dict[str, Any] = {
        "utilities": {name: shutil.which(name) for name in ("noise",)},
        "dictionaries": [], "objects": [], "outputs": [],
    }
    if case is None:
        return found
    case = Path(case)
    for relative in ("system/noiseDict", "noiseDict", "system/noise"):
        if (case / relative).is_file():
            found["dictionaries"].append(str(case / relative))
    for entry in function_objects(case):
        lowered = entry["type"].lower()
        if any(kind.lower() == lowered for kind in ACOUSTIC_TYPES):
            entry["role"] = "acoustic"
            found["objects"].append(entry)
        elif any(kind.lower() == lowered for kind in HISTORY_TYPES):
            entry["role"] = "pressure history"
            found["objects"].append(entry)
    base = case / "postProcessing"
    if base.is_dir():
        for path in sorted(base.iterdir()):
            name = path.name.lower()
            if "noise" in name or "fw" in name or "acoustic" in name or "curle" in name:
                found["outputs"].append(str(path))
    return found


# -- putting it together -------------------------------------------------------


def analyse(source: Path, column: str = "", density: float | None = None,
            already_pa: bool = False, segments: int = 4, start: float | None = None,
            lowest_band: float | None = None) -> dict[str, Any]:
    """The whole reading: convention, spectrum, bands, and what the case already has.

    The order is deliberate. The steady check and the unit check come before any
    arithmetic, because both of them are refusals and a refusal that arrives after a
    table of decibels has been printed is a refusal nobody reads.
    """
    source = Path(source)
    case = source if source.is_dir() else None
    steady = steady_reading(case) if case is not None else {
        "application": "", "steady": False,
        "why": "no case directory was given, so system/controlDict was not read: "
               "whether the time column is seconds is the caller's to know"}
    if steady["steady"]:
        raise results.Unavailable(steady["why"])
    convention = pressure_convention(case, density, already_pa)

    history = read_pressure(source, column)
    name = history["column"]
    times = results.column(history, history["columns"][0])
    values = results.column(history, name)
    cropped = None
    if start is not None:
        keep = times >= start
        if int(keep.sum()) < MIN_SAMPLES:
            raise results.Unavailable(
                f"--from {start:g} leaves {int(keep.sum())} rows of {times.size}, under "
                f"the {MIN_SAMPLES} a spectrum needs")
        cropped = {"from": float(start), "dropped": int(times.size - keep.sum())}
        times, values = times[keep], values[keep]
    if times.size < MIN_SAMPLES:
        raise results.Unavailable(
            f"{times.size} rows in {history['series']}, under the {MIN_SAMPLES} a "
            "spectrum needs")

    grid, resampled, interpolated = spectra.uniform_samples(times, values)
    factor = convention["density"] if convention["kind"] != "static" else 1.0
    pascals = resampled * float(factor)
    spectrum = spl_spectrum(grid, pascals, segments=segments)
    bands = third_octave(spectrum, segments, lowest=lowest_band)
    peak_bin = int(np.argmax(spectrum["density"]))
    loudest = max(bands["bands"], key=lambda entry: entry["level"], default=None)
    stated = dict(convention)
    stated["dimensions"] = list(convention["dimensions"]) if convention["dimensions"] else None

    return {
        "source": str(source),
        "series": history["series"],
        "column": name,
        "columns": list(history["columns"]),
        "alternatives": history.get("alternatives", []),
        "probes": {str(index): list(position)
                   for index, position in history.get("probes", {}).items()},
        "sources": history.get("sources", []),
        "superseded": history.get("superseded", []),
        "solver": steady,
        "convention": stated,
        "reference_pressure": P_REF,
        "record": {"samples": int(times.size), "t0": float(times[0]), "t1": float(times[-1]),
                   "span": float(grid[-1] - grid[0]), "dt": spectrum["dt"],
                   "resampled": interpolated, "cropped": cropped,
                   "rate": 1.0 / spectrum["dt"] if spectrum["dt"] else float("nan")},
        "spectrum": {"resolution": spectrum["resolution"], "nyquist": spectrum["nyquist"],
                     "nperseg": spectrum["nperseg"], "segments": spectrum["segments"],
                     "overall_spl": spectrum["overall"],
                     "mean_square": spectrum["mean_square"],
                     "rms": math.sqrt(spectrum["mean_square"]),
                     "peak": {"frequency": float(spectrum["frequency"][peak_bin]),
                              "level": float(spectrum["bin_level"][peak_bin])}},
        "third_octave": {"level": bands["level"], "asked": bands["asked"],
                         "loudest": loudest,
                         "bands": [{key: entry[key] for key in
                                    ("index", "nominal", "centre", "lower", "upper",
                                     "bins", "level")} for entry in bands["bands"]],
                         "unresolved": [{"nominal": entry["nominal"],
                                         "needs_span": entry["needs_span"]}
                                        for entry in bands["unresolved"]]},
        "far_field": where(case),
        "wav": None,
        "_arrays": {"t": grid, "pressure": pascals,
                    "frequency": spectrum["frequency"],
                    "bin_level": spectrum["bin_level"],
                    "bands": bands["bands"]},
    }


LIMITS = (
    "this is a near-field pressure history, not a propagated far-field prediction: no "
    "Ffowcs Williams-Hawkings surface integral is performed here and no distance law is "
    "applied, so the level is the level at the sampling point",
    "the spectrum holds only what the write interval resolved -- nothing above the "
    "record's own Nyquist frequency is in it, however loud it was",
    "a .wav written here is not calibrated sound: the scale factor is stated and the "
    "playback chain's gain is unknown",
)


def report(found: dict[str, Any]) -> str:
    """The reading in words, every dB figure carrying the convention behind it."""
    record, spectrum = found["record"], found["spectrum"]
    convention, bands = found["convention"], found["third_octave"]
    basis = convention["basis"]
    reference = f"re {found['reference_pressure']:g} Pa"
    lines = [f"series: {found['series']}  column: {found['column']}",
             f"pressure: {basis}"]
    if convention["field"]:
        lines.append(f"  convention read from {convention['field']}"
                     + (f" -> {' '.join(f'{v:g}' for v in convention['dimensions'])}"
                        if convention["dimensions"] else ""))
    index = found["column"][5:] if found["column"].startswith("probe") else ""
    if index in found["probes"]:
        lines.append("  sampled at ("
                     + " ".join(f"{value:g}" for value in found["probes"][index]) + ")")
    if found["solver"]["application"]:
        lines.append(f"  solver: {found['solver']['application']}")
    lines.append(
        f"record: {record['samples']} samples, t = {record['t0']:g} to {record['t1']:g} "
        f"(span {record['span']:g}), step {record['dt']:.4g} -> {record['rate']:.4g} Hz "
        f"sampling, Nyquist {spectrum['nyquist']:.4g} Hz")
    if record["cropped"]:
        lines.append(f"  --from {record['cropped']['from']:g} dropped the first "
                     f"{record['cropped']['dropped']} rows")
    if record["resampled"]:
        lines.append("  the time step varies, so the series was interpolated onto the "
                     "mean step before the transform")
    if found["superseded"]:
        for note in found["superseded"]:
            lines.append(f"  {note}")
    if found["alternatives"]:
        lines.append("  other pressure series in this case: "
                     + ", ".join(found["alternatives"]))

    lines.append("")
    lines.append(f"overall SPL: {spectrum['overall_spl']:.2f} dB {reference} "
                 f"[{basis}]")
    lines.append(f"  from an RMS fluctuation of {spectrum['rms']:.6g} Pa about the "
                 "record's own trend, which is what a mean of 101325 Pa is not")
    lines.append(f"peak bin: {spectrum['peak']['frequency']:.4g} Hz at "
                 f"{spectrum['peak']['level']:.2f} dB {reference} [{basis}], bin width "
                 f"{spectrum['resolution']:.4g} Hz")
    lines.append(f"one-third-octave bands, summed: {bands['level']:.2f} dB {reference} "
                 f"[{basis}]")
    shortfall = spectrum["overall_spl"] - bands["level"]
    if shortfall > 0.1:
        lines.append(f"  {shortfall:.2f} dB under the overall level, which is the power "
                     "in bands too narrow for this record to resolve -- listed below "
                     "rather than folded in")

    lines.append("")
    lines.append(f"one-third-octave levels in dB {reference}, {basis}")
    lines.append("  band (Hz)    exact        bins   level")
    for entry in bands["bands"]:
        lines.append(f"  {entry['nominal']:>9g}  {entry['centre']:>10.4g}  "
                     f"{entry['bins']:>6d}  {entry['level']:>7.2f}")
    if bands["loudest"]:
        lines.append(f"  loudest band: {bands['loudest']['nominal']:g} Hz at "
                     f"{bands['loudest']['level']:.2f} dB {reference} [{basis}]")
    if bands["unresolved"]:
        # The highest unresolved band is the one nearly within reach, so its span is
        # the number worth printing: the lowest one would need a far longer record.
        nearest = bands["unresolved"][-1]
        lines.append(f"  {len(bands['unresolved'])} band(s) at and below "
                     f"{nearest['nominal']:g} Hz hold no bin of this spectrum and are "
                     f"not reported; the {nearest['nominal']:g} Hz band needs about "
                     f"{nearest['needs_span']:.4g} s of record at these settings")
    if bands["asked"] and not bands["asked"]["resolved"]:
        asked = bands["asked"]
        lines.append(f"  the lowest band asked for, {asked['nominal']:g} Hz, is not "
                     f"resolved by a {record['span']:g} s record: it needs about "
                     f"{asked['needs_span']:.4g} s at these settings")

    wav = found.get("wav")
    if wav:
        lines.append("")
        lines.append(f"sound: {wav['path']}")
        lines.append(f"  16-bit mono PCM at {wav['rate']} Hz, {wav['frames']} frames = "
                     f"{wav['seconds']:.4g} audio seconds from a "
                     f"{wav['physical_seconds']:.4g} s record (stretch "
                     f"{wav['stretch']:g}x)")
        lines.append(f"  scale factor {wav['scale']:.6g} counts/Pa on a peak of "
                     f"{wav['peak_pa']:.6g} Pa -- applied, stated, and NOT a calibration: "
                     "this is not calibrated sound")
        lines.append(f"  pitch: the record's {wav['record_rate']:.4g} Hz sampling plays "
                     f"at {wav['playback_rate']:.4g} Hz, so every frequency is divided by "
                     f"{wav['stretch']:g}; the pitch is only right if the time column was "
                     "in seconds")
        if wav["aliased"]:
            lines.append("  the record's playback rate is above the audio rate, so the "
                         "interpolation aliases what is above the audio Nyquist")

    far = found["far_field"]
    lines.append("")
    lines.append("far field and what the case already ran")
    for name, path in far["utilities"].items():
        lines.append(f"  {name}: {path or 'not on PATH here'}")
    for path in far["dictionaries"]:
        lines.append(f"  dictionary: {path}")
    for entry in far["objects"]:
        lines.append(f"  function object: {entry['name']} ({entry['type']}, "
                     f"{entry['role']}) in system/{entry['file']}")
    if far["outputs"]:
        for path in far["outputs"]:
            lines.append(f"  output already written: {path} -- that is the result to "
                         "quote; nothing here recomputes it")
    elif not far["objects"]:
        lines.append("  no acoustic analogy or noise utility output in this case")

    lines.append("")
    lines.append("what this does not tell you")
    for limit in LIMITS:
        lines.append(f"  - {limit}")
    return "\n".join(lines)


def plot(found: dict[str, Any], target: Path) -> Path:
    """The narrowband spectrum and the one-third-octave bars, in one figure.

    Log frequency on both: a decade of flow noise and a decade of nothing look the
    same on a linear axis, and the bands are logarithmic by construction.
    """
    arrays = found["_arrays"]
    frequency, bin_level = arrays["frequency"], arrays["bin_level"]
    bands = arrays["bands"]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 8))
    keep = frequency > 0
    top.semilogx(frequency[keep], bin_level[keep], linewidth=0.9)
    peak = found["spectrum"]["peak"]
    top.plot([peak["frequency"]], [peak["level"]], "o", color="crimson")
    top.annotate(f"{peak['frequency']:.4g} Hz, {peak['level']:.1f} dB",
                 (peak["frequency"], peak["level"]),
                 textcoords="offset points", xytext=(8, 6), color="crimson")
    top.set_xlabel("frequency (Hz)")
    top.set_ylabel(f"SPL per {found['spectrum']['resolution']:.4g} Hz bin (dB)")
    top.set_title(f"{found['series']} / {found['column']}  --  overall "
                  f"{found['spectrum']['overall_spl']:.2f} dB re "
                  f"{found['reference_pressure']:g} Pa\n{found['convention']['basis']}")
    top.grid(True, which="both", alpha=0.3)
    if bands:
        centres = [entry["centre"] for entry in bands]
        levels = [entry["level"] if np.isfinite(entry["level"]) else 0.0 for entry in bands]
        widths = [entry["upper"] - entry["lower"] for entry in bands]
        bottom.bar(centres, levels, width=widths, align="center",
                   color="steelblue", edgecolor="white", linewidth=0.5)
        bottom.set_xscale("log")
        bottom.set_xticks(centres)
        bottom.set_xticklabels([f"{entry['nominal']:g}" for entry in bands],
                               rotation=90, fontsize=7)
        bottom.set_xlabel("one-third-octave band, ISO 266 nominal centre (Hz)")
    bottom.set_ylabel(f"band SPL (dB re {found['reference_pressure']:g} Pa)")
    bottom.set_title("one-third-octave bands -- " + found["convention"]["basis"])
    bottom.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=130)
    plt.close(figure)
    return target


def payload(found: dict[str, Any]) -> dict[str, Any]:
    """The findings without the arrays, for a sidecar another script reads."""
    return {key: value for key, value in found.items() if key != "_arrays"}


def from_cwd(path: Path) -> Path:
    """A relative `--png`, `--wav` or `--json` resolved against the invocation
    directory, never against the case -- the toolbox's one convention
    (`mesh_look.from_cwd`)."""
    return path if path.is_absolute() else Path.cwd() / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Noise spectra and a .wav from a pressure history: SPL against "
                    "frequency, one-third-octave bands, overall SPL.",
        epilog="Exits 0 whatever it finds. A kinematic pressure with no --density is "
               "refused rather than guessed, and a steady run has no time axis and so "
               "no sound.")
    parser.add_argument("source", type=Path, nargs="?",
                        help="a case directory (its pressure history is found) or a "
                             "single probes/.dat/CSV file")
    parser.add_argument("--column", default="",
                        help="which column holds the pressure (default: the first "
                             "after the time column)")
    parser.add_argument("--list", action="store_true",
                        help="print the pressure series and columns available, and stop")
    parser.add_argument("--where", action="store_true",
                        help="print what acoustic tooling is on this instance and in "
                             "this case, and stop")
    parser.add_argument("--density", type=float, default=None,
                        help="density in kg/m3, for a case whose p is kinematic "
                             "(p/rho in m2/s2); nothing is assumed without it")
    parser.add_argument("--pa", action="store_true",
                        help="the column is already static pressure in pascals")
    parser.add_argument("--from", dest="start", type=float, default=None,
                        help="ignore rows before this time, for a startup transient")
    parser.add_argument("--segments", type=int, default=4,
                        help="Welch segments to average (default 4); fewer segments "
                             "buys frequency resolution with variance")
    parser.add_argument("--min-band", dest="min_band", type=float, default=None,
                        help="the lowest one-third-octave band wanted, as its nominal "
                             "centre in Hz; a record too short for it says so")
    parser.add_argument("--wav", type=Path, default=None,
                        help="write the predicted sound here (relative to where you run)")
    parser.add_argument("--rate", type=int, default=AUDIO_RATE,
                        help=f"audio frames per second in the .wav (default {AUDIO_RATE})")
    parser.add_argument("--stretch", type=float, default=1.0,
                        help="audio seconds per physical second (default 1.0, real "
                             "time); 8 plays it eight times slower and divides every "
                             "frequency by eight")
    parser.add_argument("--png", type=Path, default=None,
                        help="write the spectrum and band figure here")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the findings here as JSON")
    args = parser.parse_args(argv)

    if args.where and args.source is None:
        print(json.dumps(where(None), indent=2))
        return 0
    if args.source is None:
        parser.error("a case directory or series file is needed (or --where on its own)")

    try:
        if args.where:
            print(json.dumps(where(args.source if args.source.is_dir() else None), indent=2))
            return 0
        if args.list:
            if args.source.is_dir():
                grouped = find_pressure_files(args.source)
                for key in sorted(grouped):
                    print(f"series: {key}  ({len(grouped[key])} file(s))")
                if not grouped:
                    print("no pressure history under postProcessing/")
                    return 0
            history = read_pressure(args.source)
            print(f"reading: {history['series']}")
            print(f"columns: {', '.join(history['columns'])}")
            for index, position in sorted(history.get("probes", {}).items()):
                print(f"  probe{index} at (" + " ".join(f"{v:g}" for v in position) + ")")
            return 0
        found = analyse(args.source, column=args.column, density=args.density,
                        already_pa=args.pa, segments=max(1, args.segments),
                        start=args.start, lowest_band=args.min_band)
        if args.wav is not None:
            found["wav"] = write_wav(found["_arrays"]["t"], found["_arrays"]["pressure"],
                                     from_cwd(args.wav), rate=args.rate,
                                     stretch=args.stretch)
    except results.Unavailable as exc:
        # A refusal is a reading, and a reading is what this offers; `spectra.py` and
        # `layer_report.py` exit 0 on theirs for the same reason. Nothing is written:
        # a .wav of a steady run's iteration counter would be the wrong answer in a
        # file somebody can play.
        print(f"not measured: {exc}")
        return 0

    print(report(found))
    if args.png is not None:
        print(f"\nfigure: {plot(found, from_cwd(args.png))}")
    if args.json is not None:
        target = from_cwd(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload(found), indent=2, default=float),
                          encoding="utf-8")
        print(f"json: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
