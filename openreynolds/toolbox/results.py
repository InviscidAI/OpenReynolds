#!/usr/bin/env python3
"""A finished case turned into the standard set of pictures and plots, in one call.

The pictures a study needs at the end are nearly always the same five or six, and
they get rewritten from scratch every time anyway -- a `view_fields.py` here, a
`plots.py` there, each one a fresh chance to slice on the wrong normal or to forget
that the forces were written into two directories after a restart. So the set is
written down once as a table and named: a preset is a list of outputs, and running
one produces all of them.

what a preset is

`PRESETS` maps a preset name to a list of `Output` rows, and each row says only what
it is (`pressure`, `velocity`, `vorticity`, ...), which producer draws it, and the
options that producer needs. Adding a preset is a new key in that dict; adding an
output to one is a new row. Nothing else in the file changes, and `--list` prints
the table, so what a preset will produce is answerable before it runs.

what happens when an output cannot be made

Half a results directory is worth more than an exception. Every output is attempted
independently: one that cannot be produced -- the case never wrote `p`, no forces
were logged, the solver log has no residual lines yet -- is recorded as skipped with
the reason it was skipped, and the rest still run. The reasons end up in the summary
alongside the files, so what is missing is as visible as what is there.

what is registered

Every file written is registered with `study_state` under its kind, so `study_state
latest vorticity` finds it afterwards, and the `render` phase is marked. The run
finishes by writing `results.md` -- what was produced, what was skipped and why, and
the headline numbers (final residuals, the force coefficients) -- registered as
`report`.

on the last value of a coefficient

For a shedding case the last row of `forceCoeffs.dat` is whichever phase of the cycle
the final write happened to land on, which is why the mean over the tail of the record
is reported next to it rather than instead of it. Neither number is called converged
here; that reading is yours.

    python3 results.py --list
    python3 results.py /work/case --preset external-flow-2d
    python3 results.py /work/case --preset transient-wake --time 12.5 --out /work/case/results
    python3 results.py /work/case --preset mesh-validation --normal y
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import log_digest  # noqa: E402
import study_state  # noqa: E402


class Unavailable(Exception):
    """Raised by a producer when the case does not contain what it needs.

    Separate from a plain exception on purpose: this one is an expected answer
    ("no forces were logged") and becomes a skipped row, while anything else is a
    fault in the producer and becomes a failed row. Both leave the other outputs
    running.
    """


# -- the preset table ---------------------------------------------------------------


class Output:
    """One thing a preset produces. Data, not behaviour: `producer` is a key into
    `PRODUCERS` and `options` is whatever that producer reads.

    Written out by hand rather than as a dataclass because these scripts get loaded
    by path (`importlib.util.spec_from_file_location`) as often as they are run, and
    a dataclass whose annotations are strings -- which `from __future__ import
    annotations` makes them -- looks its own module up in `sys.modules` while it is
    being built, where a path-loaded module is not.
    """

    __slots__ = ("name", "kind", "producer", "about", "options")

    def __init__(self, name: str, kind: str, producer: str, about: str = "",
                 options: dict | None = None):
        self.name = name
        self.kind = kind
        self.producer = producer
        self.about = about
        self.options = dict(options or {})

    def __repr__(self) -> str:
        return f"Output({self.name!r}, {self.kind!r}, {self.producer!r})"


PRESET_ABOUT = {
    "external-flow-2d": "a 2D external case -- flow past something, forces on it",
    "duct-flow": "internal flow through a duct or channel, no forces expected",
    "transient-wake": "a time-resolved wake: vorticity first, and the coefficient history",
    "vehicle-aero": "a 3D external vehicle case, mesh cut included",
    "mesh-validation": "the mesh alone, before anything has been solved on it",
}

PRESETS: dict[str, list[Output]] = {
    "external-flow-2d": [
        Output("pressure", "pressure", "field", "static pressure on the mid slice",
               {"field": "p", "cmap": "coolwarm", "symmetric": True}),
        Output("velocity-magnitude", "velocity", "field", "|U| on the mid slice",
               {"field": "U", "component": "mag", "cmap": "viridis"}),
        Output("velocity-x", "velocity", "field", "streamwise component, signed, so reversed flow reads",
               {"field": "U", "component": "x", "cmap": "coolwarm", "symmetric": True}),
        Output("vorticity", "vorticity", "vorticity", "out-of-plane vorticity", {"component": "z"}),
        Output("streamlines", "streamlines", "streamlines", "streamlines over the slice", {}),
        Output("residuals", "residuals", "residuals", "initial residual per equation", {}),
        Output("forces", "forces", "forces", "force / coefficient history", {}),
    ],
    "duct-flow": [
        Output("pressure", "pressure", "field", "static pressure along the duct",
               {"field": "p", "cmap": "coolwarm"}),
        Output("velocity-magnitude", "velocity", "field", "|U| on the mid slice",
               {"field": "U", "component": "mag", "cmap": "viridis"}),
        Output("velocity-x", "velocity", "field", "streamwise component",
               {"field": "U", "component": "x", "cmap": "coolwarm", "symmetric": True}),
        Output("velocity-y", "velocity", "field", "cross-stream component, where secondary flow shows",
               {"field": "U", "component": "y", "cmap": "coolwarm", "symmetric": True}),
        Output("streamlines", "streamlines", "streamlines", "streamlines over the slice", {}),
        Output("residuals", "residuals", "residuals", "initial residual per equation", {}),
    ],
    "transient-wake": [
        Output("vorticity", "vorticity", "vorticity", "out-of-plane vorticity, the wake structure",
               {"component": "z"}),
        Output("velocity-magnitude", "velocity", "field", "|U| on the mid slice",
               {"field": "U", "component": "mag", "cmap": "viridis"}),
        Output("pressure", "pressure", "field", "static pressure on the mid slice",
               {"field": "p", "cmap": "coolwarm", "symmetric": True}),
        Output("streamlines", "streamlines", "streamlines", "streamlines over the slice", {}),
        Output("residuals", "residuals", "residuals", "initial residual per equation", {}),
        Output("forces", "forces", "forces", "coefficient history, where the shedding shows", {}),
    ],
    "vehicle-aero": [
        Output("pressure", "pressure", "field", "static pressure on the centreline slice",
               {"field": "p", "cmap": "coolwarm", "symmetric": True}),
        Output("velocity-magnitude", "velocity", "field", "|U| on the centreline slice",
               {"field": "U", "component": "mag", "cmap": "viridis"}),
        Output("velocity-x", "velocity", "field", "streamwise component, signed, so the wake reads",
               {"field": "U", "component": "x", "cmap": "coolwarm", "symmetric": True}),
        Output("vorticity", "vorticity", "vorticity", "vorticity magnitude", {"component": "mag"}),
        Output("streamlines", "streamlines", "streamlines", "streamlines over the slice", {}),
        Output("mesh-cut", "mesh-full", "mesh-cut", "the mesh on the same slice, refinement visible",
               {}),
        Output("residuals", "residuals", "residuals", "initial residual per equation", {}),
        Output("forces", "forces", "forces", "force / coefficient history", {}),
    ],
    "mesh-validation": [
        Output("mesh-cut-z", "mesh-full", "mesh-cut", "the mesh on a z-normal cut", {"normal": "z"}),
        Output("mesh-cut-y", "mesh-full", "mesh-cut", "the mesh on a y-normal cut", {"normal": "y"}),
        Output("mesh-quality", "other", "mesh-quality",
               "histograms of cell volume and the quality measures VTK can compute", {}),
    ],
}


def preset_outputs(preset: str) -> list[Output]:
    """The rows of one preset, or a message naming the ones that exist."""
    try:
        return PRESETS[preset]
    except KeyError:
        raise SystemExit(
            f"no preset named {preset!r}; available: {', '.join(sorted(PRESETS))}"
        ) from None


def describe_presets() -> str:
    """The table as text, for `--list`."""
    lines = ["presets"]
    for preset in sorted(PRESETS):
        lines.append("")
        lines.append(f"{preset}   {PRESET_ABOUT.get(preset, '')}".rstrip())
        width = max(len(row.name) for row in PRESETS[preset])
        for row in PRESETS[preset]:
            lines.append(f"  {row.name:<{width}}  {row.kind:<12}  {row.about}".rstrip())
    lines.append("")
    lines.append(
        "An output whose inputs are not in the case is reported as skipped with the "
        "reason; the others still run."
    )
    return "\n".join(lines)


# -- postProcessing .dat files ------------------------------------------------------
#
# forceCoeffs and forces are written in the same shape by every version, and the shape
# is only nearly regular: a block of `#` lines, the last of which is the column header,
# then whitespace-separated numbers. What varies is the column set (Cd/Cl/Cm in one
# version, plus front/rear splits in another), and whether the values are scalars or
# parenthesised vectors. So the header is read rather than assumed.


def split_header_names(line: str) -> list[str]:
    """Column names out of a `# Time ...` header line.

    Parentheses carry meaning here and are not noise. `Cd(f)` is one column named for
    the front half of the drag; `forces(pressure viscous porous)` is three groups of
    three; a bare `(total_x total_y total_z)` is three columns with no prefix at all.
    Flattening them with a prefix keeps every column addressable by name.
    """
    text = line.lstrip()
    if text.startswith("#"):
        text = text[1:]

    names: list[str] = []
    token = ""
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "(":
            depth = 1
            end = index + 1
            while end < length and depth:
                if text[end] == "(":
                    depth += 1
                elif text[end] == ")":
                    depth -= 1
                end += 1
            inner = text[index + 1:end - 1].replace("(", " ").replace(")", " ")
            inner_names = inner.split()
            prefix = token.strip()
            token = ""
            if not inner_names:
                if prefix:
                    names.append(prefix)
            elif prefix:
                names.extend(f"{prefix}_{word}" for word in inner_names)
            else:
                names.extend(inner_names)
            index = end
            continue
        if char.isspace():
            if token.strip():
                names.append(token.strip())
            token = ""
            index += 1
            continue
        token += char
        index += 1
    if token.strip():
        names.append(token.strip())
    return names


def data_values(line: str) -> list[float]:
    """The numbers on one data line, parentheses and commas treated as separators.

    A line with anything non-numeric on it returns nothing rather than a short row:
    a partial line (a job killed mid-write) that yielded three numbers instead of
    nineteen would otherwise shift every column after it.
    """
    cleaned = line.replace("(", " ").replace(")", " ").replace(",", " ")
    values: list[float] = []
    for token in cleaned.split():
        try:
            values.append(float(token))
        except ValueError:
            return []
    return values


def align_columns(names: Sequence[str], width: int) -> list[str]:
    """Header names stretched or trimmed to the number of columns actually written.

    The mismatch that matters is `forces(pressure viscous porous)`: three names in the
    header, nine numbers on the line, because each is a vector. When the count divides
    evenly the names are expanded component-wise; anything else is padded with `cN` so
    the data is still readable under a made-up name rather than dropped.
    """
    names = list(names)
    if width <= 0:
        return []
    if not names:
        return ["Time"] + [f"c{i}" for i in range(1, width)]
    if len(names) == width:
        return names

    head, rest = names[0], names[1:]
    remaining = width - 1
    if rest and remaining > len(rest) and remaining % len(rest) == 0:
        factor = remaining // len(rest)
        suffixes = ("x", "y", "z") if factor == 3 else tuple(str(i) for i in range(factor))
        return [head] + [f"{name}_{suffix}" for name in rest for suffix in suffixes]

    if width < len(names):
        return names[:width]
    return names + [f"c{i}" for i in range(len(names), width)]


def parse_dat(text: str) -> dict[str, Any]:
    """One postProcessing `.dat` file: columns, rows, and the header's key/value lines.

    The `# key : value` lines (magUInf, lRef, Aref, CofR) are kept as metadata rather
    than mistaken for column names -- they are what a coefficient means, and a plot of
    Cd with no magUInf behind it is a plot of an unknown quantity.
    """
    meta: dict[str, str] = {}
    header: list[str] = []
    rows: list[list[float]] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if rows:
                continue
            body = stripped.lstrip("#").strip()
            if not body:
                continue
            key, sep, value = body.partition(":")
            if sep and key.strip() and len(key.split()) <= 3:
                meta[key.strip()] = value.strip()
                continue
            names = split_header_names(body)
            if names:
                header = names
            continue
        values = data_values(stripped)
        if values:
            rows.append(values)

    width = 0
    if rows:
        counts: dict[int, int] = {}
        for row in rows:
            counts[len(row)] = counts.get(len(row), 0) + 1
        width = max(counts, key=lambda size: (counts[size], size))
        rows = [row for row in rows if len(row) == width]

    return {"columns": align_columns(header, width), "rows": rows, "meta": meta}


def time_key(name: str) -> tuple[int, float, str]:
    """Sort key for a postProcessing time directory. `0`, `0.5` and `100` sort as
    numbers; anything else sorts after them by name."""
    try:
        return (0, float(name), "")
    except ValueError:
        return (1, 0.0, name)


def find_force_files(case: Path | str) -> dict[str, list[Path]]:
    """Every force / coefficient `.dat` under `postProcessing`, grouped by series.

    A restart writes a second time directory beside the first
    (`postProcessing/forces/0/` and `postProcessing/forces/240/`) and each holds a
    file with the same name. They are one history and are returned as one list in
    time order; reading only the newest directory is how half a run's coefficients go
    missing from a plot.
    """
    base = Path(case) / "postProcessing"
    if not base.is_dir():
        return {}
    grouped: dict[str, list[Path]] = {}
    for dat in sorted(base.rglob("*.dat")):
        if not dat.is_file():
            continue
        if not dat.stem.lower().startswith(("force", "coefficient")):
            continue
        parts = dat.relative_to(base).parts
        if len(parts) < 2:
            continue
        function = "/".join(parts[:-2]) if len(parts) >= 3 else parts[0]
        grouped.setdefault(f"{function}/{dat.name}", []).append(dat)
    for paths in grouped.values():
        paths.sort(key=lambda path: time_key(path.parent.name))
    return grouped


def choose_force_series(grouped: dict[str, list[Path]]) -> str:
    """Which series to plot when a case logged several.

    Coefficients before raw forces: Cd is comparable with a published number and
    the newtons on a half-model are not.
    """
    if not grouped:
        raise Unavailable("no force or coefficient .dat under postProcessing/")

    def rank(key: str) -> tuple[int, str]:
        lowered = key.lower()
        if "coeff" in lowered or "coefficient" in lowered:
            return (0, lowered)
        if "moment" in lowered:
            return (2, lowered)
        return (1, lowered)

    return sorted(grouped, key=rank)[0]


def read_history(paths: Iterable[Path]) -> dict[str, Any]:
    """Several files of one series read and merged into a single history.

    Overlapping times happen whenever a run is restarted from a write earlier than
    the last one it had reached: both files then hold rows for the same times. The
    later file wins, because it is the one describing the run that is still on disk.
    """
    parsed = [(Path(path), parse_dat(Path(path).read_text(errors="replace"))) for path in paths]
    usable = [(path, data) for path, data in parsed if data["rows"]]
    if not usable:
        return {
            "columns": parsed[-1][1]["columns"] if parsed else [],
            "rows": [],
            "meta": parsed[-1][1]["meta"] if parsed else {},
            "sources": [str(path) for path, _ in parsed],
            "ignored": [],
        }

    columns = usable[-1][1]["columns"]
    merged: dict[float, list[float]] = {}
    meta: dict[str, str] = {}
    sources: list[str] = []
    ignored: list[str] = []
    for path, data in usable:
        if data["columns"] != columns:
            ignored.append(f"{path.name} in {path.parent.name}/ has a different column set")
            continue
        meta.update(data["meta"])
        sources.append(str(path))
        for row in data["rows"]:
            merged[row[0]] = row

    return {
        "columns": columns,
        "rows": [merged[key] for key in sorted(merged)],
        "meta": meta,
        "sources": sources,
        "ignored": ignored,
    }


def column(history: dict[str, Any], name: str) -> np.ndarray:
    """One named column as a float array; empty when the column is not there."""
    try:
        index = list(history["columns"]).index(name)
    except ValueError:
        return np.empty(0)
    return np.array([row[index] for row in history["rows"]], dtype=float)


COEFFICIENT = re.compile(r"^C[A-Za-z]*$")


def plot_columns(columns: Sequence[str]) -> list[str]:
    """Which columns are worth drawing on one axes.

    A forceCoeffs file has Cd, Cl, CmPitch and then the front/rear splits of each;
    a forces file has eighteen numbers. All of them on one plot is a grey band, so
    the whole-body coefficients come first, then the totals, then the pressure part,
    and only failing all of those the first few columns as they came.
    """
    rest = list(columns)[1:]
    if not rest:
        return []
    coefficients = [name for name in rest if COEFFICIENT.match(name)]
    if coefficients:
        return coefficients
    totals = [name for name in rest if name.lower().startswith("total")]
    if totals:
        return totals
    pressure = [name for name in rest if "pressure" in name.lower()]
    if pressure:
        return pressure
    return rest[:6]


def tail_mean(values: np.ndarray, fraction: float = 0.25) -> float:
    """The mean of the last `fraction` of a series.

    For a shedding case the final row is one phase of the cycle and says more about
    when the write happened than about the flow; the tail mean is the number people
    mean when they quote a Cd. It is reported next to the last value, not instead of
    it -- a series still climbing has a tail mean too, and it means nothing.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    keep = max(1, int(round(values.size * fraction)))
    return float(np.mean(values[-keep:]))


def force_headline(history: dict[str, Any], columns: Sequence[str], fraction: float = 0.25) -> dict[str, str]:
    """Last value and tail mean per plotted column, as text for the summary."""
    notes: dict[str, str] = {}
    times = column(history, history["columns"][0]) if history["columns"] else np.empty(0)
    if times.size:
        notes["force record"] = f"{times.size} rows, t = {times[0]:g} to {times[-1]:g}"
    percent = int(round(fraction * 100))
    for name in columns:
        values = column(history, name)
        if not values.size:
            continue
        notes[f"{name} (last)"] = f"{values[-1]:.5g}"
        notes[f"{name} (mean of last {percent}%)"] = f"{tail_mean(values, fraction):.5g}"
    for key in ("magUInf", "lRef", "Aref", "rhoInf"):
        if key in history.get("meta", {}):
            notes[key] = history["meta"][key]
    return notes


# -- the solver log ------------------------------------------------------------------


def looks_like_solver_log(path: Path, probe_bytes: int = 400_000) -> bool:
    """Whether a file has solver residual lines in its opening.

    A case directory usually holds several logs -- blockMesh, snappyHexMesh,
    checkMesh, the solver -- and only one of them has residuals in it. The first few
    hundred kilobytes are enough: a solver writes its first `Solving for` inside the
    first time step, and reading the whole of a multi-gigabyte log to find that out
    would cost more than the plot.
    """
    try:
        with Path(path).open("r", errors="replace") as handle:
            return "Solving for" in handle.read(probe_bytes)
    except OSError:
        return False


def find_solver_log(case: Path | str) -> Path | None:
    """The newest file in the case (or its `logs/`) that has residual lines in it."""
    case = Path(case)
    candidates: list[Path] = []
    for directory in (case, case / "logs"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            name = path.name.lower()
            if not (name.startswith("log") or name.endswith(".log")):
                continue
            if path.suffix.lower() in (".png", ".csv", ".json"):
                continue
            candidates.append(path)
    solver = [path for path in candidates if looks_like_solver_log(path)]
    if not solver:
        return None
    return max(solver, key=lambda path: path.stat().st_mtime)


# -- numbers used by the renderers ---------------------------------------------------


def resolve_time(requested: str | float | None, times: Sequence[float]) -> tuple[float | None, str]:
    """Which write time to draw, and a sentence saying which was chosen and why.

    `latest` is the default because it is what "the result" means; a number is
    matched to the nearest write rather than required to exist, since a request for
    `t = 12.5` on a case that wrote every 0.2 s is a request for the write nearest
    it, not an error.
    """
    values = [float(t) for t in times]
    if not values:
        return None, "no write times in this case"
    if requested is None:
        return values[-1], f"latest of {len(values)} write times"
    text = str(requested).strip().lower()
    if text in ("latest", "last", ""):
        return values[-1], f"latest of {len(values)} write times"
    if text == "first":
        return values[0], f"first of {len(values)} write times"
    try:
        wanted = float(text)
    except ValueError:
        raise SystemExit(f"--time takes a number, 'latest' or 'first'; got {requested!r}") from None
    chosen = min(values, key=lambda value: abs(value - wanted))
    if chosen == wanted:
        return chosen, f"requested time, one of {len(values)} writes"
    return chosen, f"nearest write to {wanted:g} (of {len(values)})"


def robust_clim(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    """A colour range that ignores the tails.

    One cell against a wall with a pressure spike three orders of magnitude above the
    field sets the whole colour bar to itself and the picture comes back a single
    flat colour. Percentiles rather than min/max, so that cell is saturated and the
    rest of the field is visible.
    """
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0)
    lower = float(np.percentile(finite, low))
    upper = float(np.percentile(finite, high))
    if lower == upper:
        spread = abs(lower) * 0.05 or 0.5
        return (lower - spread, upper + spread)
    return (lower, upper)


def symmetric_clim(values: np.ndarray, high: float = 98.0) -> tuple[float, float]:
    """A colour range centred on zero, for anything signed.

    On a diverging colour map an off-centre range puts the neutral colour at some
    arbitrary non-zero value, and the eye reads the sign of the field wrong -- which
    matters most for exactly the fields drawn this way: pressure about freestream,
    streamwise velocity in a recirculation, vorticity.
    """
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    extent = float(np.percentile(np.abs(finite), high))
    if extent == 0.0:
        extent = float(np.max(np.abs(finite))) or 1.0
    return (-extent, extent)


# -- plots that need no case data ----------------------------------------------------


def force_plot(history: dict[str, Any], columns: Sequence[str], out: Path, title: str = "") -> Path:
    """The chosen coefficient columns against time, with the tail mean drawn on.

    The dashed line is the mean over the last quarter of the record. It is drawn
    because the eye reads an oscillating series against whatever it last touched,
    and the number that gets quoted is the mean.
    """
    times = column(history, history["columns"][0])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(9, 5))
    for name in columns:
        values = column(history, name)
        if values.size != times.size or not values.size:
            continue
        line, = axes.plot(times, values, linewidth=1.0, label=name)
        mean = tail_mean(values)
        if np.isfinite(mean):
            axes.axhline(mean, color=line.get_color(), linestyle="--", linewidth=0.7, alpha=0.6)
    axes.set_xlabel("time")
    axes.set_ylabel("coefficient / force")
    axes.grid(True, alpha=0.3)
    axes.legend(loc="best", fontsize="small", ncol=2)
    if title:
        axes.set_title(f"{title}   (dashed: mean of last 25%)", fontsize="medium")
    figure.tight_layout()
    figure.savefig(out, dpi=120)
    plt.close(figure)
    return out


def quality_histograms(measures: dict[str, np.ndarray], out: Path, title: str = "") -> Path:
    """One histogram per quality measure, log-binned where the values span decades.

    Cell volume on a snappy mesh runs over five or six orders of magnitude between
    the layer cells and the far field, and on linear bins the whole mesh lands in the
    first bar. Anything strictly positive whose range spans more than two decades is
    binned in the log instead.
    """
    usable = []
    for name, values in measures.items():
        array = np.asarray(values, dtype=float).ravel()
        array = array[np.isfinite(array)]
        if array.size:
            usable.append((name, array))
    if not usable:
        raise Unavailable("no quality measure could be computed on this mesh")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = min(2, len(usable))
    rows = (len(usable) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(6.0 * columns, 3.6 * rows), squeeze=False)
    for index, (name, values) in enumerate(usable):
        axis = axes[index // columns][index % columns]
        positive = values[values > 0]
        spans_decades = positive.size == values.size and positive.size > 0 and (
            positive.max() / positive.min() > 100.0
        )
        if spans_decades:
            bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 60)
            axis.set_xscale("log")
        else:
            bins = 60
        axis.hist(values, bins=bins, color="steelblue")
        axis.set_yscale("log")
        axis.set_title(name, fontsize="medium")
        axis.set_ylabel("cells")
        axis.grid(True, alpha=0.3)
        axis.text(
            0.98, 0.95,
            f"min {values.min():.3g}\nmax {values.max():.3g}\nmedian {np.median(values):.3g}",
            transform=axis.transAxes, ha="right", va="top", fontsize=8,
        )
    for index in range(len(usable), rows * columns):
        axes[index // columns][index % columns].axis("off")
    if title:
        figure.suptitle(title, fontsize="medium")
    figure.tight_layout()
    figure.savefig(out, dpi=120)
    plt.close(figure)
    return out


# -- the pyvista side ----------------------------------------------------------------
#
# pyvista is imported inside these functions rather than at the top of the file, so
# the parsing above can be imported and tested anywhere. Nothing outside this section
# touches it.

NORMALS = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def _pyvista():
    import pyvista as pv

    pv.OFF_SCREEN = True
    return pv


VIEWUP = {"z": (0.0, 1.0, 0.0), "y": (0.0, 0.0, 1.0), "x": (0.0, 0.0, 1.0)}
"""Which way is up, per slice normal. Without this VTK picks, and for a z-normal
slice it picked +x: every picture of a left-to-right flow came out rotated a quarter
turn, so a cylinder wake read as though the flow went upward. The physics was right
and the picture was not, which is the worse of the two failures -- a wrong number
gets checked and a wrong orientation gets believed."""


def aim(plotter, normal: str) -> None:
    """Point the camera down the slice normal, with x across the page.

    `plotter.camera_position = "z"` reads as if it would do this and does not:
    the string form of `camera_position` takes a plane ("xy", "xz", ...), and a
    single axis letter is not one of them. Naming the direction as a vector says
    what is meant and holds for every pyvista version -- and naming the up vector
    with it is what keeps the streamwise axis horizontal.
    """
    plotter.view_vector(NORMALS[normal], viewup=VIEWUP.get(normal, (0.0, 1.0, 0.0)))


def scalar_bar(plotter, title: str) -> None:
    """A colour bar under the picture rather than across it.

    pyvista's default places the bar inside the render window, and with a parallel
    projection filling the frame that put the tick labels on top of the flow -- the
    numbers were unreadable and so was the part of the wake behind them.
    """
    plotter.add_scalar_bar(
        title=title, n_labels=5, vertical=False,
        position_x=0.15, position_y=0.02, width=0.7, height=0.06,
        title_font_size=14, label_font_size=12,
    )


def internal_mesh(block):
    """The internalMesh block, whatever nesting this case produced."""
    pv = _pyvista()
    if isinstance(block, pv.MultiBlock):
        if "internalMesh" in (block.keys() or []):
            return block["internalMesh"]
        for item in block:
            if item is None:
                continue
            found = internal_mesh(item)
            if found is not None:
                return found
        return None
    return block


def open_case(case: Path, requested: str | float | None):
    """The internal mesh at the chosen write time, plus what that time was."""
    pv = _pyvista()
    case = Path(case)
    foam = case / f"{case.name}.foam"
    if not foam.exists():
        foam.write_text("")
    reader = pv.OpenFOAMReader(str(foam))
    reader.cell_to_point_creation = True
    times = [float(value) for value in reader.time_values]
    chosen, note = resolve_time(requested, times)
    if chosen is not None:
        reader.set_active_time_value(chosen)
    mesh = internal_mesh(reader.read())
    if mesh is None:
        raise Unavailable("no internalMesh in this case -- it has not been meshed")
    return mesh, chosen, note


def slice_at(mesh, normal: str):
    origin = np.array(mesh.center)
    return mesh.slice(normal=NORMALS[normal], origin=origin)


def scalar_from(cut, field_name: str, component: str | None) -> str:
    """A per-point scalar name to colour by, deriving a component or magnitude.

    Raises `Unavailable` naming what the case does hold, because "field 'p' is not
    here" and "this case wrote p_rgh" are the same situation and the second sentence
    is the useful one.
    """
    data = cut.point_data.get(field_name)
    if data is None:
        data = cut.cell_data.get(field_name)
    if data is None:
        present = ", ".join(sorted(set(cut.point_data.keys()) | set(cut.cell_data.keys())))
        raise Unavailable(f"the case has no field '{field_name}' at this time; it has: {present or 'nothing'}")
    array = np.asarray(data)
    if array.ndim == 1 or array.shape[1] == 1:
        return field_name
    if component in (None, "mag"):
        name = f"|{field_name}|"
        cut[name] = np.linalg.norm(array, axis=1)
        return name
    index = {"x": 0, "y": 1, "z": 2}.get(str(component))
    if index is None or index >= array.shape[1]:
        raise Unavailable(f"'{field_name}' has no component '{component}'")
    name = f"{field_name}_{component}"
    cut[name] = array[:, index]
    return name


def render_scalar(cut, scalars: str, out: Path, *, normal: str, cmap: str,
                  clim: tuple[float, float] | None, title: str) -> Path:
    pv = _pyvista()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 850))
    plotter.add_mesh(cut, scalars=scalars, cmap=cmap, clim=clim, show_edges=False)
    scalar_bar(plotter, scalars)
    aim(plotter, normal)
    plotter.enable_parallel_projection()
    plotter.add_text(title, font_size=9)
    plotter.screenshot(str(out))
    plotter.close()
    return out


def render_mesh_cut(cut, out: Path, *, normal: str, title: str) -> Path:
    pv = _pyvista()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 850))
    plotter.add_mesh(cut, color="white", show_edges=True, edge_color="black", line_width=0.3)
    aim(plotter, normal)
    plotter.enable_parallel_projection()
    plotter.add_text(title, font_size=9)
    plotter.screenshot(str(out))
    plotter.close()
    return out


QUALITY_MEASURES = ("aspect_ratio", "skew", "scaled_jacobian", "max_angle")
"""What VTK is asked for. Not every one exists for every cell type -- a polyhedral
snappy cell answers to fewer of them than a hexahedron -- so each is attempted alone
and the ones that fail are left out rather than taking the plot with them."""


def mesh_quality_arrays(mesh) -> dict[str, np.ndarray]:
    """Cell volume plus whatever quality measures this VTK build computes here."""
    measures: dict[str, np.ndarray] = {}
    try:
        sized = mesh.compute_cell_sizes(length=False, area=False, volume=True)
        volumes = np.asarray(sized.cell_data["Volume"], dtype=float)
        volumes = volumes[np.isfinite(volumes)]
        if volumes.size:
            measures["cell volume"] = np.abs(volumes)
    except Exception:
        pass

    quality = getattr(mesh, "cell_quality", None) or getattr(mesh, "compute_cell_quality", None)
    if quality is None:
        return measures
    for measure in QUALITY_MEASURES:
        try:
            computed = quality(measure)
            values = None
            for key in (measure, "CellQuality", "quality"):
                if key in computed.cell_data:
                    values = np.asarray(computed.cell_data[key], dtype=float)
                    break
            if values is None or not values.size:
                continue
            values = values[np.isfinite(values)]
            if values.size:
                measures[measure.replace("_", " ")] = values
        except Exception:
            continue
    return measures


def streamline_geometry(mesh, cut, normal: str):
    """Streamlines over the slice, by whichever of the two routes this case allows.

    The evenly spaced 2D seeding gives much the better picture but wants a slice that
    is genuinely planar in xy; on anything else it raises, and a source-line seeding
    through the volume is used instead.
    """
    if normal == "z":
        try:
            return cut.streamlines_evenly_spaced_2D(vectors="U", start_position=cut.center,
                                                    separating_distance=3.0, separating_distance_ratio=0.2)
        except Exception:
            pass
    bounds = np.asarray(mesh.bounds, dtype=float)
    span = float(np.linalg.norm(bounds[1::2] - bounds[0::2]))
    return mesh.streamlines(
        vectors="U",
        n_points=250,
        source_radius=span * 0.4,
        source_center=mesh.center,
        max_time=span * 20.0,
    )


# -- producers -----------------------------------------------------------------------
#
# Each takes the run context and one Output row, writes one file and returns its path
# with any numbers worth putting in the summary. Anything the case does not contain is
# an `Unavailable` naming what was missing.


def produce_field(ctx, spec: Output):
    cut = ctx.cut(spec.options.get("normal"))
    field_name = spec.options.get("field", "U")
    component = spec.options.get("component")
    scalars = scalar_from(cut, field_name, component)
    values = np.asarray(cut.point_data[scalars] if scalars in cut.point_data else cut.cell_data[scalars])
    clim = symmetric_clim(values) if spec.options.get("symmetric") else robust_clim(values)
    normal = spec.options.get("normal") or ctx.normal
    out = ctx.out_dir / f"{spec.name}.png"
    render_scalar(
        cut, scalars, out,
        normal=normal, cmap=spec.options.get("cmap", "viridis"), clim=clim,
        title=f"{scalars}   {normal}-normal slice   t = {ctx.time_label}",
    )
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    notes = {}
    if finite.size:
        notes[f"{scalars} range"] = f"{finite.min():.5g} to {finite.max():.5g}"
    return out, notes


def produce_vorticity(ctx, spec: Output):
    mesh = ctx.mesh()
    if "U" not in mesh.point_data and "U" not in mesh.cell_data:
        raise Unavailable("vorticity needs U, which this case did not write")
    if "vorticity" in mesh.point_data:
        derived = mesh
    else:
        derived = mesh.compute_derivative(scalars="U", vorticity=True)
        if "vorticity" not in derived.point_data:
            raise Unavailable("VTK would not compute a vorticity from U on this mesh")
    vectors = np.asarray(derived.point_data["vorticity"], dtype=float)
    component = spec.options.get("component", "z")
    if component == "mag":
        name = "|vorticity|"
        derived.point_data[name] = np.linalg.norm(vectors, axis=1)
        clim = robust_clim(derived.point_data[name])
        cmap = "inferno"
    else:
        index = {"x": 0, "y": 1, "z": 2}[component]
        name = f"vorticity_{component}"
        derived.point_data[name] = vectors[:, index]
        clim = symmetric_clim(derived.point_data[name])
        cmap = "RdBu_r"
    normal = spec.options.get("normal") or ctx.normal
    cut = slice_at(derived, normal)
    out = ctx.out_dir / f"{spec.name}.png"
    render_scalar(cut, name, out, normal=normal, cmap=cmap, clim=clim,
                  title=f"{name}   {normal}-normal slice   t = {ctx.time_label}")
    return out, {f"{name} range": f"{clim[0]:.5g} to {clim[1]:.5g} (2-98 percentile)"}


def produce_streamlines(ctx, spec: Output):
    mesh = ctx.mesh()
    if "U" not in mesh.point_data and "U" not in mesh.cell_data:
        raise Unavailable("streamlines need U, which this case did not write")
    normal = spec.options.get("normal") or ctx.normal
    cut = ctx.cut(normal)
    lines = streamline_geometry(mesh, cut, normal)
    if lines is None or lines.n_points == 0:
        raise Unavailable("no streamline could be integrated through this field")
    pv = _pyvista()
    out = ctx.out_dir / f"{spec.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 850))
    plotter.add_mesh(cut, color="whitesmoke", show_edges=False, opacity=0.6)
    plotter.add_mesh(lines, scalars="U", cmap="viridis", line_width=1.5)
    aim(plotter, normal)
    plotter.enable_parallel_projection()
    plotter.add_text(f"streamlines   {normal}-normal   t = {ctx.time_label}", font_size=9)
    plotter.screenshot(str(out))
    plotter.close()
    return out, {}


def produce_mesh_cut(ctx, spec: Output):
    normal = spec.options.get("normal") or ctx.normal
    cut = ctx.cut(normal)
    mesh = ctx.mesh()
    out = ctx.out_dir / f"{spec.name}.png"
    render_mesh_cut(cut, out, normal=normal, title=f"mesh   {normal}-normal cut")
    return out, {"cells": f"{mesh.n_cells:,}", "points": f"{mesh.n_points:,}"}


def produce_mesh_quality(ctx, spec: Output):
    mesh = ctx.mesh()
    measures = mesh_quality_arrays(mesh)
    if not measures:
        raise Unavailable("VTK computed no quality measure for this mesh's cell types")
    out = ctx.out_dir / f"{spec.name}.png"
    quality_histograms(measures, out, title=f"{ctx.case_name} mesh quality")
    notes = {}
    if "cell volume" in measures:
        volumes = measures["cell volume"]
        notes["cell volume min / max"] = f"{volumes.min():.4g} / {volumes.max():.4g}"
    for name, values in measures.items():
        if name != "cell volume":
            notes[f"{name} max"] = f"{values.max():.4g}"
    return out, notes


def produce_residuals(ctx, spec: Output):
    log = ctx.solver_log()
    if log is None:
        raise Unavailable(
            f"no log with 'Solving for' lines in {ctx.case.name}/ or {ctx.case.name}/logs/"
        )
    data = log_digest.digest(log)
    if not data["residuals"]:
        raise Unavailable(f"{log.name} has no residual lines yet")
    out = ctx.out_dir / f"{spec.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    log_digest.plot(data["residuals"], out)
    notes = {"solver log": log.name}
    if data["times"]:
        notes["time steps in log"] = f"{len(data['times'])} (last Time = {data['times'][-1]:g})"
    for name in sorted(data["final_residual"]):
        notes[f"final residual {name}"] = f"{data['final_residual'][name]:.3e}"
    if data["continuity"]:
        notes["cumulative continuity error"] = f"{data['continuity'][2]:.3e}"
    if data["bounding"]:
        notes["bounding messages"] = ", ".join(
            f"{name} x{count}" for name, count in sorted(data["bounding"].items())
        )
    return out, notes


def produce_forces(ctx, spec: Output):
    grouped = find_force_files(ctx.case)
    key = choose_force_series(grouped)
    history = read_history(grouped[key])
    if not history["rows"]:
        raise Unavailable(f"{key} has a header but no data rows yet")
    columns = plot_columns(history["columns"])
    if not columns:
        raise Unavailable(f"{key} has no column other than time")
    out = ctx.out_dir / f"{spec.name}.png"
    force_plot(history, columns, out, title=key)
    notes = {"force series": key}
    if len(history["sources"]) > 1:
        notes["merged from"] = f"{len(history['sources'])} time directories"
    notes.update(force_headline(history, columns))
    if history["ignored"]:
        notes["ignored"] = "; ".join(history["ignored"])
    return out, notes


PRODUCERS: dict[str, Callable[[Any, Output], tuple[Path, dict]]] = {
    "field": produce_field,
    "vorticity": produce_vorticity,
    "streamlines": produce_streamlines,
    "mesh-cut": produce_mesh_cut,
    "mesh-quality": produce_mesh_quality,
    "residuals": produce_residuals,
    "forces": produce_forces,
}


# -- running a preset ----------------------------------------------------------------


class Outcome:
    """What became of one row of the preset: produced, skipped or failed, and the
    reason when it is one of the last two."""

    __slots__ = ("spec", "status", "path", "reason", "notes")

    def __init__(self, spec: Output, status: str, path: Path | None = None,
                 reason: str = "", notes: dict | None = None):
        self.spec = spec
        self.status = status
        self.path = path
        self.reason = reason
        self.notes = dict(notes or {})

    def __repr__(self) -> str:
        return f"Outcome({self.spec.name!r}, {self.status!r}, {self.reason!r})"


def run_output(ctx, spec: Output, producers: dict | None = None) -> Outcome:
    """One output, attempted. Never raises: a missing input is a skipped row and an
    unexpected error is a failed one, so the outputs after it still get their turn."""
    table = PRODUCERS if producers is None else producers
    producer = table.get(spec.producer)
    if producer is None:
        return Outcome(spec, "skipped", reason=f"no producer named '{spec.producer}'")
    try:
        path, notes = producer(ctx, spec)
    except Unavailable as exc:
        return Outcome(spec, "skipped", reason=str(exc))
    except Exception as exc:  # a broken producer must not cost the other outputs
        return Outcome(spec, "failed", reason=f"{type(exc).__name__}: {exc}")
    if path is None:
        return Outcome(spec, "skipped", reason="the producer wrote no file")
    path = Path(path)
    ctx.record(spec.kind, path, label=spec.about or spec.name)
    return Outcome(spec, "produced", path=path, notes=dict(notes or {}))


def run_preset(ctx, preset: str, producers: dict | None = None,
               only: Sequence[str] | None = None) -> list[Outcome]:
    specs = preset_outputs(preset)
    if only:
        wanted = set(only)
        specs = [spec for spec in specs if spec.name in wanted]
    return [run_output(ctx, spec, producers) for spec in specs]


def phase_result(outcomes: Sequence[Outcome]) -> tuple[str, str]:
    """The status the `render` phase ends in, and a note for it."""
    produced = [outcome for outcome in outcomes if outcome.status == "produced"]
    skipped = [outcome for outcome in outcomes if outcome.status == "skipped"]
    failed = [outcome for outcome in outcomes if outcome.status == "failed"]
    note = f"{len(produced)} produced, {len(skipped)} skipped, {len(failed)} failed"
    if failed and not produced:
        return "failed", note
    if produced:
        return "done", note
    return "skipped", note


def relative_to(path: Path | None, root: Path | None) -> str:
    if path is None:
        return ""
    if root is None:
        return str(path)
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except (ValueError, OSError):
        return str(path)


def summary_markdown(preset: str, outcomes: Sequence[Outcome], *, case: str = "",
                     time_label: str = "", time_note: str = "", root: Path | None = None,
                     out_dir: Path | None = None) -> str:
    """The results directory as one page: what is there, what is not, and why.

    The skipped rows carry their reason into the summary rather than being left out
    of it. "There is no forces plot" and "there is no forces plot because nothing
    under postProcessing logged any" are different pieces of information, and only
    the second one says whether to go and add the function object.
    """
    produced = [outcome for outcome in outcomes if outcome.status == "produced"]
    skipped = [outcome for outcome in outcomes if outcome.status == "skipped"]
    failed = [outcome for outcome in outcomes if outcome.status == "failed"]

    lines = [f"# results -- {preset}", ""]
    if PRESET_ABOUT.get(preset):
        lines.append(PRESET_ABOUT[preset])
        lines.append("")
    if case:
        lines.append(f"- case: `{case}`")
    if time_label:
        lines.append(f"- time: {time_label}" + (f" ({time_note})" if time_note else ""))
    if out_dir is not None:
        lines.append(f"- written to: `{relative_to(Path(out_dir), root) or out_dir}`")
    lines.append(
        f"- {len(produced)} produced, {len(skipped)} skipped, {len(failed)} failed"
    )
    lines.append("")

    if produced:
        lines.append("## produced")
        lines.append("")
        lines.append("| output | kind | file |")
        lines.append("|---|---|---|")
        for outcome in produced:
            shown = relative_to(outcome.path, root) or (outcome.path.name if outcome.path else "")
            lines.append(f"| {outcome.spec.name} | {outcome.spec.kind} | `{shown}` |")
        lines.append("")

    if skipped:
        lines.append("## skipped")
        lines.append("")
        for outcome in skipped:
            lines.append(f"- **{outcome.spec.name}** -- {outcome.reason}")
        lines.append("")

    if failed:
        lines.append("## failed")
        lines.append("")
        for outcome in failed:
            lines.append(f"- **{outcome.spec.name}** -- {outcome.reason}")
        lines.append("")

    headline = [outcome for outcome in produced if outcome.notes]
    if headline:
        lines.append("## numbers")
        lines.append("")
        for outcome in headline:
            lines.append(f"### {outcome.spec.name}")
            lines.append("")
            for key, value in outcome.notes.items():
                lines.append(f"- {key}: {value}")
            lines.append("")

    lines.append(
        "These are the numbers as written. Whether they are converged, mesh "
        "independent or right is not decided here."
    )
    return "\n".join(lines).rstrip() + "\n"


# -- the run context -----------------------------------------------------------------


class Context:
    """Everything the producers share: the case, where files go, and the case data,
    opened once and only if something asks for it.

    A preset whose outputs are all log-based (residuals, forces) never touches
    pyvista at all this way, which is the difference between a results run on a case
    that has not been reconstructed failing outright and it producing the two plots
    it can.
    """

    def __init__(self, case: Path, out_dir: Path, *, normal: str = "z",
                 requested_time: str | float | None = "latest", log: Path | None = None,
                 root: Path | None = None):
        self.case = Path(case)
        self.out_dir = Path(out_dir)
        self.normal = normal
        self.requested_time = requested_time
        self.explicit_log = Path(log) if log else None
        self.root = Path(root) if root else study_state.find_root(self.case)
        self.case_name = self.case.name
        self.time_value: float | None = None
        self.time_label = "?"
        self.time_note = ""
        self._mesh = None
        self._mesh_error: Exception | None = None
        self._cuts: dict[str, Any] = {}
        self._log_searched = False
        self._log: Path | None = None

    def mesh(self):
        """The case data, read once. A failure to read it is remembered and re-raised
        rather than retried: opening a case costs the same whether it works or not,
        and six outputs asking for a mesh that is not there would pay it six times."""
        if self._mesh_error is not None:
            raise self._mesh_error
        if self._mesh is None:
            try:
                mesh, chosen, note = open_case(self.case, self.requested_time)
            except Exception as exc:
                self._mesh_error = exc
                raise
            self._mesh = mesh
            self.time_value = chosen
            self.time_label = "?" if chosen is None else f"{chosen:g}"
            self.time_note = note
        return self._mesh

    def cut(self, normal: str | None = None):
        normal = normal or self.normal
        if normal not in self._cuts:
            self._cuts[normal] = slice_at(self.mesh(), normal)
        return self._cuts[normal]

    def solver_log(self) -> Path | None:
        if self.explicit_log is not None:
            return self.explicit_log if self.explicit_log.exists() else None
        if not self._log_searched:
            self._log = find_solver_log(self.case)
            self._log_searched = True
        return self._log

    def record(self, kind: str, path: Path, label: str = "") -> None:
        study_state.record(kind, path, root=self.root, case=self.case_name, label=label)


# -- the command line ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case", nargs="?", type=Path, help="the case directory")
    parser.add_argument("--preset", default="", help=f"one of: {', '.join(sorted(PRESETS))}")
    parser.add_argument("--list", action="store_true", help="print the presets and what each produces")
    parser.add_argument("--time", default="latest", help="a write time, 'latest' or 'first'")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default <case>/results)")
    parser.add_argument("--normal", default="z", choices=sorted(NORMALS), help="slice normal")
    parser.add_argument("--log", type=Path, default=None, help="solver log, if it is not in the case")
    parser.add_argument("--only", nargs="+", default=None, help="run only these outputs of the preset")
    parser.add_argument("--no-state", action="store_true",
                        help="do not touch the study manifest or phase table")
    args = parser.parse_args(argv)

    if args.list:
        print(describe_presets())
        return 0
    if args.case is None or not args.preset:
        parser.error("a case directory and --preset are needed (or --list)")
    if not args.case.is_dir():
        raise SystemExit(f"{args.case} is not a directory")
    specs = preset_outputs(args.preset)

    # A mistyped --only would otherwise run nothing and report a clean, empty
    # results directory, which reads as "the case had none of it" rather than as
    # a typo. Same for a --log path that is not there: silently falling back to
    # the search would blame the case for a wrong argument.
    if args.only:
        known = [spec.name for spec in specs]
        unknown = [name for name in args.only if name not in known]
        if unknown:
            raise SystemExit(
                f"--only names {', '.join(unknown)}, which {args.preset} does not produce; "
                f"it produces: {', '.join(known)}"
            )
    if args.log is not None and not args.log.is_file():
        raise SystemExit(f"--log {args.log} is not a file")

    out_dir = args.out or args.case / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    context = Context(
        args.case, out_dir, normal=args.normal, requested_time=args.time, log=args.log
    )
    if args.no_state:
        context.record = lambda *a, **k: None  # type: ignore[assignment]
    else:
        study_state.set_phase("render", "running", root=context.root, case=context.case_name)

    print(f"{args.preset} -> {out_dir}", flush=True)
    outcomes = []
    for spec in preset_outputs(args.preset):
        if args.only and spec.name not in set(args.only):
            continue
        outcome = run_output(context, spec)
        outcomes.append(outcome)
        if outcome.status == "produced":
            print(f"  {spec.name}: {outcome.path}", flush=True)
        else:
            print(f"  {spec.name}: {outcome.status} -- {outcome.reason}", flush=True)

    # Nothing that reads the case data may have run (a preset of log-based outputs,
    # or a case with no write times), and "time: ?" says less than no line at all.
    solved_time = context.time_label if context.time_value is not None else ""
    summary = summary_markdown(
        args.preset, outcomes,
        case=context.case_name, time_label=solved_time, time_note=context.time_note,
        root=context.root, out_dir=out_dir,
    )
    summary_path = out_dir / "results.md"
    summary_path.write_text(summary, encoding="utf-8")
    status, note = phase_result(outcomes)
    if not args.no_state:
        context.record("report", summary_path, label=f"{args.preset} results summary")
        study_state.set_phase("render", status, root=context.root,
                              case=context.case_name, note=f"{args.preset}: {note}")

    print(f"\n{note}")
    print(summary_path)
    return 0 if status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
