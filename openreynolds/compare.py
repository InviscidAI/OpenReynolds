"""Two finished studies read off disk and put side by side.

Second runs happen: a mesh is refined, a turbulence model swapped, a boundary
condition corrected, and the question afterwards is always the same one -- what
actually changed. Answering it meant opening two `results.md` files in two
directories and reading two numbers out of each by eye, which is how a study gets
quoted with the other run's Cd.

Nothing here judges. It reports what the two runs recorded, and the difference
between numbers that are the same quantity.

WHAT IS NOT COMPARED IS SAID OUT LOUD. Every section reports a quantity present on
one side only as missing from the other, and two claims under different unit labels
as not comparable rather than converting them. That is not caution for its own sake:
the first version of `toolbox/claims.py` dropped the pairs it could not compare and
printed its all-clear over the silence, on the very pair it had been written to catch
(`toolbox/claims.py:394`). A comparison tool's one unforgivable failure mode is
saying "no differences" about a quantity it never looked at.

AND ITS MIRROR IMAGE IS A FALSE ALARM. On two real Ahmed-body runs (#141) this
printed `CmRoll by 71.6%`, `CmYaw by 115.3%` and `Cs by 91.5%` -- three quantities
that are identically zero on a symmetric body, compared at 1e-20 against 1e-20 by a
relative rule with no floor -- and nine `checkMesh` lines that say two meshes are two
meshes, while `Cd 0.4236 against 0.4067`, a 4.0% change in drag and the entire point
of the second run, was missing from the summary because 0.040 < the 5% stamp
tolerance it had borrowed from `claims.py`. A verdict nobody can trust is worth as
little as no verdict, so `TOL` is a force-coefficient tolerance rather than a
figure-stamp one, `NEGLIGIBLE` is the floor, and both are printed with the answer.

AND EVERY NUMBER CARRIES THE CASE IT CAME FROM. On a three-case study (#143) side
a's `Cd`/`Cl`/`Cp` were read from `case/postProcessing/bladeF` and its `total_x/y/z`
from a different case directory, `case_axisX`, whose sibling is named
`case_axisX_ruledout` -- spliced into one table with no case column, while the
primary case's own 13-row force history was never compared and never mentioned.
Provenance is part of the number here: the case and the series are in the table, a
series the chooser passed over is named, a case that is marked as ruled out is not
read unless it is all there is, and a quantity whose two sides come from differently
named cases says so in the summary line about it.

THE READERS ARE THE TOOLBOX'S. `postProcessing` restarts, the superseded rows of an
abandoned run, the coefficient-before-newtons preference, `checkMesh`'s one-star
advisories, what a residual series is doing -- all of that is already solved in
`openreynolds/toolbox/`, correctly and with the scars to prove it, and a second
implementation here would be a second set of those bugs. The toolbox is data rather
than a package (it is synced to a workspace, where this package does not exist), so
the scripts are imported by path, the way `buildup/probes.py` and `tests/
test_toolbox.py` already import them. `results.py` and `log_digest.py` pull in numpy
and matplotlib, which belong to the workspace image and to `openreynolds[toolbox]`
rather than to the base install, so those two sections report themselves unavailable
on a machine without them instead of taking the whole command down.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

from .store import Store

TOOLBOX = Path(__file__).resolve().parent / "toolbox"

TOL = 0.01
"""What counts as a difference here, and `--tol` overrides it.

One per cent, not `claims.DEFAULT_TOL`'s five. Five is right for a figure stamp
against an answer -- a plot label and a number that disagree by 5% disagree about
something -- and wrong for a force coefficient, where a mesh study is called
converged when the coefficient stops moving by about a per cent and 4% on Cd is the
finding, not the noise. Borrowing the stamp tolerance is what hid #141's drag
change. It is stated in the output, because "no differences" means nothing without
the number it was measured against.
"""

NEGLIGIBLE = 1e-3
"""The floor, as a fraction of the largest coefficient the two runs logged.

`claims.apart` normalises on the larger magnitude, which keeps it symmetric and
stops it dividing by a near-zero -- and makes 3.3e-20 against 1.9e-20 "71.6%
apart". On a symmetric body the roll moment, the yaw moment and the side force are
identically zero and what is logged for them is solver residue, so the scale has to
come from the data: a quantity whose magnitude on both sides is under a thousandth
of the biggest coefficient in the same comparison (Cd = 0.42, in #141) is zero on
both sides, and zero against zero is unchanged. A thousandth and not an absolute
epsilon, because a run in newtons on a full-scale hull logs numbers in the
thousands and one in coefficients logs numbers near one.
"""

RULED_OUT = "ruledout"
"""A case directory whose name carries this was tried and rejected by whoever ran the
study, and is not read while another case exists -- #143 compared `case_axisX`, whose
sibling was `case_axisX_ruledout`, in place of the case the study was named for. It is
named in the notes either way: skipping it silently is the same failure as choosing it
silently."""

PROCESSOR = re.compile(r"^processor\d+$", re.IGNORECASE)
"""`processor0` .. `processor3` are one decomposed case, not four cases."""

NUMERIC_HINT = "needs numpy and matplotlib: pip install 'openreynolds[toolbox]'"
"""Said instead of the forces and convergence sections when the toolbox's own
dependencies are not on this machine."""


def _toolbox(*names: str) -> list[Any]:
    """Toolbox scripts, imported into this process by path.

    `sys.path`, not `importlib.util.spec_from_file_location`, because `claims.py`
    and `results.py` import their own siblings by bare name.
    """
    if str(TOOLBOX) not in sys.path:
        sys.path.insert(0, str(TOOLBOX))
    return [importlib.import_module(name) for name in names]


# -- finding things in a mirrored study ----------------------------------------------


def cases(root: Path) -> list[Path]:
    """Every case directory in the mirror: one with a `postProcessing/` under it.

    A decomposed run's `processor0/postProcessing` belongs to the case above it, so
    those fold into their parent: four processor directories reported as four cases is
    how a single-case study came back as "the first of 6 cases".
    """
    found = set()
    for path in root.rglob("postProcessing"):
        if not path.is_dir():
            continue
        case = path.parent
        if PROCESSOR.match(case.name):
            case = case.parent
        found.add(case)
    return sorted(found)


def _live_cases(found: list[Path]) -> tuple[list[Path], list[str]]:
    """The cases worth reading, and what was said about the ones left out.

    A `*_ruledout` directory is skipped while any other case exists, and read anyway
    when it is all there is -- with a note in both cases.
    """
    live = [case for case in found if RULED_OUT not in case.name.lower().replace("_", "")]
    if found and not live:
        return found, ["every case directory here is named as ruled out, so they were "
                       "read anyway: there is nothing else to read"]
    return live, [f"{case.name} is named as ruled out, so it was not read"
                  for case in found if case not in live]


def _newest(paths: list[Path]) -> Path | None:
    def when(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    return max(paths, key=when) if paths else None


def solver_log(root: Path) -> Path | None:
    """The newest file anywhere in the mirror with solver residual lines in it.

    `results.find_solver_log` looks in one case directory and its `logs/`; a mirror
    holds the whole tree and the case is not known in advance, so the same probe is
    run over the tree.
    """
    results, = _toolbox("results")
    found = [
        path for path in root.rglob("*")
        if path.is_file()
        and (path.name.lower().startswith("log") or path.name.lower().endswith(".log"))
        and path.suffix.lower() not in (".png", ".csv", ".json")
        and results.looks_like_solver_log(path)
    ]
    return _newest(found)


def mesh_log(root: Path) -> Path | None:
    """The newest `checkMesh` output in the mirror."""
    return _newest([
        path for path in root.rglob("*")
        if path.is_file() and "checkmesh" in path.name.lower()
    ])


def state_roots(root: Path) -> list[Path]:
    """Every directory in the mirror that keeps study state, for `study_state`.

    `study_state.find_root` walks *up* from where it is pointed, so pointing it at
    the mirror's top finds nothing: the `.reynolds/` sits further down, beside the
    case, wherever the workspace put it.
    """
    return sorted(path.parent for path in root.rglob(".reynolds") if path.is_dir())


# -- the sections --------------------------------------------------------------------


def identity(studies_dir: Path, study_id: str) -> dict[str, Any]:
    """What the study says about itself, out of `session.json` and the transcript."""
    store = Store(studies_dir, study_id)
    session = store.session
    messages = studies_dir / study_id / "messages.jsonl"
    count = 0
    if messages.is_file():
        count = sum(1 for line in messages.read_text(
            encoding="utf-8", errors="replace").splitlines() if line.strip())
    return {
        "study_id": session.study_id or study_id,
        "title": session.title,
        "created_at": session.created_at,
        "model": session.model,
        "provider": session.provider,
        "solver": store.recorded_solver or "",
        "instance_id": session.instance_id,
        "messages": count,
        "dir": str(studies_dir / study_id),
    }


def forces(root: Path) -> dict[str, Any]:
    """Each force or coefficient column the study logged: last value and tail mean.

    One series per case, chosen by `results.choose_force_series` -- coefficients
    before newtons, which is its call and not one to make twice. Keyed by the column
    name (`Cd`), because that is the thing two studies have in common; the function
    object they were written under usually is not -- so each row carries the `case`
    and the `series` it was read from, and whatever was passed over is in `notes`.

    Keyed by column name and not by `(case, column)`: the two studies being compared
    name their cases differently (`case` against `case_Fr0316`), and keying on the
    case would pair nothing with anything. The case travels with the number instead.
    """
    try:
        results, = _toolbox("results")
    except ImportError as exc:
        return {"unavailable": f"{exc.name}: {NUMERIC_HINT}", "series": {}, "notes": []}
    series: dict[str, dict[str, Any]] = {}
    found = cases(root)
    live, notes = _live_cases(found)
    for case in live:
        grouped = results.find_force_files(case)
        if not grouped:
            notes.append(f"{case.name} logged no force or coefficient file")
            continue
        key = results.choose_force_series(grouped)
        # Every series the chooser passed over, not only one whose column names
        # happen to repeat in a later case: #143's 13-row `forces/force.dat` lost to
        # a one-row `bladeF/force.dat` in the same case and was never mentioned.
        for passed_over in sorted(set(grouped) - {key}):
            notes.append(f"{case.name}/{passed_over} is not compared: "
                         f"{case.name}/{key} was chosen for this case")
        history = results.read_history(grouped[key])
        taken, said = 0, len(notes)
        for name in results.plot_columns(history["columns"]):
            values = results.column(history, name)
            if not values.size:
                continue
            if name in series:
                notes.append(f"{name} is also in {case.name}/{key}, which is not compared")
                continue
            if values.size == 1:
                notes.append(f"{name} in {case.name}/{key} has one data row, so its "
                             "mean is that row rather than a mean")
            taken += 1
            series[name] = {
                "last": float(values[-1]),
                "tail_mean": float(results.tail_mean(values)),
                "rows": int(values.size),
                "series": key,
                "case": case.name,
            }
        if not taken and len(notes) == said:
            # Unless the loop above already said why, column by column.
            notes.append(f"{case.name}/{key} contributed nothing that is compared")
        notes.extend(history["ignored"])
        notes.extend(history["superseded"])
    mixed = sorted({row["case"] for row in series.values()})
    if len(mixed) > 1:
        notes.append("these numbers are not all from one case -- "
                     + "; ".join(f"{case}: " + ", ".join(
                         name for name, row in sorted(series.items()) if row["case"] == case)
                       for case in mixed)
                     + " -- so the case column, not the table, says what each one is of")
    return {"series": series, "notes": notes, "unavailable": ""}


def convergence(root: Path) -> dict[str, Any]:
    """How the run ended, and what each field's residuals were doing when it did."""
    try:
        # `results` too, because `solver_log` below runs its probe and the two scripts
        # want a dependency each (numpy, matplotlib); either one absent is the same
        # answer and is better given here than raised halfway through the section.
        log_digest, _results = _toolbox("log_digest", "results")
    except ImportError as exc:
        return {"unavailable": f"{exc.name}: {NUMERIC_HINT}", "ended": [], "fields": {}}
    log = solver_log(root)
    if log is None:
        return {"unavailable": "no solver log in the mirror", "ended": [], "fields": {}}
    data = log_digest.digest(log)
    shape = log_digest.residual_shape(data.get("residuals") or {})
    fields = {
        field: {
            "shape": kind,
            "last": shape["last"].get(field),
            "best": shape["best"].get(field),
            "level": shape["level"].get(field),
            "since": shape["since"].get(field),
        }
        for field, kind in shape["fields"].items()
    }
    return {
        "log": log.name,
        "ended": log_digest.how_it_ended(data, log_digest.requested_end_time(log)),
        "shape": shape["shape"],
        "steps": len(data.get("times") or []),
        "fields": fields,
        "unavailable": "",
    }


MESH_METRICS = (
    ("cells", lambda d: d["counts"].get("cells")),
    ("faces", lambda d: d["counts"].get("faces")),
    ("points", lambda d: d["counts"].get("points")),
    ("hexahedra", lambda d: d["counts"].get("hexahedra")),
    ("tetrahedra", lambda d: d["counts"].get("tetrahedra")),
    ("non-orthogonality max/avg", lambda d: d.get("non_ortho")),
    ("max skewness", lambda d: d.get("skewness")),
    ("max aspect ratio", lambda d: d.get("aspect_ratio")),
    ("max cell openness", lambda d: d.get("openness")),
    ("min face determinant", lambda d: d.get("face_determinant")),
    ("min/max cell volume", lambda d: d.get("cell_volume")),
)
"""The `checkMesh` numbers worth putting next to each other, in `mesh_digest`'s own
terms. A metric the log does not carry is absent from the study's dict and reported
as missing rather than as a zero."""


def mesh(root: Path) -> dict[str, Any]:
    """`checkMesh`'s metrics, its failures and its one-star advisories."""
    mesh_digest, = _toolbox("mesh_digest")
    log = mesh_log(root)
    if log is None:
        return {"unavailable": "no checkMesh log in the mirror", "metrics": {}}
    data = mesh_digest.parse(log.read_text(encoding="utf-8", errors="replace"))
    metrics: dict[str, Any] = {}
    for label, pick in MESH_METRICS:
        value = pick(data)
        if value is not None:
            metrics[label] = list(value) if isinstance(value, tuple) else value
    return {
        "log": log.name,
        "metrics": metrics,
        "failures": data.get("failures") or [],
        "warnings": data.get("warnings") or [],
        "unavailable": "",
    }


def stamped_claims(root: Path) -> list[dict[str, Any]]:
    """Every claim this study carries, from the manifest and from the PNGs themselves.

    `claims.collect` over each state root, and then the images again over the whole
    mirror -- `renders/` sits beside `files/` rather than under the study root the
    manifest knows about, and a figure fetched home is still stamped.
    """
    claims, study_state = _toolbox("claims", "study_state")
    found: dict[tuple[str, str], dict[str, Any]] = {}

    def take(figure: Path, rows: Any) -> None:
        for row in rows or []:
            if not isinstance(row, dict) or "value" not in row:
                continue
            entry = dict(row)
            entry["figure"] = figure.name
            found[(figure.name, str(entry.get("name", "")))] = entry

    for state in state_roots(root):
        for row in claims.collect(root=state):
            take(Path(str(row.get("figure") or "")), [row])
    for png in sorted(root.rglob("*.png")):
        if study_state.STATE_DIR in png.parts:
            continue
        take(png, claims.read_png(png))
    return [found[key] for key in sorted(found)]


def artifacts(root: Path) -> dict[str, int]:
    """How many artifacts of each kind the study registered."""
    study_state, = _toolbox("study_state")
    kinds: dict[str, int] = {}
    seen: set[str] = set()
    for state in state_roots(root):
        for row in study_state.artifacts(root=state):
            key = f"{row.get('kind')}\x00{row.get('path')}"
            if key in seen:
                continue
            seen.add(key)
            kinds[str(row.get("kind") or "other")] = kinds.get(str(row.get("kind") or "other"), 0) + 1
    return dict(sorted(kinds.items()))


# -- putting two of them together ----------------------------------------------------


def _pairs(left: dict[str, Any], right: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Every key either side has, in order, with `None` where one has nothing.

    The whole union, because a key on one side only is the answer to the question
    rather than something to leave out of it.
    """
    return [(key, left.get(key), right.get(key))
            for key in sorted(set(left) | set(right))]


def _difference(a: float | None, b: float | None) -> dict[str, Any]:
    """Absolute and relative, the relative one by `claims.apart` -- normalised on the
    larger magnitude, so it is symmetric and does not divide by a near-zero. One rule
    for "how far apart are these" in this package, and it is already written down."""
    claims, = _toolbox("claims")
    if a is None or b is None:
        return {"absolute": None, "relative": None}
    return {"absolute": float(b) - float(a), "relative": claims.apart(float(a), float(b))}


def compare_claims(left: list[dict[str, Any]], right: list[dict[str, Any]],
                   tol: float | None = None) -> list[dict[str, Any]]:
    """Each claim name on either side, with the two values and whether they agree.

    `claims.unit_key` and `claims.apart` and this module's `TOL`, in that order --
    comparability before tolerance, which is `claims.contradictions`'s rule and the
    whole of what it learned: two numbers under different unit labels are not
    converted and not called close, they are reported as not comparable. A tolerance
    applied before the labels were looked at is how 6.32 `h` and 6.30 `m` came back
    as an all-clear.

    `TOL` rather than `claims.DEFAULT_TOL`: two studies' stamped numbers for the same
    named quantity are two measurements of it, which is the `--tol` question, not the
    does-this-plot-label-match-the-answer question 5% was written for.
    """
    claims, = _toolbox("claims")
    tol = TOL if tol is None else tol
    by_name: dict[str, tuple[list[dict], list[dict]]] = {}
    for side, rows in ((0, left), (1, right)):
        for row in rows:
            name = str(row.get("name", ""))
            by_name.setdefault(name, ([], []))[side].append(row)

    out: list[dict[str, Any]] = []
    for name, (mine, theirs) in sorted(by_name.items()):
        if not mine or not theirs:
            held = (mine or theirs)[0]
            out.append({
                "name": name,
                "missing": "b" if not theirs else "a",
                "comparable": False,
                "agree": None,
                "a": _claim_row(mine[0]) if mine else None,
                "b": _claim_row(theirs[0]) if theirs else None,
                "apart": None,
                "units": str(held.get("units", "")),
            })
            continue
        for one in mine:
            for other in theirs:
                same = claims.unit_key(one) == claims.unit_key(other)
                gap = claims.apart(float(one["value"]), float(other["value"]))
                out.append({
                    "name": name,
                    "missing": None,
                    "comparable": same,
                    "agree": (gap <= tol) if same else None,
                    "a": _claim_row(one),
                    "b": _claim_row(other),
                    "apart": gap if same else None,
                })
    return out


def _claim_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "value": float(row["value"]),
        "units": str(row.get("units", "")),
        "symbol": str(row.get("symbol", "")),
        "figure": str(row.get("figure", "")),
        "source": str(row.get("source", "")),
    }


def _scale(rows: list[dict[str, Any]]) -> tuple[str, float]:
    """The biggest coefficient magnitude in this comparison, and whose it is.

    The data's own scale, which is what `NEGLIGIBLE` is a fraction of. Taken over both
    sides and every paired series, so that it is one number for the whole table and
    the same number whichever way round the two studies were named.
    """
    biggest = ("", 0.0)
    for row in rows:
        for side in ("a", "b"):
            held = row[side]
            if not held:
                continue
            size = abs(float(held["tail_mean"]))
            if size == size and size > biggest[1]:  # size == size: not a NaN
                biggest = (row["name"], size)
    return biggest


def _provenance(row: dict[str, Any]) -> str:
    """What has to be said in the same breath as a force difference: the cases it was
    read from when they are not the same case, and a side whose "mean" is one row.

    The case names and not the series names: two studies naming their function objects
    differently (`forceCoeffs` against `forceCoeffs1`) is ordinary and is in the table
    anyway, while a difference whose two sides are two different cases is the thing
    #143 printed as a finding about one.
    """
    bits = []
    a, b = row["a"], row["b"]
    if a["case"] != b["case"]:
        bits.append(f"a from {a['case']}, b from {b['case']}")
    for side in ("a", "b"):
        if row[side]["rows"] == 1:
            bits.append(f"{side} has one data row, so its mean is that row")
    return f" ({'; '.join(bits)})" if bits else ""


def summary(report: dict[str, Any], tol: float | None = None) -> dict[str, Any]:
    """What the report amounts to: how much was compared, what differs, and -- the
    part that has to be in the same sentence -- what could not be compared at all.

    `compared` without `one_sided` and `not_comparable` beside it is the all-clear
    `claims.py` printed over the pair it was built to catch. The three counts travel
    together so that "no differences" can never be said on its own.

    `tol` and `negligible` travel with them for the same reason, from the other side.
    A verdict is only as good as the threshold behind it, so the threshold is in the
    answer; and a quantity put under the floor is listed with the numbers that put it
    there, because "treated as zero" is a decision this made and not a fact about the
    study (#141).
    """
    tol = TOL if tol is None else tol
    compared = 0
    differ: list[str] = []
    one_sided: list[str] = []
    not_comparable: list[str] = []
    negligible: list[str] = []

    def only(which: str, what: str) -> str:
        return f"{what} only in {'a' if which == 'b' else 'b'}"

    big_name, scale = _scale(report["forces"]["series"])
    for row in report["forces"]["series"]:
        if row["missing"]:
            one_sided.append(only(row["missing"], f"force series {row['name']}"))
            continue
        compared += 1
        relative = row["tail_mean"]["relative"]
        sides = [float(row[side]["tail_mean"]) for side in ("a", "b")]
        if scale and max(abs(value) for value in sides) <= NEGLIGIBLE * scale:
            negligible.append(
                f"{row['name']}: {sides[0]:.3g} against {sides[1]:.3g}, both under "
                f"{NEGLIGIBLE:g} of {big_name} = {scale:.3g}, so both are zero here "
                f"and this is not treated as a difference")
            continue
        if relative is not None and relative > tol:
            differ.append(
                f"{row['name']} by {relative * 100:.1f}% on the tail mean "
                f"({sides[0]:.5g} against {sides[1]:.5g})" + _provenance(row))

    for row in report["convergence"]["fields"]:
        if row["missing"]:
            one_sided.append(only(row["missing"], f"residuals for {row['field']}"))
            continue
        compared += 1
        if row["a"]["shape"] != row["b"]["shape"]:
            differ.append(f"{row['field']} residuals {row['a']['shape']} against "
                          f"{row['b']['shape']}")

    def plain(value: Any) -> str:
        """A metric as it is printed, not as it reprs: `mesh_digest` gives the paired
        numbers (max and average non-orthogonality) as a pair, and a list's repr in a
        sentence about two meshes reads as a bug."""
        return " / ".join(str(item) for item in value) if isinstance(value, list) else str(value)

    for row in report["mesh"]["metrics"]:
        if row["missing"]:
            one_sided.append(only(row["missing"], f"checkMesh {row['metric']}"))
            continue
        compared += 1
        if row["a"] != row["b"]:
            differ.append(f"checkMesh {row['metric']}: {plain(row['a'])} "
                          f"against {plain(row['b'])}")

    for row in report["claims"]:
        if row["missing"]:
            one_sided.append(only(row["missing"], f"claim {row['name']}"))
        elif not row["comparable"]:
            not_comparable.append(
                f"claim {row['name']}: {row['a']['units'] or 'no unit'} against "
                f"{row['b']['units'] or 'no unit'}, so the values were not converted")
        else:
            compared += 1
            if not row["agree"]:
                differ.append(f"claim {row['name']} by {row['apart'] * 100:.1f}%")

    for row in report["artifacts"]:
        if row["missing"]:
            one_sided.append(only(row["missing"], f"{row['kind']} artifacts"))

    for section in ("forces", "convergence", "mesh"):
        for side, why in report[section]["unavailable"].items():
            if why:
                not_comparable.append(f"{section} for {side}: {why}")

    return {
        "compared": compared,
        "tol": tol,
        "differ": differ,
        "negligible": negligible,
        "one_sided": one_sided,
        "not_comparable": not_comparable,
    }


def compare(studies_dir: Path, left: str, right: str,
            tol: float | None = None) -> dict[str, Any]:
    """Everything two studies recorded, side by side, as one object.

    `a` is the first id given and `b` the second throughout, and a difference is
    always b - a. `tol` is what counts as a difference, `TOL` unless `--tol` said
    otherwise, and it comes back inside `summary` so that a reader of the report knows
    what the verdict was measured against.
    """
    tol = TOL if tol is None else tol
    studies_dir = Path(studies_dir)
    roots = {side: studies_dir / study for side, study in (("a", left), ("b", right))}
    # Before `identity`, which opens a `Store` -- and a Store creates the directory it
    # is pointed at, so a mistyped id would be answered with an empty study rather
    # than with a complaint, and would leave a directory behind to confuse the next
    # `studies` listing.
    for root in roots.values():
        if not (root / "session.json").is_file():
            raise FileNotFoundError(str(root))
    mirrors = {side: (root / "files" if (root / "files").is_dir() else root)
               for side, root in roots.items()}

    force = {side: forces(root) for side, root in mirrors.items()}
    series = []
    for name, mine, theirs in _pairs(force["a"]["series"], force["b"]["series"]):
        series.append({
            "name": name,
            "a": mine,
            "b": theirs,
            "missing": None if (mine and theirs) else ("b" if mine else "a"),
            "last": _difference(mine and mine["last"], theirs and theirs["last"]),
            "tail_mean": _difference(mine and mine["tail_mean"],
                                     theirs and theirs["tail_mean"]),
        })

    runs = {side: convergence(root) for side, root in mirrors.items()}
    residuals = [
        {"field": field, "a": mine, "b": theirs,
         "missing": None if (mine and theirs) else ("b" if mine else "a")}
        for field, mine, theirs in _pairs(runs["a"]["fields"], runs["b"]["fields"])
    ]

    meshes = {side: mesh(root) for side, root in mirrors.items()}
    declared = [label for label, _pick in MESH_METRICS]
    metrics = sorted(
        ({"metric": label, "a": mine, "b": theirs,
          "missing": None if (mine is not None and theirs is not None)
                     else ("b" if mine is not None else "a")}
         for label, mine, theirs in _pairs(meshes["a"]["metrics"], meshes["b"]["metrics"])),
        # In `MESH_METRICS` order -- cells first, which is what anybody comparing two
        # meshes looks at -- rather than alphabetically, which puts aspect ratio there.
        key=lambda row: (declared.index(row["metric"]) if row["metric"] in declared
                         else len(declared), row["metric"]),
    )

    kinds = {side: artifacts(root) for side, root in mirrors.items()}
    produced = [
        {"kind": kind, "a": mine or 0, "b": theirs or 0,
         "missing": None if (mine and theirs) else ("b" if mine else "a")}
        for kind, mine, theirs in _pairs(kinds["a"], kinds["b"])
    ]

    report = {
        "studies": {side: identity(studies_dir, study)
                    for side, study in (("a", left), ("b", right))},
        "forces": {
            "series": series,
            "unavailable": {side: force[side]["unavailable"] for side in ("a", "b")},
            "notes": {side: force[side]["notes"] for side in ("a", "b")},
        },
        "convergence": {
            "ended": {side: runs[side]["ended"] for side in ("a", "b")},
            "shape": {side: runs[side].get("shape", "") for side in ("a", "b")},
            "steps": {side: runs[side].get("steps", 0) for side in ("a", "b")},
            "fields": residuals,
            "unavailable": {side: runs[side]["unavailable"] for side in ("a", "b")},
        },
        "mesh": {
            "metrics": metrics,
            "failures": {side: meshes[side].get("failures", []) for side in ("a", "b")},
            "warnings": {side: meshes[side].get("warnings", []) for side in ("a", "b")},
            "unavailable": {side: meshes[side]["unavailable"] for side in ("a", "b")},
        },
        "claims": compare_claims(stamped_claims(mirrors["a"]),
                                 stamped_claims(mirrors["b"]), tol),
        "artifacts": produced,
    }
    report["summary"] = summary(report, tol)
    return report
