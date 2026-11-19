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
import sys
from pathlib import Path
from typing import Any

from .store import Store

TOOLBOX = Path(__file__).resolve().parent / "toolbox"

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
    """Every case directory in the mirror: one with a `postProcessing/` under it."""
    return sorted({path.parent for path in root.rglob("postProcessing") if path.is_dir()})


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
    object they were written under usually is not.
    """
    try:
        results, = _toolbox("results")
    except ImportError as exc:
        return {"unavailable": f"{exc.name}: {NUMERIC_HINT}", "series": {}, "notes": []}
    series: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    for case in cases(root):
        grouped = results.find_force_files(case)
        if not grouped:
            continue
        key = results.choose_force_series(grouped)
        history = results.read_history(grouped[key])
        for name in results.plot_columns(history["columns"]):
            values = results.column(history, name)
            if not values.size:
                continue
            if name in series:
                notes.append(f"{name} is also in {case.name}/{key}, which is not compared")
                continue
            series[name] = {
                "last": float(values[-1]),
                "tail_mean": float(results.tail_mean(values)),
                "rows": int(values.size),
                "series": key,
                "case": case.name,
            }
        notes.extend(history["ignored"])
        notes.extend(history["superseded"])
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

    `claims.unit_key` and `claims.apart` and `claims.DEFAULT_TOL`, in that order --
    comparability before tolerance, which is `claims.contradictions`'s rule and the
    whole of what it learned: two numbers under different unit labels are not
    converted and not called close, they are reported as not comparable. A tolerance
    applied before the labels were looked at is how 6.32 `h` and 6.30 `m` came back
    as an all-clear.
    """
    claims, = _toolbox("claims")
    tol = claims.DEFAULT_TOL if tol is None else tol
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


def summary(report: dict[str, Any], tol: float | None = None) -> dict[str, Any]:
    """What the report amounts to: how much was compared, what differs, and -- the
    part that has to be in the same sentence -- what could not be compared at all.

    `compared` without `one_sided` and `not_comparable` beside it is the all-clear
    `claims.py` printed over the pair it was built to catch. The three counts travel
    together so that "no differences" can never be said on its own.
    """
    claims, = _toolbox("claims")
    tol = claims.DEFAULT_TOL if tol is None else tol
    compared = 0
    differ: list[str] = []
    one_sided: list[str] = []
    not_comparable: list[str] = []

    def only(which: str, what: str) -> str:
        return f"{what} only in {'a' if which == 'b' else 'b'}"

    for row in report["forces"]["series"]:
        if row["missing"]:
            one_sided.append(only(row["missing"], f"force series {row['name']}"))
            continue
        compared += 1
        relative = row["tail_mean"]["relative"]
        if relative is not None and relative > tol:
            differ.append(f"{row['name']} by {relative * 100:.1f}% on the tail mean")

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
        "differ": differ,
        "one_sided": one_sided,
        "not_comparable": not_comparable,
    }


def compare(studies_dir: Path, left: str, right: str,
            tol: float | None = None) -> dict[str, Any]:
    """Everything two studies recorded, side by side, as one object.

    `a` is the first id given and `b` the second throughout, and a difference is
    always b - a.
    """
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
