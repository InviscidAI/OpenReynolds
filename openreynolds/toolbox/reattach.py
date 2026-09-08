#!/usr/bin/env python3
"""Where the flow leaves a wall and where it comes back, measured once.

The backward-facing-step study computed its reattachment length twice, with two
scripts it wrote itself, and got two answers. `reattach.py` found the wall-shear
sign changes at 0.21, 6.32, 7.86 and 9.97 and read x_r/h = 6.32, which is right:
0.21 is the corner eddy, 6.32 closes the primary bubble, and it sits 4% above
Gartling's 6.1. `analyze.py` -- the one that drew the delivered figure -- read
7.5667, 24% above the benchmark, and annotated the picture with it. Both scripts
existed because there was no reattachment helper here to reuse, so the study wrote
the measurement twice and had no way to tell which one it should have believed.

This is that helper. There is one implementation, it says what it did, and it
carries its own number into the figure it draws (`claims.py`), so the next step can
put the figure and the answer side by side instead of hoping they match.

what it measures

Sign changes of the streamwise wall shear along a wall patch. That much both hand-
written scripts did. The part `analyze.py` got wrong is what to do with the four of
them, and it is the part written down here:

- The crossings themselves are convention-free. Whether OpenFOAM's
  `wallShearStress` comes out positive or negative for forward flow depends on the
  sign convention of the function object, and nothing here needs to know it: which
  sign means ATTACHED is measured, off the velocity field, from two facts that are
  read rather than assumed -- that tau tracks the near-wall velocity along a wall
  up to one constant sign, and that the mean velocity over the whole domain says
  which way downstream is. With no velocity to read (`--from-file`, an unwritten
  `U`) it falls back to the sign covering more of the wall, which is right when
  most of a wall is attached and says so when most of it is not.
  `--forward-sign` settles it by hand. The report always names the basis used and
  whether the downstream end of the patch agrees with it.
- The PRIMARY reattachment is the downstream end of the LONGEST reversed-flow run.
  Not the first sign change, which on a backward-facing step is the corner eddy
  (0.21); not the last, which is the far end of a secondary bubble (9.97). The
  report prints those two naive answers next to the primary one, precisely so that
  a number computed some other way can be checked against them rather than merely
  disagreed with. "Longest" is a rule and not a law: on a geometry whose downstream
  separation is the longer one -- a diffuser, a stalled aerofoil with a long
  trailing bubble -- it names that one, and the report says so whenever there is
  more than one reversed region for it to have chosen between.
- A bubble still reversed at the last station on the patch has not reattached
  anywhere the mesh can see, and no length is reported for it. A wall long enough
  to hold the number is part of the measurement.

The location's resolution is the spacing of the wall faces around the crossing, and
it is reported with the number, because a reattachment interpolated between two
faces half a step height apart is not a four-figure answer.

where the shear comes from

`<time>/wallShearStress` for the named patch, paired face-for-face with the patch
face centres out of `constant/polyMesh` -- so a case that ran the `wallShearStress`
function object needs nothing else. On a 3D patch the faces are grouped into
stations of constant streamwise coordinate and averaged across the span, and the
report says how many stations carry both signs across the span, because a
reattachment line that is not straight is not one number.

`--from-file` reads a two-column table instead -- a sampled `.raw` surface, or
whatever a script already wrote -- so a number produced another way can be run
through the same arithmetic rather than argued with.

what it says the number is in

`--height` divides, `--origin` shifts, and NEITHER tells this script what unit the
mesh is in: it has no way to know, so it does not say. With a `--height` OTHER THAN 1
the answer is in step heights and is labelled `h`; without one it is in whatever the
wall's own coordinate was, and the label is empty rather than guessed. `--height 1`
counts as without one: 1.0 is the flag's argparse default, only the number reaches
the labelling, and dividing a coordinate by 1 does not make it dimensionless. A table
that is already in x/h gets its label from `--units h`, which is the flag for saying
what this cannot read. That matters past pedantry because `claims.py` will only
compare two numbers carrying the same label, so a label invented here switches off
the comparison over there -- which is what the first version of this file did,
stamping "m" onto a table that was already in x/h.

It runs no solver, edits nothing that was already in the case, and exits 0 whatever
it finds; a wall with no separation on it is an answer. The only files it creates are
the `--plot` figure and, when the study keeps a `.reynolds/`, that figure's line in
the manifest -- both of which land inside the case directory when that is where you
pointed it.

    python3 reattach.py /work/case --patch lowerWall --height 0.0094
    python3 reattach.py /work/case --patch floor --origin 0.02 --height 0.0094 --plot bfs.png
    python3 reattach.py --from-file wss.raw --columns 0 3 --units h
    python3 reattach.py /work/case --patch lowerWall --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claims  # noqa: E402  (sibling scripts, not a package)
import layer_report  # noqa: E402
import locate  # noqa: E402

AXES = {"x": 0, "y": 1, "z": 2}

STATION_TOL = 1e-6
"""Two faces are at the same streamwise station when their coordinates differ by
less than this fraction of the patch's streamwise extent. On a structured wall the
difference is exactly zero; on a snappy wall it is float noise, and the gap between
genuine stations is larger than this by many orders."""

CORNER_FRACTION = 0.25
"""A structure at the upstream end of the wall shorter than this much of the primary
bubble is named a corner eddy rather than a bubble of its own. On the step the two
were 0.21 and 6.11 -- a ratio of 0.034 -- so a quarter is a wide margin around a
distinction that is not close in practice, and it is a name in the report rather
than anything the primary reattachment depends on."""

CLAIM_NAME = "reattachment-length"

CLAIM_SYMBOL = "x_r"
"""How this quantity is written -- in the report, in the caption drawn on the figure,
and in the stamp -- and it is deliberately not `x_r/h`.

A claim carries `symbol` and `units` as two fields and `claims.caption` prints both,
so a symbol of `x_r/h` beside a unit of `h` renders `x_r/h = 6.32 h`: a ratio that has
already divided by the step height, with the step height re-attached as its unit.
Measured, that is what shipped -- one run printed `x_r = 6.32 h` in the report and
stamped `x_r/h = 6.32 h` into the figure, two symbols for one number and the stamped
one self-contradictory. The unit label is where "in step heights" is said, because the
unit label is what `claims.py` compares on; the symbol says which quantity it is.

`x_r/h` stays in the claim's aliases, so an answer that writes the ratio -- which is
how every benchmark for this case is written -- is still found in the prose search."""


class Unmeasurable(Exception):
    """The wall shear could not be read. Not the same as a wall with no separation."""


# -- reading the wall shear off a case ----------------------------------------------


def field_times(case: Path, field: str) -> list[str]:
    """Time directory names that hold `field`, in numeric order.

    Both spellings, plain and gzipped: a case written `writeCompression on` keeps
    `wallShearStress.gz`, and refusing that would price the measurement out of
    exactly the long runs that compress their output.
    """
    out: list[tuple[float, str]] = []
    try:
        children = list(case.iterdir())
    except OSError:
        return []
    for child in children:
        if not child.is_dir() or not locate.is_time_name(child.name):
            continue
        if (child / field).is_file() or (child / f"{field}.gz").is_file():
            out.append((float(child.name), child.name))
    return [name for _, name in sorted(out)]


def patch_block(text: str, patch: str, names: list[str], start: int) -> tuple[int, int] | None:
    """Where one patch's entry sits inside `boundaryField`, as `(from, to)`.

    Bounded by the next patch's name rather than by brace counting, because a
    `writeFormat binary` field carries arbitrary bytes between the braces and a
    counter walking through them is counting noise. The names come from
    `constant/polyMesh/boundary`, so the bound is the mesh's own list.

    Not tested against a real binary field, and the failure it would have is
    nameable: a byte run inside a blob that happens to spell another patch's name
    followed by `{` would end this block early. Every fixture here is ascii. If a
    binary case ever comes back short, that is the first thing to look at.
    """
    def where(name: str) -> int:
        found = re.search(
            rf"(?<![\w.\-]){re.escape(name)}\s*[\r\n\s]*\{{", text[start:], re.S
        )
        return start + found.start() if found else -1

    here = where(patch)
    if here < 0:
        return None
    later = [pos for other in names if other != patch for pos in (where(other),) if pos > here]
    return here, min(later) if later else len(text)


_NONUNIFORM = re.compile(r"value\s+nonuniform\s+List<vector>")
_UNIFORM = re.compile(
    r"value\s+uniform\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)"
)


def read_patch_vectors(path: Path, patch: str, names: list[str], n_faces: int) -> np.ndarray:
    """The per-face vector this field carries on `patch`, `(nFaces, 3)`.

    The values are in the patch's own face order, which is the order
    `startFace .. startFace + nFaces` addresses in the mesh, so they pair with the
    face centres without a lookup.
    """
    fobj = locate.foam_file(path)
    if fobj is None:
        raise Unmeasurable(f"cannot read {path.as_posix()}")
    at = fobj.text.find("boundaryField", fobj.body_at)
    if at < 0:
        raise Unmeasurable(f"{path.as_posix()} has no boundaryField")
    span = patch_block(fobj.text, patch, names, at)
    if span is None:
        raise Unmeasurable(
            f"{path.as_posix()} has no entry named '{patch}' under boundaryField -- a "
            "group entry such as \".*\" carries no per-face values, so sample the patch "
            "to a raw surface and pass --from-file"
        )
    begin, end = span
    match = _NONUNIFORM.search(fobj.text, begin, end)
    if match:
        flat, _ = locate.read_list(fobj, match.end(), "scalar", per_item=3)
        values = np.asarray(flat, dtype=np.float64)
        if values.size != n_faces * 3:
            raise Unmeasurable(
                f"'{patch}' has {n_faces} faces but its value list holds "
                f"{values.size // 3} vectors"
            )
        return values.reshape(-1, 3)
    match = _UNIFORM.search(fobj.text, begin, end)
    if match:
        # A uniform value on a wall is the field before anything was computed on it;
        # it has no sign change anywhere and saying so beats reporting "no separation".
        raise Unmeasurable(
            f"'{patch}' carries a uniform value in {path.name} -- the function object "
            "has not written per-face data for this patch yet"
        )
    raise Unmeasurable(f"no readable value entry for '{patch}' in {path.as_posix()}")


BULK_FLOW_FRACTION = 0.05
"""How much of the velocity magnitude has to be net streamwise before the domain
mean is taken as saying which way downstream is. Below it -- a closed cavity, a
case at rest -- there is no bulk direction to read and the question is handed back
rather than answered from noise."""

WEAK_AGREEMENT = 0.65
"""Below this share of faces where tau and the near-wall velocity have the same
sign, the correlation the sign relation rests on is called weak in the report. Half
is a coin toss; on a wall where the relation holds it is near 1."""


def forward_sign_from_velocity(
    case: Path, mesh: dict, faces: np.ndarray, tau: np.ndarray, axis: int, time: str,
) -> tuple[int, str] | None:
    """Which sign of tau means attached flow, measured off the velocity field.

    Two facts, both read rather than assumed, and neither of them the sign
    convention of the `wallShearStress` function object:

    - Along one wall, tau and the near-wall velocity are the same quantity up to a
      constant of one sign or the other (tau ~ +/- nu u/h), so the sign of their
      product summed over the patch is that constant. It is a strong correlation:
      every face votes, and the faces inside the bubble vote with the rest.
    - Which way downstream is comes from the mean streamwise velocity over the WHOLE
      internal field, not from the wall. That distinction is the whole point of
      reading it here: on a wall truncated before the flow reattaches, most of the
      wall is reversed and anything that asks the wall gets the answer backwards.

    None when there is no readable velocity, or when the domain has no net
    streamwise flow to point with.
    """
    data = locate.read_internal_field(case / time / "U")
    if data is None or data.get("kind") != "nonuniform" or data.get("components") != 3:
        return None
    values = np.asarray(data["values"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        return None
    component = values[:, axis]
    speed = float(np.sqrt((values ** 2).sum(axis=1).mean()))
    mean = float(component.mean())
    if not speed or abs(mean) < BULK_FLOW_FRACTION * speed:
        return None
    owners = np.asarray(mesh["owner"], dtype=np.int64)[faces]
    if owners.size == 0 or int(owners.max()) >= component.size:
        return None
    near = component[owners]
    together = float(np.sum(near * np.asarray(tau, dtype=np.float64)))
    if together == 0.0:
        return None
    convention = 1 if together > 0 else -1
    downstream = 1 if mean > 0 else -1
    agree = float(np.mean(np.sign(near) == np.sign(np.asarray(tau) * convention)))
    # The vote is a sum of `u * tau` and is NOT area-weighted -- face size never
    # enters it -- so a handful of faces carrying large |u.tau| can decide it while
    # most faces disagree. Around half the faces agreeing, or fewer, means tau and
    # the near-wall velocity are barely correlated -- a wall the flow crosses, or a
    # `U` from a different time -- and the answer is then a coin toss dressed as a
    # measurement, so it says so. The fraction itself is printed beside the words,
    # which is why the words have to cover the whole range below the threshold and
    # not just the near-half case.
    #
    # A face whose near-wall U reads exactly 0.0 counts as a disagreement here
    # (`np.sign(0)` is 0, which equals neither +1 nor -1), so a patch with stagnant
    # faces on it deflates the fraction toward the warning. That is the safe
    # direction -- it over-warns rather than under-warns -- and worth knowing before
    # anyone tunes WEAK_AGREEMENT.
    weak = (
        " -- weakly correlated, so this basis is weak and --forward-sign is worth giving"
        if agree < WEAK_AGREEMENT else ""
    )
    return (
        convention * downstream,
        f"U behind the wall, which tracks tau at {agree:.0%} of the faces{weak}, "
        f"and a domain mean flowing {'+' if downstream > 0 else '-'}{'xyz'[axis]}",
    )


def stations(x: np.ndarray, tau: np.ndarray) -> dict[str, Any]:
    """Faces grouped into stations of constant streamwise coordinate.

    A 2D case one cell thick has two faces per station and a 3D floor has a whole
    span of them; averaging across the span is the only way to get one curve, and
    the count of stations whose span is not unanimous in sign is reported with it,
    because a reattachment line that wanders in the span is not the single number
    this returns.
    """
    order = np.argsort(x, kind="stable")
    x, tau = np.asarray(x)[order], np.asarray(tau)[order]
    extent = float(x[-1] - x[0]) if x.size else 0.0
    tol = max(abs(extent) * STATION_TOL, 1e-30)
    edges = np.flatnonzero(np.diff(x) > tol) + 1
    groups = np.split(np.arange(x.size), edges)
    xs = np.array([float(x[group].mean()) for group in groups])
    taus = np.array([float(tau[group].mean()) for group in groups])
    mixed = sum(
        1 for group in groups
        if group.size > 1 and tau[group].min() < 0.0 < tau[group].max()
    )
    return {
        "x": xs,
        "tau": taus,
        "faces": int(x.size),
        "stations": int(xs.size),
        "mixed_sign_stations": int(mixed),
    }


def wall_profile(
    case: Path, patch: str, *, direction: str = "x", time: str | None = None,
    field: str = "wallShearStress",
) -> dict[str, Any]:
    """The streamwise wall shear along one patch, station by station."""
    mesh = locate.read_mesh(case / "constant" / "polyMesh")
    if isinstance(mesh, str):
        raise Unmeasurable(mesh)
    entry = mesh["boundary"].get(patch)
    if entry is None:
        have = ", ".join(sorted(mesh["boundary"])) or "none"
        raise Unmeasurable(f"no patch '{patch}' in this mesh (patches: {have})")
    if not entry["nFaces"]:
        raise Unmeasurable(f"patch '{patch}' has no faces")

    available = field_times(case, field)
    if not available:
        raise Unmeasurable(
            f"no time directory holds {field} -- add the wallShearStress function "
            "object and re-run, or write it once with `postProcess -func wallShearStress`"
        )
    chosen = time if time in available else available[-1]
    if time is not None and time not in available:
        raise Unmeasurable(
            f"no {field} at time {time} (have: {', '.join(available)})"
        )
    source = case / chosen / field
    if not source.is_file():
        source = case / chosen / f"{field}.gz"

    faces = np.arange(entry["startFace"], entry["startFace"] + entry["nFaces"])
    centres, _ = layer_report.face_geometry(mesh, faces)
    vectors = read_patch_vectors(source, patch, list(mesh["boundary"]), entry["nFaces"])
    axis = AXES[direction]
    profile = stations(centres[:, axis], vectors[:, axis])
    profile.update({"case": str(case), "patch": patch, "time": chosen,
                    "field": field, "direction": direction, "source": str(source),
                    "forward": None, "forward_basis": ""})
    measured = forward_sign_from_velocity(case, mesh, faces, vectors[:, axis], axis, chosen)
    if measured is not None:
        profile["forward"], profile["forward_basis"] = measured
    return profile


def profile_from_file(path: Path, columns: tuple[int, int]) -> dict[str, Any]:
    """A two-column table read as `(x, tau)`.

    Column defaults 0 and 3 are the layout of an OpenFOAM `.raw` sampled surface --
    `x y z v_x v_y v_z` -- which is the other route to wall shear on a patch, and
    the one a study takes when the field itself was not written. `#` comments and
    blank lines are skipped; commas count as separators, so a CSV works too.
    """
    rows: list[tuple[float, float]] = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.split("#", 1)[0].strip().replace(",", " ")
        if not line:
            continue
        parts = line.split()
        if max(columns) >= len(parts):
            continue
        try:
            rows.append((float(parts[columns[0]]), float(parts[columns[1]])))
        except ValueError:
            continue  # a header line, not a fault
    if len(rows) < 2:
        raise Unmeasurable(
            f"{path.as_posix()}: fewer than two usable rows at columns {columns}"
        )
    data = np.asarray(rows)
    profile = stations(data[:, 0], data[:, 1])
    # `forward`/`forward_basis` are declared here as well as in `wall_profile`, and
    # always None/"" because a bare table carries no velocity to measure the sign
    # off. Without them the `--json` payload's `profile` object grew and lost keys
    # between a case run and a `--from-file` run -- the same KeyError the `result`
    # object below is careful not to hand its consumer, one level up.
    profile.update({"case": "", "patch": str(path), "time": "", "field": "column table",
                    "direction": "x", "source": str(path),
                    "forward": None, "forward_basis": ""})
    return profile


# -- the arithmetic, which is the part that was got wrong ---------------------------


def crossings(x: np.ndarray, tau: np.ndarray) -> list[dict[str, Any]]:
    """Every zero crossing of `tau(x)`, located by linear interpolation.

    Stations where tau is exactly zero are stepped over rather than counted as
    crossings: a face that reads zero to machine precision is between two signs,
    not a structure, and counting it would have put a fifth "sign change" in a list
    of four.
    """
    x, tau = np.asarray(x, float), np.asarray(tau, float)
    keep = np.flatnonzero(tau != 0.0)
    x, tau = x[keep], tau[keep]
    out: list[dict[str, Any]] = []
    for i in range(x.size - 1):
        if (tau[i] > 0.0) == (tau[i + 1] > 0.0):
            continue
        span = tau[i + 1] - tau[i]
        where = x[i] if span == 0.0 else x[i] + (x[i + 1] - x[i]) * (-tau[i] / span)
        out.append({
            "x": float(where),
            "from": 1 if tau[i] > 0 else -1,
            "to": 1 if tau[i + 1] > 0 else -1,
            # The two faces the answer sits between. Reporting the number without
            # this is reporting four figures off a wall sampled to two.
            "resolution": float(abs(x[i + 1] - x[i])),
        })
    return out


def forward_sign_of(x: np.ndarray, tau: np.ndarray) -> tuple[int, str]:
    """Which sign of tau means attached flow, decided from tau alone.

    The fallback, used when there is no velocity field to measure it off --
    `--from-file`, or a case whose `U` was never written. Most of a wall is attached,
    so attached is the sign that covers more of it, measured by length along the
    wall rather than by station count, so a wall refined inside the bubble does not
    vote itself separated.

    It has one failure and `measure` names it: a wall truncated before its flow
    reattaches is mostly bubble, and the majority is then the bubble.
    `forward_sign_from_velocity` is the answer to that and is tried first.
    """
    x, tau = np.asarray(x, float), np.asarray(tau, float)
    if x.size < 2:
        return (1 if tau.size and tau[0] >= 0 else -1), "a single station"
    width = np.gradient(x)
    positive = float(width[tau > 0.0].sum())
    negative = float(width[tau < 0.0].sum())
    if positive == negative:
        last = tau[np.flatnonzero(tau != 0.0)[-1]] if np.any(tau != 0.0) else 1.0
        return (1 if last > 0 else -1), "the downstream end (the two signs cover equal wall)"
    sign = 1 if positive > negative else -1
    share = max(positive, negative) / max(positive + negative, 1e-300)
    return sign, f"the sign covering {share:.0%} of the wall"


def runs(x: np.ndarray, tau: np.ndarray, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The wall cut into runs of one sign, bounded by the crossings.

    The first and last runs are marked open, because a run that reaches the end of
    the patch has no measured end: whatever it is doing, it is doing it past where
    the mesh can see.
    """
    x, tau = np.asarray(x, float), np.asarray(tau, float)
    if x.size == 0:
        return []
    edges = [float(x[0])] + [point["x"] for point in points] + [float(x[-1])]
    out: list[dict[str, Any]] = []
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        inside = tau[(x >= start) & (x <= end)]
        inside = inside[inside != 0.0]
        if inside.size == 0:
            continue
        out.append({
            "sign": 1 if inside[0] > 0 else -1,
            "start": start,
            "end": end,
            "length": float(end - start),
            "open_start": i == 0,
            "open_end": i == len(edges) - 2,
        })
    return out


def measure(
    x: np.ndarray, tau: np.ndarray, *, origin: float = 0.0, height: float = 1.0,
    forward: int | None = None, basis: str = "given on the command line",
) -> dict[str, Any]:
    """Separation and reattachment on this wall, with the workings.

    `origin` is where the wall's own coordinate starts counting -- the step face --
    and `height` the length the answer is quoted in, so the reported `x_r` is
    `(x - origin) / height`. Both default to a no-op, and a report that did not get
    a step height says the number is raw.
    """
    x, tau = np.asarray(x, float), np.asarray(tau, float)
    points = crossings(x, tau)
    guessed = forward is None
    if guessed:
        forward, basis = forward_sign_of(x, tau)
    forward = 1 if forward >= 0 else -1

    def scaled(value: float | None) -> float | None:
        return None if value is None else float((value - origin) / height)

    for point in points:
        point["kind"] = "reattachment" if point["to"] == forward else "separation"
        point["scaled"] = scaled(point["x"])

    spans = runs(x, tau, points)
    for span in spans:
        span["attached"] = span["sign"] == forward
        span["scaled_start"] = scaled(span["start"])
        span["scaled_end"] = scaled(span["end"])
        span["scaled_length"] = float(span["length"] / height)

    separated = [span for span in spans if not span["attached"]]
    result: dict[str, Any] = {
        "forward_sign": forward,
        "forward_basis": basis,
        "crossings": points,
        "runs": spans,
        "origin": float(origin),
        "height": float(height),
        "separated": bool(separated),
        "reattachment": None,
        "reattachment_scaled": None,
        "resolution": None,
        "separation": None,
        "separation_scaled": None,
        "bubble_length_scaled": None,
        # Declared here rather than only on the path that fills them in, because a
        # `--json` payload whose `result` object grows and loses keys between runs
        # hands its consumer a KeyError instead of a null. Every key this function
        # can produce exists on every `result` it returns. The `profile` object
        # beside it in that payload is held to the same rule by its two producers
        # declaring the same keys (`wall_profile`, `profile_from_file`); nothing
        # here enforces that across the two.
        "first_crossing_scaled": None,
        "last_crossing_scaled": None,
        "structures": [],
        "notes": [],
    }

    # The downstream end of the patch is a second, independent read on which sign is
    # attached. When it disagrees with the length vote the number is still reported,
    # but the disagreement is not swallowed -- it is the one case where the basis
    # above is the wrong way round.
    tail = tau[np.flatnonzero(tau != 0.0)[-1]] if np.any(tau != 0.0) else 0.0
    if tail != 0.0 and (1 if tail > 0 else -1) != forward:
        result["notes"].append(
            "the last station on the patch has the sign this reads as REVERSED -- "
            "either the wall ends inside a separation or --forward-sign is the other way"
        )

    if not separated:
        result["notes"].append("no reversed-flow region on this patch: nothing separates")
        # A wall with no sign change on it and no measured forward sign is the one
        # place this reads as "all attached" when it could equally be "all reversed":
        # the length vote picks the majority, and with one sign present the majority
        # IS that sign whichever it means. A wall lying entirely inside a separation
        # is byte-for-byte indistinguishable from a wall the flow never left, and the
        # note is the only thing between that and a confident "nothing separates".
        if guessed and not points:
            result["notes"].append(
                "every station on this wall carries the SAME sign, so the wall shear "
                "alone cannot tell attached from reversed here -- a wall lying entirely "
                "inside a separation reads exactly like this one. --forward-sign settles "
                "it, and a written U settles it without being asked"
            )
        return result

    primary = max(separated, key=lambda span: span["length"])

    # "Longest" is a rule, and on a wall with one bubble on it there is nothing for a
    # rule to get wrong. With two or more it is a choice, and a choice made by a
    # script that never said it was choosing is how the wrong structure gets
    # published: on a diffuser or a stalled aerofoil the downstream separation can be
    # the longer one. So the rule is stated wherever it actually decided something.
    if len(separated) > 1:
        result["notes"].append(
            f"{len(separated)} reversed regions on this wall; the one reported is the "
            "LONGEST, which is the primary bubble on a backward-facing step but is a "
            "rule rather than a law -- on a geometry whose downstream separation is the "
            "longer one it names that instead, and the list below says which is which"
        )

    # A bubble that starts at the first face on the wall is the symptom of the one
    # case the length vote gets wrong. On a wall truncated before the flow
    # reattaches, most of the wall IS the bubble, so the majority is the bubble, the
    # vote calls it attached, and what comes back is the short structure against the
    # step -- the corner eddy, reported as the reattachment, which is exactly the
    # 0.21-for-6.32 substitution F-36 turned on. It can also be a genuine
    # leading-edge bubble, so this is a note and not a refusal; the velocity field
    # settles it properly when there is one (see `forward_sign_from_velocity`).
    span = float(x[-1] - x[0])
    if guessed and primary["open_start"]:
        result["notes"].append(
            f"the reversed region this reads as the primary bubble starts at the first "
            f"face on the wall and covers {primary['length'] / span:.0%} of it -- if the "
            "wall was cut off before the flow reattached, the sign vote is the wrong way "
            "round and this is the corner eddy; --forward-sign settles it, and a written "
            "U settles it without being asked"
        )
    result["structures"] = _name_structures(spans, primary, forward)
    if primary["open_end"]:
        result["notes"].append(
            f"the longest reversed region runs to the end of the patch at "
            f"{primary['scaled_end']:.4g} and does not close on it -- reattachment is "
            "downstream of the mesh, so no length is reported"
        )
        result["separation"] = None if primary["open_start"] else primary["start"]
        result["separation_scaled"] = scaled(result["separation"])
        return result

    at = primary["end"]
    result["reattachment"] = float(at)
    result["reattachment_scaled"] = scaled(at)
    nearest = min(points, key=lambda point: abs(point["x"] - at))
    # Half the spacing, because the crossing was interpolated between those two faces
    # and could have been anywhere between them.
    result["resolution"] = float(0.5 * nearest["resolution"] / height)
    if not primary["open_start"]:
        result["separation"] = float(primary["start"])
        result["separation_scaled"] = scaled(primary["start"])
    result["bubble_length_scaled"] = float(primary["length"] / height)

    # The two answers a script that stopped at "the sign changes here" would give.
    # Printed rather than merely avoided, because the point of this file is that a
    # number arrived at some other way can be checked against the same list.
    result["first_crossing_scaled"] = points[0]["scaled"]
    result["last_crossing_scaled"] = points[-1]["scaled"]
    return result


def _name_structures(
    spans: list[dict[str, Any]], primary: dict[str, Any], forward: int
) -> list[dict[str, Any]]:
    """Every run given the name a person would use for it.

    The corner eddy is the thing `analyze.py` did not have a name for, and not
    having a name for it is how its 0.21 ended up in the same list as the answer.
    On a step it is a short ATTACHED run against the step face -- the counter-
    rotating eddy in the corner drives the near-wall flow downstream again -- which
    is why "the first sign change" is a separation and not the reattachment.
    """
    named: list[dict[str, Any]] = []
    for span in spans:
        if span is primary:
            what = "primary recirculation"
        elif span["attached"] and span["open_start"] and span["length"] < CORNER_FRACTION * primary["length"]:
            what = "corner eddy"
        elif span["attached"]:
            what = "attached"
        elif span["start"] < primary["start"]:
            what = "reversed region upstream of the primary bubble"
        else:
            what = "secondary bubble"
        named.append({**span, "what": what})
    return named


# -- reporting ----------------------------------------------------------------------


def digits_for(resolution: float | None) -> int | None:
    """How many decimals the wall spacing supports, or None when it is not known.

    `analyze.py` wrote 7.5667 for a quantity its own wall could not place better
    than a face apart, and four decimals is its own small claim: it says the number
    is known to a ten-thousandth of a step height. The claim is rounded to what the
    mesh can actually see, and the unrounded value stays in the result for anyone
    who wants it.

    None rather than a generous default when the spacing is zero or not finite: the
    first version returned 6 there, which is a millionth of a step height asserted
    in exactly the case where nothing had established the spacing at all -- the
    file's own rule, inverted. A caller that gets None leaves the value alone and
    says the error bar is missing.
    """
    if resolution is None or not (resolution > 0.0) or not np.isfinite(resolution):
        return None
    return max(0, int(np.ceil(-np.log10(resolution))))


def quoted_units(result: dict[str, Any], override: str | None = None) -> str:
    """The unit label for this measurement, for the report and the claim alike.

    Dividing by a `--height` is what makes the answer dimensionless, so that alone
    decides it; shifting the `--origin` moves where zero is and changes no unit.
    Without a height the number is in whatever the wall's coordinate was, which this
    script has no way to read, so the label is empty -- "not stated" -- rather than
    "m".

    Both the mistakes here were shipped and both mattered. Saying "m" on a table
    already in x/h made `claims.py contradictions` skip the F-36 pair as a unit
    mismatch and print its all-clear over it. Deciding on `height or origin` meant
    `--origin` alone stamped `x_r/h` on a metres-valued number while the report line
    beside it said `m` -- one number, two units, one run, which is F-36's own shape.
    One function now, called by both, so they cannot say different things about the
    unit again. The symbol is the same arrangement one level up: `CLAIM_SYMBOL` is a
    constant both the report and the claim read, because a third version of this
    drift had the report writing `x_r` while the stamp wrote `x_r/h`.

    What this cannot do is tell an explicit `--height 1` from no `--height` at all:
    the flag's argparse default is 1.0 and only the number reaches here, so a run
    given `--height 1` is labelled as an unscaled one. That is the honest reading --
    dividing by 1 does not make a coordinate dimensionless -- and `--units h` is how
    a coordinate that is already in step heights gets its label.
    """
    if override is not None:
        return override
    return "h" if result["height"] != 1.0 else ""


def quoted_value(result: dict[str, Any]) -> float | None:
    """The one number this script quotes, rounded to what the wall can see.

    The report prints this and the claim carries this. They were computed separately
    once, and printed 6.324 next to a figure stamped 6.32 -- 0.06% apart, inside any
    tolerance, and still two numbers for one quantity in one run.
    """
    at = result["reattachment_scaled"]
    if at is None:
        return None
    digits = digits_for(result["resolution"])
    return float(at) if digits is None else round(float(at), digits)


def as_claim(
    profile: dict[str, Any], result: dict[str, Any], units: str | None = None
) -> dict[str, Any] | None:
    """The measurement as a claim a figure can carry, or None when there is none."""
    value = quoted_value(result)
    if value is None:
        return None
    unit = quoted_units(result, units)
    known = digits_for(result["resolution"]) is not None
    return claims.claim(
        CLAIM_NAME,
        value,
        symbol=CLAIM_SYMBOL,
        units=unit,
        # `--from-file` puts the whole table path in `patch`, which is right in the
        # report's heading and absurd inside a one-line claim, and an empty `t=` on
        # a profile that has no time is a field with nothing in it. The source has
        # to survive being read on a figure, so it is kept to what identifies the
        # measurement: the script, the wall, and the time if there was one.
        source=" ".join(part for part in (
            "reattach.py",
            Path(str(profile.get("patch") or "")).name,
            f"at t={profile['time']}" if profile.get("time") else "",
        ) if part),
        note=(
            f"downstream end of the longest reversed-flow run; "
            f"+/-{result['resolution']:.3g} from the wall face spacing"
            if known else
            "downstream end of the longest reversed-flow run; the face spacing there "
            "could not be read, so this value is neither rounded nor bounded"
        ),
        aliases=["reattachment length", "reattachment", "x_r/h", "x/h", "xr/h"],
    )


def report(profile: dict[str, Any], result: dict[str, Any],
           units: str | None = None) -> str:
    unit = quoted_units(result, units)
    shown = f" {unit}" if unit else ""
    lines = [
        f"# reattachment on {profile['patch']}"
        + (f"  ({profile['field']} at t={profile['time']})" if profile.get("time") else ""),
        f"{profile['faces']} faces in {profile['stations']} stations along "
        f"{profile['direction']}; attached is tau {'>' if result['forward_sign'] > 0 else '<'} 0, "
        f"from {result['forward_basis']}",
    ]
    if profile.get("mixed_sign_stations"):
        lines.append(
            f"{profile['mixed_sign_stations']} of {profile['stations']} stations carry BOTH "
            "signs across the span -- the reattachment is a line, not a point, and the "
            "span-averaged number below is one reading of it"
        )
    if result["height"] != 1.0 or result["origin"] != 0.0:
        lines.append(
            f"quoted as (x - {result['origin']:g}) / {result['height']:g}"
        )

    lines.append("")
    value = quoted_value(result)
    if value is not None:
        bound = (
            f"   +/- {result['resolution']:.3g} (half the local face spacing)"
            if digits_for(result["resolution"]) is not None
            else "   (the face spacing here could not be read, so this carries no error bar)"
        )
        # `quoted_value` and `CLAIM_SYMBOL`, not the raw result and not a literal:
        # the claim stamped into the figure is this same call and this same constant.
        # The number used to be formatted separately (6.324 in text against 6.32 on
        # the picture) and the symbol used to be written out here (`x_r` in the
        # report against `x_r/h` in the stamp, on the same run).
        lines.append(f"{CLAIM_SYMBOL} = {value:g}{shown}{bound}")
        if result["separation_scaled"] is not None:
            lines.append(
                f"separation at {result['separation_scaled']:.4g}{shown}; "
                f"bubble {result['bubble_length_scaled']:.4g}{shown} long"
            )
        # Only the naive answers that are actually WRONG here. The first version
        # printed "neither is the primary reattachment" whenever EITHER differed,
        # so on a wall with one bubble -- crossings [separation, reattachment] --
        # it announced that the last sign change, which is the right answer, was
        # not the right answer. A reader checking their own number against that
        # line was steered off it, which is this file's failure inverted.
        primary = result["reattachment_scaled"]
        naive = [
            (which, other) for which, other in
            (("FIRST", result["first_crossing_scaled"]),
             ("LAST", result["last_crossing_scaled"]))
            if other is not None and abs(other - primary) > 1e-12
        ]
        if len(naive) == 2:
            lines.append(
                f"a script taking the FIRST sign change would say {naive[0][1]:.4g} and one "
                f"taking the LAST would say {naive[1][1]:.4g}; neither is the primary "
                "reattachment, and if your other number is one of them, that is why"
            )
        elif naive:
            which, other = naive[0]
            agrees = "LAST" if which == "FIRST" else "FIRST"
            lines.append(
                f"a script taking the {which} sign change would say {other:.4g}, which is "
                f"not the primary reattachment; the {agrees} sign change on this wall IS "
                "the primary one, so that rule agrees here by luck and will not on a wall "
                "carrying a second bubble"
            )
    for note in result["notes"]:
        lines.append(note)

    if result["crossings"]:
        lines.append("")
        lines.append(f"## sign changes ({len(result['crossings'])})")
        # Both columns in the quoted unit. `crossings` records `resolution` in the
        # mesh's own coordinate and `measure` divides only its own copy of it, so
        # this column used to print raw spacing beside an `x` already divided by
        # --height: one run's headline read `+/- 0.025 (half the local face spacing)`
        # over a table row of 0.00047, which is 0.025 h and 0.00047 m, the two
        # differing by the height with nothing saying so. That is this file's own
        # subject inside the report that argues it.
        #
        # The `x` column is the unrounded crossing and the headline above is rounded
        # to what the spacing supports, so one run prints 7.97 there and 7.975 here.
        # The column says unrounded rather than being rounded to match, because the
        # spacing that justifies the rounding is the next column along.
        head = f"{'x (unrounded)':>14}  {'kind':<14} face spacing there"
        if unit:
            head += f"   (both columns in {unit})"
        lines.append(head)
        for point in result["crossings"]:
            lines.append(
                f"{point['scaled']:>14.4g}  {point['kind']:<14} "
                f"{point['resolution'] / result['height']:.3g}"
            )

    if result["structures"]:
        lines.append("")
        lines.append("## what is on this wall, from the upstream end")
        for span in result["structures"]:
            edge = ""
            if span["open_start"] or span["open_end"]:
                edge = "   (runs off the patch)"
            lines.append(
                f"  {span['scaled_start']:>10.4g} .. {span['scaled_end']:<10.4g} "
                f"{span['what']}{edge}"
            )
    return "\n".join(lines)


def plot(profile: dict[str, Any], result: dict[str, Any], out: Path,
         claim_row: dict[str, Any] | None) -> Path:
    """The wall-shear curve with every crossing on it, and the answer written on it.

    The caption is `claims.caption(claim_row)` -- the same call that produces the
    stamp -- so the number drawn on the picture is the number the picture carries.
    The delivered figure in F-36 was annotated 7.57 while its own script printed
    7.5667 and the answer said 6.4; three numbers, one quantity, and nothing tying
    any of them together.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    origin, height = result["origin"], result["height"]
    x = (np.asarray(profile["x"], float) - origin) / height
    tau = np.asarray(profile["tau"], float)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.plot(x, tau, color="#1f77b4", linewidth=1.2, label="streamwise wall shear")

    for span in result["structures"]:
        if span["attached"]:
            continue
        ax.axvspan(span["scaled_start"], span["scaled_end"], color="#d62728", alpha=0.08)

    for point in result["crossings"]:
        ax.axvline(point["scaled"], color="0.6", linewidth=0.7, linestyle=":")
        ax.annotate(f"{point['scaled']:.3g}\n{point['kind'][:3]}",
                    xy=(point["scaled"], 0.0), xytext=(0, -28),
                    textcoords="offset points", ha="center", fontsize=7, color="0.35")

    if claim_row is not None:
        at = result["reattachment_scaled"]
        ax.axvline(at, color="#2ca02c", linewidth=1.6)
        ax.annotate(claims.caption(claim_row), xy=(at, 0.0), xytext=(6, 12),
                    textcoords="offset points", color="#2ca02c", fontsize=10,
                    fontweight="bold")

    # The axis is labelled off the claim when there is one, so the axis, the caption
    # and the stamp all say the same thing about what the number is in.
    dimensionless = (claim_row or {}).get("units") == "h" if claim_row else height != 1.0
    ax.set_xlabel("x / h" if dimensionless else "x")
    ax.set_ylabel(f"{profile['field']} along {profile['direction']}")
    title = f"{profile['patch']}"
    if profile.get("time"):
        title += f" at t={profile['time']}"
    if claim_row is not None:
        title += f" — {claims.caption(claim_row)}"
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# -- the command line ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("case", type=Path, nargs="?", default=None)
    parser.add_argument("--patch", default="", help="the wall to walk along")
    parser.add_argument("--field", default="wallShearStress")
    parser.add_argument("--time", default=None, help="defaults to the latest that has the field")
    parser.add_argument("--direction", default="x", choices=sorted(AXES),
                        help="the streamwise axis (default x)")
    parser.add_argument("--origin", type=float, default=0.0,
                        help="where the wall's coordinate starts counting, e.g. the step face")
    parser.add_argument("--height", type=float, default=1.0,
                        help="the length the answer is quoted in, e.g. the step height. "
                             "The default 1 divides by nothing and is labelled as an "
                             "unscaled number; a coordinate already in x/h wants "
                             "--units h, not --height 1")
    parser.add_argument("--units", default=None,
                        help="what the number is in, when you know and this cannot: "
                             "a table already sampled in x/h read with no --height is "
                             "dimensionless, and only you can say so. Sets the label on "
                             "the report, the figure and the claim, and claims.py only "
                             "compares two numbers whose labels match")
    parser.add_argument("--forward-sign", type=int, default=None, choices=[-1, 1],
                        help="override which sign of tau means attached flow")
    parser.add_argument("--from-file", type=Path, default=None,
                        help="a table of wall shear instead of a case")
    parser.add_argument("--columns", nargs=2, type=int, default=(0, 3),
                        metavar=("X", "TAU"), help="columns in --from-file (default 0 3, a .raw)")
    parser.add_argument("--plot", type=Path, default=None,
                        help="draw the curve here; the figure carries the number")
    parser.add_argument("--no-stamp", action="store_true",
                        help="draw the figure without stamping the claim into it")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.from_file is not None:
            profile = profile_from_file(args.from_file, tuple(args.columns))
        elif args.case is not None and args.patch:
            profile = wall_profile(args.case, args.patch, direction=args.direction,
                                   time=args.time, field=args.field)
        else:
            parser.error("give a case and --patch, or --from-file")
    except Unmeasurable as exc:
        # Unmeasurable is a fact about the case, and the exit code stays 0 for the
        # same reason layer_report's does: not being able to measure something is a
        # reading, and a reading is what this offers.
        print(f"not measured: {exc}")
        return 0

    forward, basis = args.forward_sign, "given on the command line"
    if forward is None and profile.get("forward") is not None:
        forward, basis = profile["forward"], profile["forward_basis"]
    result = measure(profile["x"], profile["tau"], origin=args.origin,
                     height=args.height, forward=forward, basis=basis)
    claim_row = as_claim(profile, result, args.units)

    figure = None
    if args.plot is not None:
        figure = plot(profile, result, args.plot, claim_row)
        if claim_row is not None and not args.no_stamp:
            attached = claims.attach(figure, [claim_row],
                                     root=args.case or figure.parent,
                                     case=args.case.name if args.case else "")
            claim_row["figure"] = str(figure)
            claim_row["recorded"] = attached["recorded"]

    if args.json:
        payload = {
            "profile": {key: value for key, value in profile.items() if key not in ("x", "tau")},
            "x": [float(value) for value in profile["x"]],
            "tau": [float(value) for value in profile["tau"]],
            "result": result,
            "claim": claim_row,
            "figure": str(figure) if figure else None,
        }
        print(json.dumps(payload, indent=2, default=float))
        return 0

    print(report(profile, result, args.units))
    if figure is not None:
        carried = " (it carries the number above)" if claim_row and not args.no_stamp else ""
        print(f"\nfigure: {figure}{carried}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
