#!/usr/bin/env python3
"""A potential-flow lift estimate for an aerofoil or a wing, in about a second.

There was no inviscid estimator here. `potentialFoam` is on the image but is used as a
warm start for a SIMPLE run (`case_gen.py` writes the `Phi` solver entry for exactly
that), so the cheapest lift number a study could get still cost a mesh and a solve. The
arithmetic below needs neither: a few hundred panels and one dense solve of a few
hundred unknowns, which is milliseconds. That makes it worth reaching for *before* a
case exists -- to see whether the incidence being asked about is anywhere near the
linear range, to get a Cl to size a reference area or a force expectation against, or to
have a number the eventual RANS answer can be put beside.

What makes it checkable is that the answer is already known from outside any code: a
thin symmetric section's lift slope is 2*pi per radian and its quarter-chord moment is
zero, a cambered NACA 4-digit's zero-lift angle is a closed-form integral over its own
camber line, and an elliptically loaded wing's induced drag is Cl^2/(pi*AR). The tests
beside this file check it against those rather than against itself.

It is pure arithmetic through numpy -- no solver, no mesh, no network -- so it runs on a
laptop as happily as on the instance.

What it cannot do is printed with every answer, not only written here. There is no
viscosity in any of it: no drag beyond an optional flat-plate friction figure that is
labelled as the flat plate's and not this section's, no stall, no separation, no
trailing-edge recovery, and no Reynolds number dependence. Attached, thin, small
incidence. Past a few degrees the answer and the wind tunnel part company, and the
script says so where the number is, because a Cl quoted without that is the kind of
thing that ends up in a report.

    python3 panel.py --naca 2412 --alpha 5
    python3 panel.py --naca 0012 --sweep -4 10 1 --png figures
    python3 panel.py --dat clarky.dat --alpha 3 --json panel.json
    python3 panel.py --naca 2412 --alpha 5 --wing --aspect-ratio 8 --taper 0.5 --twist -2
    python3 panel.py --naca 0012 --alpha 5 --reynolds 3e6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Printed with every answer. The docstring is for whoever opens the file; this is for
# whoever reads the output, which is the one who is about to quote the number.
LIMITS = [
    "no viscosity anywhere in this: no stall, no separation, no boundary layer, and so",
    "  no profile drag -- the pressure drag below is a numerics check (d'Alembert says",
    "  it is zero), not a drag estimate",
    "linear and attached only: the answer tracks a wind tunnel to within a few percent",
    "  for a few degrees either side of the zero-lift angle and then diverges, early on",
    "  a thick or heavily cambered section and later on a thin one",
    "no Reynolds number: nothing here knows one, so nothing here changes with it",
    "the 3D numbers below, where asked for, are lifting-line theory -- a straight,",
    "  unswept, moderate-to-high aspect ratio wing, induced drag only",
]

NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")


class Unusable(Exception):
    """The geometry cannot be panelled, and saying which is more use than patching it."""


# -- geometry ------------------------------------------------------------------


def naca4(designation: str, n_panels: int = 160) -> tuple[np.ndarray, dict]:
    """Nodes of a NACA 4-digit section, from the published definition of the family.

    Thickness and camber are the standard polynomials; the surface points are the
    thickness laid off *normal* to the camber line rather than vertically, which is what
    the NACA definition says and is a visible difference on a 4412 near the nose. The
    last thickness coefficient is -0.1036 rather than -0.1015, which closes the trailing
    edge exactly -- the open-trailing-edge form leaves a 0.21% chord gap, and a gap is a
    contour this method would have to decide what to do about. Which form was used is
    reported, because the two give slightly different trailing-edge pressures.

    Cosine spacing in x, so panels cluster where the curvature is: the nose, where a
    uniform distribution puts the stagnation point several percent of chord out.
    """
    digits = designation.strip().lstrip("naca").lstrip("NACA").strip()
    if not (len(digits) == 4 and digits.isdigit()):
        raise Unusable(f"a NACA 4-digit designation is four digits; got {designation!r}")
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:]) / 100.0
    if n_panels < 20:
        raise Unusable(f"{n_panels} panels is too few to mean anything; 20 is the floor")

    # One node per panel plus the closing one; the x stations are half of that, nose to
    # tail, and each station gives an upper and a lower point.
    n_stations = n_panels // 2 + 1
    beta = np.linspace(0.0, np.pi, n_stations)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                    + 0.2843 * x**3 - 0.1036 * x**4)

    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        fore = x <= p
        yc = np.where(fore,
                      m / p**2 * (2.0 * p * x - x**2),
                      m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2))
        dyc = np.where(fore,
                       2.0 * m / p**2 * (p - x),
                       2.0 * m / (1.0 - p) ** 2 * (p - x))

    phi = np.arctan(dyc)
    xu, yu = x - yt * np.sin(phi), yc + yt * np.cos(phi)
    xl, yl = x + yt * np.sin(phi), yc - yt * np.cos(phi)

    # Selig order: trailing edge, forward over the upper surface, back along the lower.
    nodes = np.vstack([
        np.column_stack([xu[::-1], yu[::-1]]),
        np.column_stack([xl[1:], yl[1:]]),
    ])
    nodes[0] = nodes[-1] = [1.0, 0.0]  # the closed-trailing-edge polynomial's own value
    meta = {
        "source": f"NACA {digits} generated analytically",
        "camber": m, "camber_position": p, "thickness": t,
        "trailing_edge": "closed (-0.1036 x^4 thickness form)",
        "spacing": "cosine in x",
    }
    return nodes, meta


def read_dat(path: Path, close_tol: float = 0.01) -> tuple[np.ndarray, dict]:
    """An aerofoil coordinate file, in either layout that turns up in the wild.

    Selig is one continuous list from the trailing edge, forward over the upper surface
    and back along the lower. Lednicer is two blocks each running nose to tail, under a
    header line giving the two point counts. The same aerofoil in the two layouts is the
    same aerofoil, and reading one as the other produces a contour that crosses itself
    and a lift coefficient with no warning attached -- so which layout was read is
    detected and then *stated*, since the detection is a guess from the numbers and the
    person holding the file can tell at a glance whether the guess was right.

    The detection: a first pair whose values are both well above 1 is a Lednicer count
    header (no aerofoil ordinate is 61.0). Failing that, a coordinate list that starts
    near the nose and runs aft is Lednicer with the header omitted, and one that starts
    at the tail is Selig.

    A contour whose ends do not meet is refused by name rather than closed quietly: a
    file read in the wrong layout, or truncated, arrives here as a large gap, and
    patching it would turn a broken file into a plausible number. A real finite trailing
    edge is a small gap, and those two points are moved to their midpoint and the move
    is reported.

    A coordinate line is numbers and separators and NOTHING else, which is #140: a
    title line carries two numbers as often as not -- `E63  (4.25%)`, `S1223 12%`,
    `zone T="... at 40.18%c ... between 0 and 1"` -- and taking the first one for a
    pair put (63, 4.25) at the head of a plain Selig file, where both values above 1.5
    read as a Lednicer count header. The file was then split, reversed and closed
    across a gap it did not have, and reported as Lednicer with two fabricated notes:
    Cl 3.4% high and Cm 8.8% high on `airfoil_e63.dat`. A genuine Lednicer header is
    two bare numbers on a line of its own, so the stricter rule keeps reading it.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        # The same `Unusable` path every other bad input takes, rather than a traceback
        # out of a tool whose contract is to report a reading or refuse by name.
        raise Unusable(f"{path} cannot be read: {exc.strerror or exc}") from exc
    pairs, name = [], ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        numeric = stripped.replace("D", "E").replace("d", "e")
        found = NUMBER.findall(numeric)
        if len(found) >= 2 and _only_numbers(numeric, found):
            pairs.append((float(found[0]), float(found[1])))
        elif not pairs and not name:
            name = stripped
    if len(pairs) < 20:
        raise Unusable(f"{path} holds {len(pairs)} coordinate pairs, too few to panel")

    header, notes = None, []
    if pairs[0][0] > 1.5 and pairs[0][1] > 1.5:
        header = (int(round(pairs[0][0])), int(round(pairs[0][1])))
        pairs = pairs[1:]

    points = np.asarray(pairs, dtype=float)
    if header is not None:
        n_upper, n_lower = header
        if n_upper + n_lower != len(points):
            notes.append(f"the header asks for {n_upper}+{n_lower} points and the file "
                         f"holds {len(points)}; the header was used as far as it goes")
            n_upper = min(n_upper, len(points) - 1)
            n_lower = len(points) - n_upper
        upper, lower = points[:n_upper], points[n_upper:n_upper + n_lower]
        layout = "Lednicer (two nose-to-tail blocks under a count header)"
        nodes = _join_blocks(upper, lower)
    elif points[0, 0] < 0.5:
        # No header, and it starts at the nose: two nose-to-tail blocks, split where x
        # turns back on itself.
        turn = int(np.argmax(points[:, 0])) + 1
        upper, lower = points[:turn], points[turn:]
        if len(lower) < 5:
            raise Unusable(
                f"{path} starts at the leading edge and never comes back to it, so it is "
                "neither a Selig list nor a Lednicer pair of blocks")
        layout = "Lednicer (two nose-to-tail blocks, no count header)"
        nodes = _join_blocks(upper, lower)
    else:
        layout = "Selig (one list, trailing edge over the upper surface and back)"
        nodes = points

    chord = float(nodes[:, 0].max() - nodes[:, 0].min())
    if chord <= 0:
        raise Unusable(f"{path} has no chord: every x is {nodes[0, 0]}")
    gap = float(np.hypot(*(nodes[-1] - nodes[0])) / chord)
    if gap > close_tol:
        raise Unusable(
            f"{path} does not close: the first point ({nodes[0, 0]:.5f}, {nodes[0, 1]:.5f}) "
            f"and the last ({nodes[-1, 0]:.5f}, {nodes[-1, 1]:.5f}) are {gap:.4f} chord "
            f"apart, over the {close_tol:.4f} this will close by hand. Read as {layout}; "
            "a file read in the wrong layout, or truncated, looks exactly like this")
    if gap > 0:
        midpoint = 0.5 * (nodes[0] + nodes[-1])
        notes.append(f"trailing edge closed: the two end points were {gap:.5f} chord apart "
                     f"and were moved to their midpoint ({midpoint[0]:.5f}, "
                     f"{midpoint[1]:.5f})")
        nodes = nodes.copy()
        nodes[0] = nodes[-1] = midpoint

    meta = {
        "source": f"{path} -- {name}" if name else str(path),
        "layout": layout,
        "notes": notes,
        "trailing_edge": "closed" if gap == 0 else f"closed by hand from a {gap:.5f}c gap",
        "spacing": "as the file gives it",
    }
    return nodes, meta


def _only_numbers(line: str, found: list[str]) -> bool:
    """Whether `line` is the numbers `found` and separators and nothing else.

    The separators a coordinate file uses: spaces, tabs, commas and semicolons. Any
    other character left over is prose, and a line with prose on it is a title, a
    comment or a Tecplot `zone` header rather than a point.
    """
    rest = line
    for token in found:
        rest = rest.replace(token, " ", 1)
    return not rest.strip(" \t,;")


def _join_blocks(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Two nose-to-tail blocks into one Selig list, without doubling the nose point."""
    upper = upper[::-1]  # now tail to nose
    if len(lower) and np.allclose(upper[-1], lower[0], atol=1e-9):
        lower = lower[1:]
    return np.vstack([upper, lower])


def panelise(nodes: np.ndarray) -> dict:
    """Panel midpoints, tangents and outward normals, with the contour wound clockwise.

    Clockwise is the one choice that matters: with it, `(-sin theta, cos theta)` is the
    outward normal on every panel and panel 1 and panel N are the two that meet at the
    trailing edge, which is where the Kutta condition goes. Selig order is anticlockwise,
    so most files arrive reversed and get turned round here rather than at four call
    sites.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) < 21:
        raise Unusable(f"a contour wants at least 21 (x, y) nodes; got {nodes.shape}")
    if not np.allclose(nodes[0], nodes[-1]):
        raise Unusable("the contour does not close; read it through read_dat, which says "
                       "by how much and refuses rather than patching")

    x, y = nodes[:, 0], nodes[:, 1]
    signed_area = 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    reversed_here = signed_area > 0
    if reversed_here:
        nodes = nodes[::-1]

    a, b = nodes[:-1], nodes[1:]
    delta = b - a
    length = np.hypot(delta[:, 0], delta[:, 1])
    if np.any(length <= 0):
        bad = int(np.argmin(length))
        raise Unusable(f"node {bad} and {bad + 1} are the same point, so panel {bad} has "
                       "no length; duplicate points want removing before panelling")
    theta = np.arctan2(delta[:, 1], delta[:, 0])
    return {
        "nodes": nodes,
        "mid": 0.5 * (a + b),
        "theta": theta,
        "length": length,
        "normal": np.column_stack([-np.sin(theta), np.cos(theta)]),
        "tangent": np.column_stack([np.cos(theta), np.sin(theta)]),
        "chord": float(nodes[:, 0].max() - nodes[:, 0].min()),
        "perimeter": float(length.sum()),
        "area": abs(signed_area),
        "reversed": reversed_here,
        "n": len(length),
    }


# -- the 2D solution -----------------------------------------------------------


def influence(geom: dict) -> dict:
    """Unit source and unit vortex panel velocities at every control point.

    Constant strength per panel, integrated in closed form in the panel's own frame --
    `ln(r0/r1)/(2*pi)` along it and the subtended angle across it -- which is the
    textbook Hess-Smith kernel. The vortex panel is the source panel rotated a quarter
    turn, so one integration serves both.

    The diagonal is written in rather than evaluated: the control point sits exactly on
    its own panel, where the formula's limit is clean (no tangential velocity, half the
    strength normal to it) but `arctan2` of a signed zero is not -- a roundoff-negative
    eta flips the self-term's sign and the whole solution with it.
    """
    mid, theta, length = geom["mid"], geom["theta"], geom["length"]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    start = geom["nodes"][:-1]

    dx = mid[:, 0, None] - start[None, :, 0]
    dy = mid[:, 1, None] - start[None, :, 1]
    xi = dx * cos_t[None, :] + dy * sin_t[None, :]
    eta = -dx * sin_t[None, :] + dy * cos_t[None, :]

    r0 = np.hypot(xi, eta)
    r1 = np.hypot(xi - length[None, :], eta)
    diag = np.arange(geom["n"])
    safe = np.where(r1 > 0, r1, 1.0)
    u_src = np.log(np.where(r0 > 0, r0, 1.0) / safe) / (2.0 * np.pi)
    v_src = (np.arctan2(eta, xi - length[None, :]) - np.arctan2(eta, xi)) / (2.0 * np.pi)
    u_src[diag, diag] = 0.0
    v_src[diag, diag] = 0.5
    u_vor, v_vor = -v_src, u_src

    def to_global(u, v):
        return (u * cos_t[None, :] - v * sin_t[None, :],
                u * sin_t[None, :] + v * cos_t[None, :])

    us_x, us_y = to_global(u_src, v_src)
    uv_x, uv_y = to_global(u_vor, v_vor)
    nx, ny = geom["normal"][:, 0], geom["normal"][:, 1]
    return {
        "source_normal": us_x * nx[:, None] + us_y * ny[:, None],
        "source_tangent": us_x * cos_t[:, None] + us_y * sin_t[:, None],
        "vortex_normal": (uv_x * nx[:, None] + uv_y * ny[:, None]).sum(axis=1),
        "vortex_tangent": (uv_x * cos_t[:, None] + uv_y * sin_t[:, None]).sum(axis=1),
    }


def solve(geom: dict, alpha_deg: float, coeff: dict | None = None,
          moment_about: float = 0.25) -> dict:
    """One incidence: Cp on every panel, Cl, Cm and where the flow stagnates.

    N source strengths and one global vortex strength against N flow-tangency conditions
    and the Kutta condition, which here is the Hess-Smith form -- equal and opposite
    tangential velocity on the two panels meeting at the trailing edge, so the trailing
    edge is not a singularity and the circulation is determined rather than chosen.
    `numpy.linalg.solve` on a dense (N+1) system; scipy is not on the image and is not
    wanted for a few hundred unknowns.

    Cl and Cm come from integrating Cp over the panels, which is the same arithmetic a
    solver's `forceCoeffs` does and so is directly comparable with one. The circulation
    gives Cl a second way, through Kutta-Joukowski, and the two agreeing is a check on
    the linear solve that costs one line. Pressure drag is reported for the same reason:
    d'Alembert's paradox says a closed body in potential flow has none, so whatever comes
    out is the discretisation error, which is the honest way to read a panel count.
    """
    coeff = coeff or influence(geom)
    n = geom["n"]
    alpha = np.radians(alpha_deg)
    theta = geom["theta"]

    matrix = np.empty((n + 1, n + 1))
    matrix[:n, :n] = coeff["source_normal"]
    matrix[:n, n] = coeff["vortex_normal"]
    matrix[n, :n] = coeff["source_tangent"][0, :] + coeff["source_tangent"][-1, :]
    matrix[n, n] = coeff["vortex_tangent"][0] + coeff["vortex_tangent"][-1]

    rhs = np.empty(n + 1)
    rhs[:n] = -np.sin(alpha - theta)
    rhs[n] = -(np.cos(alpha - theta[0]) + np.cos(alpha - theta[-1]))

    solution = np.linalg.solve(matrix, rhs)
    sigma, gamma = solution[:n], float(solution[n])

    vt = (np.cos(alpha - theta) + coeff["source_tangent"] @ sigma
          + gamma * coeff["vortex_tangent"])
    cp = 1.0 - vt**2

    chord = geom["chord"]
    normal, mid = geom["normal"], geom["mid"]
    dfx = -cp * normal[:, 0] * geom["length"]
    dfy = -cp * normal[:, 1] * geom["length"]
    cfx, cfy = float(dfx.sum()) / chord, float(dfy.sum()) / chord
    cl = cfy * np.cos(alpha) - cfx * np.sin(alpha)
    cd_pressure = cfx * np.cos(alpha) + cfy * np.sin(alpha)
    x_ref = geom["nodes"][:, 0].min() + moment_about * chord
    cm = -float(np.sum((mid[:, 0] - x_ref) * dfy - mid[:, 1] * dfx)) / chord**2

    nose = int(np.argmin(mid[:, 0]))
    return {
        "alpha_deg": float(alpha_deg),
        "cl": float(cl),
        "cl_from_circulation": float(-2.0 * gamma * geom["perimeter"] / chord),
        "cm": float(cm),
        "moment_about": float(moment_about),
        "cd_pressure": float(cd_pressure),
        "gamma": gamma,
        "cp": cp,
        "vt": vt,
        "x": mid[:, 0].copy(),
        "y": mid[:, 1].copy(),
        "upper": slice(nose, n),
        "lower": slice(0, nose),
        "cp_min": float(cp.min()),
        "stagnation": stagnation_point(geom, vt),
        "panels": n,
    }


def stagnation_point(geom: dict, vt: np.ndarray) -> dict | None:
    """Where the tangential velocity changes sign, nearest the nose.

    The wrap from the last panel to the first is skipped on purpose: the Kutta condition
    makes those two tangential velocities equal and opposite by construction, so the
    trailing edge always reads as a sign change and never is a stagnation point.
    """
    sign_change = np.nonzero(np.sign(vt[:-1]) * np.sign(vt[1:]) < 0)[0]
    if sign_change.size == 0:
        return None
    mid = geom["mid"]
    best = sign_change[int(np.argmin(mid[sign_change, 0]))]
    v0, v1 = vt[best], vt[best + 1]
    frac = float(v0 / (v0 - v1)) if v0 != v1 else 0.5
    point = mid[best] + frac * (mid[best + 1] - mid[best])
    nose = int(np.argmin(mid[:, 0]))
    return {
        "x": float(point[0]),
        "y": float(point[1]),
        "x_over_chord": float((point[0] - geom["nodes"][:, 0].min()) / geom["chord"]),
        # Which side it is on is the panel index against the nose panel, not the sign of
        # y: a cambered section's lower surface near the nose sits above y = 0.
        "surface": "lower" if best < nose else "upper",
        "panel": int(best),
        "at_the_nose": abs(float(point[0]) - float(mid[nose, 0])) < 0.01 * geom["chord"],
    }


def sweep(geom: dict, alphas, moment_about: float = 0.25) -> list[dict]:
    """A lift curve. The influence coefficients are geometry only, so they are built once
    and every incidence is one back-substitution -- which is why a sweep costs about what
    a single angle does."""
    coeff = influence(geom)
    return [solve(geom, float(a), coeff=coeff, moment_about=moment_about) for a in alphas]


def lift_slope(geom: dict, about_deg: float = 0.0, delta: float = 1.0,
               coeff: dict | None = None) -> dict:
    """Section lift slope and zero-lift angle, by central difference about an incidence.

    The matrix does not depend on incidence and the right-hand side is linear in the two
    free-stream components, so Cl is linear in `sin(alpha)` and `cos(alpha)` -- a sine,
    not a straight line. Over a few degrees that is a straight line to three figures, but
    the slope does fall off as `cos(alpha)` and the difference shows up by ten degrees, so
    the incidence the slope was taken about is reported with it rather than left implied.
    Quoted about zero by default, which is where a section's slope is normally quoted.

    This is what the lifting line takes as its section property, so the 3D answer rides
    on this aerofoil's own 2D solution rather than on a hardcoded 2*pi.
    """
    coeff = coeff or influence(geom)
    low = solve(geom, about_deg - delta, coeff=coeff)
    high = solve(geom, about_deg + delta, coeff=coeff)
    per_deg = (high["cl"] - low["cl"]) / (2.0 * delta)
    per_rad = per_deg * 180.0 / np.pi
    mid = solve(geom, about_deg, coeff=coeff)
    return {
        "per_radian": float(per_rad),
        "per_degree": float(per_deg),
        "taken_about_deg": float(about_deg),
        "zero_lift_alpha_deg": float(about_deg - mid["cl"] / per_deg) if per_deg else 0.0,
        "thin_aerofoil_ratio": float(per_rad / (2.0 * np.pi)),
    }


# -- the wing ------------------------------------------------------------------


def lifting_line(alpha_deg: float, slope_per_rad: float, zero_lift_deg: float,
                 aspect_ratio: float, taper: float = 1.0, twist_deg: float = 0.0,
                 modes: int = 24) -> dict:
    """Prandtl's lifting line for a straight tapered twisted wing, as a small solve.

    The section slope and zero-lift angle come from the 2D solution above, so this is the
    same aerofoil given a finite span rather than a second aerofoil. Glauert's series:
    the bound circulation is a sine series in the spanwise angle, the downwash each mode
    induces is known term by term, and matching flow tangency at as many stations as
    modes gives a dense `modes x modes` system -- 24 of them by default, which is a
    micro-solve and converges on the second decimal of Cl well before that.

    The two results worth putting against a book are both exact: Cl is `pi*AR*A1` and the
    induced drag is `pi*AR*sum(n*An^2)`, which for an elliptic wing reduce to
    `Cl^2/(pi*AR)`.

    Only odd modes matter on a symmetric wing, and the even ones come back at roundoff;
    they are kept rather than dropped because an asymmetric twist is a one-line change to
    `alpha_geometric` and would need them.
    """
    if aspect_ratio <= 0:
        raise Unusable(f"aspect ratio {aspect_ratio} is not a wing")
    if not 0.0 < taper <= 1.0:
        raise Unusable(f"taper ratio {taper} is outside (0, 1]: it is tip chord over root")
    if modes < 4:
        raise Unusable(f"{modes} Fourier modes is too few; 4 is the floor")

    span = 1.0
    root_chord = 2.0 * span / (aspect_ratio * (1.0 + taper))
    station = np.arange(1, modes + 1) * np.pi / (modes + 1)
    # y = -(b/2) cos(theta), so this is 2|y|/b: 0 at the root, 1 at either tip. Station 1
    # is a tip and station `modes` is the other one, which is why the spanwise table is
    # sorted before it is printed rather than walked in index order.
    eta = np.abs(np.cos(station))
    chord = root_chord * (1.0 - (1.0 - taper) * eta)
    alpha_geometric = np.radians(alpha_deg + twist_deg * eta)
    alpha_absolute = alpha_geometric - np.radians(zero_lift_deg)

    # Gamma = 2*b*V*sum(An*sin(n*theta)) is the normalisation that makes Cl = pi*AR*A1
    # and Cdi = pi*AR*sum(n*An^2) exact, and it fixes this coefficient at c*a0/(4b). The
    # 1/(8b) form in some texts goes with Gamma = 4*b*V*sum(...) and halves Cl here; the
    # elliptic-wing closed form a0*alpha/(1 + a0/(pi*AR)) is what settles which.
    mu = slope_per_rad * chord / (4.0 * span)
    orders = np.arange(1, modes + 1)
    sines = np.sin(np.outer(station, orders))
    matrix = sines * (1.0 + mu[:, None] * orders[None, :] / np.sin(station)[:, None])
    amplitudes = np.linalg.solve(matrix, mu * alpha_absolute)

    cl = float(np.pi * aspect_ratio * amplitudes[0])
    cdi = float(np.pi * aspect_ratio * np.sum(orders * amplitudes**2))
    circulation = 2.0 * span * (sines @ amplitudes)
    section_cl = 2.0 * circulation / chord
    downwash = np.degrees((sines * orders[None, :]) @ amplitudes / np.sin(station))
    return {
        # Carried out with the answer because of #147: the wing's Cl is only readable
        # beside the incidence it is for, and nothing on the page used to say which.
        "alpha_deg": float(alpha_deg),
        "cl": cl,
        "cd_induced": cdi,
        "span_efficiency": float(cl**2 / (np.pi * aspect_ratio * cdi)) if cdi > 0 else None,
        "elliptic_cd_induced": float(cl**2 / (np.pi * aspect_ratio)),
        "cl_2d": float(slope_per_rad * np.radians(alpha_deg - zero_lift_deg)),
        "aspect_ratio": float(aspect_ratio),
        "taper": float(taper),
        "twist_deg": float(twist_deg),
        "modes": int(modes),
        "station_eta": eta,
        "station_chord": chord,
        "section_cl": section_cl,
        "induced_alpha_deg": downwash,
        "amplitudes": amplitudes,
    }


def flat_plate_friction(reynolds: float, wetted_ratio: float = 2.04) -> dict:
    """A flat plate's skin friction at this Reynolds number. Not this section's drag.

    Here under protest and labelled everywhere it appears. The Blasius laminar line
    (1.328/sqrt(Re)) and the Prandtl-Schlichting turbulent one (0.074/Re^0.2) are flat
    plate results: no pressure gradient, no thickness, no separation, no transition
    modelled. On a real section at a real Reynolds number they bracket the profile drag
    loosely enough to be worth a sanity check and far too loosely to be worth quoting,
    which is what the labels say.
    """
    if reynolds <= 0:
        raise Unusable(f"Reynolds number {reynolds} is not positive")
    laminar = 1.328 / np.sqrt(reynolds)
    turbulent = 0.074 / reynolds**0.2
    return {
        "reynolds": float(reynolds),
        "wetted_ratio": float(wetted_ratio),
        "cf_laminar": float(laminar),
        "cf_turbulent": float(turbulent),
        "cd_laminar": float(laminar * wetted_ratio),
        "cd_turbulent": float(turbulent * wetted_ratio),
        "label": "flat-plate skin friction on the wetted area, NOT this section's drag: "
                 "no pressure gradient, no thickness, no transition, no separation",
    }


# -- reporting -----------------------------------------------------------------


def report(geom: dict, meta: dict, results: list[dict], slope: dict,
           wing: list[dict] | None = None, friction: dict | None = None,
           figures: list[Path] | None = None) -> str:
    out = ["# aerofoil", f"  {meta['source']}"]
    if "layout" in meta:
        out.append(f"  read as: {meta['layout']}")
    out.append(f"  trailing edge: {meta['trailing_edge']}")
    out.append(f"  {geom['n']} panels, {meta['spacing']}, chord {geom['chord']:.6g}, "
               f"enclosed area {geom['area']:.6g}")
    if geom["reversed"]:
        out.append("  the contour was wound anticlockwise and was reversed for the solve")
    for note in meta.get("notes", []):
        out.append(f"  note: {note}")

    out += ["", "# method",
            "  constant-strength source panels with one global vortex, Kutta condition as",
            "  equal and opposite tangential velocity on the two trailing-edge panels",
            f"  (Hess-Smith); dense {geom['n'] + 1}-unknown solve through numpy.linalg"]

    out += ["", "# section",
            f"  lift slope      {slope['per_radian']:.4f} /rad "
            f"({slope['per_degree']:.5f} /deg), {slope['thin_aerofoil_ratio']:.3f} x 2*pi, "
            f"taken about {slope['taken_about_deg']:+.3g} deg",
            f"  zero-lift alpha {slope['zero_lift_alpha_deg']:+.3f} deg",
            "  thin-aerofoil theory gives 2*pi /rad; thickness raises it, which is the",
            "  ratio above, and is the one place this is expected to sit off the theory"]

    out += ["", f"# incidence (moment about {results[0]['moment_about']:.3g}c)",
            "   alpha        Cl       Cm     Cp_min   Cd_pressure   stagnation x/c"]
    for res in results:
        stag = res["stagnation"]
        where = f"{stag['x_over_chord']:.4f} ({stag['surface']})" if stag else "not found"
        out.append(f"  {res['alpha_deg']:+7.2f}  {res['cl']:+8.4f}  {res['cm']:+7.4f}  "
                   f"{res['cp_min']:+8.3f}  {res['cd_pressure']:+11.2e}   {where}")
    worst = max(abs(r["cd_pressure"]) for r in results)
    drift = max(abs(r["cl"] - r["cl_from_circulation"]) for r in results)
    out += [f"  pressure drag should be zero (d'Alembert); largest here {worst:.2e}, which",
            f"  is the panel count's error. Cl from pressure and from circulation differ",
            f"  by at most {drift:.2e}."]

    if len(results) == 1:
        res = results[0]
        out += ["", "# pressure distribution (every 10th panel; --json has all of them)",
                "    x/c      Cp_upper     x/c      Cp_lower"]
        upper_x, upper_cp = res["x"][res["upper"]][::-1], res["cp"][res["upper"]][::-1]
        lower_x, lower_cp = res["x"][res["lower"]], res["cp"][res["lower"]]
        for k in range(0, max(len(upper_x), len(lower_x)), 10):
            left = (f"  {upper_x[k] / geom['chord']:6.4f}  {upper_cp[k]:+10.4f}"
                    if k < len(upper_x) else " " * 20)
            right = (f"   {lower_x[k] / geom['chord']:6.4f}  {lower_cp[k]:+10.4f}"
                     if k < len(lower_x) else "")
            out.append(left + right)

    if wing:
        first = wing[0]
        out += ["", "# wing (lifting line)",
                f"  aspect ratio {first['aspect_ratio']:.3g}, taper {first['taper']:.3g}, "
                f"tip twist {first['twist_deg']:+.3g} deg, {first['modes']} modes",
                # #147: the incidence is printed, and it is every incidence the section
                # above was solved at rather than one the page never named.
                "   alpha        Cl      Cl_2d   Cd_induced   elliptic Cdi        e"]
        for one in wing:
            efficiency = (f"{one['span_efficiency']:8.4f}"
                          if one["span_efficiency"] is not None else "       -")
            out.append(f"  {one['alpha_deg']:+7.2f}  {one['cl']:+8.4f}  "
                       f"{one['cl_2d']:+8.4f}   {one['cd_induced']:10.5f}   "
                       f"{one['elliptic_cd_induced']:10.5f} {efficiency}")
        if len(wing) == 1:
            out += [f"  spanwise section Cl at alpha {first['alpha_deg']:+.2f} deg, "
                    "root to tip:",
                    "   2|y|/b   chord     Cl     induced alpha"]
            order = np.argsort(first["station_eta"])
            for k in order[::max(1, len(order) // 8)]:
                out.append(f"   {first['station_eta'][k]:6.3f}  "
                           f"{first['station_chord'][k]:7.4f}  "
                           f"{first['section_cl'][k]:+6.3f}  "
                           f"{first['induced_alpha_deg'][k]:+8.3f} deg")
        else:
            out.append("  the spanwise loading is in --json, one set per incidence; the "
                       "table above is the integral of each")
        out += ["  induced drag is the only drag here: no profile drag, no viscosity.",
                "  Cl is converged in the mode count by about 16 modes at any ordinary",
                "  aspect ratio; the induced drag and the span efficiency converge slowest",
                "  on a square tip, so those two are worth re-running at twice the modes."]

    if friction is not None:
        out += ["", f"# flat-plate friction at Re {friction['reynolds']:.3g} -- "
                "a side calculation",
                f"  laminar   Cf {friction['cf_laminar']:.5f} -> "
                f"Cd {friction['cd_laminar']:.5f}  (Blasius, 1.328/sqrt(Re))",
                f"  turbulent Cf {friction['cf_turbulent']:.5f} -> "
                f"Cd {friction['cd_turbulent']:.5f}  (0.074/Re^0.2)",
                f"  {friction['label']}"]

    out += ["", "# what this cannot tell you"]
    out += [f"  {line}" for line in LIMITS]
    if figures:
        out += [""] + [f"figure: {path}" for path in figures]
    return "\n".join(out)


def payload(geom: dict, meta: dict, results: list[dict], slope: dict,
            wing: list[dict] | None, friction: dict | None,
            figures: list[Path] | None) -> dict:
    """The same facts for a script to read. `limits` is in here too: a sidecar that
    carried the numbers and left the caveats on stdout would be a sidecar that strips
    them."""
    def one(res):
        return {
            "alpha_deg": res["alpha_deg"], "cl": res["cl"], "cm": res["cm"],
            "cl_from_circulation": res["cl_from_circulation"],
            "cd_pressure": res["cd_pressure"], "cp_min": res["cp_min"],
            "moment_about": res["moment_about"], "stagnation": res["stagnation"],
            "x_over_chord": (res["x"] / geom["chord"]).tolist(),
            "y_over_chord": (res["y"] / geom["chord"]).tolist(),
            "cp": res["cp"].tolist(),
            "upper_panels": [res["upper"].start, res["upper"].stop],
            "lower_panels": [res["lower"].start, res["lower"].stop],
        }

    data = {
        "method": "Hess-Smith constant-strength source panels with one global vortex, "
                  "Kutta condition on the trailing-edge panel pair",
        "geometry": {k: meta[k] for k in meta if k != "notes"},
        "notes": meta.get("notes", []),
        "panels": geom["n"],
        "chord": geom["chord"],
        "enclosed_area": geom["area"],
        "section": slope,
        "incidence": [one(res) for res in results],
        "limits": LIMITS,
    }
    if wing:
        # A list, one per incidence, and a list of one when there is one incidence:
        # a key whose type changed with the flags would be a key nobody can read.
        data["wing"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in one.items()} for one in wing]
    if friction is not None:
        data["flat_plate_friction"] = friction
    if figures:
        data["figures"] = [str(path) for path in figures]
    return data


def plot(geom: dict, meta: dict, results: list[dict], slope: dict,
         out_dir: Path) -> list[Path]:
    """Cp against x for each incidence, and the lift curve when there is a sweep.

    Cp is drawn with the axis inverted, which is the convention everywhere this would be
    compared against; the caption carries the limits in short form so a figure that
    escapes the terminal does not escape them.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    caption = "potential flow: no viscosity, no stall, no separation, linear range only"
    # The source of a file section is an absolute path, which runs off the end of a
    # figure; the basename and whatever name the file carried is what identifies it.
    title = str(meta["source"]).replace("\\", "/").split("/")[-1]

    for res in results[: 12 if len(results) > 1 else 1]:
        figure, (top, bottom) = plt.subplots(
            2, 1, figsize=(7.0, 6.4), height_ratios=[3, 1], sharex=True)
        chord = geom["chord"]
        top.plot(res["x"][res["upper"]] / chord, res["cp"][res["upper"]],
                 "-o", ms=2.4, lw=1.1, label="upper")
        top.plot(res["x"][res["lower"]] / chord, res["cp"][res["lower"]],
                 "-o", ms=2.4, lw=1.1, label="lower")
        top.axhline(0.0, color="0.7", lw=0.8)
        top.invert_yaxis()
        top.set_ylabel("$C_p$")
        top.legend(loc="lower right", fontsize=8)
        top.grid(alpha=0.25)
        stag = res["stagnation"]
        if stag:
            top.plot([stag["x_over_chord"]], [1.0], "k*", ms=10,
                     label="stagnation")
            top.annotate(f"stagnation x/c {stag['x_over_chord']:.4f}",
                         (stag["x_over_chord"], 1.0), textcoords="offset points",
                         xytext=(8, -4), fontsize=8)
        top.set_title(f"{title}  alpha {res['alpha_deg']:+.2f} deg  "
                      f"Cl {res['cl']:+.4f}  Cm {res['cm']:+.4f}", fontsize=10)
        bottom.fill(geom["nodes"][:, 0] / chord, geom["nodes"][:, 1] / chord,
                    color="0.85", ec="0.3", lw=0.9)
        bottom.set_aspect("equal")
        bottom.set_xlabel("x/c")
        bottom.grid(alpha=0.25)
        figure.text(0.5, 0.005, caption, ha="center", fontsize=7.5, color="0.35")
        figure.tight_layout(rect=(0, 0.02, 1, 1))
        path = out_dir / f"cp_alpha{res['alpha_deg']:+.2f}.png".replace("+", "p").replace(
            "-", "m")
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)

    if len(results) > 1:
        alphas = np.array([res["alpha_deg"] for res in results])
        cls = np.array([res["cl"] for res in results])
        cms = np.array([res["cm"] for res in results])
        figure, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.2))
        left.plot(alphas, cls, "-o", ms=3.4, lw=1.3, label="this panel solution")
        thin = 2.0 * np.pi * np.radians(alphas - slope["zero_lift_alpha_deg"])
        left.plot(alphas, thin, "--", lw=1.0, color="0.45",
                  label=r"thin aerofoil, $2\pi(\alpha-\alpha_{L0})$")
        left.axhline(0.0, color="0.7", lw=0.8)
        left.axvline(slope["zero_lift_alpha_deg"], color="0.7", lw=0.8)
        left.set_xlabel("alpha (deg)")
        left.set_ylabel("$C_l$")
        left.legend(fontsize=8)
        left.grid(alpha=0.25)
        left.set_title(f"lift slope {slope['per_radian']:.3f}/rad, "
                       f"zero lift {slope['zero_lift_alpha_deg']:+.2f} deg", fontsize=10)
        right.plot(alphas, cms, "-o", ms=3.4, lw=1.3, color="C3")
        right.axhline(0.0, color="0.7", lw=0.8)
        right.set_xlabel("alpha (deg)")
        right.set_ylabel(f"$C_m$ about {results[0]['moment_about']:.3g}c")
        right.grid(alpha=0.25)
        figure.text(0.5, 0.005, caption + " -- the curve has no top", ha="center",
                    fontsize=7.5, color="0.35")
        figure.tight_layout(rect=(0, 0.03, 1, 1))
        path = out_dir / "lift_curve.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)
    return written


# -- command line --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    section = parser.add_mutually_exclusive_group(required=True)
    section.add_argument("--naca", help="a 4-digit designation, e.g. 2412")
    section.add_argument("--dat", type=Path,
                         help="a coordinate file, Selig or Lednicer layout")
    parser.add_argument("--panels", type=int, default=160,
                        help="panels for --naca (default 160; --dat uses its own points)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="incidence in degrees (default 0)")
    parser.add_argument("--sweep", type=float, nargs=3, metavar=("LO", "HI", "STEP"),
                        default=None, help="a lift curve instead of one incidence")
    parser.add_argument("--moment-about", type=float, default=0.25,
                        help="moment reference as a fraction of chord (default 0.25)")
    parser.add_argument("--close-tol", type=float, default=0.01,
                        help="trailing-edge gap, in chords, this will close by hand "
                             "before refusing the file (default 0.01)")
    parser.add_argument("--wing", action="store_true",
                        help="also run the lifting line, using this section's slope")
    parser.add_argument("--aspect-ratio", type=float, default=8.0)
    parser.add_argument("--taper", type=float, default=1.0,
                        help="tip chord over root chord (default 1, rectangular)")
    parser.add_argument("--twist", type=float, default=0.0,
                        help="tip incidence relative to the root, degrees; negative is "
                             "washout (default 0)")
    parser.add_argument("--modes", type=int, default=24,
                        help="Fourier modes in the lifting-line solve (default 24)")
    parser.add_argument("--reynolds", type=float, default=None,
                        help="add a flat-plate friction figure, labelled as one")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the same facts as a sidecar")
    parser.add_argument("--png", type=Path, default=None,
                        help="directory for the Cp and lift-curve figures")
    args = parser.parse_args(argv)

    try:
        if args.naca:
            nodes, meta = naca4(args.naca, args.panels)
        else:
            nodes, meta = read_dat(args.dat, close_tol=args.close_tol)
        geom = panelise(nodes)
        if args.sweep:
            low, high, step = args.sweep
            if step <= 0:
                raise Unusable(f"--sweep step {step} must be positive")
            alphas = np.arange(low, high + 0.5 * step, step)
        else:
            alphas = np.array([args.alpha])
        results = sweep(geom, alphas, moment_about=args.moment_about)
        # About zero, not about the incidence asked for: this is the section property the
        # lift slope is quoted as everywhere it would be compared, and the curve's fall-off
        # with cos(alpha) would otherwise make the same section report two slopes.
        slope = lift_slope(geom, about_deg=0.0)
        # #147: at every incidence the section was solved at, not at `--alpha` while
        # `--sweep` covered something else. There is one list of incidences and the
        # wing is on it, so there is no second convention for the page to leave unsaid.
        wing = [
            lifting_line(float(alpha), slope["per_radian"],
                         slope["zero_lift_alpha_deg"], args.aspect_ratio,
                         taper=args.taper, twist_deg=args.twist, modes=args.modes)
            for alpha in alphas
        ] if args.wing else None
        friction = flat_plate_friction(args.reynolds) if args.reynolds else None
    except Unusable as exc:
        # Exit 0: not being able to panel a file is a reading about the file, and a
        # reading is what this offers. The refusal names the file and the reason.
        print(f"not estimated: {exc}")
        print("\n# what this cannot tell you")
        for line in LIMITS:
            print(f"  {line}")
        return 0

    figures = plot(geom, meta, results, slope, args.png) if args.png else None
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            payload(geom, meta, results, slope, wing, friction, figures),
            indent=2, default=float))
    print(report(geom, meta, results, slope, wing, friction, figures))
    if args.json:
        print(f"\njson: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
