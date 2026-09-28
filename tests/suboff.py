"""The DARPA SUBOFF bare hull as an OpenFOAM half model, for the regression tests.

The hull is the analytic definition -- Groves, Huang and Chang, "Geometric
Characteristics of DARPA SUBOFF Models", DTRC/SHD-1298-01 (1989): bow, parallel middle
body, afterbody and cap, in feet at model scale -- the same equations the R01 study's
STL was generated from. Written here as the half of it on y >= 0: a `hull` wall patch
and a `symmetry` patch in the plane y = 0 that meets the hull along its cut, which is
exactly the mesh R01 solved on. Only `points`, `faces` and `boundary` are written;
nothing here needs cells.

The reference numbers are the ones R01 was graded against: towing-tank resistance
87.4 N at 3.046 m/s, Ct 3.15e-3 on S = 5.988 m2 (Crook, DTRC/SHD-1298-07, via Liu and
Huang's summary, Table 14), and the force R01's solve reported on the half, 43.74 N.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FT = 0.3048
RMAX, XB, XM, XA, XC = 0.8333333, 3.3333330, 10.6458330, 13.9791670, 14.2916670
CB1, CB2, CB3 = 1.126395101, 0.442874707, 1.0 / 2.1
RH, K0, K1 = 0.1175, 10.0, 44.6244

LENGTH_M = XC * FT
SPEED = 3.046
RHO = 998.0
REYNOLDS = 1.2e7
"""The Reynolds number the case was graded at."""
NU = SPEED * LENGTH_M / REYNOLDS

PUBLISHED_DRAG_N = 87.4
PUBLISHED_CT = 3.15e-3
PUBLISHED_WETTED_M2 = 5.988
HALF_MODEL_FORCE_N = 43.74
"""What OpenFOAM's `forces` reported on R01's half model (F-43)."""
R01_AREA_M2 = 5.98757
"""The whole hull's wetted area R01 measured off its STL and divided the half force by."""


def radius_ft(x):
    x = np.asarray(x, dtype=float)
    r = np.zeros_like(x)
    m = x < XB
    a, b = 0.3 * x[m] - 1.0, 1.2 * x[m] + 1.0
    r[m] = RMAX * np.clip(CB1 * x[m] * a ** 4 + CB2 * x[m] ** 2 * a ** 3 + 1.0 - a ** 4 * b,
                          0.0, None) ** CB3
    m = (x >= XB) & (x < XM)
    r[m] = RMAX
    m = (x >= XM) & (x < XA)
    xi = (XA - x[m]) / 3.333333
    poly = (RH ** 2 + RH * K0 * xi ** 2
            + (20.0 - 20.0 * RH ** 2 - 4.0 * RH * K0 - K1 / 3.0) * xi ** 3
            + (-45.0 + 45.0 * RH ** 2 + 6.0 * RH * K0 + K1) * xi ** 4
            + (36.0 - 36.0 * RH ** 2 - 4.0 * RH * K0 - K1) * xi ** 5
            + (-10.0 + 10.0 * RH ** 2 + RH * K0 + K1 / 3.0) * xi ** 6)
    r[m] = RMAX * np.sqrt(np.clip(poly, 0.0, None))
    m = x >= XA
    r[m] = RH * RMAX * np.sqrt(np.clip(1.0 - (3.2 * x[m] - 44.733333) ** 2, 0.0, None))
    return r


def stations_ft(n_bow=120, n_mid=30, n_aft=140, n_cap=30):
    def cluster(a, b, n):
        s = (1.0 - np.cos(np.linspace(0.0, np.pi, n))) / 2.0
        return a + (b - a) * s

    x = np.concatenate([cluster(0.0, XB, n_bow), np.linspace(XB, XM, n_mid + 2)[1:-1],
                        cluster(XM, XA, n_aft), cluster(XA, XC, n_cap)])
    return np.unique(np.clip(x, 0.0, XC))


def write_half_hull(case: Path, n_theta: int = 48, far: float = 1.5,
                    symmetry_type: str = "symmetry") -> Path:
    """`case/constant/polyMesh` holding the hull's y >= 0 half and its symmetry plane.

    `symmetry_type` is the plane's patch type; pass "patch" to have no mirror at all."""
    x = stations_ft() * FT
    r = radius_ft(x / FT) * FT
    r[0] = r[-1] = 0.0
    theta = np.linspace(0.0, np.pi, n_theta + 1)
    points: list[tuple[float, float, float]] = []
    ring: dict[tuple[int, int], int] = {}

    def point(p):
        points.append(tuple(float(c) for c in p))
        return len(points) - 1

    nose, tail = point((x[0], 0.0, 0.0)), point((x[-1], 0.0, 0.0))
    for i in range(1, len(x) - 1):
        for j, t in enumerate(theta):
            ring[i, j] = point((x[i], r[i] * np.sin(t), r[i] * np.cos(t)))

    def at(i, j):
        return nose if i == 0 else tail if i == len(x) - 1 else ring[i, j]

    hull = []
    for i in range(len(x) - 1):
        for j in range(n_theta):
            quad = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]
            face = [p for k, p in enumerate(quad) if p not in quad[:k]]
            hull.append(face)

    # The plane y = 0, above and below the hull, sharing the hull's cut points.
    top = {i: point((x[i], 0.0, far)) for i in range(len(x))}
    bottom = {i: point((x[i], 0.0, -far)) for i in range(len(x))}
    symmetry = []
    for i in range(len(x) - 1):
        symmetry.append([at(i, 0), top[i], top[i + 1], at(i + 1, 0)])
        symmetry.append([at(i, n_theta), at(i + 1, n_theta), bottom[i + 1], bottom[i]])

    poly = Path(case) / "constant" / "polyMesh"
    poly.mkdir(parents=True, exist_ok=True)
    (poly / "points").write_text(_file("vectorField", "points", [
        f"({p[0]:.9g} {p[1]:.9g} {p[2]:.9g})" for p in points]), encoding="utf-8")
    (poly / "faces").write_text(_file("faceList", "faces", [
        f"{len(f)}({' '.join(map(str, f))})" for f in hull + symmetry]), encoding="utf-8")
    boundary = (f"2\n(\n    hull\n    {{\n        type wall;\n        nFaces {len(hull)};\n"
                f"        startFace 0;\n    }}\n    symmetry\n    {{\n        type {symmetry_type};\n"
                f"        nFaces {len(symmetry)};\n        startFace {len(hull)};\n    }}\n)\n")
    (poly / "boundary").write_text(_header("polyBoundaryMesh", "boundary") + boundary,
                                   encoding="utf-8")
    return Path(case)


def _header(cls: str, obj: str) -> str:
    return ("FoamFile\n{\n    version 2.0;\n    format ascii;\n"
            f"    class {cls};\n    location \"constant/polyMesh\";\n    object {obj};\n}}\n\n")


def _file(cls: str, obj: str, rows: list[str]) -> str:
    return _header(cls, obj) + f"{len(rows)}\n(\n" + "\n".join(rows) + "\n)\n"
