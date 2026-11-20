#!/usr/bin/env python3
"""An atmospheric boundary layer inlet, written from a site file or from numbers.

OpenFOAM already has these conditions, so this does not hand-roll them. ESI v2512
ships `atmBoundaryLayerInletVelocity`, `atmBoundaryLayerInletK`,
`atmBoundaryLayerInletEpsilon` and `atmBoundaryLayerInletOmega` in
`$FOAM_SRC/atmosphericModels/derivedFvPatchFields/`, and the solver evaluates the
profiles itself at every face centre. A `codedFixedValue` would re-implement them in
a string, compile it at run time, and be wrong in a way nothing checks. What this
writes is the dictionary those conditions read.

The profiles, from `atmBoundaryLayer.H` (the generalisation is YGCJ; RH is the
special case C1=0, C2=1, which is the default):

    u*    = kappa Uref / ln((Zref + z0)/z0)
    zEff  = z - d + z0,  z measured from the lowest point of the inlet patch
    U     = flowDir (u*/kappa) ln(zEff/z0)
    k     = u*^2 / sqrt(Cmu) sqrt(C1 ln(zEff/z0) + C2)
    eps   = u*^3 / (kappa zEff) sqrt(C1 ln(zEff/z0) + C2)
    omega = u* / (kappa sqrt(Cmu) zEff)

Two things about those that are easy to get wrong, and both are reported:

  * `z` is height above the **inlet patch's own minimum**, not above z = 0 and not
    above the ground. A domain whose inlet starts at z = 300 m still has the profile
    starting from zero there.
  * `Zref` goes into `u*` with no `d` in it, so when the displacement height is not
    zero, `Uref` is the speed at `z = Zref + d` above the patch minimum, not at
    `z = Zref`. With the default `d = 0` the two coincide.

It decides nothing on your behalf. `z0` has no default: it is the one number that
moves every profile on the page and no value is right for an unknown site, so this
refuses without `--z0` or `--roughness` and prints the Davenport classes. `Uref`,
`Zref`, `flowDir` come from a site file when there is one and from flags otherwise.

A site file comes from the CLI on your own machine, because this workspace has no
outbound network:

    openreynolds site --lat 51.5045 --lon -0.0865 --radius 500 -o site.json
    openreynolds push site.json

    python3 site_abl.py --site site.json --roughness closed --print
    python3 site_abl.py --site site.json --roughness closed --figure abl.png
    python3 site_abl.py <case> --site site.json --roughness closed --write
    python3 site_abl.py <case> --uref 10 --zref 10 --z0 1.0 --write --ground terrain
    python3 site_abl.py --uref 10 --zref 10 --z0 0.1 --table profile.csv --json

`--write` creates `<case>/0/include/ABLConditions` and points the named inlet patch
at the four conditions in whichever of `0/U`, `0/k`, `0/epsilon`, `0/omega` are
there, leaving every other patch's entry byte-identical. Running it twice on the
same case produces the same bytes. `--ground <patch>` additionally writes the
roughness wall functions on that patch.

Exit code is 0 whatever is found, and 2 when the inputs are refused.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# -- the roughness table -------------------------------------------------------------
#
# Davenport's classification as revised by Wieringa, which is the table the WMO guide
# carries. Offered so that a class can be named instead of a number being invented;
# the number is still yours.

ROUGHNESS: dict[str, tuple[float, str]] = {
    "sea": (0.0002, "open sea, fetch at least 5 km; mudflats, snow, ice"),
    "smooth": (0.005, "featureless land, no obstacles, negligible vegetation; beach"),
    "open": (0.03, "level country, low vegetation, isolated obstacles 20 obstacle heights apart"),
    "roughly-open": (0.10, "cultivated land, low crops, obstacles 15 heights apart"),
    "rough": (0.25, "scattered obstacles 10 heights apart; low orchards, bushy scrub"),
    "very-rough": (0.5, "obstacle groups 8 heights apart; orchards, low dense buildings"),
    "closed": (1.0, "obstacle groups about one height apart; mature forest, villages, suburbs"),
    "chaotic": (2.0, "city centres with a mix of low and high-rise buildings"),
}

KAPPA = 0.41
"""`atmBoundaryLayer.C`: `getCheckOrDefault<scalar>("kappa", 0.41, ...)`. Note that
the Hargreaves-Wright consistency result is quoted at kappa = 0.4, not 0.41."""

CMU = 0.09
"""`getCheckOrDefault<scalar>("Cmu", 0.09, ...)`, the standard k-epsilon value."""

C1_DEFAULT = 0.0
C2_DEFAULT = 1.0
"""`getOrDefault("C1", 0.0)` and `getOrDefault("C2", 1.0)`: the Richards-Hoxey case,
in which k does not vary with height."""

INLET_TYPES = {
    "U": "atmBoundaryLayerInletVelocity",
    "k": "atmBoundaryLayerInletK",
    "epsilon": "atmBoundaryLayerInletEpsilon",
    "omega": "atmBoundaryLayerInletOmega",
}

GROUND_TYPES = {
    "nut": ("atmNutkWallFunction", True),
    "epsilon": ("atmEpsilonWallFunction", True),
    "omega": ("atmOmegaWallFunction", True),
    "k": ("kqRWallFunction", False),
}
"""The roughness wall functions, and whether each takes a `z0` entry. The three
`atm*` ones require it (`atmNutkWallFunction.H`: "z0 | Surface roughness length [m]
| PatchFunction1<scalar> | yes"); `kqRWallFunction` is what the turbineSiting
tutorial puts on its terrain patch for k and takes no roughness."""

INCLUDE_PATH = "include/ABLConditions"

SCHEMA_PREFIX = "openreynolds.site/"

BANNER = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Website:  www.openfoam.com                      |
|   \\  /    A nd           | Written:  openreynolds site_abl.py               |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/"""

FOOTER = "// ************************************************************************* //"


class Refused(Exception):
    """The inputs are not the ones this reads. Exit 2, say what to pass."""


# -- the profiles ------------------------------------------------------------------
#
# Transcribed from atmBoundaryLayer.C, not from a textbook, because what matters is
# what the solver will compute and not what the literature says it should. The one
# liberty taken is `max(z0, ROOTVSMALL)`, which the source applies to z0 before
# everything else and which here is a refusal instead: a z0 of zero is a mistake
# worth naming rather than quietly turning into 1e-150.


def ustar(uref: float, zref: float, z0: float, kappa: float = KAPPA) -> float:
    """`kappa_*Uref/log((Zref + z0)/z0)`, atmBoundaryLayer.C."""
    return kappa * uref / math.log((zref + z0) / z0)


def z_eff(z: float, z0: float, d: float) -> float:
    """`(zDir & pCf) - groundMin - d + z0`, with z already relative to groundMin."""
    return z - d + z0


def profile(z: float, uref: float, zref: float, z0: float, d: float = 0.0,
            kappa: float = KAPPA, cmu: float = CMU, c1: float = C1_DEFAULT,
            c2: float = C2_DEFAULT) -> dict[str, float]:
    """U, k, epsilon and omega at one height, by the solver's own expressions."""
    us = ustar(uref, zref, z0, kappa)
    ze = z_eff(z, z0, d)
    if ze <= 0:
        raise Refused(
            f"refused: z = {z:g} m with d = {d:g} m and z0 = {z0:g} m puts z - d + z0 "
            f"at {ze:g}, which the log is not defined at -- a displacement height "
            f"above the first sampled point is the usual cause"
        )
    shape = math.sqrt(c1 * math.log(ze / z0) + c2)
    return {
        "z": z,
        "z_eff": ze,
        "U": (us / kappa) * math.log(ze / z0),
        "k": us * us / math.sqrt(cmu) * shape,
        "epsilon": us ** 3 / (kappa * ze) * shape,
        "omega": us / (kappa * math.sqrt(cmu) * ze),
    }


def heights(zref: float, top: float, count: int = 24, d: float = 0.0) -> list[float]:
    """A geometric ladder from just above the profile's floor to `top`.

    Geometric because the profile does all of its bending in the first few metres: an
    even spacing over 200 m puts two points in the part that matters.

    The floor is `d`, not zero. The ladder used to start at 0.05 m whatever `d` was,
    and the first realistic displacement height it met -- 15 m, for a city centre --
    put every low sample below `d`, where the log is undefined, so the whole table
    refused rather than printing. The ladder is offset by `d` and the sample at
    `zref + d` is forced in, because that is where `Uref` actually occurs.
    """
    rise = max(0.05, zref / 200.0)
    low, high = d + rise, max(d + rise * 2, top)
    ladder = [low * (high / low) ** (i / (count - 1)) for i in range(count)]
    ladder.append(zref + d)
    return sorted(set(round(z, 4) for z in ladder))


# -- what the inlet needs told ------------------------------------------------------


def resolve(args: argparse.Namespace) -> dict[str, Any]:
    """Uref, Zref, z0, d, flowDir, zDir, and where each one came from."""
    source: dict[str, str] = {}
    site: dict[str, Any] = {}
    if args.site:
        site = load_site(Path(args.site))

    z0 = args.z0
    if z0 is None and args.roughness:
        key = args.roughness.strip().lower()
        if key not in ROUGHNESS:
            raise Refused(
                f"refused: no roughness class {args.roughness!r}. The classes are "
                + ", ".join(sorted(ROUGHNESS))
            )
        z0 = ROUGHNESS[key][0]
        source["z0"] = f"--roughness {key} ({ROUGHNESS[key][1]})"
    elif z0 is not None:
        source["z0"] = "--z0"
    if z0 is None:
        raise Refused(refusal_for_z0(site))
    if z0 <= 0:
        raise Refused(
            f"refused: z0 = {z0:g}. The log law divides by ln(z/z0) and is undefined "
            f"at zero roughness; the smoothest Davenport class is sea at 0.0002 m"
        )

    uref, zref, flow_dir = args.uref, args.zref, None
    if site:
        wind = wind_from_site(site, args.statistic, args.sector)
        if uref is None:
            uref = wind["speed"]
            source["Uref"] = wind["from"]
        if zref is None:
            zref = wind["height"]
            source["Zref"] = wind["from"]
        flow_dir = wind["flow_dir"]
        source["flowDir"] = wind["from"]
    if uref is None or zref is None:
        raise Refused(
            "refused: no reference wind. Either --site <file> with a weather block, "
            "or both --uref <m/s> and --zref <m>"
        )
    source.setdefault("Uref", "--uref")
    source.setdefault("Zref", "--zref")
    if zref < 0:
        raise Refused(
            f"refused: Zref = {zref:g}. atmBoundaryLayer::Ustar aborts on a negative "
            f"Zref (`FatalErrorInFunction`), so this stops here instead"
        )

    if args.flow_dir:
        flow_dir = parse_vector(args.flow_dir, "--flow-dir")
        source["flowDir"] = "--flow-dir"
    if flow_dir is None:
        flow_dir = [1.0, 0.0, 0.0]
        source["flowDir"] = "the default, +x"
    z_dir = parse_vector(args.z_dir, "--z-dir") if args.z_dir else [0.0, 0.0, 1.0]
    source["zDir"] = "--z-dir" if args.z_dir else "the default, +z"
    source["d"] = "--d" if args.d else "the default, 0"

    return {
        "Uref": float(uref),
        "Zref": float(zref),
        "z0": float(z0),
        "d": float(args.d),
        "flowDir": flow_dir,
        "zDir": z_dir,
        "kappa": float(args.kappa),
        "Cmu": float(args.cmu),
        "C1": float(args.c1),
        "C2": float(args.c2),
        "from": source,
        "site": site,
    }


def refusal_for_z0(site: dict[str, Any]) -> str:
    lines = [
        "refused: no roughness length. z0 moves every profile this writes and no "
        "value is right for a site this has not seen, so it is yours to name:",
        "",
        "    --z0 <metres>        a number you have",
        "    --roughness <class>  one of the Davenport classes below",
        "",
    ]
    for key, (value, description) in ROUGHNESS.items():
        lines.append(f"    {key:<14} {value:<8g} {description}")
    shear = ((site.get("weather") or {}).get("shear") or {})
    if shear.get("fitted_z0_m") is not None:
        lines += [
            "",
            f"  the site file fits z0 = {shear['fitted_z0_m']:g} m to the reanalysis "
            f"10 m and 100 m means, which is the grid cell's roughness and not the",
            "  site's -- an order of magnitude to sanity-check against, not a value "
            "to pass on.",
        ]
    return "\n".join(lines)


def parse_vector(text: str, flag: str) -> list[float]:
    parts = [p for p in re.split(r"[,\s()]+", text.strip()) if p]
    if len(parts) != 3:
        raise Refused(f"refused: {flag} wants three numbers, got {text!r}")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise Refused(f"refused: {flag} is not three numbers: {text!r}") from exc


# -- the site file -----------------------------------------------------------------


def load_site(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Refused(
            f"refused: no site file at {path.as_posix()}.\n"
            f"  It is fetched on your own machine, because this workspace has no "
            f"outbound network:\n"
            f"    openreynolds site --lat <deg> --lon <deg> --radius <m> -o site.json\n"
            f"    openreynolds push site.json\n"
            f"  Or skip it entirely and pass --uref, --zref and --z0."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Refused(f"refused: {path.as_posix()} is not readable JSON: {exc}") from exc
    schema = str(data.get("schema") or "")
    if not schema.startswith(SCHEMA_PREFIX):
        raise Refused(
            f"refused: {path.as_posix()} says schema {schema!r}; this reads "
            f"{SCHEMA_PREFIX}1 files, which `openreynolds site` writes"
        )
    return data


def wind_from_site(site: dict[str, Any], statistic: str, sector: float | None) -> dict[str, Any]:
    """The reference speed, its height and the flow direction, out of a site file."""
    weather = site.get("weather") or {}
    if weather.get("failed"):
        raise Refused(
            f"refused: the site file's weather block records a failure rather than "
            f"data: {weather['failed']}"
        )
    if not weather:
        raise Refused(
            "refused: the site file has no weather block, so there is no reference "
            "wind in it; --uref and --zref say it directly"
        )
    sectors = weather.get("sectors") or []
    chosen: dict[str, Any] | None = None
    if sector is not None:
        chosen = min(sectors, key=lambda s: abs(((s["centre_deg"] - sector + 180) % 360) - 180)) \
            if sectors else None
        if chosen is None:
            raise Refused("refused: the site file has no sectors to choose from")
        label = f"sector centred {chosen['centre_deg']:g} deg"
    else:
        prevailing = weather.get("prevailing") or {}
        if not prevailing:
            raise Refused("refused: the site file names no prevailing sector")
        chosen = next((s for s in sectors
                       if s["centre_deg"] == prevailing["centre_deg"]), None)
        if chosen is None:
            raise Refused("refused: the prevailing sector is not among the sectors")
        label = (f"the prevailing sector, centred {chosen['centre_deg']:g} deg, "
                 f"{chosen.get('frequency', 0) * 100:.0f}% of hours")

    stats = chosen.get("speed_10m") or {}
    if statistic not in stats:
        raise Refused(
            f"refused: the site file's {label} has no {statistic!r}; it has "
            + ", ".join(k for k in stats if k != "hours")
        )
    speed = stats[statistic]
    if not speed:
        raise Refused(f"refused: {label} has {statistic} = {speed!r}")

    centre = chosen["centre_deg"]
    towards = math.radians((centre + 180.0) % 360.0)
    flow_dir = [round(math.sin(towards), 6), round(math.cos(towards), 6), 0.0]
    return {
        "speed": float(speed),
        "height": 10.0,
        "flow_dir": flow_dir,
        "from": (f"site file: {statistic} of the 10 m wind over {label} "
                 f"({chosen.get('hours', 0)} hours)"),
    }


# -- the dictionaries --------------------------------------------------------------


def vector_text(values: list[float]) -> str:
    return "(" + " ".join(f"{v + 0.0:g}" for v in values) + ")"


def conditions_text(inputs: dict[str, Any]) -> str:
    """`0/include/ABLConditions`, in the shape the turbineSiting tutorial uses it.

    Only the six mandatory entries plus whichever optionals were moved off their
    defaults: an entry written at its own default reads as a decision that was made,
    and reading `kappa 0.41;` tells nobody whether it was chosen or inherited.
    """
    us = ustar(inputs["Uref"], inputs["Zref"], inputs["z0"], inputs["kappa"])
    lines = [
        BANNER,
        "",
        f"// Uref, Zref and flowDir: {inputs['from'].get('Uref')}",
        f"// z0: {inputs['from'].get('z0')}",
        f"// u* = kappa Uref / ln((Zref + z0)/z0) = {us:.6g} m/s",
        "",
        f"Uref                 {inputs['Uref']:g};",
        f"Zref                 {inputs['Zref']:g};",
        f"zDir                 {vector_text(inputs['zDir'])};",
        f"flowDir              {vector_text(inputs['flowDir'])};",
        f"z0                   uniform {inputs['z0']:g};",
        f"d                    uniform {inputs['d']:g};",
    ]
    for key, default in (("kappa", KAPPA), ("Cmu", CMU), ("C1", C1_DEFAULT), ("C2", C2_DEFAULT)):
        if inputs[key] != default:
            lines.append(f"{key:<20} {inputs[key]:g};")
    lines += ["", FOOTER, ""]
    return "\n".join(lines)


def inlet_block(field: str, indent: str = "    ") -> str:
    """The patch entry: the type, the conditions include, and a `value`.

    The `value` is not decoration and it is not what the turbineSiting tutorial
    writes. `atmBoundaryLayerInletEpsilonFvPatchScalarField.C:72-75` reads it only
    `if (!initABL_)`, so by the source a tutorial-shaped entry with no `value` should
    be enough -- and on a real ESI build it is not. Measured on OpenFOAM v2412
    (build `_b8cf4d35-20260127`), reading a field whose inlet entry is the two
    tutorial lines fails:

        --> FOAM FATAL IO ERROR
        Required entry 'value' : missing for patch inlet in dictionary
        "0/epsilon/boundaryField/inlet"

    and it fails the same way with `initABL true;` added, inside the patch entry or
    in the include. So the entry carries one. `$internalField` is the macro the
    tutorials use elsewhere for exactly this: a placeholder the condition overwrites
    the first time it is evaluated.
    """
    inner = indent + "    "
    return (f"{indent}{{\n"
            f"{inner}type            {INLET_TYPES[field]};\n"
            f'{inner}#include        "{INCLUDE_PATH}"\n'
            f"{inner}value           $internalField;\n"
            f"{indent}}}")


def ground_block(field: str, z0: float, indent: str = "    ") -> str:
    kind, takes_z0 = GROUND_TYPES[field]
    inner = indent + "    "
    lines = [f"{indent}{{", f"{inner}type            {kind};"]
    if takes_z0:
        lines.append(f"{inner}z0              uniform {z0:g};")
    if field == "nut":
        lines.append(f"{inner}value           uniform 0;")
    else:
        lines.append(f"{inner}value           $internalField;")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


# -- editing a field file ----------------------------------------------------------
#
# A field file is replaced at one patch and nowhere else, by finding the patch name
# and matching braces from there. Not a parser: a parser for OpenFOAM's dictionary
# grammar is a project, and what is needed here is to find one balanced block and put
# another in its place. A name that cannot be found unambiguously is refused rather
# than guessed at, because the wrong patch rewritten is a case that runs and lies.


def find_patch_block(text: str, patch: str) -> tuple[int, int, str]:
    """Where `patch`'s `{ ... }` is: the brace positions and the leading indent."""
    pattern = re.compile(
        r"^(?P<indent>[ \t]*)(?P<name>\"?" + re.escape(patch) + r"\"?)[ \t]*$",
        re.M,
    )
    found = [m for m in pattern.finditer(text)]
    if not found:
        inline = re.compile(
            r"^(?P<indent>[ \t]*)(?P<name>\"?" + re.escape(patch) + r"\"?)[ \t]*\{",
            re.M,
        )
        found = [m for m in inline.finditer(text)]
    if not found:
        raise KeyError(patch)
    if len(found) > 1:
        raise Refused(
            f"refused: the name {patch!r} appears {len(found)} times at the start of "
            f"a line, so which block is the patch is not clear; --print writes the "
            f"entries out instead and leaves the file alone"
        )
    match = found[0]
    opening = text.find("{", match.end() - 1)
    if opening < 0:
        raise Refused(f"refused: {patch!r} is named but has no {{ after it")
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return opening, index + 1, match.group("indent")
    raise Refused(f"refused: {patch!r}'s block is never closed")


def replace_patch(text: str, patch: str, body_for: Any) -> str:
    opening, closing, indent = find_patch_block(text, patch)
    return text[:opening] + body_for(indent).lstrip() + text[closing:]


def ensure_include(text: str) -> tuple[str, bool]:
    """Put the conditions include at the top of `boundaryField`, once.

    `nut` needs it even though no ABL inlet condition goes on `nut`: its roughness
    wall function is written with `z0 uniform <value>` here, but the tutorial writes
    `$z0` and resolves it from this include, and a case edited by hand afterwards
    will reach for the same macro. Having the include there costs nothing and means
    `$z0`, `$Uref` and the rest resolve wherever someone expects them to.
    """
    if INCLUDE_PATH in text:
        return text, False
    match = re.search(r"^boundaryField\s*$\s*^\{[ \t]*$", text, re.M)
    if not match:
        match = re.search(r"^boundaryField[ \t]*\{", text, re.M)
        if not match:
            return text, False
    insert = match.end()
    return text[:insert] + f'\n    #include "{INCLUDE_PATH}"\n' + text[insert:], True


# -- writing into a case -----------------------------------------------------------


def time_dir(case: Path) -> Path:
    for name in ("0", "0.orig"):
        if (case / name).is_dir():
            return case / name
    raise Refused(
        f"refused: no 0/ or 0.orig/ under {case.as_posix()}, so there are no field "
        f"files to point at the inlet"
    )


def write_case(case: Path, inputs: dict[str, Any], inlet: str,
               ground: str = "") -> list[str]:
    """Write the include file and re-point the patches. Returns what happened."""
    zero = time_dir(case)
    report: list[str] = []

    include = zero / INCLUDE_PATH
    include.parent.mkdir(parents=True, exist_ok=True)
    text = conditions_text(inputs)
    before = include.read_text(encoding="utf-8") if include.is_file() else ""
    include.write_text(text, encoding="utf-8")
    verb = "unchanged" if before == text else ("rewrote" if before else "wrote")
    report.append(f"{verb} {include.relative_to(case).as_posix()}")

    for field in ("U", "k", "epsilon", "omega"):
        path = zero / field
        if not path.is_file():
            report.append(f"no 0/{field}, so nothing to point at the inlet")
            continue
        text = path.read_text(encoding="utf-8")
        try:
            new = replace_patch(text, inlet, lambda indent: inlet_block(field, indent))
        except KeyError:
            report.append(f"0/{field} has no patch {inlet!r}, left alone")
            continue
        if new == text:
            report.append(f"0/{field}: {inlet} already {INLET_TYPES[field]}")
        else:
            path.write_text(new, encoding="utf-8")
            report.append(f"0/{field}: {inlet} -> {INLET_TYPES[field]}")

    nut = zero / "nut"
    if nut.is_file():
        text = nut.read_text(encoding="utf-8")
        text, added = ensure_include(text)
        if added:
            nut.write_text(text, encoding="utf-8")
            report.append(f'0/nut: added #include "{INCLUDE_PATH}"')

    if ground:
        for field, (kind, _z0) in GROUND_TYPES.items():
            path = zero / field
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            try:
                new = replace_patch(text, ground,
                                    lambda indent, f=field: ground_block(f, inputs["z0"], indent))
            except KeyError:
                report.append(f"0/{field} has no patch {ground!r}, left alone")
                continue
            if new != text:
                path.write_text(new, encoding="utf-8")
                report.append(f"0/{field}: {ground} -> {kind}")
            else:
                report.append(f"0/{field}: {ground} already {kind}")
    return report


# -- reporting ---------------------------------------------------------------------


def table(inputs: dict[str, Any], top: float) -> list[dict[str, float]]:
    return [profile(z, inputs["Uref"], inputs["Zref"], inputs["z0"], inputs["d"],
                    inputs["kappa"], inputs["Cmu"], inputs["C1"], inputs["C2"])
            for z in heights(inputs["Zref"], top, d=inputs["d"])]


def notes(inputs: dict[str, Any]) -> list[str]:
    """Facts about what was written. No verdicts: whether any of them matters here
    depends on the case, and the case is not this script's to judge."""
    us = ustar(inputs["Uref"], inputs["Zref"], inputs["z0"], inputs["kappa"])
    out = [
        f"u* = {us:.4g} m/s, from kappa {inputs['kappa']:g} x Uref "
        f"{inputs['Uref']:g} / ln(({inputs['Zref']:g} + {inputs['z0']:g})/{inputs['z0']:g})",
        f"z in every expression is height above the inlet patch's own lowest point, "
        f"not above z = 0",
    ]
    if inputs["C1"] == 0.0 and inputs["C2"] == 1.0:
        out.append(
            f"k is constant with height at u*^2/sqrt(Cmu) = "
            f"{us * us / math.sqrt(inputs['Cmu']):.4g} m2/s2: that is what C1 = 0 and "
            f"C2 = 1 mean, and it is the Richards-Hoxey form"
        )
    else:
        out.append(
            f"C1 = {inputs['C1']:g}, C2 = {inputs['C2']:g}, so k varies with height "
            f"(the YGCJ generalisation rather than Richards-Hoxey)"
        )
    if inputs["d"]:
        out.append(
            f"d = {inputs['d']:g} m and Ustar() has no d in it, so Uref "
            f"{inputs['Uref']:g} m/s is the speed at z = Zref + d = "
            f"{inputs['Zref'] + inputs['d']:g} m above the patch minimum"
        )
    out.append(
        "these profiles satisfy the k equation exactly; they satisfy the epsilon "
        "equation only when sigmaEpsilon is 1.11 at kappa 0.4 (Hargreaves and "
        "Wright 2007, p. 358), which is not the kEpsilon default of 1.3 -- "
        "atmBoundaryLayer.H states it and leaves it alone, and so does this"
    )
    # Sampled at d + 2 z0, not at 2 z0: with a displacement height the second is
    # below the profile's own floor and `profile` refuses it, which turned a report
    # into a traceback the first time `notes` was called with `--d`.
    probe = inputs["d"] + inputs["z0"] * 2
    first = profile(probe, inputs["Uref"], inputs["Zref"], inputs["z0"],
                    inputs["d"], inputs["kappa"], inputs["Cmu"], inputs["C1"], inputs["C2"])
    out.append(
        f"U is zero at z = d = {inputs['d']:g} m (z - d + z0 is then z0 and the log of "
        f"one is zero), negative below it, and undefined at or below z = d - z0 = "
        f"{inputs['d'] - inputs['z0']:g} m; at z = {probe:g} m (d + 2 z0) it is "
        f"U = {first['U']:.3g} m/s"
    )
    return out


def figure(rows: list[dict[str, float]], inputs: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    us = ustar(inputs["Uref"], inputs["Zref"], inputs["z0"], inputs["kappa"])
    z = [r["z"] for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(13, 5), sharey=True)
    for axis, key, label in (
        (axes[0], "U", "U  [m/s]"),
        (axes[1], "k", "k  [m$^2$/s$^2$]"),
        (axes[2], "epsilon", r"$\epsilon$  [m$^2$/s$^3$]"),
        (axes[3], "omega", r"$\omega$  [1/s]"),
    ):
        axis.plot([r[key] for r in rows], z, marker="o", markersize=3)
        axis.set_xlabel(label)
        axis.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("z above the inlet patch minimum  [m]")
    axes[0].set_yscale("log")
    axes[2].set_xscale("log")
    axes[3].set_xscale("log")
    axes[0].axhline(inputs["Zref"], color="crimson", lw=1, ls="--")
    axes[0].axvline(inputs["Uref"], color="crimson", lw=1, ls="--")
    axes[0].annotate(f"Uref {inputs['Uref']:g} m/s at Zref {inputs['Zref']:g} m",
                     (inputs["Uref"], inputs["Zref"]), textcoords="offset points",
                     xytext=(6, 6), color="crimson", fontsize=8)
    fig.suptitle(
        f"atmBoundaryLayerInlet: z0 = {inputs['z0']:g} m, d = {inputs['d']:g} m, "
        f"u* = {us:.4g} m/s, kappa = {inputs['kappa']:g}, Cmu = {inputs['Cmu']:g}, "
        f"C1 = {inputs['C1']:g}, C2 = {inputs['C2']:g}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="An atmospheric boundary layer inlet for an OpenFOAM case.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="z0 has no default; --roughness lists the Davenport classes.",
    )
    parser.add_argument("case", nargs="?", type=Path, help="The case directory, for --write.")
    parser.add_argument("--site", default="", help="A site.json from `openreynolds site`.")
    parser.add_argument("--statistic", default="mean",
                        choices=("mean", "p50", "p90", "p98", "max"),
                        help="Which of the site file's sector speeds to use as Uref.")
    parser.add_argument("--sector", type=float, default=None,
                        help="Use the sector nearest this bearing instead of the prevailing one.")
    parser.add_argument("--uref", type=float, default=None, help="Reference speed, m/s.")
    parser.add_argument("--zref", type=float, default=None, help="Reference height, m.")
    parser.add_argument("--z0", type=float, default=None, help="Roughness length, m.")
    parser.add_argument("--roughness", default="",
                        help="A Davenport class instead of --z0: " + ", ".join(sorted(ROUGHNESS)))
    parser.add_argument("--d", type=float, default=0.0, help="Displacement height, m.")
    parser.add_argument("--flow-dir", default="", help="Streamwise direction, e.g. '1 0 0'.")
    parser.add_argument("--z-dir", default="", help="Ground-normal direction, default '0 0 1'.")
    parser.add_argument("--kappa", type=float, default=KAPPA, help=f"Default {KAPPA}.")
    parser.add_argument("--cmu", type=float, default=CMU, help=f"Default {CMU}.")
    parser.add_argument("--c1", type=float, default=C1_DEFAULT, help=f"Default {C1_DEFAULT}.")
    parser.add_argument("--c2", type=float, default=C2_DEFAULT, help=f"Default {C2_DEFAULT}.")
    parser.add_argument("--inlet", default="inlet", help="The inlet patch name.")
    parser.add_argument("--ground", default="",
                        help="A ground patch to put the roughness wall functions on.")
    parser.add_argument("--top", type=float, default=None,
                        help="Highest height in the table and figure. Default 20 x Zref.")
    parser.add_argument("--write", action="store_true", help="Edit the case.")
    parser.add_argument("--print", dest="show", action="store_true",
                        help="Print the dictionary entries and change nothing.")
    parser.add_argument("--table", type=Path, default=None, help="Write the profile as CSV.")
    parser.add_argument("--figure", type=Path, default=None, help="Write the profile as a PNG.")
    parser.add_argument("--json", dest="as_json", action="store_true", help="One JSON object.")
    args = parser.parse_args(argv)

    try:
        inputs = resolve(args)
        top = args.top if args.top else inputs["Zref"] * 20
        rows = table(inputs, top)
        report: list[str] = []
        if args.write:
            if not args.case:
                raise Refused("refused: --write needs a case directory")
            if not args.case.is_dir():
                raise Refused(f"refused: no directory at {args.case.as_posix()}")
            report = write_case(args.case, inputs, args.inlet, args.ground)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.table:
        with args.table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.figure:
        figure(rows, inputs, args.figure)

    payload = {
        "inputs": {k: v for k, v in inputs.items() if k != "site"},
        "ustar": ustar(inputs["Uref"], inputs["Zref"], inputs["z0"], inputs["kappa"]),
        "inlet_types": INLET_TYPES,
        "profile": rows,
        "notes": notes(inputs),
        "wrote": report,
    }
    if args.as_json:
        print(json.dumps(payload, indent=2))
        return 0

    for key in ("Uref", "Zref", "z0", "d"):
        print(f"{key:<8} {inputs[key]:<12g} {inputs['from'].get(key, '')}")
    print(f"{'flowDir':<8} {vector_text(inputs['flowDir']):<12} {inputs['from'].get('flowDir', '')}")
    print(f"{'zDir':<8} {vector_text(inputs['zDir']):<12} {inputs['from'].get('zDir', '')}")
    print("")
    for line in payload["notes"]:
        print(f"  {line}")
    print("")
    print(f"  {'z [m]':>9} {'U [m/s]':>9} {'k':>10} {'epsilon':>11} {'omega':>10}")
    for row in rows:
        print(f"  {row['z']:>9.3f} {row['U']:>9.4f} {row['k']:>10.5g} "
              f"{row['epsilon']:>11.5g} {row['omega']:>10.5g}")
    if args.show:
        print("")
        print(f"--- {INCLUDE_PATH} " + "-" * 40)
        print(conditions_text(inputs))
        for field in INLET_TYPES:
            print(f"--- 0/{field}, patch {args.inlet} " + "-" * 30)
            print(f"    {args.inlet}")
            print(inlet_block(field))
        if args.ground:
            for field in GROUND_TYPES:
                print(f"--- 0/{field}, patch {args.ground} " + "-" * 30)
                print(f"    {args.ground}")
                print(ground_block(field, inputs["z0"]))
    for line in report:
        print(f"  {line}")
    if args.table:
        print(f"  wrote {args.table.as_posix()}")
    if args.figure:
        print(f"  wrote {args.figure.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
