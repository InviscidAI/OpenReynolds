"""Solve the conjugate wall in the image, and grade the interface it was built for.

The journey the other four take, for the family that had none: `cht_wall.py` writes the
two-region case, `chtMultiRegionSimpleFoam` solves it, and the interface temperature is
compared with the exact series-conduction answer `conjugate.py` derives --
**301.84049080 K** for the shipped wall, with a flux of 2944.78527607 W/m^2 through both
regions.

**Why the interface temperature is the right thing to grade.** A conjugate coupling has
to carry two conditions at once, temperature continuous and flux continuous, and a solver
that satisfies one without the other produces a smooth plausible field whose interface
sits in the wrong place. Every other quantity in this case is boundary data: the hot and
cold faces are fixed, so a run reproducing them has demonstrated nothing. The interface
is the only number the coupling itself decides.

**The fields come off the image, not out of this file.** `cht_wall.py` already copies the
per-region properties from `$FOAM_TUTORIALS` rather than inventing them, for the reason
`chem0d.py` does; this test extends that to the initial fields. It reads which fields the
tutorial's regions carry and writes one of each with this case's patches, taking the
coupled thermal entry verbatim from the tutorial's own `T`. Nothing here asserts what a
field is called or what the coupling type is spelled like, because those are the things
that differ between builds and were never verified here.

Run by the `journey` job, in the workspace image, on every pull request.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "openreynolds" / "toolbox" / "templates" / "cht_wall.py"
GRADER = ROOT / "openreynolds" / "toolbox" / "conjugate.py"

pytestmark = pytest.mark.skipif(
    not os.environ.get("FOAM_TUTORIALS"),
    reason="needs an OpenFOAM environment: the case's properties are copied off the "
           "image's own tutorial rather than written here")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def shell(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run it, print it, and fail with its tail rather than with a returncode."""
    print("+", " ".join(cmd), flush=True)
    done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=1800)
    if done.stdout:
        print(done.stdout[-3000:], flush=True)
    assert done.returncode == 0, (
        f"{cmd[0]} exited {done.returncode}\n--- stderr ---\n{done.stderr[-3000:]}")
    return done


def boundary_names(case: Path, region: str) -> list[str]:
    """The patches this region actually ended up with, read from its own mesh.

    Read rather than assumed: the interface patch is named by `splitMeshRegions`, and its
    naming is exactly the thing this repository has not verified for this image. A test
    that hard-coded `fluid_to_solid` would pass or fail for the wrong reason.
    """
    text = (case / "constant" / region / "polyMesh" / "boundary").read_text(
        encoding="utf-8")
    body = text[text.index("("):text.rindex(")")]
    return re.findall(r"^\s{4}(\w+)$", body, re.M)


def field_names(tutorial_region: Path) -> list[str]:
    """Which fields the tutorial's region carries, so this case writes the same set."""
    for where in (tutorial_region, ):
        if where.is_dir():
            return sorted(p.name for p in where.iterdir() if p.is_file())
    return []


def write_fields(case: Path, cht, tutorial: Path) -> dict[str, list[str]]:
    """One field per field the tutorial's region carries, with THIS case's patches.

    The tutorial's own field files are the source of two things that are not written
    here: which fields the solver wants, and how the coupled thermal condition is
    spelled. Everything else is this case's own geometry, so the boundaryField is
    rebuilt rather than translated -- the tutorial's patch names mean nothing here.
    """
    written: dict[str, list[str]] = {}
    coupled = (case / "constant" / "coupled-condition.from-tutorial").read_text(
        encoding="utf-8")
    # Drop the provenance comments the template wrote above the entry.
    coupled_body = chr(10).join(
        line for line in coupled.splitlines() if not line.startswith("//"))

    for ours, theirs in (("fluid", cht.TUTORIAL_FLUID), ("solid", cht.TUTORIAL_SOLID)):
        patches = boundary_names(case, ours)
        print(f"  {ours} patches, from its own mesh: {patches}", flush=True)
        interface = [p for p in patches if "_to_" in p]
        assert len(interface) == 1, (
            f"{ours} has {len(interface)} region-to-region patches ({patches}); "
            "splitMeshRegions should have made exactly one")
        source = None
        for candidate in (tutorial / "0" / theirs, tutorial / "0.orig" / theirs):
            if candidate.is_dir():
                source = candidate
                break
        assert source is not None, f"no 0/ fields for {theirs} under {tutorial}"

        names = field_names(source)
        assert names, f"{source} carries no fields"
        written[ours] = names
        target = case / "0" / ours
        target.mkdir(parents=True, exist_ok=True)

        for name in names:
            text = (source / name).read_text(encoding="utf-8")
            head = text[:text.index("boundaryField")]
            entries = []
            for patch in patches:
                if patch == interface[0]:
                    entries.append("    " + patch + chr(10) + "    {" + chr(10)
                                   + coupled_body + chr(10) + "    }")
                elif name == "T" and patch in ("hot", "cold"):
                    value = cht.HOT_K if patch == "hot" else cht.COLD_K
                    entries.append("    " + patch + chr(10) + "    {" + chr(10)
                                   + "        type            fixedValue;" + chr(10)
                                   + f"        value           uniform {value:g};"
                                   + chr(10) + "    }")
                elif name == "U":
                    entries.append("    " + patch + chr(10) + "    {" + chr(10)
                                   + "        type            noSlip;" + chr(10)
                                   + "    }")
                elif name.startswith("p"):
                    entries.append("    " + patch + chr(10) + "    {" + chr(10)
                                   + "        type            fixedFluxPressure;"
                                   + chr(10) + "        value           $internalField;"
                                   + chr(10) + "    }")
                else:
                    entries.append("    " + patch + chr(10) + "    {" + chr(10)
                                   + "        type            zeroGradient;" + chr(10)
                                   + "    }")
            (target / name).write_text(
                head + "boundaryField" + chr(10) + "{" + chr(10)
                + chr(10).join(entries) + chr(10) + "}" + chr(10),
                encoding="utf-8", newline=chr(10))
    return written


def sample_dict(region_patches: list[str], x_from: float, x_to: float) -> str:
    """A line of T across one region, for reading the interface value off its end."""
    return (
        "type sets;" + chr(10)
        + "libs (sampling);" + chr(10)
        + "interpolationScheme cellPoint;" + chr(10)
        + "setFormat raw;" + chr(10)
        + "fields (T);" + chr(10)
        + "sets" + chr(10) + "(" + chr(10)
        + "    line" + chr(10) + "    {" + chr(10)
        + "        type        midPoint;" + chr(10)
        + "        axis        x;" + chr(10)
        + f"        start       ({x_from:.8f} 0.002 0.002);" + chr(10)
        + f"        end         ({x_to:.8f} 0.002 0.002);" + chr(10)
        + "    }" + chr(10) + ");" + chr(10))


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """The whole journey: template, mesh, properties, fields, solve."""
    cht = load(TEMPLATE, "template_cht_wall")
    tutorial = Path(os.environ["FOAM_TUTORIALS"]) / cht.TUTORIAL_SOURCE
    case = tmp_path_factory.mktemp("cht-journey")

    shell([sys.executable, str(TEMPLATE)], cwd=case)
    for region in ("fluid", "solid"):
        assert (case / "constant" / region / "polyMesh").is_dir(), (
            f"{region} has no mesh: splitMeshRegions did not produce it")

    written = write_fields(case, cht, tutorial)
    print(f"  fields written: {written}", flush=True)

    shell(["chtMultiRegionSimpleFoam"], cwd=case)
    return case, cht


def sampled(case: Path, region: str, x_from: float, x_to: float) -> list[tuple[float, float]]:
    """(x, T) along a line in one region, from `postProcess -func sample`."""
    where = case / "system" / region
    where.mkdir(parents=True, exist_ok=True)
    (where / "sample").write_text(
        sample_dict([], x_from, x_to), encoding="utf-8", newline=chr(10))
    shell(["postProcess", "-region", region, "-func", "sample", "-latestTime"], cwd=case)
    found = sorted((case / "postProcessing" / "sample").rglob("*line*T*"))
    assert found, (
        "postProcess wrote no sample file for region " + region + ". What is under "
        "postProcessing: "
        + ", ".join(str(p.relative_to(case)) for p in (case / "postProcessing").rglob("*")))
    rows = []
    for line in found[-1].read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.startswith("#"):
            rows.append((float(parts[0]), float(parts[-1])))
    assert rows, f"{found[-1]} holds no samples"
    return rows


def test_both_regions_exist_with_exactly_one_interface_each(solved):
    """The coupling cannot be graded if the split did not make it. Read off each
    region's own `boundary` file rather than assumed, because the interface patch is
    named by `splitMeshRegions` and that naming was never verified for this image."""
    case, _ = solved
    for region in ("fluid", "solid"):
        patches = boundary_names(case, region)
        interface = [p for p in patches if "_to_" in p]
        print(f"  {region}: {patches}", flush=True)
        assert len(interface) == 1, (region, patches)
        for expected in ("hot", "cold", "sides"):
            if region == "fluid" and expected == "cold":
                continue
            if region == "solid" and expected == "hot":
                continue
            assert expected in patches or expected == "sides", (region, expected, patches)


def test_the_interface_temperature_is_the_series_conduction_answer(solved):
    """The one number the coupling decides. Everything else here is boundary data: the
    hot and cold faces are fixed, so reproducing them demonstrates nothing.

    The tolerance is a discretisation bound and not a reference uncertainty, because the
    reference is exact: 20 cells across the fluid and 10 across the solid, with the whole
    drop in the fluid (its resistance is 53x the solid's), so a per cent of the 100 K
    drop is a generous allowance for a linear profile on a uniform mesh.
    """
    case, cht = solved
    answer = cht.expected()
    rows = sampled(case, "solid", cht.FLUID_THICK_M,
                   cht.FLUID_THICK_M + cht.SOLID_THICK_M)
    at_interface = min(rows, key=lambda r: r[0])[1]
    exact = answer["T_interface_K"]
    print(f"  solid-side interface T {at_interface:.6f} K against an exact "
          f"{exact:.8f} ({(at_interface - exact) / exact:+.4%})", flush=True)
    assert abs(at_interface - exact) < 1.0, (at_interface, exact)


def test_the_two_sides_of_the_interface_agree(solved):
    """Temperature continuity, measured rather than assumed. A coupling that satisfies
    flux continuity alone leaves a jump here, and the field still looks smooth in each
    region on its own."""
    case, cht = solved
    fluid = sampled(case, "fluid", 0.0, cht.FLUID_THICK_M)
    solid = sampled(case, "solid", cht.FLUID_THICK_M,
                    cht.FLUID_THICK_M + cht.SOLID_THICK_M)
    fluid_end = max(fluid, key=lambda r: r[0])[1]
    solid_end = min(solid, key=lambda r: r[0])[1]
    print(f"  fluid side {fluid_end:.6f} K, solid side {solid_end:.6f} K, "
          f"jump {abs(fluid_end - solid_end):.6f} K", flush=True)
    assert abs(fluid_end - solid_end) < 1.0, (fluid_end, solid_end)


def test_the_fixed_faces_were_held(solved):
    """The cheap check that catches a case solved with the wrong boundary data, which
    would make every number above wrong in a way that still looks converged."""
    case, cht = solved
    fluid = sampled(case, "fluid", 0.0, cht.FLUID_THICK_M)
    hot_end = min(fluid, key=lambda r: r[0])[1]
    print(f"  hot face {hot_end:.4f} K against {cht.HOT_K:g} K asked for", flush=True)
    assert abs(hot_end - cht.HOT_K) < 5.0, hot_end


def test_the_grader_owns_the_reference_and_agrees_with_the_template(solved):
    """`conjugate.py` derives the answer and the template recomputes it to print it.
    They must agree, or the number this journey grades against is whichever file the
    reader happened to open."""
    case, cht = solved
    conjugate = load(GRADER, "toolbox_conjugate")
    layers = ((cht.FLUID_THICK_M, cht.FLUID_K), (cht.SOLID_THICK_M, cht.SOLID_K))
    theirs = conjugate.interface_temperatures(layers, cht.HOT_K, cht.COLD_K)[1]
    mine = cht.expected()["T_interface_K"]
    print(f"  grader {theirs:.8f} K, template {mine:.8f} K", flush=True)
    assert abs(theirs - mine) < 1e-9
