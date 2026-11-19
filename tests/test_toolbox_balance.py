"""`balance.py`: the per-patch split, the closure, and K and f.

Every number below is known by construction rather than read back off the script. The
case is `test_toolbox_locate.py`'s hand-written ascii polyMesh -- a 10 x 6 x 1 box of
0.1 m cells, so a boundary face is 0.01 m^2 -- with its single `outlet` patch relabelled
as three, which is a manifold as far as the boundary file is concerned and needs no second
mesh writer. `U = (1 + j, 0, 0)` in cell row j and `p = 10 - 5 x` at the cell centres,
both `zeroGradient` at the boundary, so each patch reads the cells behind it:

- the six inlet faces carry 1 + 2 + ... + 6 = 21 units, so the inflow is 0.21 m^3/s;
- the outlets take rows (0, 1), (2, 3) and (4, 5), which is 3, 7 and 11 of those 21 --
  14.2857%, 33.3333% and 52.3810%, summing to 100%;
- the wall faces are all perpendicular to U, so their flux is exactly zero;
- the flux-weighted velocity head at a station is 0.5 * sum (1+j)^3 / sum (1+j) =
  0.5 * 441 / 21 = 10.5 m^2/s^2, which is what separates the static and total drops.

`--outlet-scale s` multiplies U in the last cell column, which is the only column the
outlet patches read, so the outflow becomes 0.21 s with the inflow untouched: a case that
does not close, by exactly the amount the test asks for.

The arithmetic that needs no mesh -- the shares, the closure, K, f, and the refusals that
name a flag -- is exercised on plain dicts rather than through a reader, so what is being
checked is the accounting and not the fixture.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLBOX = HERE.parents[0] / "openreynolds" / "toolbox"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


balance = _load("toolbox_balance", TOOLBOX / "balance.py")

NX, NY, NZ = 10, 6, 1
CELL = 0.1
FACE_AREA = CELL * CELL
INFLOW = 0.21  # 0.01 m^2 x (1 + 2 + ... + 6) m/s
HEAD = 10.5  # 0.5 * sum (1+j)^3 / sum (1+j), the flux-weighted velocity head
OUTLETS = {"outlet_a": 3 / 21, "outlet_b": 7 / 21, "outlet_c": 11 / 21}
KINEMATIC = "[0 2 -2 0 0 0 0]"
PASCAL = "[1 -1 -2 0 0 0 0]"


# -- the arithmetic, on plain dicts ------------------------------------------------------

def rows(*fluxes: tuple[str, float]) -> list[dict]:
    """Patch records with nothing in them but what `split` reads."""
    return [{"patch": name, "volume_flux": flux, "mass_flux": None} for name, flux in fluxes]


def test_the_shares_of_an_outflow_set_sum_to_a_hundred():
    table = rows(("in", -1.0), ("a", 0.2), ("b", 0.3), ("c", 0.5))
    balance.split(table, 1e-6)
    shares = {row["patch"]: row["share_pct"] for row in table}
    assert sum(shares[name] for name in ("a", "b", "c")) == pytest.approx(100.0)
    assert shares["in"] == pytest.approx(100.0)  # of the inflow, which it is all of


def test_a_patch_with_no_flux_takes_no_share_at_all():
    """A wall's 1e-18 is not a 0.0000001% outlet, and in a share list it is noise."""
    table = rows(("in", -1.0), ("out", 1.0), ("walls", 1e-18))
    report = balance.split(table, 1e-6)
    closed = next(row for row in table if row["patch"] == "walls")
    assert closed["direction"] == "closed" and closed["share_pct"] is None
    assert report["closed_patches"] == ["walls"]
    assert report["outflow"] == pytest.approx(1.0)


def test_the_closure_error_is_in_and_out_with_the_number_not_a_verdict():
    report = balance.split(rows(("in", -1.0), ("out", 0.9)), 1e-6)
    assert report["inflow"] == pytest.approx(1.0)
    assert report["outflow"] == pytest.approx(0.9)
    assert report["closure_error"] == pytest.approx(0.1)
    assert report["closure_error_relative"] == pytest.approx(0.1 / 0.95)
    assert report["inflow_patches"] == ["in"] and report["outflow_patches"] == ["out"]
    assert "verdict" not in report and "ok" not in report


def test_direction_comes_from_the_sign_and_not_from_the_name():
    table = rows(("outlet_west", -1.0), ("inlet_east", 1.0))
    balance.split(table, 1e-6)
    assert [row["direction"] for row in table] == ["inflow", "outflow"]


def test_k_is_the_total_drop_over_the_dynamic_head():
    drop = {"dp_static": 10.0, "dp_total": 12.5}
    got = balance.coefficients(drop, "kinematic", "m^2 s^-2", None, 5.0, None, None)
    assert got["K"] == pytest.approx(12.5 / (0.5 * 5.0 ** 2))
    assert "cancels" in got["dynamic_head_basis"]


def test_k_on_a_pascal_case_carries_the_density_and_a_kinematic_one_does_not():
    drop = {"dp_static": 10.0, "dp_total": 12.5}
    pascal = balance.coefficients(drop, "Pa", "kg m^-1 s^-2", 1000.0, 5.0, None, None)
    assert pascal["K"] == pytest.approx(12.5 / (0.5 * 1000.0 * 25.0))
    kinematic = balance.coefficients(drop, "kinematic", "m^2 s^-2", None, 5.0, None, None)
    assert pascal["K"] == pytest.approx(kinematic["K"] / 1000.0)


def test_a_density_that_cannot_be_read_is_refused_by_name():
    """`grade.py:_need`'s rule: a K off an assumed rho is out by a factor of 1000 for
    water and looks exactly as measured as a right one."""
    with pytest.raises(SystemExit) as err:
        balance.coefficients({"dp_static": 1.0, "dp_total": 1.0}, "Pa", "kg m^-1 s^-2",
                             None, 5.0, None, None)
    assert "--rho" in str(err.value.code) and "density" in str(err.value.code)


def test_the_darcy_factor_refuses_a_length_or_a_diameter_by_name():
    drop = {"dp_static": 10.0, "dp_total": 12.5}
    got = balance.coefficients(drop, "kinematic", "m^2 s^-2", None, 5.0, 2.0, 0.1)
    assert got["f_darcy"] == pytest.approx(10.0 / ((2.0 / 0.1) * 0.5 * 25.0))
    with pytest.raises(SystemExit) as err:
        balance.coefficients(drop, "kinematic", "m^2 s^-2", None, 5.0, 2.0, None)
    assert "--dh" in str(err.value.code)
    with pytest.raises(SystemExit) as err:
        balance.coefficients(drop, "kinematic", "m^2 s^-2", None, 5.0, None, 0.1)
    assert "--length" in str(err.value.code)


def test_no_reference_velocity_means_no_coefficient_rather_than_a_guessed_one():
    assert balance.coefficients({"dp_static": 1.0, "dp_total": 1.0}, "kinematic",
                                "m^2 s^-2", None, None, None, None) == {}


# -- the case ----------------------------------------------------------------------------

pytest.importorskip("pyvista", reason="the patches are read through pyvista",
                    exc_type=ImportError)

write_box_mesh = _load("balance_locate_fixture", HERE / "test_toolbox_locate.py").write_box_mesh

PATCH_ENTRY = re.compile(r"(\w+)\s*\{\s*type\s+(\w+);\s*nFaces\s+(\d+);\s*startFace\s+(\d+);\s*\}")


def relabel(case: Path, patch: str, pieces: list[tuple[str, int]]) -> None:
    """Relabel one patch of the box mesh as several, in place.

    The faces do not move: a manifold and a duct differ here only in how the boundary
    file groups the same outlet faces, so this gets three outlets out of the one mesh
    writer the suite already has rather than a second one to keep correct.
    """
    path = case / "constant" / "polyMesh" / "boundary"
    text = path.read_text()
    out = []
    for name, kind, count, start in PATCH_ENTRY.findall(text):
        count, start = int(count), int(start)
        if name != patch:
            out.append((name, kind, count, start))
            continue
        at = start
        for piece, take in pieces:
            out.append((piece, kind, take, at))
            at += take
        assert at == start + count, f"the pieces of {patch} must add up to its {count} faces"
    body = "".join(f"    {name} {{ type {kind}; nFaces {count}; startFace {start}; }}\n"
                   for name, kind, count, start in out)
    path.write_text(f"{text.splitlines()[0]}\n{len(out)}\n(\n{body})\n")


def patch_names(case: Path) -> list[str]:
    text = (case / "constant" / "polyMesh" / "boundary").read_text()
    return [name for name, _, _, _ in PATCH_ENTRY.findall(text)]


def write_field(case: Path, time: str, name: str, cls: str, dims: str, values: list[str],
                patches: list[str]) -> None:
    kind = "vector" if cls == "volVectorField" else "scalar"
    boundary = " ".join(f"{patch} {{ type zeroGradient; }}" for patch in patches)
    directory = case / time
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        f"FoamFile {{ version 2.0; format ascii; class {cls}; object {name}; }}\n"
        f"dimensions {dims};\n"
        f"internalField nonuniform List<{kind}> {len(values)}\n(\n" + "\n".join(values) + "\n);\n"
        f"boundaryField {{ {boundary} }}\n"
    )


def manifold(root: Path, name: str = "manifold", *, dims: str = KINEMATIC,
             scale: float = 1.0, p_scale: float = 1.0, inlet: str = "inlet") -> Path:
    """The three-outlet case. `scale` multiplies U in the last cell column only."""
    case = root / name
    write_box_mesh(case, NX, NY, NZ, (CELL, CELL, CELL))
    relabel(case, "inlet", [(inlet, NY * NZ)])
    relabel(case, "outlet", [("outlet_a", 2), ("outlet_b", 2), ("outlet_c", 2)])
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text("application simpleFoam;\n")
    patches = patch_names(case)
    cells = [(i, j) for j in range(NY) for i in range(NX)]
    write_field(case, "100", "p", "volScalarField", dims,
                [f"{p_scale * (10 - 5 * (i + 0.5) * CELL):.10g}" for i, _ in cells], patches)
    write_field(case, "100", "U", "volVectorField", "[0 1 -1 0 0 0 0]",
                [f"({(scale if i == NX - 1 else 1.0) * (1 + j):.10g} 0 0)" for i, j in cells],
                patches)
    return case


def run(case: Path, *argv, out: Path | None = None) -> dict:
    """The script as a session runs it, with the `--json` sidecar read back."""
    sidecar = out or (case / "balance.json")
    assert balance.main([str(case), *[str(a) for a in argv], "--json", str(sidecar)]) == 0
    return json.loads(sidecar.read_text())


def by_patch(report: dict) -> dict[str, dict]:
    return {row["patch"]: row for row in report["patches"]}


def test_a_three_outlet_manifold_splits_the_flow_and_the_shares_sum_to_a_hundred(tmp_path):
    report = run(manifold(tmp_path))
    found = by_patch(report)
    for name, share in OUTLETS.items():
        assert found[name]["direction"] == "outflow"
        assert found[name]["share_pct"] == pytest.approx(100 * share, rel=1e-4)
        assert found[name]["volume_flux"] == pytest.approx(INFLOW * share, rel=1e-4)
    assert sum(found[name]["share_pct"] for name in OUTLETS) == pytest.approx(100.0, rel=1e-6)
    assert report["balance"]["outflow_patches"] == sorted(OUTLETS)


def test_an_inflow_is_identified_by_its_flux_sign_however_it_is_named(tmp_path):
    """The patch at x = 0 is called `outlet_west` here and the flow goes in through it."""
    report = run(manifold(tmp_path, inlet="outlet_west"))
    found = by_patch(report)
    assert found["outlet_west"]["direction"] == "inflow"
    assert found["outlet_west"]["volume_flux"] == pytest.approx(-INFLOW, rel=1e-4)
    assert found["outlet_west"]["share_pct"] == pytest.approx(100.0)
    assert report["balance"]["inflow_patches"] == ["outlet_west"]


def test_a_wall_carries_zero_flux_and_does_not_pollute_the_shares(tmp_path):
    report = run(manifold(tmp_path))
    walls = by_patch(report)["walls"]
    assert walls["volume_flux"] == pytest.approx(0.0, abs=1e-12)
    assert walls["direction"] == "closed" and walls["share_pct"] is None
    assert walls["area"] > 0 and walls["p_area"] is not None
    assert walls["p_mixed"] is None, "a patch with no flux has no flux-weighted mean"
    assert report["balance"]["closed_patches"] == ["walls"]


def test_a_case_that_does_not_close_reports_the_error_and_still_exits_zero(tmp_path, capsys):
    report = run(manifold(tmp_path, scale=0.9))
    closure = report["balance"]
    assert closure["inflow"] == pytest.approx(INFLOW, rel=1e-4)
    assert closure["outflow"] == pytest.approx(0.9 * INFLOW, rel=1e-4)
    assert closure["closure_error"] == pytest.approx(0.1 * INFLOW, rel=1e-3)
    assert closure["closure_error_relative"] == pytest.approx(0.1 / 0.95, rel=1e-3)
    assert closure["flux_quantity"] == "volume"
    printed = capsys.readouterr().out
    assert "closure error" in printed
    assert "inlet" in printed and "outlet_a" in printed


def test_the_included_patches_are_named_beside_the_closure(tmp_path):
    """A case that does not close usually has a patch nobody told the analysis about, so
    the answer is only readable next to the list of what was counted."""
    case = manifold(tmp_path)
    closure = run(case)["balance"]
    counted = (closure["inflow_patches"] + closure["outflow_patches"]
               + closure["closed_patches"])
    assert sorted(counted) == sorted(patch_names(case))


def test_static_and_total_drop_differ_by_the_velocity_head_and_both_are_reported(tmp_path):
    """Scaled so the two stations have different velocity heads: 10.5 at the inlet and
    0.5 * 0.81 * (1 + 8) / 3 = 1.215 on the first outlet. Quoting either drop as the
    other is out by their difference."""
    report = run(manifold(tmp_path, scale=0.9), "--drop", "inlet", "outlet_a")
    drop = report["drop"]
    head_a = 0.5 * 0.81 * (1 + 8) / (1 + 2)
    assert drop["dp_static"] == pytest.approx(9.75 - 5.25, rel=1e-5)
    assert drop["dp_total"] == pytest.approx((9.75 + HEAD) - (5.25 + head_a), rel=1e-5)
    assert drop["dp_total"] - drop["dp_static"] == pytest.approx(HEAD - head_a, rel=1e-4)
    assert "mixed mean" in drop["weighted"]


def test_a_drop_between_two_sections_goes_through_probe(tmp_path):
    report = run(manifold(tmp_path), "--sections", 0.25, 0.75)
    assert report["drop"]["dp_static"] == pytest.approx(2.5, rel=1e-4)
    # U does not change with x on an unscaled case, so the heads cancel and only the
    # static drop is left -- which is the constant-area duct the Darcy factor assumes.
    assert report["drop"]["dp_total"] == pytest.approx(2.5, rel=1e-4)


def test_k_comes_off_the_measured_drop_and_a_given_reference_velocity(tmp_path):
    report = run(manifold(tmp_path), "--drop", "inlet", "outlet_a", "--u-ref", 2.5)
    coefficients = report["coefficients"]
    assert coefficients["dynamic_head"] == pytest.approx(0.5 * 2.5 ** 2)
    assert coefficients["K"] == pytest.approx(report["drop"]["dp_total"] / (0.5 * 2.5 ** 2),
                                              rel=1e-5)


def test_the_friction_factor_comes_off_the_static_drop_with_a_length_and_a_diameter(tmp_path):
    report = run(manifold(tmp_path), "--drop", "inlet", "outlet_a", "--u-ref", 2.5,
                 "--length", 1.0, "--dh", 0.2)
    expected = report["drop"]["dp_static"] / ((1.0 / 0.2) * 0.5 * 2.5 ** 2)
    assert report["coefficients"]["f_darcy"] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("dims, convention, units, p_scale", [
    (KINEMATIC, "kinematic", "m^2 s^-2", 1.0),
    (PASCAL, "Pa", "kg m^-1 s^-2", 1000.0),
])
def test_the_pressure_convention_is_read_off_p_s_own_dimensions(tmp_path, dims, convention,
                                                               units, p_scale):
    """The same case either way; only `dimensions` says which, and a dp read as the wrong
    one is out by rho in both directions."""
    case = manifold(tmp_path, dims=dims, p_scale=p_scale)
    extra = ["--rho", 1000.0] if convention == "Pa" else []
    report = run(case, "--drop", "inlet", "outlet_a", *extra)
    assert report["pressure_convention"] == convention
    assert report["pressure_units"] == units
    assert report["drop"]["dp_static"] == pytest.approx(p_scale * 4.5, rel=1e-5)
    head = HEAD * (1000.0 if convention == "Pa" else 1.0)
    assert report["drop"]["p_total_from"] == pytest.approx(p_scale * 9.75 + head, rel=1e-5)


def test_a_pascal_case_with_no_density_anywhere_is_refused_by_name(tmp_path):
    """p in Pa, no `rho` field, no `--rho`: the velocity head cannot be formed, so the
    total pressure cannot be, so nothing is reported rather than a kinematic head added
    to a pressure in pascals."""
    case = manifold(tmp_path, dims=PASCAL, p_scale=1000.0)
    with pytest.raises(SystemExit) as err:
        balance.main([str(case)])
    message = str(err.value.code)
    assert "--rho" in message and "rho" in message


def test_a_density_turns_the_balance_into_a_mass_balance(tmp_path):
    report = run(manifold(tmp_path), "--rho", 998.2)
    assert report["balance"]["flux_quantity"] == "mass"
    assert report["balance"]["flux_units"] == "kg s^-1"
    assert report["balance"]["inflow"] == pytest.approx(998.2 * INFLOW, rel=1e-4)
    assert by_patch(report)["outlet_b"]["mass_flux"] == pytest.approx(
        998.2 * INFLOW * OUTLETS["outlet_b"], rel=1e-4)


def test_the_table_states_the_units_of_what_it_prints(tmp_path, capsys):
    run(manifold(tmp_path), "--drop", "inlet", "outlet_a", "--u-ref", 2.5)
    printed = capsys.readouterr().out
    assert "m^2 s^-2" in printed and "m^3 s^-1" in printed
    assert "kinematic" in printed and "negative is into the domain" in printed


def test_a_patch_the_case_does_not_have_is_refused_naming_the_ones_it_does(tmp_path):
    with pytest.raises(SystemExit) as err:
        balance.main([str(manifold(tmp_path)), "--drop", "inlet", "plenum"])
    message = str(err.value.code)
    assert "plenum" in message and "outlet_a" in message


def test_a_drop_referred_to_a_wall_is_refused_rather_than_divided_by_zero(tmp_path):
    with pytest.raises(SystemExit) as err:
        balance.main([str(manifold(tmp_path)), "--drop", "inlet", "walls"])
    assert "no flux" in str(err.value.code)


def test_the_png_draws_the_split(tmp_path):
    case = manifold(tmp_path)
    png = tmp_path / "split.png"
    assert balance.main([str(case), "--drop", "inlet", "outlet_a", "--u-ref", "2.5",
                         "--png", str(png)]) == 0
    assert png.stat().st_size > 0
