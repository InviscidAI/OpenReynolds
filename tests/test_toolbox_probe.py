"""`probe.py`: any solved field, sampled where an agent asks, as JSON.

The seed-5 Tesla campaign (Archimedes `results/rematch/REFLECTION-tesla_v3-s5.md`) ran
its agents on one number per candidate -- the score -- with no way to look inside a
solve and ask where the pressure was lost. These tests exercise every op of the probe
CLI contract on a case whose answers are known by construction: a 10x4x1 box of
0.1 m cells, `p = 10 - 5 x` at the cell centres, `U = (1 + j, 0, 0)` in row j, and a
temperature `T = j` -- so the area mean of T over a section is 1.5 and its flux-weighted
mean is 2.0, the flux through any x section is 0.1 m^3/s, and none of those numbers
came from the script.

The mesh is `test_toolbox_locate.py`'s hand-written ascii polyMesh, read by the same
pyvista reader the workspace uses; skipped where pyvista is not importable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyvista", reason="probe reads cases through pyvista", exc_type=ImportError)

HERE = Path(__file__).resolve().parent
TOOLBOX = HERE.parents[0] / "openreynolds" / "toolbox"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load("toolbox_probe", TOOLBOX / "probe.py")
write_box_mesh = _load("probe_locate_fixture", HERE / "test_toolbox_locate.py").write_box_mesh

NX, NY = 10, 4
PATCHES = ("inlet", "outlet", "walls")


def _field(case: Path, time: str, name: str, cls: str, dims: str, values: list[str]) -> None:
    kind = "vector" if cls == "volVectorField" else "scalar"
    zero = "(0 0 0)" if kind == "vector" else "0"
    boundary = " ".join(f"{p} {{ type zeroGradient; }}" for p in PATCHES)
    directory = case / time
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        f"FoamFile {{ version 2.0; format ascii; class {cls}; object {name}; }}\n"
        f"dimensions {dims};\n"
        f"internalField nonuniform List<{kind}> {len(values)}\n(\n" + "\n".join(values) + "\n);\n"
        f"boundaryField {{ {boundary} }}\n// {zero}\n"
    )


def make_case(root: Path, name: str = "box", times=("100",)) -> Path:
    case = root / name
    write_box_mesh(case, NX, NY, 1, (0.1, 0.1, 0.1))
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text("application simpleFoam;\n")
    cells = [(i, j) for j in range(NY) for i in range(NX)]
    for offset, time in enumerate(times):
        _field(case, time, "p", "volScalarField", "[0 2 -2 0 0 0 0]",
               [f"{10 - 5 * (i + 0.5) * 0.1 + offset:.10g}" for i, j in cells])
        _field(case, time, "U", "volVectorField", "[0 1 -1 0 0 0 0]",
               [f"({1 + j} 0 0)" for i, j in cells])
        _field(case, time, "T", "volScalarField", "[0 0 0 1 0 0 0]", [f"{j}" for i, j in cells])
    return case


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    return make_case(tmp_path_factory.mktemp("probe"))


def run(argv, capsys):
    code = probe.main([str(a) for a in argv])
    out = capsys.readouterr().out
    return code, (json.loads(out) if code == 0 else out)


def test_fields_are_read_off_the_case_with_their_kind_and_units(case, capsys):
    code, got = run([case, "--fields"], capsys)
    assert code == 0 and got["op"] == "fields" and got["time"] == 100.0
    assert got["data"]["U"]["kind"] == "vector"
    assert got["data"]["p"]["kind"] == "scalar" and got["data"]["p"]["units"] == "m^2 s^-2"
    assert got["data"]["T"]["units"] == "K"
    assert got["data"]["T"]["min"] == 0 and got["data"]["T"]["max"] == 3


def test_a_line_samples_a_scalar_and_a_vector_with_its_components(case, tmp_path, capsys):
    png = tmp_path / "line.png"
    code, got = run([case, "--line", 0.15, 0.25, 0.85, 0.25, "--field", "p", "--n", 8,
                     "--png", png], capsys)
    assert code == 0 and got["units"] == "m^2 s^-2"
    values = got["data"]["value"]
    assert len(values) == 8 and len(got["data"]["x"]) == 8
    assert values[0] == pytest.approx(10 - 5 * 0.15, abs=0.05)
    assert values[-1] == pytest.approx(10 - 5 * 0.85, abs=0.05)
    assert png.stat().st_size > 0
    code, got = run([case, "--line", 0.15, 0.25, 0.85, 0.25, "--field", "U", "--n", 5], capsys)
    assert set(got["data"]) >= {"s", "x", "y", "z", "U_x", "U_y", "U_z", "U_mag"}
    assert got["data"]["U_x"][2] == pytest.approx(3.0, abs=1e-6)


def test_points_are_sampled_where_asked(case, capsys):
    code, got = run([case, "--points", "0.55,0.25;0.35,0.15", "--field", "p"], capsys)
    assert code == 0
    assert got["data"]["value"] == pytest.approx([10 - 5 * 0.55, 10 - 5 * 0.35], abs=0.05)


def test_a_point_outside_the_domain_is_null_not_zero(case, capsys):
    code, got = run([case, "--points", "5,5", "--field", "p"], capsys)
    assert code == 0 and got["data"]["value"] == [None]


@pytest.mark.parametrize("stat, expected", [
    ("mean", 10 - 5 * 0.25), ("min", 10 - 5 * 0.45), ("max", 10 - 5 * 0.05),
    ("integral", (10 - 5 * 0.25) * 0.5 * 0.4 * 0.1),
])
def test_a_region_reduces_the_cells_inside_it(case, capsys, stat, expected):
    code, got = run([case, "--region", 0, 0, 0.5, 0.4, "--field", "p", "--stat", stat], capsys)
    assert code == 0 and got["data"]["n_cells"] == 20
    assert got["data"]["value"] == pytest.approx(expected, rel=1e-6)


def test_a_section_mean_is_area_or_flux_weighted(case, capsys):
    code, got = run([case, "--section-mean", "--field", "T", "--at-x", 0.55], capsys)
    assert code == 0 and got["data"]["value"] == pytest.approx(1.5)
    code, got = run([case, "--section-mean", "--field", "T", "--at-x", 0.55,
                     "--weighted", "flux"], capsys)
    assert got["data"]["value"] == pytest.approx(2.0)


def test_between_is_the_downstream_mean_minus_the_upstream_one(case, capsys):
    code, got = run([case, "--between", 0.25, 0.75, "--field", "p"], capsys)
    assert code == 0 and got["data"]["difference"] == pytest.approx(-2.5)


def test_total_pressure_is_derived_when_the_case_did_not_write_it(case, capsys):
    """Kinematic p here, so p_total = p + |U|^2/2 with no density."""
    code, got = run([case, "--section-mean", "--field", "p_total", "--at-x", 0.55], capsys)
    mean_half_u2 = sum(0.5 * (1 + j) ** 2 for j in range(NY)) / NY
    assert code == 0 and got["data"]["value"] == pytest.approx(10 - 5 * 0.55 + mean_half_u2)
    assert any("p_total" in note for note in got["notes"])


def test_flux_through_a_section(case, capsys):
    code, got = run([case, "--flux", "--at-x", 0.55], capsys)
    assert code == 0 and got["data"]["value"] == pytest.approx(0.1)
    assert got["units"] == "m^3 s^-1"


def test_streamlines_follow_the_flow_from_their_seeds(case, tmp_path, capsys):
    png = tmp_path / "stream.png"
    code, got = run([case, "--streamlines", "--seeds", "0.15,0.25", "--direction", "forward",
                     "--png", png], capsys)
    assert code == 0
    (line,) = got["data"]["lines"]
    xs = [p[0] for p in line["points"]]
    ys = [p[1] for p in line["points"]]
    assert min(xs) == pytest.approx(0.15, abs=0.02) and max(xs) > 0.9
    assert ys == pytest.approx([0.25] * len(ys), abs=1e-6)
    assert png.stat().st_size > 0


def test_time_picks_the_nearest_write(tmp_path, capsys):
    two = make_case(tmp_path, times=("50", "100"))
    code, got = run([two, "--time", 49, "--points", "0.55,0.25", "--field", "p"], capsys)
    assert got["time"] == 50.0 and got["data"]["value"][0] == pytest.approx(7.25, abs=0.05)
    code, got = run([two, "--points", "0.55,0.25", "--field", "p"], capsys)
    assert got["time"] == 100.0 and got["data"]["value"][0] == pytest.approx(8.25, abs=0.05)


def test_a_field_the_case_lacks_fails_naming_what_it_has(case, capsys):
    with pytest.raises(SystemExit) as err:
        probe.main([str(case), "--points", "0.5,0.2", "--field", "omega"])
    message = str(err.value.code)
    assert "omega" in message and "U" in message and "p" in message


def make_candidate(tmp_path, *names):
    """parametric.py's layout: build.py's mesh at the root, each case solved beside it."""
    candidate = tmp_path / "cand"
    write_box_mesh(candidate, NX, NY, 1, (0.1, 0.1, 0.1))
    for name in names:
        make_case(candidate, name)
    return candidate


def test_a_candidate_with_one_case_is_probed_through_it(tmp_path, capsys):
    candidate = make_candidate(tmp_path, "forward")
    code, got = run([candidate, "--flux", "--at-x", 0.55], capsys)
    assert code == 0 and got["case"].endswith("forward")


def test_a_candidate_with_several_cases_asks_which(tmp_path):
    candidate = make_candidate(tmp_path, "forward", "reverse")
    with pytest.raises(SystemExit) as err:
        probe.main([str(candidate), "--fields"])
    assert "forward" in str(err.value.code) and "reverse" in str(err.value.code)


def test_out_writes_the_same_json(case, tmp_path, capsys):
    out = tmp_path / "probe.json"
    code, got = run([case, "--flux", "--at-x", 0.55, "--out", out], capsys)
    assert json.loads(out.read_text()) == got


def test_both_directions_join_into_one_line_upstream_to_downstream(case, capsys):
    code, got = run([case, "--streamlines", "--seeds", "0.55,0.25;5,5"], capsys)
    first, off_mesh = got["data"]["lines"]
    xs = [p[0] for p in first["points"]]
    assert got["field"] == "U" and xs == sorted(xs) and xs[0] < 0.2 and xs[-1] > 0.8
    assert 0.55 in [pytest.approx(x, abs=1e-6) for x in xs]
    assert off_mesh["points"] == [] and got["notes"]


def test_a_case_with_no_solved_fields_fails_rather_than_listing_nothing(tmp_path):
    bare = tmp_path / "bare"
    write_box_mesh(bare, NX, NY, 1, (0.1, 0.1, 0.1))
    with pytest.raises(SystemExit) as err:
        probe.main([str(bare), "--fields"])
    assert err.value.code not in (0, None)


def test_a_decomposed_unreconstructed_case_is_refused_not_read_at_time_zero(tmp_path):
    import shutil
    split = make_case(tmp_path, times=("0", "100"))
    shutil.copytree(split / "constant", split / "processor0" / "constant")
    shutil.move(str(split / "100"), str(split / "processor0" / "100"))
    with pytest.raises(SystemExit) as err:
        probe.main([str(split), "--points", "0.55,0.25", "--field", "p"])
    assert "reconstructPar" in str(err.value.code)
