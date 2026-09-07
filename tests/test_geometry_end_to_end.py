"""The four worked scripts of DESIGN.md section 7 through the real child interpreter
(`python -I openreynolds/geometry/cli.py build --script ...`, the way the desk runs a lap),
each against its claims file in tests/fixtures/geometry/claims, and the T01 record
rebuilt through `toolbox/mesh2d.py --spec geometry.json` into a case directory (section
8's end-to-end check; there is no OpenFOAM here, so the case's files are asserted, never
run).

What is pinned and how. Every FEATURES, LEGS and CLAIMS line of section 7 is matched
after whitespace is squashed, with the numbers compared to 0.006 rather than as text:
the design's T01 leg table was printed from the probe spec whose anchor is the rounded
7.24, while the compiler places the loop at the closed form's 7.2391, so (10.06, 2.526)
in the design is (10.06, 2.526) here but (7.24, 1.5) is (7.239, 1.5). LINT is matched by
code and by the numbers the catalogue names (2.44 apart, 28.9 deg at the four lips),
not by text, because the units' recorded deviations stand: one I-LIP Finding per vertex
carrying "4 such lips" (8.1's fixture row 13 counts 4 I-LIP), the I-REFERENCE p10 that
U2 pinned, and T04's W-UNITS (7.3's abridged claims omit the request's 60 mm legs, so
the largest stated length is the 10 mm bend radius against a 74 mm extent; warnings never
block, D10). The places where the design's own text is wrong are pinned to the measured
truth and said here: T02's c9 range is x = -4..34 (the design typed 0..38, the extent
without the bends' overhang) and T03 has no probe spec yet, so it is not run.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("gmsh")
pytest.importorskip("matplotlib")

from openreynolds.geometry import runner  # noqa: E402

import geometry_scripts as scripts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "tests" / "fixtures" / "geometry" / "claims"
MESH2D = ROOT / "openreynolds" / "toolbox" / "mesh2d.py"

NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?")
TOL = 0.006
"""The design's leg points are the probe's (anchor 7.24); the compiler's are closed form (7.2391)."""


def squash(text: str) -> str:
    return " ".join(text.split())


def same_line(expected: str, actual: str, tol: float = TOL) -> bool:
    """The two lines are the same words with the same numbers to `tol`."""
    e, a = squash(expected), squash(actual)
    e_nums = [float(v) for v in NUMBER.findall(e)]
    a_nums = [float(v) for v in NUMBER.findall(a)]
    if len(e_nums) != len(a_nums) or NUMBER.sub("#", e) != NUMBER.sub("#", a):
        return False
    return all(abs(x - y) <= tol for x, y in zip(e_nums, a_nums))


def find_line(expected: str, report: str, tol: float = TOL) -> str:
    """The report line that says `expected`; an assertion naming the nearest miss otherwise."""
    lines = report.splitlines()
    for line in lines:
        if same_line(expected, line, tol):
            return line
    e_words = NUMBER.sub("#", squash(expected)).split()
    nearest = max(lines, key=lambda ln: len(set(e_words) & set(NUMBER.sub("#", squash(ln)).split())), default="")
    raise AssertionError(f"no line reads {squash(expected)!r}; nearest: {squash(nearest)!r}")


def find_prefix(expected: str, report: str, tol: float = TOL) -> str:
    """The report line that begins with `expected` (as many words as it has, numbers to
    `tol`): the MEASURED line's narrowest-passage cell carries the resolution and the
    place U2 pins, which section 7 leaves as a placeholder."""
    n = len(squash(expected).split())
    for line in report.splitlines():
        head = " ".join(squash(line).split()[:n])
        if same_line(expected, head, tol):
            return line
    raise AssertionError(f"no line begins with {squash(expected)!r}")


def row_of(report: str, claim_id: str) -> str:
    """The CLAIMS row for `claim_id`, squashed."""
    for line in report.splitlines():
        s = squash(line)
        if s.startswith(claim_id + " "):
            return s
    raise AssertionError(f"no CLAIMS row {claim_id}")


def assert_row(report: str, claim_id: str, tail: str) -> None:
    """The row for `claim_id` ends with `tail` (the measured, expected and verdict cells
    of section 7; the `says` column is the claims file's, which the design abbreviates by
    hand), numbers to TOL."""
    row = row_of(report, claim_id)
    tail_words = squash(tail)
    # compare the row's ending of the same word count
    n = len(tail_words.split())
    ending = " ".join(row.split()[-n:])
    assert same_line(tail_words, ending), f"{claim_id}: {row!r} does not end with {tail_words!r}"


@pytest.fixture(scope="module")
def laps(tmp_path_factory) -> dict[str, runner.RunOutcome]:
    """Each worked script once through the real child, with its claims file."""
    out = {}
    base = tmp_path_factory.mktemp("worked")
    for name in ("T01", "T02", "T04", "T05"):
        work = base / name
        work.mkdir()
        outcome = runner.run_script(scripts.WORKED[name], work, CLAIMS / f"{name}.json", None, timeout_s=120)
        out[name] = outcome
        out[name + "_work"] = work
    return out


# -- every lap: rc 0, a picture, a record, the files of 4.3 ------------------------------------------


@pytest.mark.parametrize("name", ["T01", "T02", "T04", "T05"])
def test_the_lap_builds_draws_and_records(laps, name):
    outcome, work = laps[name], laps[name + "_work"]
    assert outcome.rc == 0, outcome.text
    assert outcome.png is not None and len(outcome.png) > 10_000
    result = outcome.result
    assert result["ok"] is True and result["preview"] == "preview.png" and result["compliance"] is not None
    for file in ("result.json", "report.txt", "record.json", "preview.png"):
        assert (work / file).exists(), file
    record = json.loads((work / "record.json").read_text(encoding="utf-8"))
    assert record["format"] == "openreynolds.geometry/1" and record["units"] == "mm" and record["scale"] == 0.001
    assert record["ops"][-1]["name"] == "fluid" and record["script"].strip() == scripts.WORKED[name].strip()
    assert record["claims"]["claims"][0]["id"] == "c1" and record["compliance"]["rows"]
    assert (work / "report.txt").read_text(encoding="utf-8") == outcome.text == result["report"]
    order = [line.split()[0] for line in outcome.text.splitlines() if line[:1].isupper()]
    assert order == ["SCRIPT", "LINT", "FEATURES", "LEGS", "MEASURED", "PATCHES", "CLAIMS", "REFERENCE", "VERDICT"] or \
        order == ["SCRIPT", "LINT", "FEATURES", "MEASURED", "PATCHES", "CLAIMS", "REFERENCE", "VERDICT"], order


# -- 7.1 T01, lap 2c ------------------------------------------------------------------------------


def test_t01_lap_2_prints_the_section_7_1_features_legs_lint_and_claims(laps):
    text = laps["T01"].text
    assert text.startswith("SCRIPT     ran in ") and "; 1 note\n" in text.splitlines()[0] + "\n"
    assert "           note: return_angle 80 keeps 4 loops of outer r 6 inside 60" in text
    # LINT: clean, with the three information codes of 7.1 and their numbers
    assert "LINT       clean" in text
    codes = [f["code"] for f in laps["T01"].result["lint"]]
    assert sorted(set(codes)) == ["I-GAP", "I-LIP", "I-REFERENCE"] and codes.count("I-LIP") == 4
    find_line("check I-GAP Row 'loops': neighbours' walls 2.44 apart (solved gap 1.64 is along the wall's bounding footprint)", text)
    for x in (11.42, 26.08, 40.74, 55.40):
        assert any(f["code"] == "I-LIP" and abs(f["where"][0] - x) < 0.05 and abs(f["numbers"].get("deg", f["numbers"].get("solid_deg", 0)) - 28.9) < 0.1
                   for f in laps["T01"].result["lint"]), x
    assert "I-REFERENCE  fluid: passage width 3 (declared by 'main')" in text
    # FEATURES, verbatim
    find_line("FEATURES   main      Passage    width 3, one leg 60 along +x, start (0, 0) end (60, 0)", text)
    find_line("loop      Bypass     R 4.5 (outer 6, inner 3), sweep 240, return heading 260 (against +x), "
              "leave_length 3 (default: one width),", text)
    find_line("lands 4.22 upstream of its anchor, lip 4.18 downstream at 28.9 deg, footprint -5.74..+7.28, height 11.25", text)
    find_line("loops     Row        4 at pitch 14.66 (13.02 + 1.64 solved), anchors u = 7.24, 21.90, 36.56, 51.22,", text)
    find_line("span 1.50..58.50 on main.top (0..60), centred", text)
    # LEGS, the numbers to the probe's rounding
    find_line("LEGS       loops[0] (anchor 7.24; loops[1..3] identical, shifted by 14.66)", text)
    find_line("leg 1  line 3     from (7.24, 1.5) heading 20 deg (+x) to (10.06, 2.526)", text)
    find_line("leg 2  arc  r 4.5 (outer 6, inner 3) centre (8.52, 6.755) 240 deg left: heading 20 -> 260", text)
    find_line("leg 3  to y=1.5   from (4.088, 7.536) heading 260 deg (-x) to (3.024, 1.5)   lands on main.top", text)
    # MEASURED and PATCHES: the numbers the design took from the probe
    find_line("MEASURED   extent 60 x 14.25 mm (0.06 x 0.01425 m)   area 513.7 mm2   islands 4   narrowest passage "
              "2.999 (+/- 0.003) at (56.38, 2.181)   36 edges, shortest 1.50 at (0.75, 1.5)", text, tol=0.15)
    find_line("PATCHES    inlet (1 edge, 3.0) at (0, 0) = main.start     outlet (1 edge, 3.0) at (60, 0) = main.end", text)
    find_line("walls (34 edges, 294.6)", text)
    # CLAIMS: the summary and every row's measured / expected / verdict cells
    assert "CLAIMS     12 claims: 9 pass, 0 FAIL, 1 not measurable, 2 reported" in text
    assert_row(text, "c1", "main.width = 3.000 3 +/- 0.03 pass")
    assert_row(text, "c2", "main.length = 60.00 60 +/- 0.1 pass")
    assert_row(text, "c3", "loops.count = 4 4 pass")
    assert_row(text, "c4", "loops[*].width = 3.000 (x4) 3 +/- 0.03 pass")
    assert_row(text, "c5", "loops[*].outer_radius = 6.000 (x4, from the outer arc's curvature) 6 +/- 0.06 pass")
    assert_row(text, "c6", "loops[*].leave_angle = 20.0 (x4) built at the leg's wall; lip 28.9 on the arc <= 30 deg pass")
    assert_row(text, "c7", "heading 260 deg, 100 deg from the flow (+x): component -0.17 (x4) > 90 deg pass")
    assert_row(text, "c8", "inlet at (0, 0) (0, 0) +/- 0.5 pass")
    assert_row(text, "c9", "outlet at (60, 0) (60, 0) +/- 0.5 pass")
    assert_row(text, "c10", "loops[*].sweep = 240 deg (x4) reported pass")
    assert_row(text, "c11", "loops[*].return_angle = 80.0 (x4) reported pass")
    assert_row(text, "c12", "not measurable here: the finish and the main agent do these")
    assert text.splitlines()[-1] == "VERDICT    ready to COMMIT" and laps["T01"].ready


# -- 7.2 T02, the serpentine that contradicts itself --------------------------------------------------


def test_t02_prints_the_section_7_2_features_legs_and_the_c9_failure(laps):
    text = laps["T02"].text
    assert "LINT       clean" in text and not [f for f in laps["T02"].result["lint"] if f["level"] != "info"]
    find_line("FEATURES   snake     Serpentine  4 passes of 30, 3 bends r 3 (outer 4, inner 2), pass pitch 6, "
              "wall between passes 4;", text)
    find_line("start (0, 0) heading 0; end (0, 18) heading 180: on the start's side", text)
    find_line("LEGS       snake", text)
    find_line("leg 1  line 30   from (0, 0) heading 0 deg (+x) to (30, 0)", text)
    find_line("leg 2  arc  r 3 (outer 4, inner 2) centre (30, 3) 180 deg left: heading 0 -> 180", text)
    find_line("leg 3  line 30   from (30, 6) heading 180 deg (-x) to (0, 6)", text)
    find_line("leg 4  arc  r 3 (outer 4, inner 2) centre (0, 9) 180 deg right: heading 180 -> 0", text)
    find_line("leg 5  line 30   from (0, 12) heading 0 deg (+x) to (30, 12)", text)
    find_line("leg 6  arc  r 3 (outer 4, inner 2) centre (30, 15) 180 deg left: heading 0 -> 180", text)
    find_line("leg 7  line 30   from (30, 18) heading 180 deg (-x) to (0, 18)", text)
    find_prefix("MEASURED   extent 38 x 20 mm (0.038 x 0.02 m)   area 296.6 mm2   islands 0   narrowest passage 2.0", text, tol=0.15)
    assert "20 edges, shortest 2.00 at (0, 18)" in text
    find_line("PATCHES    inlet (1 edge, 2.0) at (0, 0) = snake.start    outlet (1 edge, 2.0) at (0, 18) = snake.end", text)
    assert "CLAIMS     10 claims: 8 pass, 1 FAIL, 1 not measurable" in text
    assert_row(text, "c1", "snake.width = 2.000 2 +/- 0.02 pass")
    assert_row(text, "c2", "passes = 4 4 pass")
    assert_row(text, "c3", "snake.pass_length = 30.00 (x4) 30 +/- 0.3 pass")
    assert_row(text, "c4", "bends = 3 3 pass")
    assert_row(text, "c5", "sweeps 180, 180, 180 deg 180 +/- 2 pass")
    assert_row(text, "c6", "snake.bend_radius = 3.000 (x3) 3 +/- 0.03 pass")
    assert "headings 0/180/0/180; centrelines y = 0, 6, 12, 18 (spacing 6)" in row_of(text, "c7") and row_of(text, "c7").endswith(" pass")
    assert_row(text, "c8", "inlet at (0, 0) (0, 0) +/- 0.5 pass")
    # the range is the fluid's true x extent (the bends overhang the passes by their outer radius 4)
    assert_row(text, "c9", "outlet at (0, 18): the LEFT end (x = 0 of -4..34) right FAIL")
    assert "an even number of passes ends on the inlet's side; 4 passes with 3 bends cannot end at the right." in text
    assert "5 passes (4 bends) or 3 passes (2 bends) end on the right; the request fixes 4 and 3." in text
    assert_row(text, "c10", "not measurable here: the finish and the main agent")
    last = text.splitlines()[-1]
    assert last.startswith("VERDICT    not ready: claims c9 FAIL") and "COMMIT disagrees: c9" in last
    assert laps["T02"].failing_claims == ["c9"] and not laps["T02"].ready


# -- 7.3 T04, the U duct ---------------------------------------------------------------------------


def test_t04_prints_the_section_7_3_features_legs_and_claims(laps):
    text = laps["T04"].text
    assert not [f for f in laps["T04"].result["lint"] if f["level"] == "error"]
    warned = [f["code"] for f in laps["T04"].result["lint"] if f["level"] == "warn"]
    assert warned in ([], ["W-UNITS"]), warned   # the abridged claims omit the 60 mm legs (docstring)
    find_line("FEATURES   u         Passage    width 8, 3 legs, centreline 151.4; start (0, 0) heading 0, "
              "end (0, 20) heading 180;", text)
    find_line("straight legs 1 and 3 parallel, 20 apart; both ends on x = 0 (the left)", text)
    find_line("LEGS       u", text)
    for expected in ("leg 1  line 60   from (0, 0) heading 0 deg (+x) to (60, 0)",
                     "leg 2  arc  r 10 (outer 14, inner 6) centre (60, 10) 180 deg left: heading 0 -> 180",
                     "leg 3  line 60   from (60, 20) heading 180 deg (-x) to (0, 20)"):
        # the design's "(line_to x=60)" / "(line_to x=0; open end)" tails are its own
        # annotation of the script, not a leg-table column
        assert any(squash(line).startswith(squash(expected)) for line in text.splitlines()), expected
    find_prefix("MEASURED   extent 74 x 28 mm (0.074 x 0.028 m)   area 1211 mm2   islands 0   narrowest passage 8.0", text, tol=0.5)
    assert "10 edges, shortest 8.00 at (0, 20)" in text
    find_line("PATCHES    inlet (1 edge, 8.0) at (0, 0) = u.start      outlet (1 edge, 8.0) at (0, 20) = u.end", text)
    find_line("walls (8 edges, 302.8)", text)
    assert "CLAIMS     9 claims: 9 pass" in text
    assert_row(text, "c1", "u.width = 8.000 8 +/- 0.08 pass")
    assert_row(text, "c2", "u.legs_straight = 2 (60.0, 60.0) 2 pass")
    assert_row(text, "c3", "legs 1 and 3: headings 0 / 180, 20.00 apart 20 +/- 0.2 pass")
    assert_row(text, "c4", "bends = 1 1 pass")
    assert_row(text, "c5", "u.bend_radius = 10.00 10 +/- 0.1 pass")
    assert_row(text, "c6", "sweep 180 deg 180 +/- 2 pass")
    assert_row(text, "c7", "inlet at (0, 0) (0, 0) +/- 4 pass")
    assert_row(text, "c8", "outlet at (0, 20) (0, 20) +/- 4 pass")
    assert_row(text, "c9", "open ends at (0, 0) facing -x and (0, 20) facing -x -x pass")
    assert text.splitlines()[-1] == "VERDICT    ready to COMMIT" and laps["T04"].ready


# -- 7.4 T05, a cylinder in a channel ----------------------------------------------------------------


def test_t05_prints_the_section_7_4_features_patches_and_claims(laps):
    text = laps["T05"].text
    assert "LINT       clean" in text
    codes = [f["code"] for f in laps["T05"].result["lint"]]
    assert codes == ["I-REFERENCE"] and "10 (hole diameter" in text
    find_line("FEATURES   duct      Rect       300 x 60 at origin (0, 0), centre (150, 30); width 60 (the shorter side), length 300", text)
    find_line("cyl       Disk       centre (100, 30), radius 5.000 (diameter 10, from the hole's curvature), circumference 31.42", text)
    find_line("fluid     Cut        duct - cyl: one face, 1 island (cyl)", text)
    assert "LEGS" not in text
    find_prefix("MEASURED   extent 300 x 60 mm (0.3 x 0.06 m)   area 1.792e+04 mm2   islands 1   narrowest passage 25.16",
                text, tol=0.5)
    assert "5 edges, shortest 31.42 at (95, 30)" in text and "islands at (100, 30) r 5" in text
    find_line("PATCHES    inlet (1 edge, 60.0) at (0, 30) = duct.left       outlet (1 edge, 60.0) at (300, 30) = duct.right", text)
    assert "cylinder (1 edge, 31.42)" in text and "= cyl.edge" in text
    find_line("walls (2 edges, 600.0) = duct.top, duct.bottom", text)
    assert "CLAIMS     10 claims: 10 pass" in text
    assert_row(text, "c1", "cyl.diameter = 10.00 10 +/- 0.1 pass")
    assert_row(text, "c2", "duct.height = 60.00 60 +/- 0.6 pass")
    assert_row(text, "c3", "duct.length = 300.0 300 +/- 3 pass")
    assert_row(text, "c4", "cyl.centre (100, 30); inlet at x = 0: 100.0 from it 100 +/- 1 pass")
    assert_row(text, "c5", "cyl.centre.y = 30.0 = duct mid-height 30.0 +/- 0.6 pass")
    assert_row(text, "c6", "inlet at (0, 30), the left side left pass")
    assert_row(text, "c7", "outlet at (300, 30), the right side right pass")
    assert "patch 'cylinder' = the 1 curve of cyl.edge" in row_of(text, "c8") and row_of(text, "c8").endswith(" pass")
    assert "patch 'walls' = every curve not inlet/outlet/cylinder (2)" in row_of(text, "c9") and row_of(text, "c9").endswith(" pass")
    assert_row(text, "c10", "holes = 1 1 pass")
    assert text.splitlines()[-1] == "VERDICT    ready to COMMIT" and laps["T05"].ready
    # the record's resolved rules are today's grammar, the cylinder named ON its curve (7.4, D8)
    patches = laps["T05"].result["record"]["patches"]
    assert patches[0] == {"name": "inlet", "at": "near:0,30"} and patches[1] == {"name": "outlet", "at": "near:300,30"}
    assert {"name": "cylinder", "kind": "wall", "at": "near:95,30"} in patches


# -- the record rebuilds through mesh2d.py into a case ---------------------------------------------------


def test_the_t01_record_rebuilds_through_mesh2d_into_a_case(laps, tmp_path):
    """`python openreynolds/toolbox/mesh2d.py <dir> --spec <dir>/geometry.json --scale 0.001
    --force` on the record the child wrote: the case directory with Allmesh and the mesh
    (the instance's rebuild command of 6.4; parse_spec ignores the record's extra keys,
    3.17 edit 3). OpenFOAM is not here, so the files are asserted."""
    case = tmp_path / "case"
    case.mkdir()
    record = laps["T01"].result["record"]
    (case / "geometry.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(MESH2D), str(case), "--spec", str(case / "geometry.json"),
                           "--scale", "0.001", "--force"], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    for name in ("Allmesh", "Allrun", "body.msh", "system/controlDict", "system/fvSchemes", "constant/geometry/body.json",
                 "constant/geometry/body.step", "0"):
        assert (case / name).exists(), name
    body = json.loads((case / "constant" / "geometry" / "body.json").read_text(encoding="utf-8"))
    assert body["ops"] == record["ops"] and body["claims"] == record["claims"]
    assert re.search(r"extent\s+0\.06 x 0\.01425", proc.stdout) and "islands 4" in proc.stdout
    assert re.search(r"patches\s+inlet \(1\), outlet \(1\), walls \(34\)", proc.stdout), proc.stdout[-1500:]
