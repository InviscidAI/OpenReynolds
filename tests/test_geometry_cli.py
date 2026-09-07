"""The geometry command line (DESIGN.md 3.15): build | preview | check | reference |
golden, the files a build writes (result.json of 4.3, report.txt, record.json,
preview.png), the rc-5 path with and without a `partial`, the reference gate, and the
`__main__` shim.

`build --spec` runs the ops grammar through the same lap as a script (mesh2d by path
until U1/U2 land). `build --script` is exercised in-process over a canned Plan: the
FakeSketch stands in for U1's registry and `compile.plan` is monkeypatched to return a
Plan or raise the Row refusal, so the guard, the exec, the partial path, the record and
the files are all tested today and the same tests hold the moment U1 merges (the gated
tests at the end then run the real scripts and pin section 7.1's exact blocks).
"""
from __future__ import annotations

import builtins
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from openreynolds.geometry import cli, library, sketch
from openreynolds.geometry.sketch import SketchError

from _geometry_fakes import (FIXTURES, T01_CLAIMS, T01_LAP1_SCRIPT, T01_SPEC, T05_SPEC, install, landed,
                             partial_plan, plan_of, row_refusal)

pytest.importorskip("gmsh")
pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "openreynolds" / "geometry" / "cli.py"

needs_u1 = pytest.mark.skipif(not landed("U1"), reason="needs U1's Sketch (the script must construct one)")
needs_u3 = pytest.mark.skipif(not landed("U3"), reason="needs U3's report and claims")


def write_spec(tmp_path: Path, spec: dict, name: str = "spec.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def result_of(work: Path) -> dict:
    return json.loads((work / "result.json").read_text(encoding="utf-8"))


# -- build --------------------------------------------------------------------------------------


def test_build_writes_result_json_report_record_and_preview(tmp_path, capsys):
    work = tmp_path / "lap"
    rc = cli.main(["build", "--spec", str(write_spec(tmp_path, T01_SPEC)), "--out", str(work)])
    assert rc == 0
    for name in ("result.json", "report.txt", "record.json", "preview.png"):
        assert (work / name).exists(), name
    result = result_of(work)
    assert set(result) == {"ok", "rc", "seconds", "record", "measurements", "lint", "compliance", "report",
                           "verdict", "preview", "reference", "error", "traceback", "code"}
    assert result["ok"] is True and result["rc"] == 0 and result["seconds"] > 0
    assert result["preview"] == "preview.png" and (work / "preview.png").stat().st_size > 10_000
    assert result["error"] is None and result["traceback"] is None and result["code"] is None
    assert result["reference"] is None and result["compliance"] is None
    record = result["record"]
    assert record["format"] == "openreynolds.geometry/1" and record["units"] == "mm" and record["scale"] == 0.001
    assert record["ops"] == T01_SPEC["ops"] and record["patches"] == T01_SPEC["patches"]
    assert record["measurements"]["islands"] == 4 and record["built_with"]["gmsh"]
    assert json.loads((work / "record.json").read_text(encoding="utf-8")) == record
    assert result["measurements"]["extent"][0] == pytest.approx(60.0, abs=1e-5)
    report = (work / "report.txt").read_text(encoding="utf-8")
    assert report == result["report"] == capsys.readouterr().out.rstrip("\n")
    order = [line.split()[0] for line in report.splitlines() if line[:1].isupper()]
    assert order == ["SCRIPT", "LINT", "FEATURES", "LEGS", "MEASURED", "PATCHES", "CLAIMS", "REFERENCE", "VERDICT"]
    assert "MEASURED   extent 60 x 14.25 mm (0.06 x 0.01425 m)" in report and "islands 4" in report
    assert re.search(r"leg 3  to y=1\.5\s+from \(4\.088, 7\.536\) heading 260 deg \(-x\) to \(3\.024, 1\.5\)", report)
    assert "inlet (1 edge, 3.0) at (0, 0)" in report and "outlet (1 edge, 3.0) at (60, 0)" in report


def test_build_rc_2_on_lint_errors_still_draws(tmp_path):
    work = tmp_path / "lap"
    rc = cli.main(["build", "--spec", str(FIXTURES / "tesla_real_attempt3.json"), "--out", str(work)])
    assert rc == 2
    result = result_of(work)
    assert result["ok"] is False and result["rc"] == 2 and result["record"] is not None
    assert (work / "preview.png").stat().st_size > 10_000
    codes = [f["code"] for f in result["lint"]]
    assert codes.count("E-OVERLAP") == 3
    assert result["verdict"]["ready"] is False and result["verdict"]["text"].startswith("not ready: LINT E-OVERLAP")
    assert "bypasses[0] and bypasses[1]" in result["report"]   # D29: <row>[k], 0-based


def test_build_script_over_a_canned_plan_runs_the_whole_lap(tmp_path, monkeypatch):
    """The exec, the guard (installed around the exec alone), the notes and prints, the
    build, the picture and the record, with `compile.plan` handing back T01 lap 2."""
    install(monkeypatch, plan=plan_of(T01_SPEC))
    real_import = builtins.__import__
    script = ('s = Sketch(units="mm")\n'
              'print("hello", 1.5)\n'
              's.note("return_angle 80 keeps 4 loops of outer r 6 inside 60")\n')
    work = tmp_path / "lap"
    rc = cli.exec_script(script, work, None, None, work / "preview.png")
    assert rc == 0 and builtins.__import__ is real_import
    result = result_of(work)
    assert result["ok"] and result["record"]["script"] == script
    assert result["record"]["script_sha256"] and len(result["record"]["script_sha256"]) == 64
    assert "SCRIPT     ran in" in result["report"] and "1 print" in result["report"]
    assert "           > hello 1.5" in result["report"]
    assert "           note: return_angle 80 keeps 4 loops of outer r 6 inside 60" in result["report"]
    assert (work / "preview.png").stat().st_size > 10_000
    assert (work / "script.py").exists() is False   # the runner writes script.py; the cli reads what it is given


def test_the_guard_refuses_only_the_scripts_own_frame(tmp_path, monkeypatch):
    """A kernel function that imports during the script is untouched: `Sketch.note`
    here imports `json` from the fake's frame, whose `__name__` is not `__sketch__`."""
    install(monkeypatch, plan=plan_of(T05_SPEC))

    def note(self, text):
        import os  # noqa: F401 - allowed: not the script's frame
        self.notes.append(text)

    monkeypatch.setattr(sketch.Sketch, "note", note)
    work = tmp_path / "lap"
    rc = cli.exec_script('s = Sketch(units="mm")\ns.note("kernel import ok")\n', work, None, None, None)
    assert rc == 0 and "note: kernel import ok" in result_of(work)["report"]


def test_two_sketches_is_rc_5(tmp_path, monkeypatch):
    install(monkeypatch, plan=plan_of(T05_SPEC))
    work = tmp_path / "lap"
    rc = cli.exec_script('a = Sketch(units="mm")\nb = Sketch(units="mm")\n', work, None, None, None)
    assert rc == 5
    result = result_of(work)
    assert result["code"] == "E-MANY-SKETCHES" and result["record"] is None
    assert "the script made 2 Sketch objects; exactly one" in result["report"]


def test_build_rc_5_with_a_partial_draws_the_instance_and_prints_features_and_legs(tmp_path, monkeypatch):
    """Section 7.1 lap 1: the Row refuses, and the lap still gets the single instance
    built, measured and drawn with its footprint, FEATURES and LEGS for it, the refusal
    under LINT, CLAIMS not evaluated, VERDICT naming the code."""
    install(monkeypatch, raising=row_refusal(), partial=partial_plan())
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(json.dumps(T01_CLAIMS), encoding="utf-8")
    work = tmp_path / "lap1"
    rc = cli.exec_script('s = Sketch(units="mm")\n', work, claims_path, None, work / "preview.png")
    assert rc == 5
    result = result_of(work)
    assert result["ok"] is False and result["rc"] == 5 and result["code"] == "E-ROW-FIT"
    assert result["error"] == "!! ERROR  E-ROW-FIT  Row 'loops': 4 x Bypass 'loop' do not fit on main.top (60 long)"
    assert result["compliance"] is None and result["preview"] == "preview.png"
    assert (work / "preview.png").stat().st_size > 10_000
    assert "loop" in result["record"]["features"] and result["record"]["features"]["loop"]["kind"] == "Bypass"
    assert result["record"]["features"]["loop"]["solved"]["footprint"] == [-12.46, 7.28]
    assert result["measurements"]["extent"][0] == pytest.approx(19.74, abs=0.05)
    assert [f["code"] for f in result["lint"]] == ["E-ROW-FIT"]
    assert result["lint"][0]["numbers"]["table"]["80"] == 13.02
    report = result["report"]
    assert "LINT       1 error (errors block)" in report
    assert "           !! ERROR  E-ROW-FIT  Row 'loops': 4 x Bypass 'loop' do not fit on main.top (60 long)" in report
    assert "4 x 19.74 = 78.96 needed before any gap; 60 - 2 x margin 1.5 = 57 available" in report
    assert re.search(r"^FEATURES   loop\s+Bypass", report, re.M)
    # the leg table in the design's own padding (7.1: "line 3     from", "to y=1.5   from")
    assert re.search(r"leg 1  line 3\s+from \(0, 1\.5\) heading 20 deg \(\+x\) to \(2\.819, 2\.526\)", report)
    assert re.search(r"leg 2  arc  r 4\.5 \(outer 6, inner 3\) centre \(1\.28, 6\.755\) 205 deg left: heading 20 -> 225", report)
    assert re.search(r"leg 3  to y=1\.5\s+from \(-1\.9\d*, 9\.9\d*\) heading 225 deg \(-x\) to \(-10\.3\d, 1\.5\)", report)
    assert "CLAIMS     not evaluated: the sketch did not build" in report
    assert report.splitlines()[-1] == "VERDICT    not ready: LINT E-ROW-FIT"
    assert result["verdict"] == {"text": "not ready: LINT E-ROW-FIT", "ready": False}


def test_build_rc_5_without_a_partial_is_the_refusal_alone(tmp_path, monkeypatch):
    refusal = SketchError("E-RADIUS", "Bypass 'loop'", "outer_radius 3 with width 3 leaves an inner radius of 0",
                          "the inner wall folds over; outer_radius > width (inner > 0), and >= 1.5 x width meshes cleanly")
    install(monkeypatch, raising=refusal)
    work = tmp_path / "lap"
    rc = cli.exec_script('s = Sketch(units="mm")\n', work, None, None, work / "preview.png")
    assert rc == 5
    result = result_of(work)
    assert result["record"] is None and result["measurements"] is None and result["compliance"] is None
    assert result["preview"] is None and not (work / "preview.png").exists() and not (work / "record.json").exists()
    assert result["code"] == "E-RADIUS" and result["lint"] == []
    assert result["report"] == ("SCRIPT     ran in 0.0 s; no prints\n" + str(refusal))
    assert result["error"] == str(refusal).splitlines()[0]


def test_an_unreadable_claims_file_is_rc_5(tmp_path, monkeypatch):
    install(monkeypatch, plan=plan_of(T05_SPEC))
    claims_path = tmp_path / "claims.json"
    claims_path.write_text("{not json", encoding="utf-8")
    work = tmp_path / "lap"
    rc = cli.exec_script('s = Sketch(units="mm")\n', work, claims_path, None, None)
    assert rc == 5 and result_of(work)["code"] == "E-CLAIMS"


def test_a_picture_that_fails_does_not_lose_the_lap(tmp_path, monkeypatch):
    """gmsh raises a plain Exception when the coarse triangulation fails; the build, the
    measurements and the print-back stand, the lap notes the missing picture and exits
    with the lint's code, and no half-written PNG is left for the runner to read."""
    from openreynolds.geometry import preview

    def refuse(*args, **kwargs):
        raise Exception("Mesh generation failed (simulated)")

    monkeypatch.setattr(preview, "_triangles", refuse)
    work = tmp_path / "lap"
    rc = cli.main(["build", "--spec", str(write_spec(tmp_path, T05_SPEC)), "--out", str(work)])
    assert rc == 0
    result = result_of(work)
    assert result["ok"] is True and result["record"] is not None and result["measurements"]["islands"] == 1
    assert result["preview"] is None and not (work / "preview.png").exists()
    assert "note: no picture: Exception: Mesh generation failed (simulated)" in result["report"]
    assert "MEASURED   extent 300 x 60 mm" in result["report"] and "VERDICT" in result["report"]


def test_a_kernel_exception_is_rc_3_with_a_result_json(tmp_path, monkeypatch):
    """Whatever the kernel raises outside the script (here the record writer), the child
    still answers with result.json: rc 3, code E-KERNEL, the kernel's frames in
    `traceback`, and a text that says nothing in the script is to be fixed."""
    install(monkeypatch, plan=plan_of(T05_SPEC))

    def broken(*args, **kwargs):
        raise RuntimeError("boom in the record")

    monkeypatch.setattr(cli.compile, "record", broken)
    work = tmp_path / "lap"
    rc = cli.exec_script('s = Sketch(units="mm")\nprint("kept")\n', work, None, None, work / "preview.png")
    assert rc == 3
    result = result_of(work)
    assert result["ok"] is False and result["rc"] == 3 and result["code"] == "E-KERNEL"
    assert result["error"] == "!! ERROR  E-KERNEL  the kernel failed: RuntimeError: boom in the record"
    assert "(nothing in the script to fix: an internal error; reported to the desk)" in result["report"]
    assert "           > kept" in result["report"]
    assert "boom in the record" in result["traceback"] and "cli.py" in result["traceback"]
    assert result["record"] is None and result["measurements"] is None
    # the spec lap has the same net under it
    monkeypatch.setattr(cli.compile, "record", broken)
    rc = cli.main(["build", "--spec", str(write_spec(tmp_path, T05_SPEC)), "--out", str(tmp_path / "spec")])
    assert rc == 3 and result_of(tmp_path / "spec")["code"] == "E-KERNEL"


def test_a_records_claims_go_into_the_work_dir_not_the_system_temp(tmp_path):
    spec, scale, claims_file = cli._record_to_spec({"ops": T05_SPEC["ops"], "patches": T05_SPEC["patches"],
                                                    "scale": 0.001, "claims": T01_CLAIMS}, tmp_path)
    assert claims_file == tmp_path / "claims.json" and json.loads(claims_file.read_text(encoding="utf-8")) == T01_CLAIMS
    assert scale == 0.001 and spec["ops"] == T05_SPEC["ops"]
    assert cli._record_to_spec({"ops": [], "scale": 1.0}, tmp_path)[2] is None


# -- check / preview ----------------------------------------------------------------------------


def test_check_on_the_t05_record_prints_the_tables_and_exits_0(tmp_path, capsys):
    work = tmp_path / "build"
    assert cli.main(["build", "--spec", str(write_spec(tmp_path, T05_SPEC)), "--out", str(work)]) == 0
    capsys.readouterr()
    rc = cli.main(["check", "--record", str(work / "record.json")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "MEASURED   extent 300 x 60 mm (0.3 x 0.06 m)" in out and "islands 1" in out
    assert "cylinder (1 edge, 31.42)" in out and "inlet (1 edge, 60.0) at (0, 30)" in out
    assert "outlet (1 edge, 60.0) at (300, 30)" in out and "walls (2 edges, 600.0)" in out
    assert "LINT       clean" in out
    assert "FEATURES   duct      Rect" in out and re.search(r"^\s+cyl\s+Disk", out, re.M)


def test_check_exit_2_on_the_attempt3_fixture(capsys):
    rc = cli.main(["check", "--spec", str(FIXTURES / "tesla_real_attempt3.json")])
    out = capsys.readouterr().out
    assert rc == 2
    assert out.count("!! ERROR  E-OVERLAP") == 3 and "VERDICT    not ready: LINT E-OVERLAP" in out


def test_check_refuses_a_spec_whose_fluid_ends_in_a_mirror(tmp_path, capsys):
    spec = copy.deepcopy(T01_SPEC)
    spec["ops"][-1]["name"] = "fused"
    spec["ops"].append({"op": "mirror", "name": "body", "target": "fused", "axis": "x", "at": 0})
    rc = cli.main(["check", "--spec", str(write_spec(tmp_path, spec, "mirror_last.json"))])
    out = capsys.readouterr().out
    assert rc == 5
    assert "!! ERROR  E-MIRROR-BAKE  spec: the fluid's last op is `mirror`; the classifier is wrong on a mirrored face" in out
    assert "fuse it with something after the mirror, or mirror the primitives' coordinates (the API does this itself)" in out
    # `keep: true` fuses the image with the original: that fuse clears it (D26)
    spec["ops"][-1]["keep"] = True
    rc = cli.main(["check", "--spec", str(write_spec(tmp_path, spec, "mirror_kept.json"))])
    assert rc != 5 and "E-MIRROR-BAKE" not in capsys.readouterr().out
    assert cli.mirror_baked(T01_SPEC["ops"]) is False


def test_preview_draws_a_spec_a_record_and_the_plain_picture(tmp_path):
    spec = write_spec(tmp_path, T05_SPEC)
    assert cli.main(["preview", "--spec", str(spec), "--out", str(tmp_path / "a.png")]) == 0
    assert (tmp_path / "a.png").stat().st_size > 10_000
    assert cli.main(["preview", "--spec", str(spec), "--out", str(tmp_path / "plain.png"), "--plain"]) == 0
    assert (tmp_path / "plain.png").stat().st_size > 10_000
    work = tmp_path / "build"
    assert cli.main(["build", "--spec", str(spec), "--out", str(work)]) == 0
    assert cli.main(["preview", "--record", str(work / "record.json"), "--out", str(tmp_path / "rec.png")]) == 0
    assert (tmp_path / "rec.png").stat().st_size > 10_000


# -- reference / golden / the shim ----------------------------------------------------------------


def test_reference_prints_api_summary(tmp_path, capsys, monkeypatch):
    card = "Sketch(units='mm')  -- one sketch per script\nRect(...)\n"
    monkeypatch.setattr(sketch, "api_summary", lambda: card)
    assert cli.main(["reference"]) == 0
    assert capsys.readouterr().out == card + "\n"
    monkeypatch.setattr(cli, "REFERENCE_MD", tmp_path / "reference.md")
    assert cli.main(["reference", "--write"]) == 0
    assert (tmp_path / "reference.md").read_text(encoding="utf-8") == card


def test_an_unapproved_reference_is_refused_with_rc_5(tmp_path, capsys):
    work = tmp_path / "lap"
    rc = cli.main(["build", "--spec", str(write_spec(tmp_path, T01_SPEC)), "--out", str(work),
                   "--reference", "tesla_valve"])
    assert rc == 5
    result = result_of(work)
    assert result["code"] == "E-REFERENCE-UNAPPROVED" and result["record"] is None
    assert ("!! ERROR  E-REFERENCE-UNAPPROVED  library entry 'tesla_valve' is not approved; nothing is shown as a reference\n"
            "          until a person approves it (cli golden --approve tesla_valve --by <name>)") in result["report"]
    assert not (work / "preview.png").exists()


def test_the_reference_gate_lets_an_approved_entry_through(tmp_path, monkeypatch):
    """With an approval whose sha matches the entry, the golden's outline and
    measurements reach the lap, the picture gets its panel and result.json its distance."""
    from types import SimpleNamespace
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    with cli._gmsh_session("golden") as gmsh:
        from openreynolds.geometry import preview
        plan = plan_of(T01_SPEC)
        a = cli.analyse(gmsh, plan, None)
        outline = preview.outline_polyline(gmsh, a.face, max(a.m.extent) / 500)
    (golden_dir / "tesla_valve.json").write_text(json.dumps({
        "entry": "tesla_valve", "preset": "t01", "source_sha": "abc", "outline": [[list(p) for p in loop] for loop in outline],
        "measurements": {"extent": list(a.m.extent), "patches": {"inlet": a.m.patches["inlet"]}}}), encoding="utf-8")
    entry = SimpleNamespace(name="tesla_valve")
    monkeypatch.setattr(library, "GOLDEN", golden_dir)
    monkeypatch.setattr(library, "get", lambda name: entry)
    monkeypatch.setattr(library, "approval", lambda e: {"preset": "t01", "by": "Kabir", "at": "2026-09-08T09:40:00Z", "source_sha": "abc"})
    monkeypatch.setattr(library, "source_sha", lambda e: "abc")
    work = tmp_path / "lap"
    rc = cli.main(["build", "--spec", str(write_spec(tmp_path, T01_SPEC)), "--out", str(work), "--reference", "tesla_valve"])
    assert rc == 0
    result = result_of(work)
    assert result["reference"]["entry"] == "tesla_valve" and result["reference"]["preset"] == "t01"
    assert result["reference"]["hausdorff"] == pytest.approx(0.0, abs=1e-6)
    assert "REFERENCE  library tesla_valve (preset t01, approved 2026-09-08T09:40:00Z)" in result["report"]
    # and a source that changed since the approval closes the gate again (D32)
    monkeypatch.setattr(library, "source_sha", lambda e: "def")
    rc = cli.main(["build", "--spec", str(write_spec(tmp_path, T01_SPEC)), "--out", str(tmp_path / "lap2"),
                   "--reference", "tesla_valve"])
    assert rc == 5 and result_of(tmp_path / "lap2")["code"] == "E-REFERENCE-UNAPPROVED"


def test_the_shim_does_not_shadow_stdlib_trace():
    """In the child the checkout ROOT is the path entry (parents[2]), never
    `openreynolds/` itself, so `import trace` is the stdlib module and the kernel has
    one module name; the failure it proves against is `openreynolds/trace.py`."""
    code = ("import sys, trace, openreynolds.geometry.cli as c; "
            "print(trace.__file__); print(c.__name__); print(sys.flags.isolated)")
    proc = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r}); " + code],
                          capture_output=True, text=True, timeout=120,
                          env={"PATH": "", "PYTHONUTF8": "1", "SYSTEMROOT": "C:\\Windows"})
    assert proc.returncode == 0, proc.stderr
    trace_file, name, isolated = proc.stdout.split()
    assert "openreynolds" not in trace_file and name == "openreynolds.geometry.cli" and isolated == "1"
    assert (ROOT / "openreynolds" / "trace.py").exists()   # what the parents[1] shim of the first draft would have loaded
    # and the shim itself: run by path, the cli imports as `openreynolds.geometry.cli` and answers
    proc = subprocess.run([sys.executable, "-I", str(CLI), "reference"], capture_output=True, text=True, timeout=120,
                          env={"PATH": "", "PYTHONUTF8": "1", "SYSTEMROOT": "C:\\Windows"})
    assert "ModuleNotFoundError" not in proc.stderr and "Traceback" not in proc.stderr, proc.stderr


def test_main_without_a_command_prints_help_and_exits_1(capsys):
    assert cli.main([]) == 1
    assert "build" in capsys.readouterr().out


def test_golden_check_passes_on_committed_goldens(capsys):
    approvals = library.GOLDEN / "APPROVALS.json"
    if not approvals.exists():
        pytest.skip("no APPROVALS.json yet: awaiting the library (U6) and Kabir's approval")
    if not json.loads(approvals.read_text(encoding="utf-8")):
        pytest.skip("APPROVALS.json is empty: awaiting Kabir's approval")
    assert cli.main(["golden", "--check"]) == 0, capsys.readouterr().out


# -- the worked example through the real API (after U1 / U3) ----------------------------------------


@needs_u1
@needs_u3
def test_lap_1_of_t01_prints_the_section_7_1_features_and_legs(tmp_path):
    """The exact FEATURES and LEGS blocks of section 7.1 lap 1, `[new]` markers stripped."""
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(json.dumps(T01_CLAIMS), encoding="utf-8")
    work = tmp_path / "lap1"
    rc = cli.exec_script(T01_LAP1_SCRIPT, work, claims_path, None, work / "preview.png")
    assert rc == 5
    result = result_of(work)
    assert (work / "preview.png").stat().st_size > 10_000 and result["compliance"] is None
    expected = """
FEATURES   main      Passage    width 3, one leg 60 along +x, start (0, 0) end (60, 0)
           loop      Bypass     solved: R 4.5 (outer 6, inner 3), sweep 205, return heading 225 (-x), leave_length 3 (default: one width),
                                lands 10.34 upstream of its anchor, footprint -12.46..+7.28, height 11.25
           loops     Row        NOT SOLVED (E-ROW-FIT)
LEGS       loop (one instance, anchor at u = 0 for the table)
             leg 1  line 3     from (0, 1.5) heading 20 deg (+x) to (2.819, 2.526)
             leg 2  arc  r 4.5 (outer 6, inner 3) centre (1.28, 6.755) 205 deg left: heading 20 -> 225
             leg 3  to y=1.5   from (-1.902, 9.937) heading 225 deg (-x) to (-10.34, 1.5)   lands on main.top
"""
    # the leg numbers are the design's to four significant figures (7.1 lap 2 is the
    # "exact tool output": (10.06, 2.526), (8.52, 6.755)); the lap-1 block of the design
    # rounds the same points by hand (2.82, 2.53, 6.75, -1.90, 9.94)
    squash = lambda text: [" ".join(line.split()) for line in text.strip().splitlines()]  # noqa: E731
    report = squash(result["report"])
    for line in squash(expected):
        assert line in report, line
