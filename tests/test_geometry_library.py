"""The library and its goldens (DESIGN.md 3.16, 4.5, 4.6, D16, D17, D32).

Three kinds of test: the entries as code (a source, keywords, presets, no default
geometric number), the goldens on disk (each rebuilds to its own outline and
measurements, and carries the sha of the entry's file now), and the approval gate (an
unapproved entry is never matched and never shown, whatever the request says -- the one
test here that must never be xfail, because invariant 1 is xfail until a person approves).
The `golden` cli flow runs end to end in a temporary golden directory so the committed
APPROVALS.json is never touched by a test.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openreynolds.geometry import claims, cli, library

from geometry_scripts import T01

ROOT = Path(__file__).resolve().parents[1]

NAMES = ("serpentine", "t_junction", "tesla_valve")

T01_PROMPT = ("Generate a 2D mesh of a Tesla valve: a straight channel 3 mm wide and 60 mm long with 4 bypass "
              "loops (loop channel 3 mm wide, outer radius 6 mm) that leave the main channel at a shallow angle, "
              "sweep round, and return into the main channel against the forward direction, inlet at x=0, "
              "outlet at x=60 mm. Produce the mesh, run checkMesh, render the mesh, and write results.json "
              "with cells, bbox and number of bypass loops. Do not solve.")
"""qa-runs/prompts/T01.txt (the benchmark prompts live outside the repo)."""


def presets():
    return [(e.name, p) for e in library.entries() for p in e.presets]


def approve_everything(monkeypatch):
    """Every entry approved against its file now, in memory only."""
    monkeypatch.setattr(library, "approval",
                        lambda e: {"preset": e.default_preset, "approved": True, "by": "test",
                                   "at": "2026-09-08T09:40:00Z", "source_sha": library.source_sha(e)})


# -- the entries as code -----------------------------------------------------------------


def test_the_three_entries_are_the_directory():
    assert tuple(e.name for e in library.entries()) == NAMES
    for name in NAMES:
        entry = library.get(name)
        assert entry.name == name and entry.module.__name__ == f"openreynolds.geometry.library.{name}"
        assert library.Entry.from_dict(entry.as_dict()).name == name
    with pytest.raises(library.LibraryError, match="no library entry 'venturi'; the entries are serpentine"):
        library.get("venturi")
    with pytest.raises(library.LibraryError, match="has no preset 'xl'; its presets are t01"):
        library.get("tesla_valve").build("xl")


def test_every_entry_has_a_source_and_keywords():
    for entry in library.entries():
        assert entry.source.strip(), entry.name
        assert entry.keywords and all(k == k.lower() and k.strip() for k in entry.keywords), entry.name
        assert entry.presets and all(isinstance(v, dict) and v for v in entry.presets.values()), entry.name
        if entry.claims is not None:
            claim_set, _ = claims.parse(entry.claims)
            assert claim_set.claims
    tesla = library.get("tesla_valve")
    assert "tesla" in tesla.keywords and "20260907-003655-6aab" in tesla.source and "t01" in tesla.presets
    assert tesla.presets["t01"] == {"length": 60, "width": 3, "outer_radius": 6, "leave_angle": 20, "return_angle": 80,
                                   "count": 4, "pitch": 13}, "the committed spec of the study (D16)"
    assert library.get("serpentine").presets["t02"] == {"width": 2, "passes": 4, "pass_length": 30, "bend_radius": 3}


def test_no_entry_carries_a_default_geometric_number():
    """D16: `build` may default a unit or a direction word, never a length, an angle or
    a count -- those come from a preset with a source, or from the request."""
    for entry in library.entries():
        defaults = library.signature_defaults(entry)
        assert "units" in defaults
        for name, value in defaults.items():
            assert value is None or isinstance(value, str), f"{entry.name}.build({name}={value!r})"


def test_a_preset_builds_a_sketch_and_the_request_overrides_it():
    entry = library.get("tesla_valve")
    s = entry.build()
    assert s.units == "mm" and set(s.features) >= {"main", "loop", "loops"} and s.features["loops"].count == 4
    s = entry.build("t01", count=3, units="cm")
    assert s.units == "cm" and s.features["loops"].count == 3


def test_source_sha_ignores_line_endings(tmp_path):
    module = SimpleNamespace(__file__=str(tmp_path / "e.py"))
    entry = SimpleNamespace(module=module)
    (tmp_path / "e.py").write_bytes(b"SOURCE = 'x'\nKEYWORDS = ()\n")
    lf = library.source_sha(entry)
    (tmp_path / "e.py").write_bytes(b"SOURCE = 'x'\r\nKEYWORDS = ()\r\n")
    assert library.source_sha(entry) == lf
    (tmp_path / "e.py").write_bytes(b"SOURCE = 'y'\n")
    assert library.source_sha(entry) != lf


# -- every preset through the real cli ----------------------------------------------------


@pytest.mark.parametrize("name,preset", presets())
def test_every_preset_builds_clean_through_its_own_row_and_lint(name, preset, tmp_path, capsys):
    """rc 0 from `cli preview --library`, LINT clean, no notch, the entry's own claims all
    pass (the print-back a person reads before approving), and a picture."""
    pytest.importorskip("gmsh")
    pytest.importorskip("matplotlib")
    out = tmp_path / f"{name}.png"
    assert cli.main(["preview", "--library", name, "--preset", preset, "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "LINT       clean" in text and "NOTCH" not in text and "!! " not in text
    assert "VERDICT    ready to COMMIT" in text, text
    assert out.stat().st_size > 10_000


def test_regenerate_refuses_an_entry_that_fails_its_own_claims(monkeypatch, tmp_path):
    pytest.importorskip("gmsh")
    entry = library.get("t_junction")
    wrong = json.loads(json.dumps(entry.claims))
    wrong["claims"][0]["value"] = 99
    monkeypatch.setattr(entry, "claims", wrong)
    with pytest.raises(library.LibraryError, match="fails its own claims c1"):
        library.regenerate(entry, into=tmp_path)
    assert not (tmp_path / "t_junction.json").exists()


# -- the goldens on disk -------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_golden_json_pins_the_outline_and_measurements(name, tmp_path):
    """The committed golden is the entry's file now (source sha) and rebuilds to itself:
    Hausdorff < 1e-6 span, measurements within 1e-6 relative, no FAIL row (D17)."""
    pytest.importorskip("gmsh")
    pytest.importorskip("matplotlib")
    entry = library.get(name)
    golden = json.loads((library.GOLDEN / f"{name}.json").read_text(encoding="utf-8"))
    assert (library.GOLDEN / f"{name}.png").stat().st_size > 10_000
    assert golden["entry"] == name and golden["preset"] == entry.default_preset
    assert golden["params"] == entry.presets[entry.default_preset]
    assert golden["source_sha"] == library.source_sha(entry), "the golden was made from another source: regenerate it"
    for key in ("extent", "area", "islands", "features", "patches", "params"):
        assert key in golden["measurements"], key
    assert golden["measurements"]["patches"]["inlet"]["midpoints"], "the inlet's midpoints: the reference shift"
    assert not [r["id"] for r in golden["compliance"]["rows"] if r["verdict"] == "FAIL"]
    assert not [f for f in golden["lint"] if f["level"] == "error"]
    rebuilt = library.regenerate(entry, into=tmp_path)
    span = library.span_of(golden)
    assert library.hausdorff(rebuilt["outline"], golden["outline"]) < 1e-6 * span
    assert library.measurements_agree(rebuilt["measurements"], golden["measurements"], rel=1e-6)
    assert (tmp_path / f"{name}.png").exists() and (tmp_path / f"{name}.json").exists()


def test_the_approvals_file_is_keyed_by_entry_and_in_step_with_the_code():
    """APPROVALS.json carries one record per entry with the shas of the golden on disk;
    `approved` is a bool that only a person flips, and an approved record names who and
    when. (Whether anyone has approved is not asserted: that is Kabir's, not the suite's.)"""
    approvals = library.approvals()
    assert set(approvals) == set(NAMES)
    for entry in library.entries():
        record = approvals[entry.name]
        golden = json.loads((library.GOLDEN / f"{entry.name}.json").read_text(encoding="utf-8"))
        assert record["preset"] == golden["preset"] == entry.default_preset
        assert {k: record[k] for k in ("source_sha", "measurements_sha", "outline_sha")} == library.shas(golden)
        assert isinstance(record["approved"], bool)
        if record["approved"]:
            assert record["by"] and record["at"]
            assert library.approval(entry) == record and library.is_approved(entry)
        else:
            assert library.approval(entry) is None and not library.is_approved(entry)


def test_measurements_agree_and_hausdorff():
    a = {"extent": [60.0, 14.25], "islands": 4, "features": {"main": {"width": 3.0, "kind": "Passage"}}}
    assert library.measurements_agree(a, json.loads(json.dumps(a)), 1e-6)
    assert library.measurements_agree(a, {**a, "extent": [60.0 * (1 + 1e-7), 14.25]}, 1e-6)
    assert not library.measurements_agree(a, {**a, "extent": [60.0 * (1 + 1e-5), 14.25]}, 1e-6)
    assert not library.measurements_agree(a, {**a, "islands": 5}, 1e-6)
    assert not library.measurements_agree(a, {k: v for k, v in a.items() if k != "islands"}, 1e-6)
    assert not library.measurements_agree(a, {**a, "features": {"main": {"width": 3.0, "kind": "Rect"}}}, 1e-6)
    square = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]
    shifted = [[(x + 0.25, y) for x, y in square[0]]]
    assert library.hausdorff(square, square) == 0.0
    assert library.hausdorff(square, shifted) == pytest.approx(0.25)


# -- the approval gate (never xfail) --------------------------------------------------------


def test_match_finds_tesla_from_the_t01_prompt(monkeypatch):
    approve_everything(monkeypatch)
    entry, preset = library.match(T01_PROMPT)
    assert entry.name == "tesla_valve" and preset == "t01"
    entry, preset = library.match("A SERPENTINE passage of four passes")
    assert entry.name == "serpentine" and preset == "t02"
    assert library.match("a T junction with a 6 mm branch")[0].name == "t_junction"
    assert library.match("a straight duct 60 mm long") is None
    # the preset nearest the request's numbers: the one with the most parameters named
    two = SimpleNamespace(presets={"small": {"width": 3, "length": 60}, "large": {"width": 5, "length": 120}},
                          default_preset="small")
    assert library._nearest_preset(two, "a channel 5 wide and 120 long") == "large"
    assert library._nearest_preset(two, "a channel") == "small"


def test_an_unapproved_entry_is_never_matched_or_shown(tmp_path, monkeypatch, capsys):
    """D32, NOT xfail: with no approvals -- or a regenerated record a person has not
    flipped, or an approval of another source -- `match()` is None on the T01 prompt, the
    desk's reference is None, and the child with `--reference tesla_valve` exits 5 with
    E-REFERENCE-UNAPPROVED and no picture."""
    golden = tmp_path / "golden"
    golden.mkdir()
    monkeypatch.setattr(library, "GOLDEN", golden)
    assert library.match(T01_PROMPT) is None
    entry = library.get("tesla_valve")
    record = {"preset": "t01", "approved": False, "by": None, "at": None, "source_sha": library.source_sha(entry)}
    (golden / "APPROVALS.json").write_text(json.dumps({"tesla_valve": record}), encoding="utf-8")
    assert library.approval(entry) is None and library.match(T01_PROMPT) is None
    stale = dict(record, approved=True, by="someone", at="2026-09-08T09:40:00Z", source_sha="0" * 64)
    (golden / "APPROVALS.json").write_text(json.dumps({"tesla_valve": stale}), encoding="utf-8")
    assert library.approval(entry) == stale and not library.is_approved(entry)
    assert library.match(T01_PROMPT) is None

    from openreynolds.geometry import desk
    assert "library reference" not in desk.claims_message(T01_PROMPT, reference=None).lower()

    script = tmp_path / "t01.py"
    script.write_text(T01, encoding="utf-8")
    work = tmp_path / "lap"
    rc = cli.main(["build", "--script", str(script), "--out", str(work), "--reference", "tesla_valve"])
    assert rc == 5
    result = json.loads((work / "result.json").read_text(encoding="utf-8"))
    assert result["code"] == "E-REFERENCE-UNAPPROVED" and result["record"] is None
    assert "library entry 'tesla_valve' is not approved; nothing is shown as a reference" in result["report"]
    assert not (work / "preview.png").exists()
    capsys.readouterr()


def test_the_reference_gate_shows_an_approved_golden_beside_a_lap(tmp_path, monkeypatch, capsys):
    """With the committed golden and an approval in memory, the T01 script's lap carries
    the REFERENCE block with the entry's numbers and a Hausdorff distance measured after
    the inlet centres are made to coincide (the golden's row is at pitch 13, the script's
    at the solved 14.66, so the distance is small but not zero)."""
    pytest.importorskip("gmsh")
    pytest.importorskip("matplotlib")
    approve_everything(monkeypatch)
    script = tmp_path / "t01.py"
    script.write_text(T01, encoding="utf-8")
    work = tmp_path / "lap"
    assert cli.main(["build", "--script", str(script), "--out", str(work), "--reference", "tesla_valve"]) == 0
    result = json.loads((work / "result.json").read_text(encoding="utf-8"))
    assert result["reference"]["entry"] == "tesla_valve" and result["reference"]["preset"] == "t01"
    assert 0.0 < result["reference"]["hausdorff"] < 10.0
    report = result["report"]
    assert "REFERENCE  library tesla_valve (preset t01, approved 2026-09-08T09:40:00Z): length 60, width 3" in report
    assert "reference vs candidate: extent 60 x 14.25 vs 60 x 14.25, islands 4 vs 4; Hausdorff" in report
    assert (work / "preview.png").stat().st_size > 10_000
    capsys.readouterr()


# -- the golden flow through the cli, in a temporary directory ---------------------------------


def test_the_golden_flow_regenerates_checks_and_approves(tmp_path, monkeypatch, capsys):
    """`--regenerate NAME` writes json + png and an unapproved record; `--check` refuses an
    unapproved entry; `--approve NAME --by X` flips the flag with who and when; `--check`
    then passes; a source that changes afterwards fails it again; a regeneration that
    reproduces the golden leaves the approval standing."""
    pytest.importorskip("gmsh")
    pytest.importorskip("matplotlib")
    golden = tmp_path / "golden"
    monkeypatch.setattr(library, "GOLDEN", golden)
    monkeypatch.setattr(library, "_entry_names", lambda: ["t_junction"])

    assert cli.main(["golden", "--approve", "t_junction", "--by", "Tester"]) == 1
    assert "no golden for t_junction" in capsys.readouterr().out

    assert cli.main(["golden", "--regenerate", "t_junction"]) == 0
    out = capsys.readouterr().out
    assert "regenerated t_junction (plain)" in out and "awaiting approval" in out
    assert (golden / "t_junction.json").exists() and (golden / "t_junction.png").stat().st_size > 10_000
    record = library.approvals()["t_junction"]
    assert record["approved"] is False and record["by"] is None and record["at"] is None
    assert record["source_sha"] == library.source_sha(library.get("t_junction"))

    assert cli.main(["golden", "--check"]) == 1
    assert "t_junction: not approved" in capsys.readouterr().out
    assert library.match("a t-junction") is None

    assert cli.main(["golden", "--approve", "t_junction", "--by", "Tester"]) == 0
    assert "approved t_junction (plain) by Tester" in capsys.readouterr().out
    record = library.approvals()["t_junction"]
    assert record["approved"] is True and record["by"] == "Tester" and record["at"].endswith("Z")
    assert library.is_approved(library.get("t_junction"))
    assert library.match("a T-junction 100 long")[0].name == "t_junction"

    assert cli.main(["golden", "--check"]) == 0
    assert "t_junction: matches its golden" in capsys.readouterr().out

    assert cli.main(["golden", "--regenerate", "t_junction"]) == 0
    assert "the approval stands" in capsys.readouterr().out
    assert library.approvals()["t_junction"]["approved"] is True

    monkeypatch.setattr(library, "source_sha", lambda e: "f" * 64)
    assert cli.main(["golden", "--check"]) == 1
    assert "the source changed since its approval" in capsys.readouterr().out
    assert library.match("a t-junction") is None
    assert cli.main(["golden", "--approve", "t_junction", "--by", "Tester"]) == 1
    assert "made from another source; regenerate it before approving" in capsys.readouterr().out
