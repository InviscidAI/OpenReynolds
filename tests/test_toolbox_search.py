"""`search.py`: one retrieval interface over both tiers.

Every hit says which tier it came from and why it matched, a distribution is reported
per tier, and every query is logged with its whole ranked list."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def corpus():
    return load("corpus")


@pytest.fixture(scope="module")
def search(corpus):
    return load("search")


def dictionary(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("// a dictionary\n" + body, encoding="utf-8")
    return path


def row(**overrides) -> dict:
    """An index row, in the shape `corpus.py` writes (held to it below)."""
    base = {
        "path": "/tutorials/incompressible/simpleFoam/aCase",
        "tier": "vendor",
        "solver": {"executable": "simpleFoam", "module": None},
        "runs": True,
        "turbulence": {"simulation_type": "RAS", "model": "kEpsilon"},
        "regions": {},
        "regime": {"class": "incompressible-steady", "compressible": False,
                   "steady": True, "Re": None, "Ma": None, "shedding_risk": None},
        "mesh_type": "blockMesh",
        "bc_map": {},
        "of_version": "v2512",
        "of_fork": "esi",
        "verdict": None,
        "provenance": {"indexed_at": "2026-09-03T00:00:00Z", "schema_version": 1},
    }
    base.update(overrides)
    return base


def earned_row(**overrides) -> dict:
    base = row(path="/work/a-study", tier="earned", verdict="failed", study_id="a-study",
               case="case", cases=["case"], artifacts={"mesh-full": 1}, rungs=[], notes=[])
    base.update(overrides)
    return base


def test_the_row_factory_matches_what_corpus_writes(corpus, tmp_path):
    dictionary(tmp_path / "tutorials" / "family" / "aCase" / "system" / "controlDict",
               "application  simpleFoam;\n")
    real, _ = corpus.harvest_tutorials(tmp_path / "tutorials")
    assert set(real[0]) == set(row())


def test_the_row_and_stamp_shapes_are_pinned_to_the_schema_version(corpus, tmp_path):
    """A field added without bumping `SCHEMA_VERSION` leaves every index on disk stale
    and undetectably so. This makes the bump conscious rather than remembered."""
    assert corpus.SCHEMA_VERSION == 4
    assert set(earned_row()) - set(row()) == {"study_id", "case", "cases", "artifacts",
                                              "rungs", "notes"}
    dictionary(tmp_path / "tutorials" / "family" / "aCase" / "system" / "controlDict",
               "application  simpleFoam;\n")
    stamp = corpus.build(tmp_path / "tutorials", tmp_path / "corpus", work=tmp_path / "work")
    assert set(stamp) == {"of_version", "of_fork", "built_at", "counts", "schema_version",
                          "tutorials", "work", "studies"}


# -- staleness -----------------------------------------------------------------


def stamp_of(corpus, **overrides) -> dict:
    return {"of_version": "v2512", "schema_version": corpus.SCHEMA_VERSION, **overrides}


def test_staleness_from_the_stamp(search, corpus):
    assert search.staleness(None, "v2512")
    assert search.staleness(stamp_of(corpus), "v2512") is None
    other = search.staleness(stamp_of(corpus, of_version="v2406"), "v2512")
    assert "v2406" in other and "v2512" in other
    assert "schema" in search.staleness(
        stamp_of(corpus, schema_version=corpus.SCHEMA_VERSION - 1), "v2512")


def test_not_knowing_the_live_version_does_not_force_a_rebuild(search, corpus):
    """Rebuilding on it would rebuild forever, each index stamped `unknown`."""
    assert search.staleness(stamp_of(corpus), "unknown") is None


@pytest.fixture
def built(tmp_path, monkeypatch):
    """A tutorial tree on disk with one case in it, and an empty `work`."""
    monkeypatch.setenv("WM_PROJECT_VERSION", "v2512")
    case = tmp_path / "tutorials" / "incompressible" / "simpleFoam" / "pitzDaily"
    dictionary(case / "system" / "controlDict", "application  simpleFoam;\n")
    dictionary(case / "constant" / "turbulenceProperties",
               "simulationType RAS;\nRAS\n{\n    RASModel  kEpsilon;\n}\n")
    (tmp_path / "work").mkdir()
    return tmp_path


def ensure(search, built, **kwargs):
    args = {"tutorials": built / "tutorials", "work": built / "work", **kwargs}
    return search.ensure(built / "corpus", **args)


def test_an_index_is_built_on_first_use_and_reused_after(search, built):
    rows, reason = ensure(search, built)
    assert reason and [r["solver"]["executable"] for r in rows] == ["simpleFoam"]
    stamped = (built / "corpus" / "corpus.stamp.json").read_text()
    rows, reason = ensure(search, built)
    assert reason is None and rows
    assert (built / "corpus" / "corpus.stamp.json").read_text() == stamped
    assert ensure(search, built, rebuild=True)[1] == "asked for"


def test_a_missing_index_file_is_a_reason_to_rebuild(search, built):
    """The stamp is not the index: a deleted `.jsonl` behind a valid stamp answered
    "nothing matched" with exit 0."""
    ensure(search, built)
    (built / "corpus" / "tutorials.index.jsonl").unlink()
    rows, reason = ensure(search, built)
    assert "index" in reason and rows


def test_an_index_built_from_another_tree_is_stale(search, built, tmp_path):
    ensure(search, built)
    other = tmp_path / "other-tutorials"
    dictionary(other / "family" / "bCase" / "system" / "controlDict",
               "application  pimpleFoam;\n")
    rows, reason = ensure(search, built, tutorials=other)
    assert "tree" in reason
    assert [r["solver"]["executable"] for r in rows] == ["pimpleFoam"]


def test_another_work_tree_makes_the_index_stale(search, built, tmp_path):
    """The stamp recorded `work`, and nothing compared it."""
    ensure(search, built)
    (tmp_path / "other-work").mkdir()
    _, reason = ensure(search, built, work=tmp_path / "other-work")
    assert "other studies" in reason


def test_a_study_finishing_makes_the_index_stale(search, built):
    """A study finishing moves none of schema, tree or version, so without the
    `studies` fingerprint the earned tier was built once and never again."""
    rows, _ = ensure(search, built)
    assert [r["tier"] for r in rows] == ["vendor"]
    study = built / "work" / "a-study"
    (study / ".reynolds").mkdir(parents=True)
    (study / ".reynolds" / "phases.json").write_text('{"phases": []}', encoding="utf-8")
    dictionary(study / "case" / "system" / "controlDict", "application  pimpleFoam;\n")
    rows, reason = ensure(search, built)
    assert reason == "the studies have changed since the index was built"
    assert sorted(r["tier"] for r in rows) == ["earned", "vendor"]


def test_if_stale_builds_once_and_then_leaves_a_current_index_alone(corpus, built, capsys):
    """What the harness runs at toolbox sync."""
    args = ["build", "--if-stale", "--tutorials", str(built / "tutorials"),
            "--work", str(built / "work"), "--out", str(built / "corpus")]
    assert corpus.main(args) == 0
    assert "rebuilding: no index has been built yet" in capsys.readouterr().out
    assert corpus.main(args) == 0
    assert "index is current" in capsys.readouterr().out


def test_an_unset_tutorials_variable_does_not_build_from_the_working_directory(
    search, tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("FOAM_TUTORIALS", raising=False)
    monkeypatch.chdir(tmp_path)
    dictionary(tmp_path / "a-stray-case" / "system" / "controlDict", "application  simpleFoam;\n")
    assert search.main(["--corpus", str(tmp_path / "corpus"), "regime", "simpleFoam"]) == 1
    assert "FOAM_TUTORIALS" in capsys.readouterr().err
    assert not (tmp_path / "corpus").exists()


def test_a_query_against_an_existing_index_needs_no_tutorial_tree(
    search, built, monkeypatch, capsys
):
    """The guard belongs on building, not on asking."""
    ensure(search, built)
    monkeypatch.delenv("FOAM_TUTORIALS", raising=False)
    code = search.main(["--corpus", str(built / "corpus"), "--work", str(built / "work"),
                        "regime", "simpleFoam"])
    assert code == 0
    assert "simpleFoam" in capsys.readouterr().out


# -- ranking, and why ----------------------------------------------------------


def test_a_field_match_outranks_a_path_match_and_says_why(search):
    """`incompressible/simpleFoam/other` names simpleFoam by filing, not by running it."""
    solver = row(path="/tutorials/x/y/aCase")
    named = row(path="/tutorials/incompressible/simpleFoam/other",
                solver={"executable": "potentialFoam", "module": None})
    hits = search.search([named, solver], "simpleFoam")
    assert [(h["path"], h["why"]) for h in hits] == [
        (solver["path"], ["solver=simpleFoam"]), (named["path"], ["text~simpleFoam"])]


def test_more_fields_matched_scores_higher(search):
    both = row(path="/a")
    one = row(path="/b", turbulence={"simulation_type": "RAS", "model": "kOmegaSST"})
    assert [h["path"] for h in search.search([one, both], "RAS kEpsilon")] == ["/a", "/b"]


def test_the_regime_class_matches_on_its_parts_and_only_once(search):
    """A one-word class (`steady`) used to score twice and outrank a case matching two
    fields. Found by running the design document's own query."""
    one_word = row(path="/a", regime=dict(row()["regime"], **{"class": "steady"}))
    hits = search.search([one_word, row(path="/b")], "incompressible steady")
    assert [h["path"] for h in hits] == ["/b", "/a"]
    assert "regime=steady" in hits[1]["why"]


def test_a_word_that_matches_nothing_is_reported_against_the_tier_searched(search):
    """Internal versus external is not derivable from a tutorial; and `failure`
    searches only the earned tier, so a vendor-only word is unmatched there."""
    rows = [row(path="/tutorials/incompressible/simpleFoam/pitzDaily"), earned_row()]
    assert search.unmatched_tokens(rows, "internal incompressible steady") == ["internal"]
    assert search.unmatched_tokens(rows, "pitzDaily") == []
    assert search.unmatched_tokens(rows, "pitzDaily", tier="earned") == ["pitzDaily"]


def test_every_hit_carries_its_tier_and_a_search_can_be_held_to_one(search):
    rows = [row(), earned_row(verdict="completed")]
    assert sorted(h["tier"] for h in search.search(rows, "simpleFoam")) == ["earned", "vendor"]
    assert [h["tier"] for h in search.search(rows, "simpleFoam", tier="earned")] == ["earned"]


def test_an_identical_tutorial_and_study_rank_the_same_across_tiers(search):
    """Notes, artifacts, rungs and the verdict exist only on earned rows, and scored
    across both tiers they ranked a study above an identical tutorial."""
    study = earned_row(verdict="completed", notes=["solve: converged with kEpsilon"],
                       artifacts={"mesh-full": 1, "solve": 1})
    hits = search.search([row(path="/tutorials/a/aCase"), study],
                         "simpleFoam converged completed solve")
    assert [h["score"] for h in hits] == [3, 3]


def test_a_failure_query_searches_the_earned_tier_and_its_own_fields(search):
    """Tutorials do not fail, they ship. The sentence that identifies a failure is
    the one written into a phase note."""
    study = earned_row(notes=["solve: diverged at t=0.31, trailing-edge collapse"])
    hits = search.failures([row(), study], "trailing-edge failed")
    assert [h["path"] for h in hits] == ["/work/a-study"]
    assert hits[0]["why"] == ["text~trailing-edge", "verdict=failed"]


def test_mesh_only_cases_are_held_back_from_a_solver_query_and_counted(search):
    """22 cases do not solve; dropping them silently would decide for the reader."""
    mesher = row(path="/m", solver={"executable": "snappyHexMesh", "module": None},
                 runs=False)
    hits, held = search.search_with_held([mesher, row(path="/s")], "snappyHexMesh simpleFoam",
                                         runnable_only=True)
    assert [h["path"] for h in hits] == ["/s"] and held == 1


# -- keyword -------------------------------------------------------------------


def case_setting(root: Path, value: str) -> Path:
    dictionary(root / "system" / "fvSchemes",
               "divSchemes\n{\n    div(phi,U)  " + value + ";\n}\n")
    return root


def test_a_distribution_counts_what_the_cases_set_now(search, tmp_path):
    """Read out of the cases, so it is current with what is on disk."""
    rows = [{"path": str(case_setting(tmp_path / f"c{i}", "Gauss upwind")), "tier": "vendor"}
            for i in range(3)]
    case_setting(tmp_path / "c0", "Gauss linear")
    found = search.distribution(rows, "div(phi,U)")
    assert found["total"] == 3
    assert {v["value"]: v["count"] for v in found["values"]} == \
        {"Gauss upwind": 2, "Gauss linear": 1}
    assert search.distribution(rows, "div(phi,neverSet)") == \
        {"key": "div(phi,neverSet)", "total": 0, "values": []}


def test_a_distribution_over_both_tiers_is_reported_per_tier(search, tmp_path, capsys):
    """One table over both counted this instance's own study as tutorial consensus."""
    vendor = case_setting(tmp_path / "tut" / "aCase", "Gauss linearUpwind grad(U)")
    case_setting(tmp_path / "work" / "a-study" / "case", "Gauss upwind")
    rows = [{"path": str(vendor), "tier": "vendor"},
            {"path": str(tmp_path / "work" / "a-study"), "tier": "earned", "case": "case"}]
    found = search.distributions(rows, "div(phi,U)")
    assert [(e["tier"], e["values"][0]["value"]) for e in found] == [
        ("vendor", "Gauss linearUpwind grad(U)"), ("earned", "Gauss upwind")]
    search.show_distributions(found)
    out = capsys.readouterr().out
    assert "[vendor tier]" in out and "[earned tier]" in out


# -- provenance ----------------------------------------------------------------


@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "corpus"


def records(out_dir) -> list[dict]:
    text = (Path(out_dir) / "retrievals.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def offered(path: str, tier: str = "vendor") -> list[dict]:
    return [{"path": path, "tier": tier, "score": 3, "why": []}]


def test_a_query_records_its_whole_ranked_list_and_nothing_taken(search, log_dir):
    """A vendor hit and an earned hit side by side is the signature the log exists for,
    and a log of winners cannot show it. One hit is still not a hit adopted."""
    hits = search.search([row(path="/tutorial"), earned_row(verdict="completed")],
                         "simpleFoam")
    search.record_retrieval(log_dir, kind="regime", text="simpleFoam", hits=hits, matched=2)
    [entry] = records(log_dir)
    assert entry["query"] == {"kind": "regime", "text": "simpleFoam", "tier": None,
                              "unmatched": []}
    assert {h["tier"] for h in entry["hits"]} == {"vendor", "earned"}
    assert entry["taken"] is None and entry["t"] and entry["id"]


def test_taken_attaches_to_the_retrieval_that_actually_offered_it(search, log_dir):
    """Three queries, and the case adopted came from the first."""
    search.record_retrieval(log_dir, kind="regime", text="seeds",
                            hits=offered(str(Path("/tutorials/simpleFoam/motorBike"))),
                            matched=1)
    search.record_retrieval(log_dir, kind="keyword", text="div(phi,U)", hits=[], matched=9)
    search.record_retrieval(log_dir, kind="failure", text="diverged",
                            hits=offered("/study", "earned"), matched=1)
    # Written the way a shell writes it, not the way the hit was recorded.
    search.record_taken(log_dir, path="/tutorials/simpleFoam/motorBike")
    assert [r["taken"] for r in search.read_retrievals(log_dir)] == \
        ["/tutorials/simpleFoam/motorBike", None, None]


def test_taking_something_no_query_offered_is_recorded_unattached(search, log_dir):
    search.record_retrieval(log_dir, kind="regime", text="seeds", hits=offered("/a"),
                            matched=1)
    search.record_taken(log_dir, path="/somewhere-else")
    assert search.read_retrievals(log_dir)[0]["taken"] is None
    assert records(log_dir)[-1] == {"t": records(log_dir)[-1]["t"], "kind": "taken",
                                    "of": None, "path": "/somewhere-else"}


def test_a_log_that_cannot_be_written_does_not_break_the_query(search, tmp_path):
    blocked = tmp_path / "corpus"
    blocked.write_text("not a directory", encoding="utf-8")
    assert search.record_retrieval(blocked, kind="regime", text="x", hits=[], matched=0) is None
    assert search.record_taken(blocked, path="/b") is False


def test_a_stale_lock_does_not_silence_the_log_forever(search, log_dir, monkeypatch):
    """A process killed between taking the lock and removing it leaves the file there."""
    monkeypatch.setattr(search, "LOCK_ATTEMPTS", 2)
    monkeypatch.setattr(search, "LOCK_WAIT_S", 0)
    log_dir.mkdir(parents=True)
    (log_dir / "retrievals.jsonl.lock").write_text("held by nobody", encoding="utf-8")
    assert search.record_retrieval(log_dir, kind="regime", text="one", hits=[], matched=0)
    assert len(search.read_retrievals(log_dir)) == 1
