"""`search.py`: one retrieval interface over both tiers.

Three things are being held here, and only the first is about finding anything.

**Every hit says which tier it came from.** That is R2 at query time, and it is the
whole enforcement point: a tutorial is a demonstration and a past study is this
system's own output, and a result set that does not say which is which is how the two
quietly become one thing.

**Every hit says why it matched.** A corpus with no benchmark tier under it cannot
demonstrate that its ranking is right, so the ranking has to be inspectable instead --
a score nobody can explain is exactly where a closed loop hides.

**A distribution reports its own concentration.** 89 of 557 vendor rows are one solver
from one tutorial family. An unweighted "what do cases use for this" reports that
family's house style as the consensus of the corpus, with nothing on screen to say so.
"""

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


BANNER = "// a dictionary\n"


def dictionary(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BANNER + body, encoding="utf-8")
    return path


def row(**overrides) -> dict:
    """An index row, in the shape `corpus.py` writes.

    Kept honest by `test_the_row_factory_matches_what_corpus_writes`, which builds a
    real one and compares the fields. A factory that drifts from the writer would let
    every ranking test here pass against a row shape that does not exist.
    """
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
    base = row(
        path="/work/a-study",
        tier="earned",
        verdict="failed",
        study_id="a-study",
        case="case",
        cases=["case"],
        artifacts={"mesh-full": 1},
        rungs=[],
        notes=[],
    )
    base.update(overrides)
    return base


# -- the factory is held to the writer -----------------------------------------


def test_the_row_factory_matches_what_corpus_writes(corpus, search, tmp_path):
    """The ranking tests below are readable because they build rows by hand. That is
    only safe if the hand-built row has the shape the harvester writes."""
    case = tmp_path / "tutorials" / "family" / "aCase"
    dictionary(case / "system" / "controlDict", "application  simpleFoam;\n")
    real, _ = corpus.harvest_tutorials(tmp_path / "tutorials")
    assert set(real[0]) == set(row())


# -- staleness -----------------------------------------------------------------


def test_no_stamp_at_all_is_a_reason_to_build(search):
    assert search.staleness(None, "v2512")


def test_a_matching_stamp_is_not_stale(search, corpus):
    stamp = {"of_version": "v2512", "schema_version": corpus.SCHEMA_VERSION}
    assert search.staleness(stamp, "v2512") is None


def test_an_index_built_against_another_openfoam_is_stale(search, corpus):
    """The drift the design document names. A stale index is worse than no index
    because it looks authoritative: it answers, and the answer is about a version of
    OpenFOAM that is not the one running."""
    stamp = {"of_version": "v2406", "schema_version": corpus.SCHEMA_VERSION}
    reason = search.staleness(stamp, "v2512")
    assert reason and "v2406" in reason and "v2512" in reason


def test_an_index_built_by_an_older_schema_is_stale(search, corpus):
    stamp = {"of_version": "v2512", "schema_version": corpus.SCHEMA_VERSION - 1}
    reason = search.staleness(stamp, "v2512")
    assert reason and "schema" in reason.lower()


def test_an_index_built_without_knowing_the_version_is_stale_once_we_know(search, corpus):
    stamp = {"of_version": "unknown", "schema_version": corpus.SCHEMA_VERSION}
    assert search.staleness(stamp, "v2512")


def test_not_knowing_the_live_version_does_not_force_a_rebuild(search, corpus):
    """`$WM_PROJECT_VERSION` unset is the harness not knowing, not the index being
    wrong. Rebuilding on it would rebuild on every query, forever, and the rebuild
    would produce an index stamped `unknown` that is stale by the same rule."""
    stamp = {"of_version": "v2512", "schema_version": corpus.SCHEMA_VERSION}
    assert search.staleness(stamp, "unknown") is None


# -- ensure --------------------------------------------------------------------


@pytest.fixture
def built(corpus, search, tmp_path, monkeypatch):
    """A real corpus on disk: one tutorial case, one study."""
    monkeypatch.setenv("WM_PROJECT_VERSION", "v2512")
    tutorials = tmp_path / "tutorials" / "incompressible" / "simpleFoam" / "pitzDaily"
    dictionary(tutorials / "system" / "controlDict", "application  simpleFoam;\n")
    dictionary(tutorials / "system" / "fvSchemes",
               "divSchemes\n{\n    div(phi,U)  bounded Gauss linearUpwind grad(U);\n}\n")
    dictionary(tutorials / "constant" / "turbulenceProperties",
               "simulationType RAS;\nRAS\n{\n    RASModel  kEpsilon;\n}\n")
    dictionary(tutorials / "constant" / "transportProperties", "nu  1e-05;\n")
    return tmp_path


def test_a_missing_index_is_built_on_first_use(search, built):
    rows, reason = search.ensure(built / "corpus", tutorials=built / "tutorials",
                                 work=built / "work")
    assert reason
    assert (built / "corpus" / "tutorials.index.jsonl").exists()
    assert [r["solver"]["executable"] for r in rows] == ["simpleFoam"]


def test_a_second_query_does_not_rebuild(search, built):
    out = built / "corpus"
    search.ensure(out, tutorials=built / "tutorials", work=built / "work")
    stamped = json.loads((out / "corpus.stamp.json").read_text())["built_at"]
    rows, reason = search.ensure(out, tutorials=built / "tutorials", work=built / "work")
    assert reason is None
    assert json.loads((out / "corpus.stamp.json").read_text())["built_at"] == stamped
    assert rows


def test_rebuild_is_forced_when_asked_for(search, built):
    out = built / "corpus"
    search.ensure(out, tutorials=built / "tutorials", work=built / "work")
    _, reason = search.ensure(out, tutorials=built / "tutorials", work=built / "work",
                              rebuild=True)
    assert reason == "asked for"


def test_both_tiers_are_loaded_and_stay_labelled(search, corpus, built, tmp_path):
    study = built / "work" / "a-study"
    (study / ".reynolds").mkdir(parents=True)
    dictionary(study / "case" / "system" / "controlDict", "application  pimpleFoam;\n")
    rows, _ = search.ensure(built / "corpus", tutorials=built / "tutorials",
                            work=built / "work")
    assert sorted(r["tier"] for r in rows) == ["earned", "vendor"]


# -- ranking, and why ----------------------------------------------------------


def test_a_field_match_outranks_a_free_text_match(search):
    """The rule the design document states: field matches score above free-text ones.
    A case whose solver *is* simpleFoam is a better answer to "simpleFoam" than a case
    that merely lives in a directory of that name."""
    solver = row(path="/tutorials/x/y/aCase", solver={"executable": "simpleFoam",
                                                      "module": None})
    named = row(path="/tutorials/incompressible/simpleFoam/other",
                solver={"executable": "potentialFoam", "module": None})
    hits = search.search([named, solver], "simpleFoam")
    assert [hit["path"] for hit in hits] == [solver["path"], named["path"]]
    assert hits[0]["score"] > hits[1]["score"]


def test_every_hit_says_why_it_matched(search):
    hits = search.search([row()], "RAS kEpsilon")
    assert hits[0]["why"]
    assert "turbulence=RAS" in hits[0]["why"]
    assert "model=kEpsilon" in hits[0]["why"]


def test_a_free_text_match_says_it_was_free_text(search):
    """A hit that matched on nothing but a directory name has to look different from
    one that matched a field, or the score is the only thing distinguishing them and
    the score is not on the screen."""
    hits = search.search([row(path="/tutorials/x/pitzDaily")], "pitzDaily")
    assert hits[0]["why"] == ["text~pitzDaily"]


def test_more_tokens_matched_scores_higher(search):
    both = row(path="/a", turbulence={"simulation_type": "RAS", "model": "kEpsilon"})
    one = row(path="/b", turbulence={"simulation_type": "RAS", "model": "kOmegaSST"})
    hits = search.search([one, both], "RAS kEpsilon")
    assert [hit["path"] for hit in hits] == ["/a", "/b"]


def test_a_row_matching_nothing_is_not_a_hit(search):
    assert search.search([row()], "supersonic combustion") == []


def test_the_regime_class_matches_on_its_parts(search):
    """`incompressible-steady` should answer "steady", because that is how the query
    in the design document is written: "internal incompressible steady"."""
    hits = search.search([row()], "steady")
    assert hits and "regime=steady" in hits[0]["why"]


def test_the_query_is_case_insensitive(search):
    assert search.search([row()], "ras KEPSILON")


# -- R2 at query time ----------------------------------------------------------


def test_every_hit_carries_its_tier(search):
    hits = search.search([row(), earned_row(verdict="completed")], "simpleFoam")
    assert len(hits) == 2
    assert {hit["tier"] for hit in hits} == {"vendor", "earned"}
    assert all("tier" in hit for hit in hits)


def test_a_search_can_be_held_to_one_tier(search):
    hits = search.search([row(), earned_row()], "simpleFoam", tier="earned")
    assert [hit["tier"] for hit in hits] == ["earned"]


def test_the_two_tiers_are_never_silently_merged(search):
    """A mixed result set is fine and is the point. What must not happen is a hit
    arriving without the label that says which kind of evidence it is."""
    hits = search.search([row(), earned_row(verdict="completed")], "simpleFoam")
    for hit in hits:
        assert hit["tier"] in ("vendor", "earned")


# -- mesh-only cases -----------------------------------------------------------


def test_a_mesh_only_case_does_not_answer_a_solver_query(search):
    """22 cases in the tree do not solve -- 18 name a mesher as their application and
    4 name nothing. A mesh-only tutorial is a legitimate meshing precedent and stays
    in the index, but it cannot be a seed for a solve."""
    mesher = row(path="/m", solver={"executable": "snappyHexMesh", "module": None},
                 runs=False, mesh_type="snappyHexMesh")
    hits = search.search([mesher], "snappyHexMesh", runnable_only=True)
    assert hits == []


def test_a_mesh_only_case_is_still_findable_when_asked_for(search):
    mesher = row(path="/m", solver={"executable": "snappyHexMesh", "module": None},
                 runs=False, mesh_type="snappyHexMesh")
    assert search.search([mesher], "snappyHexMesh", runnable_only=False)


def test_the_count_of_what_was_held_back_is_reported(search):
    """Silently dropping 22 rows would be the search deciding something on the
    reader's behalf without saying so."""
    mesher = row(path="/m", solver={"executable": "snappyHexMesh", "module": None},
                 runs=False)
    solver = row(path="/s")
    _, held = search.search_with_held([mesher, solver], "snappyHexMesh simpleFoam",
                                      runnable_only=True)
    assert held == 1


# -- keyword: the value distribution -------------------------------------------


@pytest.fixture
def scheme_corpus(tmp_path):
    """Six cases, five of them from one tutorial family, disagreeing about a scheme.

    The shape that matters: `family-a` is 5 of the 6 cases and votes one way. That is
    the corpus in miniature -- 89 of 557 real rows are one solver from one family.
    """
    rows = []
    for index in range(5):
        case = tmp_path / "incompressible" / "family-a" / f"case{index}"
        dictionary(case / "system" / "fvSchemes",
                   "divSchemes\n{\n    div(phi,U)  Gauss upwind;\n}\n")
        rows.append({"path": str(case), "tier": "vendor", "runs": True})
    case = tmp_path / "incompressible" / "family-b" / "case0"
    dictionary(case / "system" / "fvSchemes",
               "divSchemes\n{\n    div(phi,U)  bounded Gauss linearUpwind grad(U);\n}\n")
    rows.append({"path": str(case), "tier": "vendor", "runs": True})
    return rows


def test_a_keyword_query_reports_the_values_and_their_counts(search, scheme_corpus):
    found = search.distribution(scheme_corpus, "div(phi,U)")
    assert found["total"] == 6
    counts = {entry["value"]: entry["count"] for entry in found["values"]}
    assert counts == {"Gauss upwind": 5, "bounded Gauss linearUpwind grad(U)": 1}


def test_the_distribution_reports_how_concentrated_it_is(search, scheme_corpus, tmp_path):
    """The finding this exists for: an unweighted count says 83% of the corpus uses
    `Gauss upwind`. It does not say that all 5 of those cases are one tutorial family,
    which is the difference between a consensus and a house style. Both numbers are
    reported, so the reader can tell which they are looking at."""
    found = search.distribution(scheme_corpus, "div(phi,U)", root=tmp_path)
    largest = found["families"]["largest"]
    assert found["families"]["total"] == 2
    assert largest["name"].endswith("family-a")
    assert largest["count"] == 5
    assert round(largest["share"], 2) == 0.83


def test_each_value_says_how_many_families_hold_it(search, scheme_corpus, tmp_path):
    """A value held by one family and a value held by twenty are not the same claim,
    however similar the counts."""
    found = search.distribution(scheme_corpus, "div(phi,U)", root=tmp_path)
    by_value = {entry["value"]: entry for entry in found["values"]}
    assert by_value["Gauss upwind"]["families"] == 1
    assert by_value["bounded Gauss linearUpwind grad(U)"]["families"] == 1


def test_a_key_nothing_sets_is_an_empty_distribution_not_an_error(search, scheme_corpus):
    found = search.distribution(scheme_corpus, "div(phi,neverSet)")
    assert found["total"] == 0
    assert found["values"] == []
    assert found["families"]["largest"] is None


def test_the_distribution_reads_the_case_rather_than_the_index(search, scheme_corpus, tmp_path):
    """Schemes are not indexed -- there are hundreds of keys per case and they would
    swamp the row. The index says which cases exist and the query reads them, so a
    distribution is always current with what is on disk rather than with what was
    true at build time."""
    case = Path(scheme_corpus[0]["path"])
    dictionary(case / "system" / "fvSchemes",
               "divSchemes\n{\n    div(phi,U)  Gauss linear;\n}\n")
    found = search.distribution(scheme_corpus, "div(phi,U)")
    counts = {entry["value"]: entry["count"] for entry in found["values"]}
    assert counts["Gauss linear"] == 1
    assert counts["Gauss upwind"] == 4


def test_a_distribution_can_be_held_to_one_tier(search, scheme_corpus):
    """R2 again: "what does the corpus use for this scheme" and "what have my own
    studies used" are different questions, and mixing them is how a house style
    becomes a fact."""
    mine = dict(scheme_corpus[0], tier="earned")
    found = search.distribution(scheme_corpus + [mine], "div(phi,U)", tier="vendor")
    assert found["total"] == 6


# -- failure -------------------------------------------------------------------


def test_a_failure_query_searches_the_earned_tier_only(search):
    """"Has this gone wrong before" is a question about this system's own history.
    A tutorial cannot answer it: tutorials do not fail, they ship."""
    hits = search.failures([row(path="/tutorial"), earned_row(path="/study")], "failed")
    assert [hit["path"] for hit in hits] == ["/study"]


def test_a_failure_query_matches_a_phase_note(search):
    """The sentence that identifies a failure is hand-written into a phase note by
    whoever hit it. That is the text a later session is trying to find again."""
    study = earned_row(notes=["solve: diverged at t=0.31, trailing-edge collapse"])
    hits = search.failures([study], "trailing-edge collapse")
    assert hits and hits[0]["path"] == study["path"]


def test_a_failure_query_matches_rung_evidence(search):
    study = earned_row(
        verdict="completed",
        rungs=[{"class": "external-2d", "rung": 2, "name": "flat-plate drag",
                "status": "fail", "value": 0.41, "known": "ITTC-57"}],
    )
    hits = search.failures([study], "flat-plate drag")
    assert hits


def test_a_failure_query_matches_the_verdict(search):
    assert search.failures([earned_row(verdict="failed")], "failed")


def test_every_failure_hit_still_carries_its_tier(search):
    hits = search.failures([earned_row()], "failed")
    assert hits[0]["tier"] == "earned"


# -- the shape a hit is recorded in --------------------------------------------


def test_a_hit_has_the_four_fields_provenance_will_record(search):
    """The design document's provenance record stores `path`, `tier`, `score` and
    `why` for every hit including the ones not taken. A hit is written in that shape
    from the start so the log is a copy rather than a translation."""
    hit = search.search([row()], "simpleFoam")[0]
    assert set(hit) == {"path", "tier", "score", "why"}


def test_a_regime_class_of_one_word_does_not_score_twice(search):
    """Found by running the design document's own query against the real corpus.

    `fields_of` offers the regime class whole *and* split on `-`, so a class that is
    a single word -- `steady`, which 21 real rows carry because their properties file
    is missing and compressibility is null -- produced the pair `regime=steady` twice
    and scored six for one concept. A case matching only `steady` as a field, with
    `incompressible` appearing merely in its directory path, then outranked a case
    that matched *both* as fields. Which is precisely backwards.
    """
    one_word = row(path="/a", regime={"class": "steady", "compressible": None,
                                      "steady": True, "Re": None, "Ma": None,
                                      "shedding_risk": None})
    both = row(path="/b", regime={"class": "incompressible-steady",
                                  "compressible": False, "steady": True,
                                  "Re": None, "Ma": None, "shedding_risk": None})
    hits = search.search([one_word, both], "incompressible steady")
    assert [hit["path"] for hit in hits] == ["/b", "/a"]


def test_a_path_coincidence_never_outranks_two_field_matches(search):
    """The same bug from the other side: every one of the 200-odd cases filed under
    `incompressible/` contains that word in its path, and a free-text hit on it must
    not add up to a field."""
    filed = row(path="/tutorials/incompressible/x/aCase",
                regime={"class": "steady", "compressible": None, "steady": True,
                        "Re": None, "Ma": None, "shedding_risk": None})
    actual = row(path="/tutorials/x/y/bCase")
    hits = search.search([filed, actual], "incompressible steady")
    assert [hit["path"] for hit in hits] == ["/tutorials/x/y/bCase",
                                             "/tutorials/incompressible/x/aCase"]


def test_a_token_that_matches_nothing_anywhere_is_reported(search):
    """"internal incompressible steady" is the query the design document writes, and
    `internal` matches nothing in this corpus -- internal versus external is not
    derivable from a tutorial, so it was never indexed. A query silently scoring
    two-thirds of what the reader asked for should say so."""
    assert search.unmatched_tokens([row()], "internal incompressible steady") == ["internal"]
    assert search.unmatched_tokens([row()], "RAS kEpsilon") == []


def test_the_row_shape_is_pinned_to_the_schema_version(corpus):
    """A field added to a row without bumping `SCHEMA_VERSION` leaves every index on
    disk stale *and undetectable* -- `staleness` compares schema versions, so an
    unchanged number means an old index is loaded and quietly answers with fields it
    does not have.

    That is not hypothetical. `notes` was added to the earned row for the `failure`
    query, the version was not bumped, and a real `failure "diverged"` query against a
    real index returned nothing at all: the rows on disk had no `notes` and nothing
    said so. This test makes the bump conscious rather than remembered.

    It has since caught its second one: version 3 added `tutorials` and `work` to the
    *stamp* so a switched tree is detectable, which is not a row change at all. The pin
    covers the stamp's shape too, because "the shape on disk" is what the version is
    actually about.
    """
    assert corpus.SCHEMA_VERSION == 3
    assert set(row()) == {
        "path", "tier", "solver", "runs", "turbulence", "regions", "regime",
        "mesh_type", "bc_map", "of_version", "of_fork", "verdict", "provenance",
    }
    assert set(earned_row()) - set(row()) == {
        "study_id", "case", "cases", "artifacts", "rungs", "notes",
    }


def test_the_stamp_shape_is_pinned_to_the_schema_version(corpus, tmp_path, monkeypatch):
    """The other half of the same pin. A stamp field that `staleness` reads is exactly
    as load-bearing as a row field, and adding one without moving the version leaves
    every index on disk unable to be checked against it."""
    monkeypatch.setenv("WM_PROJECT_VERSION", "v2512")
    tree = tmp_path / "tutorials"
    dictionary(tree / "family" / "aCase" / "system" / "controlDict",
               "application  simpleFoam;\n")

    stamp = corpus.build(tree, tmp_path / "corpus", work=tmp_path / "work")

    assert set(stamp) == {
        "of_version", "of_fork", "built_at", "counts", "schema_version",
        "tutorials", "work",
    }


# -- which family a case belongs to --------------------------------------------


def test_family_grouping_survives_a_deeply_nested_tutorial_family(search):
    """Found by running `keyword "div(-phi,Ua)"` -- a key only the adjoint tutorials
    set -- against the real tree. Every one of those 69 cases is under
    `incompressible/adjointOptimisationFoam`, so the answer should be "one family".

    Taking the two directory levels immediately above the case reported **47**
    families, with the largest named `1_Inlet_2_Outlet/levelSet`, because that family
    nests its cases several levels deeper than the others. The proxy failed hardest on
    the single family the concentration warning exists for. Grouping is relative to the
    corpus root instead, so a family is a family however deep its cases sit.
    """
    root = "/tutorials"
    cases = [
        f"{root}/incompressible/adjointOptimisationFoam/1_Inlet_2_Outlet/levelSet/a",
        f"{root}/incompressible/adjointOptimisationFoam/1_Inlet_2_Outlet/shapeOpt/b",
        f"{root}/incompressible/adjointOptimisationFoam/sensitivityMaps/deep/nested/c",
        f"{root}/incompressible/simpleFoam/pitzDaily",
    ]
    names = search.family_names([Path(case) for case in cases], root)
    assert names[Path(cases[0])] == "incompressible/adjointOptimisationFoam"
    assert names[Path(cases[2])] == "incompressible/adjointOptimisationFoam"
    assert names[Path(cases[3])] == "incompressible/simpleFoam"
    assert len(set(names.values())) == 2


def test_a_distribution_over_one_family_says_one_family(search, tmp_path):
    """The end the whole concentration report exists for: a key that only one family
    sets must not look like a corpus-wide agreement."""
    rows = []
    for index, depth in enumerate(("a/b", "c", "d/e/f")):
        case = tmp_path / "tutorials" / "incompressible" / "adjointOptimisationFoam" / depth
        dictionary(case / "system" / "fvSchemes",
                   "divSchemes\n{\n    div(-phi,Ua)  bounded Gauss upwind;\n}\n")
        rows.append({"path": str(case), "tier": "vendor", "runs": True})
    other = tmp_path / "tutorials" / "incompressible" / "simpleFoam" / "pitzDaily"
    dictionary(other / "system" / "fvSchemes", "divSchemes\n{\n    default none;\n}\n")
    rows.append({"path": str(other), "tier": "vendor", "runs": True})

    found = search.distribution(rows, "div(-phi,Ua)", root=tmp_path / "tutorials")

    assert found["total"] == 3
    assert found["values"][0]["families"] == 1
    assert found["families"]["largest"]["name"] == "incompressible/adjointOptimisationFoam"
    assert found["families"]["largest"]["count"] == 3


# =============================================================================
# Provenance
# =============================================================================
#
# The log exists for one specific failure it is meant to make visible. When a hit
# from one tier and a hit from another disagree, and the one from the earned tier is
# the one adopted, that is the closed loop tightening by a notch -- the system
# preferring its own previous output to the vendor's. It is invisible if only the
# winner is recorded, which is why the ranked list is written down whole, including
# every hit that was passed over.
#
# The other half is that nothing here guesses. `taken` is null until somebody says
# otherwise. A query that returned exactly one hit did not thereby adopt it, and the
# harness inferring that it did would be manufacturing the evidence this log exists
# to collect.


@pytest.fixture
def log_dir(tmp_path):
    return tmp_path / "corpus"


def records(search, out_dir) -> list[dict]:
    text = (Path(out_dir) / "retrievals.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# -- one record per query ------------------------------------------------------


def test_a_query_appends_one_record(search, log_dir):
    hits = search.search([row()], "simpleFoam")
    search.record_retrieval(log_dir, kind="regime", text="simpleFoam", hits=hits, matched=1)
    written = records(search, log_dir)
    assert len(written) == 1
    assert written[0]["query"] == {"kind": "regime", "text": "simpleFoam",
                                   "tier": None, "unmatched": []}
    assert written[0]["t"]
    assert written[0]["id"]


def test_a_second_query_appends_rather_than_replaces(search, log_dir):
    """Append-only. The log is the history of what was asked, and a history that
    rewrites its own last line is not one."""
    search.record_retrieval(log_dir, kind="regime", text="one", hits=[], matched=0)
    search.record_retrieval(log_dir, kind="regime", text="two", hits=[], matched=0)
    written = records(search, log_dir)
    assert [entry["query"]["text"] for entry in written] == ["one", "two"]
    assert written[0]["id"] != written[1]["id"]


def test_the_record_carries_the_fields_the_design_document_names(search, log_dir):
    hits = search.search([row()], "simpleFoam")
    search.record_retrieval(log_dir, kind="regime", text="simpleFoam", hits=hits, matched=1)
    entry = records(search, log_dir)[0]
    assert set(entry) == {"t", "id", "query", "matched", "hits", "taken"}
    assert set(entry["hits"][0]) == {"path", "tier", "score", "why"}


# -- the hits that were not taken ----------------------------------------------


def test_every_hit_is_recorded_and_not_just_the_first(search, log_dir):
    """The whole ranked list as it was shown. A log of winners cannot answer the
    question the log is for."""
    rows = [row(path="/a"), row(path="/b"), earned_row(path="/study", verdict="completed")]
    hits = search.search(rows, "simpleFoam")
    search.record_retrieval(log_dir, kind="regime", text="simpleFoam", hits=hits,
                            matched=len(hits))
    entry = records(search, log_dir)[0]
    assert sorted(h["path"] for h in entry["hits"]) == ["/a", "/b", "/study"]


def test_a_record_shows_a_vendor_and_an_earned_hit_side_by_side(search, log_dir):
    """The signature this log exists to catch: two tiers answering the same query.
    Whether the earned one was preferred is a question the log can only answer if
    both are in it."""
    rows = [row(path="/tutorial"), earned_row(path="/study", verdict="completed")]
    hits = search.search(rows, "simpleFoam")
    search.record_retrieval(log_dir, kind="regime", text="simpleFoam", hits=hits, matched=2)
    entry = records(search, log_dir)[0]
    assert {h["tier"] for h in entry["hits"]} == {"vendor", "earned"}


def test_the_total_matched_is_recorded_beside_what_was_shown(search, log_dir):
    """358 matched and 10 shown is a different retrieval from 10 matched and 10 shown,
    and the record has to be able to tell them apart later."""
    hits = search.search([row()], "simpleFoam")
    search.record_retrieval(log_dir, kind="regime", text="simpleFoam", hits=hits,
                            matched=358)
    entry = records(search, log_dir)[0]
    assert entry["matched"] == 358
    assert len(entry["hits"]) == 1


def test_the_words_that_matched_nothing_are_recorded(search, log_dir):
    search.record_retrieval(log_dir, kind="regime", text="internal incompressible",
                            hits=[], matched=0, unmatched=["internal"])
    assert records(search, log_dir)[0]["query"]["unmatched"] == ["internal"]


# -- taken is never inferred ---------------------------------------------------


def test_taken_is_null_until_somebody_says_otherwise(search, log_dir):
    hits = search.search([row()], "simpleFoam")
    search.record_retrieval(log_dir, kind="regime", text="simpleFoam", hits=hits, matched=1)
    assert records(search, log_dir)[0]["taken"] is None


def test_a_single_hit_is_still_not_assumed_taken(search, log_dir):
    """The tempting inference, and the one that would quietly corrupt the record. A
    query that returned one case did not adopt it; the agent may well have read it and
    decided against."""
    hits = search.search([row(path="/only")], "simpleFoam")
    assert len(hits) == 1
    search.record_retrieval(log_dir, kind="regime", text="simpleFoam", hits=hits, matched=1)
    assert records(search, log_dir)[0]["taken"] is None


def test_saying_what_was_taken_attaches_it_to_the_last_retrieval(search, log_dir):
    hits = search.search([row(path="/a"), row(path="/b")], "simpleFoam")
    search.record_retrieval(log_dir, kind="regime", text="simpleFoam", hits=hits, matched=2)
    search.record_taken(log_dir, path="/b")

    folded = search.read_retrievals(log_dir)
    assert len(folded) == 1
    assert folded[0]["taken"] == "/b"
    assert sorted(h["path"] for h in folded[0]["hits"]) == ["/a", "/b"]


def test_saying_what_was_taken_is_itself_appended(search, log_dir):
    """Not an edit of the earlier line. The retrieval happened, then the adoption
    happened, and both are events with their own times."""
    search.record_retrieval(log_dir, kind="regime", text="simpleFoam", hits=[], matched=0)
    search.record_taken(log_dir, path="/b")
    written = records(search, log_dir)
    assert len(written) == 2
    assert written[0]["taken"] is None
    assert written[1]["kind"] == "taken"
    assert written[1]["path"] == "/b"


def test_taken_attaches_to_its_own_retrieval_and_not_a_later_one(search, log_dir):
    """A query, an adoption from it, then another query. The adoption belongs to the
    first, and a later retrieval must not acquire it."""
    offered = [{"path": "/from-one", "tier": "vendor", "score": 3, "why": []}]
    search.record_retrieval(log_dir, kind="regime", text="one", hits=offered, matched=1)
    search.record_taken(log_dir, path="/from-one")
    search.record_retrieval(log_dir, kind="regime", text="two", hits=offered, matched=1)

    folded = search.read_retrievals(log_dir)
    assert [entry["query"]["text"] for entry in folded] == ["one", "two"]
    assert folded[0]["taken"] == "/from-one"
    assert folded[1]["taken"] is None


def test_saying_what_was_taken_with_nothing_to_attach_it_to_is_still_recorded(
    search, log_dir
):
    """A `took` with no preceding query is odd but it is not nothing: something was
    adopted. Recording it unattached keeps the fact and does not invent a query to
    hang it on."""
    search.record_taken(log_dir, path="/b")
    written = records(search, log_dir)
    assert written[0]["kind"] == "taken"
    assert written[0]["of"] is None
    assert search.read_retrievals(log_dir) == []


# -- a keyword query is a retrieval too ----------------------------------------


def test_a_distribution_records_the_values_it_reported(search, log_dir, scheme_corpus,
                                                       tmp_path):
    """A keyword query returns no cases to adopt, so `hits` is empty and the values
    are what was retrieved. Logging it matters for the same reason: "the corpus says
    most cases use X" is a claim that shaped a decision, and which values were on
    screen when it was made is the record of that."""
    found = search.distribution(scheme_corpus, "div(phi,U)", root=tmp_path)
    search.record_retrieval(log_dir, kind="keyword", text="div(phi,U)", hits=[],
                            matched=found["total"], values=found["values"])
    entry = records(search, log_dir)[0]
    assert entry["matched"] == 6
    assert {value["value"] for value in entry["values"]} == {
        "Gauss upwind", "bounded Gauss linearUpwind grad(U)",
    }


# -- the log never costs a query -----------------------------------------------


def test_a_log_that_cannot_be_written_does_not_break_the_query(search, tmp_path):
    """Provenance is a record of the work, not a precondition for it. A read-only
    corpus directory, a full disk, a path that is a file where a directory should be
    -- none of them is a reason for a retrieval to fail."""
    blocked = tmp_path / "corpus"
    blocked.write_text("not a directory", encoding="utf-8")
    assert search.record_retrieval(blocked, kind="regime", text="x", hits=[], matched=0) is None
    assert search.record_taken(blocked, path="/b") is False


def test_a_log_line_that_will_not_parse_costs_its_line_and_no_more(search, log_dir):
    search.record_retrieval(log_dir, kind="regime", text="one", hits=[], matched=0)
    with (log_dir / "retrievals.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"t": "truncated mid-writ\n')
    search.record_retrieval(log_dir, kind="regime", text="two", hits=[], matched=0)
    folded = search.read_retrievals(log_dir)
    assert [entry["query"]["text"] for entry in folded] == ["one", "two"]


def test_reading_a_log_that_is_not_there_is_empty_and_not_an_error(search, tmp_path):
    assert search.read_retrievals(tmp_path / "nothing") == []


def test_taken_attaches_to_the_retrieval_that_actually_offered_it(search, log_dir):
    """Found by running three queries and then adopting something from the first.

    Attaching an adoption to whichever retrieval happened to be most recent is a
    guess, and it was wrong the first time it was tried for real: a `regime` query, a
    `keyword` query and a `failure` query were run, the case adopted came from the
    first, and the record hung it on the third. The rule is a fact instead -- the most
    recent retrieval whose hits actually contained that path.
    """
    search.record_retrieval(log_dir, kind="regime", text="seeds",
                            hits=[{"path": "/a", "tier": "vendor", "score": 6, "why": []}],
                            matched=1)
    search.record_retrieval(log_dir, kind="keyword", text="div(phi,U)", hits=[], matched=9)
    search.record_retrieval(log_dir, kind="failure", text="diverged",
                            hits=[{"path": "/study", "tier": "earned", "score": 1,
                                   "why": []}],
                            matched=1)

    search.record_taken(log_dir, path="/a")

    folded = search.read_retrievals(log_dir)
    assert folded[0]["taken"] == "/a"
    assert folded[1]["taken"] is None
    assert folded[2]["taken"] is None


def test_the_most_recent_offer_of_a_path_is_the_one_it_attaches_to(search, log_dir):
    """Asked for twice, adopted once: the adoption belongs to the query that was
    actually in front of the reader when they chose."""
    for _ in range(2):
        search.record_retrieval(log_dir, kind="regime", text="seeds",
                                hits=[{"path": "/a", "tier": "vendor", "score": 6,
                                       "why": []}],
                                matched=1)
    search.record_taken(log_dir, path="/a")
    folded = search.read_retrievals(log_dir)
    assert folded[0]["taken"] is None
    assert folded[1]["taken"] == "/a"


def test_taking_something_no_query_offered_is_recorded_unattached(search, log_dir):
    """Also worth keeping, and arguably the more interesting record: the agent used a
    case the corpus never handed it. Hanging that on an unrelated retrieval would make
    the log say the retrieval succeeded when it did not."""
    search.record_retrieval(log_dir, kind="regime", text="seeds",
                            hits=[{"path": "/a", "tier": "vendor", "score": 6, "why": []}],
                            matched=1)
    search.record_taken(log_dir, path="/somewhere-else")

    assert search.read_retrievals(log_dir)[0]["taken"] is None
    raw = records(search, log_dir)
    assert raw[-1]["kind"] == "taken"
    assert raw[-1]["of"] is None
    assert raw[-1]["path"] == "/somewhere-else"


def test_the_same_case_written_two_ways_still_attaches(search, log_dir):
    r"""Found by adopting a path copied out of a shell rather than out of the result
    table. The hits carry `full\incompressible\simpleFoam\motorBike` and the
    adoption arrived as `full/incompressible/simpleFoam/motorBike`; compared as raw
    strings they are different cases, so the log said the corpus never offered it.

    Compared as paths they are one case, which is what they are.
    """
    offered = [{"path": str(Path("/tutorials/incompressible/simpleFoam/motorBike")),
                "tier": "vendor", "score": 9, "why": []}]
    search.record_retrieval(log_dir, kind="regime", text="seeds", hits=offered, matched=1)

    search.record_taken(log_dir, path="/tutorials/incompressible/simpleFoam/motorBike")

    assert search.read_retrievals(log_dir)[0]["taken"]


def test_a_redundant_path_segment_does_not_hide_the_match(search, log_dir):
    offered = [{"path": str(Path("/tutorials/family/aCase")), "tier": "vendor",
                "score": 3, "why": []}]
    search.record_retrieval(log_dir, kind="regime", text="seeds", hits=offered, matched=1)
    search.record_taken(log_dir, path="/tutorials/./family/other/../aCase")
    assert search.read_retrievals(log_dir)[0]["taken"]


# =============================================================================
# From the code review
# =============================================================================


def test_an_unset_tutorials_variable_does_not_build_from_the_working_directory(
    search, tmp_path, monkeypatch, capsys
):
    """`corpus.py` at least attempted a guard here; `search.py` had none, and
    `ensure()` builds on the first query without asking. So a query run with
    `$FOAM_TUTORIALS` unset walked the working directory, wrote a stamped index of
    whatever it found, and answered from it."""
    monkeypatch.delenv("FOAM_TUTORIALS", raising=False)
    monkeypatch.chdir(tmp_path)
    dictionary(tmp_path / "a-stray-case" / "system" / "controlDict",
               "application  simpleFoam;\n")

    code = search.main(["--corpus", str(tmp_path / "corpus"), "regime", "simpleFoam"])

    assert code == 1
    assert "FOAM_TUTORIALS" in capsys.readouterr().err
    assert not (tmp_path / "corpus").exists()


def test_a_query_against_an_existing_index_needs_no_tutorial_tree(search, built, capsys):
    """The guard belongs on building, not on asking. An index already on disk and not
    stale is answerable with no tree in sight -- which is the normal case for every
    query after the first."""
    search.ensure(built / "corpus", tutorials=built / "tutorials", work=built / "work")

    code = search.main(["--corpus", str(built / "corpus"), "regime", "simpleFoam"])

    assert code == 0
    assert "simpleFoam" in capsys.readouterr().out


# -- the index files themselves, not just the stamp ----------------------------


def test_a_missing_index_file_is_a_reason_to_rebuild(search, built):
    """The module docstring says the index is rebuilt when it is missing, and it was
    not: `staleness` read only `corpus.stamp.json`, and `load_rows` swallows a missing
    file per tier. Deleting the `.jsonl` while leaving a valid stamp made every query
    answer "nothing matched", with exit 0 and no rebuild, indefinitely."""
    out = built / "corpus"
    search.ensure(out, tutorials=built / "tutorials", work=built / "work")
    (out / "tutorials.index.jsonl").unlink()

    rows, reason = search.ensure(out, tutorials=built / "tutorials", work=built / "work")

    assert reason and "index" in reason.lower()
    assert (out / "tutorials.index.jsonl").exists()
    assert rows


def test_a_missing_studies_index_is_also_a_reason_to_rebuild(search, built):
    out = built / "corpus"
    search.ensure(out, tutorials=built / "tutorials", work=built / "work")
    (out / "studies.index.jsonl").unlink()
    _, reason = search.ensure(out, tutorials=built / "tutorials", work=built / "work")
    assert reason


# -- a switched tree is a stale index ------------------------------------------


def test_an_index_built_from_another_tree_is_stale(search, corpus, built, tmp_path):
    """Verified end to end by the review: an index built from the wrong tree, with the
    live version unknown, kept serving its rows after `$FOAM_TUTORIALS` was corrected,
    because the stamp matched on everything it recorded. The tree was the one thing it
    did not record."""
    out = built / "corpus"
    search.ensure(out, tutorials=built / "tutorials", work=built / "work")
    other = tmp_path / "other-tutorials"
    dictionary(other / "family" / "bCase" / "system" / "controlDict",
               "application  pimpleFoam;\n")

    rows, reason = search.ensure(out, tutorials=other, work=built / "work")

    assert reason and "tree" in reason.lower()
    assert [r["solver"]["executable"] for r in rows] == ["pimpleFoam"]


def test_the_same_tree_written_differently_is_not_stale(search, built):
    """A trailing separator or a `.` segment is the same tree, and rebuilding on it
    would rebuild on every query."""
    out = built / "corpus"
    search.ensure(out, tutorials=built / "tutorials", work=built / "work")
    _, reason = search.ensure(out, tutorials=Path(str(built / "tutorials") + "/."),
                              work=built / "work")
    assert reason is None


def test_an_index_with_no_recorded_tree_does_not_rebuild_forever(search, corpus, built):
    """An older stamp has no tree in it. That is a thing not known, not a mismatch,
    and treating it as one would rebuild on every query."""
    out = built / "corpus"
    search.ensure(out, tutorials=built / "tutorials", work=built / "work")
    stamp = json.loads((out / corpus.STAMP).read_text(encoding="utf-8"))
    del stamp["tutorials"]
    (out / corpus.STAMP).write_text(json.dumps(stamp), encoding="utf-8")

    assert search.staleness(stamp, "v2512", tree=built / "tutorials") is None


# -- R2 at the one place it was not enforced -----------------------------------


def test_a_distribution_over_both_tiers_is_reported_per_tier(search, tmp_path):
    """The finding that matters most of the seven. `keyword` called `distribution`
    with the tier defaulting to None, so a tutorial and the instance's own finished
    study were counted into one table and the footer called them all "tutorial
    families". A 50/50 split with nothing saying half of it was this system's own
    prior output is precisely the merge R2 exists to forbid -- and the capability to
    hold a distribution to one tier already existed and was simply defaulted off.
    """
    vendor = tmp_path / "tut" / "family" / "aCase"
    dictionary(vendor / "system" / "fvSchemes",
               "divSchemes\n{\n    div(phi,U)  Gauss linearUpwind grad(U);\n}\n")
    earned = tmp_path / "work" / "a-study" / "case"
    dictionary(earned / "system" / "fvSchemes",
               "divSchemes\n{\n    div(phi,U)  Gauss upwind;\n}\n")
    rows = [
        {"path": str(vendor), "tier": "vendor", "runs": True},
        {"path": str(tmp_path / "work" / "a-study"), "tier": "earned", "case": "case",
         "runs": True},
    ]

    found = search.distributions(rows, "div(phi,U)", root=tmp_path / "tut")

    assert [entry["tier"] for entry in found] == ["vendor", "earned"]
    assert found[0]["total"] == 1
    assert found[1]["total"] == 1
    assert found[0]["values"][0]["value"] == "Gauss linearUpwind grad(U)"
    assert found[1]["values"][0]["value"] == "Gauss upwind"


def test_a_distribution_held_to_one_tier_returns_only_that_tier(search, tmp_path):
    vendor = tmp_path / "tut" / "family" / "aCase"
    dictionary(vendor / "system" / "fvSchemes",
               "divSchemes\n{\n    div(phi,U)  Gauss upwind;\n}\n")
    rows = [{"path": str(vendor), "tier": "vendor", "runs": True}]
    found = search.distributions(rows, "div(phi,U)", tier="vendor",
                                 root=tmp_path / "tut")
    assert [entry["tier"] for entry in found] == ["vendor"]


def test_the_printed_distribution_names_the_tier_it_is_describing(search, tmp_path, capsys):
    """Every other command prints the tier of every hit. This one printed a table with
    no tier anywhere on it."""
    vendor = tmp_path / "tut" / "family" / "aCase"
    dictionary(vendor / "system" / "fvSchemes",
               "divSchemes\n{\n    div(phi,U)  Gauss upwind;\n}\n")
    rows = [{"path": str(vendor), "tier": "vendor", "runs": True}]

    search.show_distributions(search.distributions(rows, "div(phi,U)",
                                                   root=tmp_path / "tut"))

    assert "vendor" in capsys.readouterr().out


# -- unmatched words are counted against the tier actually searched ------------


def test_unmatched_words_are_judged_against_the_tier_being_searched(search):
    """`failure` searches the earned tier, and the unmatched-word notice was computed
    over every row of both. So `failure "pitzDaily"` -- a word that appears only in a
    vendor path -- printed a bare "nothing matched" and logged `unmatched: []`, telling
    the reader and the record that the query had been understood."""
    rows = [row(path="/tutorials/incompressible/simpleFoam/pitzDaily"),
            earned_row(path="/work/a-study")]
    assert search.unmatched_tokens(rows, "pitzDaily") == []
    assert search.unmatched_tokens(rows, "pitzDaily", tier="earned") == ["pitzDaily"]


def test_a_failure_query_says_when_the_earned_tier_has_none_of_the_words(
    search, tmp_path, capsys
):
    vendor = tmp_path / "corpus"
    vendor.mkdir()
    (vendor / "tutorials.index.jsonl").write_text(
        json.dumps(row(path="/tutorials/incompressible/simpleFoam/pitzDaily")) + "\n",
        encoding="utf-8")
    (vendor / "studies.index.jsonl").write_text(
        json.dumps(earned_row(path="/work/a-study")) + "\n", encoding="utf-8")
    import openreynolds  # noqa: F401  (keeps the import list honest for linters)

    hits = search.failures(search.load_rows(vendor), "pitzDaily")

    assert hits == []
    assert search.unmatched_tokens(search.load_rows(vendor), "pitzDaily",
                                   tier="earned") == ["pitzDaily"]


def test_the_lock_file_is_cleaned_up_after_a_write(search, log_dir):
    """A lock left behind would make the next writer wait out its whole retry budget
    before proceeding unlocked -- a log that gets slower and less safe the longer it
    is used."""
    search.record_retrieval(log_dir, kind="regime", text="one", hits=[], matched=0)
    leftover = list(log_dir.glob("*.lock"))
    assert leftover == [], leftover


def test_a_stale_lock_does_not_silence_the_log_forever(search, log_dir, monkeypatch):
    """A process killed between taking the lock and removing it leaves the file there.
    Waiting on it forever, or refusing to write, would let one dead process stop the
    provenance record permanently -- so past the retry budget the write goes ahead.

    A record written into a possible race beats a record dropped for certain.
    """
    monkeypatch.setattr(search, "LOCK_ATTEMPTS", 2)
    monkeypatch.setattr(search, "LOCK_WAIT_S", 0)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "retrievals.jsonl.lock").write_text("held by nobody", encoding="utf-8")

    assert search.record_retrieval(log_dir, kind="regime", text="one", hits=[], matched=0)
    assert len(search.read_retrievals(log_dir)) == 1


def test_a_write_still_reports_failure_when_the_directory_will_not_take_one(
    search, tmp_path
):
    """The lock must not turn an unwritable corpus into a hang or a false success."""
    blocked = tmp_path / "corpus"
    blocked.write_text("not a directory", encoding="utf-8")
    assert search.record_retrieval(blocked, kind="regime", text="x", hits=[],
                                   matched=0) is None
