"""`scratch.py` runs a case on local disk and reconstructs off the Volume. The
subprocess side needs a container; the decisions -- which names are times, what is
still pending, how a scoped reconstruct command is spelled, where a case stages to
-- are ordinary Python and are what rots, so they are tested here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"toolbox_{name}", TOOLBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scratch():
    return load("scratch")


# -- time-directory recognition ------------------------------------------------

def test_a_bare_number_is_a_time_directory(scratch):
    for name in ("0", "0.005", "1.5e-05", "100.00510749", "120"):
        assert scratch.is_time_name(name), name


def test_the_case_structure_dirs_are_not_times(scratch):
    for name in ("constant", "system", "0.orig", "processors16", "processor0", "renders"):
        assert not scratch.is_time_name(name), name


def test_time_dirs_come_back_in_numeric_not_lexical_order(scratch):
    names = ["10", "2", "100.5", "0", "system", "1.5e-05", "constant"]
    assert scratch.time_dirs(names) == ["0", "1.5e-05", "2", "10", "100.5"]


# -- where decomposed data lives -----------------------------------------------

def test_collated_processor_container_is_preferred(scratch):
    names = ["0", "constant", "system", "processors16", "processor0", "processor1"]
    assert scratch.processor_container(names) == "processors16"


def test_uncollated_falls_back_to_the_first_processor_dir(scratch):
    names = ["0", "constant", "processor0", "processor1", "processor2", "processor10"]
    assert scratch.processor_container(names) == "processor0"


def test_an_undecomposed_case_has_no_container(scratch):
    assert scratch.processor_container(["0", "constant", "system"]) is None


# -- what a reconstruct still owes ---------------------------------------------

def test_pending_times_are_the_decomposed_ones_missing_from_the_root(scratch):
    reconstructed = ["0", "constant", "system", "0.5"]
    decomposed = ["0", "0.5", "1", "1.5"]
    assert scratch.pending_times(reconstructed, decomposed) == ["1", "1.5"]


def test_zero_is_never_pending(scratch):
    """decomposePar writes a 0 per processor; reconstruct does not add it to the root,
    so a case with only 0 decomposed is not perpetually 'pending'."""
    assert scratch.pending_times(["0", "system"], ["0"]) == []


def test_nothing_pending_when_the_case_is_reconstructed(scratch):
    assert scratch.pending_times(["0", "1", "2"], ["0", "1", "2"]) == []


# -- the reconstruct command for a scope ---------------------------------------

def test_default_scope_is_newtimes_which_is_idempotent(scratch):
    assert scratch.reconstruct_argv() == ["reconstructPar", "-newTimes"]


def test_latest_time_scope(scratch):
    assert scratch.reconstruct_argv(latest=True) == ["reconstructPar", "-latestTime"]


def test_a_time_range_becomes_a_comma_pair(scratch):
    assert scratch.reconstruct_argv(time_range="120:135") == ["reconstructPar", "-time", "120,135"]
    assert scratch.reconstruct_argv(time_range="135") == ["reconstructPar", "-time", "135"]


def test_latest_and_time_win_over_newtimes(scratch):
    assert "-newTimes" not in scratch.reconstruct_argv(latest=True)
    assert "-newTimes" not in scratch.reconstruct_argv(time_range="1:2")


def test_fields_are_wrapped_in_openfoam_list_syntax(scratch):
    assert scratch.reconstruct_argv(fields="U p") == ["reconstructPar", "-newTimes", "-fields", "(U p)"]
    assert scratch.reconstruct_argv(latest=True, fields="U, p, k") == \
        ["reconstructPar", "-latestTime", "-fields", "(U p k)"]


# -- staging path and sync command --------------------------------------------

def test_scratch_dir_is_local_and_stable(scratch):
    a = scratch.scratch_dir("/work/study/case")
    assert a.startswith("/tmp/") and a == scratch.scratch_dir("/work/study/case")


def test_same_basename_different_path_do_not_collide(scratch):
    assert scratch.scratch_dir("/work/a/case") != scratch.scratch_dir("/work/b/case")
    assert "case-" in scratch.scratch_dir("/work/a/case")


def test_a_trailing_slash_does_not_change_the_target(scratch):
    assert scratch.scratch_dir("/work/study/case/") == scratch.scratch_dir("/work/study/case")


def test_rsync_copies_contents_and_only_deletes_when_asked(scratch):
    assert scratch.rsync_argv("/tmp/c", "/work/c") == ["rsync", "-a", "/tmp/c/", "/work/c"]
    assert "--delete" in scratch.rsync_argv("/tmp/c", "/work/c", delete=True)


# -- the CLI wiring ------------------------------------------------------------

def test_run_strips_the_leading_double_dash(scratch, monkeypatch):
    seen = {}

    def fake(case, command, **kw):
        seen["case"], seen["command"] = case, command
        return 0

    monkeypatch.setattr(scratch, "run_on_scratch", fake)
    scratch.main(["run", "/work/s/case", "--", "mpirun", "-np", "4", "pimpleFoam", "-parallel"])
    assert seen["case"] == "/work/s/case"
    assert seen["command"] == ["mpirun", "-np", "4", "pimpleFoam", "-parallel"]


def test_the_checkpointer_thread_starts_and_joins(scratch):
    """Regression: the stop flag was named `_stop`, which shadows threading.Thread's own
    private `_stop()` method -- so `join()` tried to call the Event and raised
    "'Event' object is not callable", killing every run at its final sync."""
    ticker = scratch._Checkpointer("/tmp/nowhere", "/tmp/nowhere2", every=3600)
    ticker.start()
    ticker.stop()
    ticker.join(timeout=5)
    assert not ticker.is_alive()


def test_reconstruct_passes_its_scope_through(scratch, monkeypatch):
    seen = {}

    def fake(case, latest=False, time_range=None, new_times=True, fields=None, **kw):
        seen.update(case=case, latest=latest, time_range=time_range, fields=fields)
        return 0

    monkeypatch.setattr(scratch, "reconstruct", fake)
    scratch.main(["reconstruct", "/work/s/case", "--latest", "--fields", "U p"])
    assert seen == {"case": "/work/s/case", "latest": True, "time_range": None, "fields": "U p"}
