"""Pairing two sweeps, and the question each case actually asked.

A sweep records `git_sha`, which says what the desk was, and until 2026-09-20 nothing
that said what the *corpus* was. Two sweeps then paired on the case's name, so a case
whose request had been rewritten between them was compared against a different question
with no sign of it in the table.

`core-shipped-20260919-124001-f33d` did exactly that to T4, T5 and T10 -- T4 and T10
rewritten when the `core+cad_export` vet withdrew its own findings against them, T5
reposed to external flow. None of the three changed verdict, so the headline survived.
That is luck, not a check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sweep_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "cad_sweep_under_test", ROOT / "scripts" / "cad_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_digest_follows_the_request_and_not_the_provenance(sweep_module, monkeypatch):
    """`## What this catches` is not a criterion, so editing it is not a reposing.

    `tests/data/prompts/README.md` is explicit that only `## Request` and the properties
    are graded against, and that grading against the provenance notes is what produced
    three withdrawn findings in the previous sweep. A digest that moved when a note was
    edited would unpair cases nobody had rewritten.
    """
    prompts = {"T1": {"request": "mesh a duct", "properties": ["one cell thick"],
                      "false_pass": "a note", "expects": "done"}}
    monkeypatch.setitem(sys.modules, "cad_accept",
                        type("m", (), {"load_prompts": staticmethod(lambda: prompts)}))

    first = sweep_module._asked(["T1"])["T1"]

    prompts["T1"]["false_pass"] = "a completely different note"
    assert sweep_module._asked(["T1"])["T1"] == first, "provenance is not a criterion"

    prompts["T1"]["properties"] = ["one cell thick", "minimum width measured"]
    assert sweep_module._asked(["T1"])["T1"] != first, "the checklist is"

    prompts["T1"]["properties"] = ["one cell thick"]
    prompts["T1"]["request"] = "mesh a duct, 10 mm wide"
    assert sweep_module._asked(["T1"])["T1"] != first, "and so is the request"


def test_every_case_in_the_corpus_gets_a_digest(sweep_module):
    """Read off the real prompts, so a case the loader cannot parse is caught here."""
    digests = sweep_module._asked([f"T{n}" for n in range(1, 27)])
    assert len(digests) == 26
    assert len(set(digests.values())) == 26, "two cases hash alike; the digest is wrong"
    assert all(len(value) == 16 for value in digests.values())


def test_the_sweep_that_found_this_now_carries_its_own_questions():
    """Backfilled, so the next sweep can pair against it mechanically.

    Recorded after the fact from the prompts as they stand, which are the ones that ran:
    nothing has touched `tests/data/prompts` since. The flag says so rather than letting
    it read as having been captured at run time.
    """
    import json

    sweep = (ROOT / "docs" / "cad-buildup" / "sweeps"
             / "core-shipped-20260919-124001-f33d" / "sweep.json")
    if not sweep.is_file():
        pytest.skip("that sweep is not in this checkout")
    data = json.loads(sweep.read_text(encoding="utf-8"))
    assert len(data.get("asked") or {}) == 26
    assert data.get("asked_backfilled") is True


def test_a_baseline_without_the_digests_says_so_rather_than_claiming_agreement():
    """Silence is not a claim that nothing moved.

    The baseline predates `asked`, so the comparison cannot be made and the table says
    to check by hand. Reporting "no cases reposed" would be the same failure as the one
    this exists to close, with a reassuring sentence attached.
    """
    source = (ROOT / "scripts" / "cad_sweep.py").read_text(encoding="utf-8")
    assert "the baseline predates `asked`" in source
    assert "cannot be detected here" in source
