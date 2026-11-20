"""The #49 eval's scenario says what it claims: one deliverable missing, the rest there.

The live eval (`benchmarks/deliverables/run.py`) is only worth its cost if the thread it
plays leaves exactly the radial-profile plot out. These hold it to that without a model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "deliverables"))
import scenario  # noqa: E402


def _shown() -> str:
    return json.dumps(scenario.messages()[1:])


def test_the_missing_plot_is_nowhere_in_what_the_model_is_shown():
    shown = _shown().lower()
    for name in ("radial_profile", "profiles.png", "collapse", "self_similar", "selfsim"):
        assert name not in shown


def test_every_other_deliverable_is_in_the_last_listing():
    listing = scenario.messages()[-1]["content"][0]["content"]
    for name in scenario.PRESENT:
        assert name in listing


def test_the_request_asks_for_the_missing_plot():
    assert "x/D = 20, 40 and 60" in scenario.REQUEST
    assert "self-similarity" in scenario.REQUEST


def test_every_tool_call_gets_its_answer():
    thread = scenario.messages()
    for asked, answered in zip(thread[1::2], thread[2::2]):
        calls = [b["id"] for b in asked["content"] if b["type"] == "tool_use"]
        results = [b["tool_use_id"] for b in answered["content"]]
        assert calls == results


def test_the_stations_reach_eighty_diameters():
    # D = 10 mm: the last sampled line must be at x = 0.8 m, or the model (rightly)
    # stops to ask why the fits claim x/D >= 30.
    assert scenario.STATIONS[-1] == "x0.80_U.xy"


def test_reading_the_listing_does_not_invent_the_plot():
    out = scenario.respond("bash", {"cmd": "ls -lh case/post"})
    assert "radial" not in out and "centerline_decay.png" in out


def test_the_eval_will_not_spend_anything_unless_it_is_told_to(capsys):
    """The one benchmark in the repository that costs money, held behind a flag.

    Every other grader reads files a solver already wrote, so the benchmark job can run
    all of them and this one cannot be reached by accident: no flag, no config load, no
    provider, no call."""
    import run  # the same benchmarks/deliverables on sys.path as `scenario` above

    assert run.main([]) == 2
    assert "--spend" in capsys.readouterr().err
    assert "--spend" in run.__doc__
