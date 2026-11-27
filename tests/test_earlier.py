"""`import_study`: an earlier study's files, into this study's own workspace (#212)."""
from __future__ import annotations

from openreynolds import earlier, tools
from openreynolds.backend.base import BackendError

STUDIES = [
    {"id": "now", "title": "tank v3", "created_at": "2026-11-26T10:00:00Z", "instance_id": "i3"},
    {"id": "s2", "title": "Tank v2, corrected inlet", "created_at": "2026-11-20T09:00:00Z", "instance_id": "i2"},
    {"id": "s1", "title": "Tank v1", "created_at": "2026-11-10T09:00:00Z", "instance_id": "i1"},
    {"id": "gone", "title": "deleted one", "created_at": "2026-11-01T09:00:00Z", "instance_id": "i0",
     "deleted_at": "2026-11-02T00:00:00Z"},
    {"id": "20261115-134546-cd85", "title": "20261115-134546-cd85", "created_at": "2026-11-15T09:00:00Z",
     "instance_id": "i4"},
    {"id": "truck", "title": "Simulate airflow around this truck at 25 m/s\n(uploaded to the workspace: /work/truck/t.stl)", "created_at": "2026-11-05T09:00:00Z",
     "instance_id": "i5"},
    {"id": "laptop", "title": "run on my laptop", "created_at": "2026-10-01T09:00:00Z", "instance_id": None},
]


class Client:
    def __init__(self, answer=None, error=None):
        self.asked = []
        self.answer = answer or {"into": "/work/now/from-s2", "from": "/work/s2", "files": 12,
                                 "bytes": 3 * 1024 * 1024, "as_of": "2026-11-20T10:31:00+00:00",
                                 "failed": []}
        self.error = error

    def list_studies(self):
        return STUDIES

    def import_study(self, study_id, from_study, path="", into=None):
        self.asked.append((study_id, from_study, path, into))
        if self.error:
            raise self.error
        return self.answer


def _earlier(client, study_id="now"):
    waited = []
    return earlier.Earlier(connect=lambda: client, study_id=lambda: study_id,
                           workspace=lambda: waited.append(1)), waited


def test_without_a_study_it_lists_the_others_newest_first():
    e, waited = _earlier(Client())
    text = e.run({})
    assert "s2  2026-11-20  Tank v2, corrected inlet" in text and "s1  2026-11-10  Tank v1" in text
    assert text.index("s2") < text.index("s1")
    assert "now" not in text.split("\n", 1)[1], "not this study itself"
    assert "deleted one" not in text, "not one its owner deleted"
    assert "laptop" not in text, "not one that never had a workspace on the service"
    assert waited == [], "a listing needs no workspace"


def test_a_study_by_id_or_by_words_from_its_title_is_copied_in():
    client = Client()
    e, waited = _earlier(client)
    text = e.run({"study": "s2"})
    assert client.asked == [("now", "s2", "", None)] and waited == [1]
    assert "Copied 12 files (3.0M)" in text and "into /work/now/from-s2" in text
    assert "as of 2026-11-20 10:31 UTC" in text
    e.run({"study": "corrected", "path": "case/constant", "into": "v2"})
    assert client.asked[-1] == ("now", "s2", "case/constant", "v2")


def test_words_that_match_several_or_none_ask_for_the_id():
    client = Client()
    e, _ = _earlier(client)
    several = e.run({"study": "tank"})
    assert "2 earlier studies" in several and "s1" in several and "s2" in several
    none = e.run({"study": "airfoil"})
    assert "No earlier study" in none and "s2" in none
    assert e.run({"study": "gone"}).startswith("No earlier study"), "a deleted one is not offered"
    assert client.asked == []


def test_a_request_in_its_own_words_finds_the_title():
    """The agent passes the person's phrase, which is rarely the title verbatim."""
    client = Client()
    e, _ = _earlier(client)
    e.run({"study": "airflow truck"})
    assert client.asked[-1][1] == "truck", "every word is in the title, in another order"
    near = e.run({"study": "my truck with a finer mesh"})
    assert near.startswith("No earlier study's title holds all") and "truck" in near
    assert "s2" not in near, "only the ones that share a word are offered"
    assert len(client.asked) == 1, "a near match is offered, not copied"


def test_a_title_that_is_only_the_id_lists_as_untitled():
    text = _earlier(Client())[0].run({})
    assert "20261115-134546-cd85  2026-11-15  (untitled)" in text
    assert "25 m/s (uploaded to the workspace" in text, "a title is listed on one line"


def test_files_that_failed_or_were_left_are_named():
    failed = [{"path": f"/work/s2/f{i}", "error": "502"} for i in range(12)]
    client = Client(answer={"into": "/work/now/from-s2", "from": "/work/s2", "files": 3, "bytes": 10,
                            "already": 4, "failed": failed, "left": 7, "complete": False})
    e, _ = _earlier(client)
    text = e.run({"study": "s2"})
    assert "12 files did not come across" in text and "/work/s2/f0: 502" in text
    assert "...and 2 more" in text and "7 files not yet tried" in text
    assert "4 were there already" in text and "copies only what is still missing" in text
    client.answer = {"into": "/work/now/from-s2", "from": "/work/s2", "files": 3, "bytes": 10, "complete": True}
    assert "Calling again" not in e.run({"study": "s2"})


def test_a_study_with_no_record_has_nothing_to_copy_into():
    client = Client()
    e, _ = _earlier(client, study_id=None)
    assert "no record on the service" in e.run({"study": "s2"})
    assert client.asked == []


def test_offered_only_where_it_can_run():
    class Ctx:
        cad = reviewer = None
        solver = "openfoam"
        earlier = None

    names = {t["name"] for t in tools.tools_for(Ctx())}
    assert "import_study" not in names
    Ctx.earlier = object()
    assert "import_study" in {t["name"] for t in tools.tools_for(Ctx())}
    Ctx.solver = "felix"
    assert "import_study" not in {t["name"] for t in tools.tools_for(Ctx())}


def test_the_service_refusing_reaches_the_model_as_an_error():
    e, _ = _earlier(Client(error=BackendError("413 too_large: import a subfolder (path)")))
    ctx = type("Ctx", (), {"earlier": e, "calls": 0})()
    result, is_error = tools.dispatch(ctx, "import_study", {"study": "s2"})
    assert is_error and "subfolder" in result
    ctx.earlier = None
    result, is_error = tools.dispatch(ctx, "import_study", {})
    assert not is_error and "only on the hosted service" in result


def test_a_timed_out_answer_says_calling_again_carries_on():
    e, _ = _earlier(Client(error=BackendError("request timed out: read", code="timeout")))
    assert "carries on" in e.run({"study": "s2"})
