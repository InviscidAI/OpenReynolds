"""An earlier study's files, brought into this study's workspace (`import_study`).

On the hosted service every study runs on a workspace of its own, which starts empty.
A follow-up ("continue from my corrected case", "use the tank from last week") used to
meet "the prior case and the named CAD files are not present in this workspace", and
nothing could bring them across: `openreynolds clone` copies only this machine's
mirror. The service keeps each workspace's checkpointed copy, and its import route
has the new workspace fetch an earlier study's files from it directly (foamd
`POST /v1/studies/{id}/workspace/import`).

The tool has two halves. Without a study it lists the person's earlier studies, so the
agent can find the one they mean; with one it copies that study, or a folder of it,
into `<this study>/from-<id>/`, every file at its path relative to the earlier study's
home.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .backend.base import BackendError
from .browse import human

LISTED = 30
"""Earlier studies shown at once, newest first."""
FAILED_SHOWN = 10


@dataclass
class Earlier:
    connect: Callable[[], Any]
    """The service's client (`FoamdClient`), made when first needed."""
    study_id: Callable[[], str | None]
    """This study's id on the service; None before its row is opened."""
    workspace: Callable[[], Any]
    """Waits for this study's workspace to be up (the copy needs it running)."""

    def run(self, args: dict[str, Any]) -> str:
        current = self.study_id()
        if not current:
            return ("This study has no record on the service, so there is nothing to copy "
                    "into: earlier studies can be brought in only on the hosted service.")
        client = self.connect()
        # Not this one, not one its owner deleted, and not one that never had a
        # workspace on the service (run on the person's own machine): nothing to copy.
        studies = [s for s in client.list_studies() or []
                   if s.get("id") != current and not s.get("deleted_at") and s.get("instance_id")]
        wanted = str(args.get("study") or "").strip()
        if not wanted:
            return listing(studies)
        found = pick(studies, wanted)
        if isinstance(found, str):
            return found
        self.workspace()
        try:
            got = client.import_study(current, str(found["id"]), str(args.get("path") or ""),
                                      str(args.get("into") or "") or None)
        except BackendError as exc:
            if exc.code != "timeout":
                raise
            return ("The answer did not arrive in time, but the copy may still be going on "
                    "the service. Calling again with the same arguments carries on: files "
                    "already there are left alone.")
        return report(found, got)


def listing(studies: list[dict[str, Any]]) -> str:
    if not studies:
        return "There are no earlier studies on this account."
    lines = [f"{s['id']}  {str(s.get('created_at') or '')[:10]}  {s.get('title') or '(untitled)'}"
             for s in studies[:LISTED]]
    more = len(studies) - LISTED
    tail = f"\n...and {more} older." if more > 0 else ""
    return ("Earlier studies, newest first (id, started, title). Call again with `study` "
            "set to the id to copy one in.\n" + "\n".join(lines) + tail)


def pick(studies: list[dict[str, Any]], wanted: str) -> dict[str, Any] | str:
    """The study `wanted` names: by id, else by a title it matches alone."""
    for s in studies:
        if s.get("id") == wanted:
            return s
    low = wanted.lower()
    titled = [s for s in studies if low in str(s.get("title") or "").lower()]
    if len(titled) == 1:
        return titled[0]
    if titled:
        return (f"{len(titled)} earlier studies have \"{wanted}\" in their title; give the id "
                "of the one meant.\n" + listing(titled).split("\n", 1)[1])
    return f"No earlier study of this account is called {wanted!r}. " + listing(studies)


def report(study: dict[str, Any], got: dict[str, Any]) -> str:
    title = study.get("title") or study["id"]
    as_of = f", as of {got['as_of'][:16].replace('T', ' ')} UTC" if got.get("as_of") else ""
    text = (f"Copied {got.get('files', 0)} files ({human(int(got.get('bytes') or 0))}) of "
            f"\"{title}\" from {got.get('from')} into {got.get('into')}{as_of}. Each file keeps "
            "its path relative to the earlier study's folder.")
    if got.get("already"):
        text += f" {got['already']} were there already and were left as they are."
    failed = got.get("failed") or []
    if failed:
        shown = "\n".join(f"  {f.get('path')}: {f.get('error')}" for f in failed[:FAILED_SHOWN])
        more = f"\n  ...and {len(failed) - FAILED_SHOWN} more" if len(failed) > FAILED_SHOWN else ""
        text += f"\n{len(failed)} files did not come across:\n{shown}{more}"
    if got.get("left"):
        text += f"\nThe copy stopped with {got['left']} files not yet tried."
    if got.get("complete") is False:
        text += "\nCalling again with the same arguments copies only what is still missing."
    return text
