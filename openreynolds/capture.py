"""Platform capture — invisible to the model, and never in its way.

Messages, fetched artifacts and any end-of-session results payload are posted to the
platform on a background thread. If the platform is unreachable the work buffers, and
what will not go is dropped with a warning. Nothing here can delay or fail a study.

Retrying is the transport's job and not this module's, which is a correction rather
than a division of labour: this worker used to try each item three more times on top of
the five `FoamdClient.request` already gives it, so one message could be posted fifteen
times. The service takes the `seq` this client assigns and its `messages` table indexes
`(study_id, seq)` without making it unique, so every repeat of a post that had already
been carried out appended the same row again — one job reply from a 3D transient run
landed in the captured transcript three times, five more messages of the same study
twice. And because there is one worker and it is serial, those repeats also held every
later message behind them: rows carry the time they were *inserted*, not the time the
agent recorded them, so a transcript read back afterwards showed a run stalled for
twenty-six minutes that was not stalled at all.
"""

from __future__ import annotations

import mimetypes
import queue
import threading
from pathlib import Path
from typing import Any, Callable

from .backend.hosted import FoamdClient

_CONTENT_CAP = 20_000
"""Characters of any single captured message body. The local mirror keeps the full text."""

_NAME_AT_MOST = 6
"""How many dropped items are named individually before the warning counts kinds instead."""

_REMEMBER_AT_MOST = 200
"""A ceiling on the labels kept, so a platform that is down for a whole study cannot
turn this into a second copy of the transcript in memory."""


class Capture:
    """A fire-and-forget uploader for one study."""

    def __init__(
        self,
        client: FoamdClient,
        study_id: str,
        *,
        warn: Callable[[str], None] | None = None,
    ):
        self.client = client
        self.study_id = study_id
        self._warn = warn or (lambda _msg: None)
        self._queue: queue.Queue[tuple[str, Callable[[], None]] | None] = queue.Queue()
        self._dropped = 0
        self._lost: list[str] = []
        """What was dropped, by name, up to `_REMEMBER_AT_MOST` of them (F-44)."""
        self._worker = threading.Thread(target=self._drain, name="capture", daemon=True)
        self._worker.start()

    @classmethod
    def start(
        cls,
        client: FoamdClient,
        title: str,
        instance_id: str,
        *,
        study_id: str | None = None,
        home: str | None = None,
        warn: Callable[[str], None] | None = None,
    ) -> Capture | None:
        """Create the remote study. Returns None if the platform will not have us.

        `study_id` is this study's own id, so the platform row, the local directory
        and the web URL are one string rather than three names for one thing."""
        try:
            # Passed only when supplied, so a client without these parameters --
            # an older service's, or a test's stand-in -- stays callable.
            extra: dict[str, str] = {}
            if study_id:
                extra["study_id"] = study_id
            if home:
                extra["home"] = home
            study_id = client.create_study(title, instance_id, **extra)
        except Exception as exc:
            if warn:
                warn(f"capture off — could not open a study ({exc})")
            return None
        return cls(client, study_id, warn=warn)

    # -- public surface --------------------------------------------------------

    def message(self, seq: int, role: str, content: Any) -> None:
        payload = {"seq": seq, "role": role, "content": _cap_content(content)}
        self._submit(f"message seq {seq} ({role})",
                     lambda: self.client.post_messages(self.study_id, [payload]))

    def artifact(self, path: Path, kind: str | None = None) -> None:
        def send() -> None:
            data = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.client.post_artifact(self.study_id, path.name, data, kind or mime)

        self._submit(f"artifact {path.name}", send)

    def result(self, payload: Any) -> None:
        self._submit("the end-of-study result",
                     lambda: self.client.post_result(self.study_id, payload))

    def close(self, timeout: float = 10.0) -> None:
        """Drain what is queued, then stop. Bounded, so it cannot hang an exit."""
        self._queue.put(None)
        self._worker.join(timeout=timeout)
        if self._dropped:
            self._warn(self._dropped_line())

    def _dropped_line(self) -> str:
        """What was lost, by name (F-44).

        The count alone answered the wrong question. The behaviour it reports is
        right -- capture is fire-and-forget by design and must never delay or fail a
        study -- and the warning was honest that something went missing, but the only
        question a person has on reading it is *what*, and a dropped message leaves a
        gap in the web transcript while a dropped artifact leaves a picture that
        exists on the laptop and nowhere else. Those are not the same loss. The queue
        already held a callable per item, so a label alongside it costs nothing.

        Long runs are summarised rather than listed: past a handful the names stop
        being readable and the kinds are what is left to act on.
        """
        head = f"capture dropped {self._dropped} item(s) — the local mirror is complete"
        if len(self._lost) <= _NAME_AT_MOST:
            return f"{head}: {', '.join(self._lost)}"
        kinds: dict[str, int] = {}
        for label in self._lost:
            kinds[label.split(" ", 1)[0]] = kinds.get(label.split(" ", 1)[0], 0) + 1
        counted = ", ".join(f"{n} {kind}(s)" for kind, n in sorted(kinds.items()))
        return f"{head}: {counted}; first was {self._lost[0]}"

    # -- worker ----------------------------------------------------------------

    def _submit(self, label: str, task: Callable[[], None]) -> None:
        self._queue.put((label, task))

    def _drain(self) -> None:
        """Post each item once, and let go of whatever will not go.

        One attempt, deliberately. The client underneath already retries what is safe
        to retry — a rate limit, a workspace still booting, a connection that was never
        made — and refuses to repeat a write whose outcome it does not know. A second
        attempt from here cannot tell those apart, so all it can add is the duplicate
        row this module's docstring is about. A message that is lost leaves a gap in a
        sequence that is dense by construction, which is a thing anyone reading the
        transcript can see; a message posted twice is a thing nobody sees.
        """
        while True:
            item = self._queue.get()
            if item is None:
                return
            label, task = item
            try:
                task()
            except Exception:
                self._dropped += 1
                if len(self._lost) < _REMEMBER_AT_MOST:
                    self._lost.append(label)


def _cap_content(content: Any) -> Any:
    """Keep captured message bodies a sane size."""
    if isinstance(content, str):
        return content[:_CONTENT_CAP]
    if isinstance(content, dict):
        return {key: _cap_content(value) for key, value in content.items()}
    if isinstance(content, list):
        return [_cap_content(item) for item in content]
    return content
