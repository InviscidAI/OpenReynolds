"""Platform capture — invisible to the model, and never in its way.

Messages, fetched artifacts and any end-of-session results payload are posted to the
platform on a background thread. If the platform is unreachable the work buffers, then
retries, then is dropped with a warning. Nothing here can delay or fail a study.
"""

from __future__ import annotations

import mimetypes
import queue
import threading
from pathlib import Path
from typing import Any, Callable

from .backend.hosted import FoamdClient

_MAX_ATTEMPTS = 3
_CONTENT_CAP = 20_000
"""Characters of any single captured message body. The local mirror keeps the full text."""


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
        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._dropped = 0
        self._worker = threading.Thread(target=self._drain, name="capture", daemon=True)
        self._worker.start()

    @classmethod
    def start(
        cls,
        client: FoamdClient,
        title: str,
        instance_id: str,
        *,
        warn: Callable[[str], None] | None = None,
    ) -> Capture | None:
        """Create the remote study. Returns None if the platform will not have us."""
        try:
            study_id = client.create_study(title, instance_id)
        except Exception as exc:
            if warn:
                warn(f"capture off — could not open a study ({exc})")
            return None
        return cls(client, study_id, warn=warn)

    # -- public surface --------------------------------------------------------

    def message(self, seq: int, role: str, content: Any) -> None:
        payload = {"seq": seq, "role": role, "content": _cap_content(content)}
        self._submit(lambda: self.client.post_messages(self.study_id, [payload]))

    def artifact(self, path: Path, kind: str | None = None) -> None:
        def send() -> None:
            data = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.client.post_artifact(self.study_id, path.name, data, kind or mime)

        self._submit(send)

    def result(self, payload: Any) -> None:
        self._submit(lambda: self.client.post_result(self.study_id, payload))

    def close(self, timeout: float = 10.0) -> None:
        """Drain what is queued, then stop. Bounded, so it cannot hang an exit."""
        self._queue.put(None)
        self._worker.join(timeout=timeout)
        if self._dropped:
            self._warn(f"capture dropped {self._dropped} item(s) — the local mirror is complete")

    # -- worker ----------------------------------------------------------------

    def _submit(self, task: Callable[[], None]) -> None:
        self._queue.put(task)

    def _drain(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                return
            for attempt in range(_MAX_ATTEMPTS):
                try:
                    task()
                    break
                except Exception:
                    if attempt == _MAX_ATTEMPTS - 1:
                        self._dropped += 1


def _cap_content(content: Any) -> Any:
    """Keep captured message bodies a sane size."""
    if isinstance(content, str):
        return content[:_CONTENT_CAP]
    if isinstance(content, dict):
        return {key: _cap_content(value) for key, value in content.items()}
    if isinstance(content, list):
        return [_cap_content(item) for item in content]
    return content
