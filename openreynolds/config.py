"""Configuration: environment first, then a user config file.

Credentials live in the user config directory with restrictive permissions — never in
the repository, and never in the study mirror.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOOL_OUTPUT = 48_000
"""Bytes of tool output shown inline before a marker takes over. Never a hard wall —
the marker always says where the rest lives and how to window into it."""

DEFAULT_LLM_TIMEOUT_S = 300.0
"""Longest gap with no bytes from the model API before the request is abandoned.

A stalled connection with no timeout is indistinguishable from a model thinking
hard, and the session simply stops with nothing said. Failing is recoverable --
the loop reports it and the thread survives -- and silence is not."""

DEFAULT_MIRROR_INTERVAL_S = 20.0
"""Seconds between background syncs of the study's files to this machine.

The mirror used to run only when a turn ended, which is exactly when nothing is
happening: a two-hour solve writes its fields, logs and renders while the model's
turn is over, and none of it reached the user's machine until the session wound
down. A live study should be on the user's disk while it is live. Zero or a
negative value turns the background sync off (turn-end syncs still run)."""

DEFAULT_NARRATE_EVERY_S = 60.0
"""Seconds between mid-run progress wakes while jobs are being watched.

A long solve used to be a silent one: the model was woken only when the job
ended, so twenty minutes of "watching 1 job(s)" was all a person saw. At this
cadence the model is woken with progress facts (elapsed time, log growth, the
last lines) and can say where things stand. Each wake is a model turn and is
priced like one; zero turns narration off."""

DEFAULT_DESK_MODEL = "claude-haiku-4-5"
"""The front-desk model: a second, cheap agent that answers the user while the main
agent is busy (`desk.py`). Haiku is ~5x cheaper than the Opus main model and fast
enough to reply in seconds. Uses the same BYOK key. `OPENREYNOLDS_DESK_MODEL` overrides;
`OPENREYNOLDS_DESK=0` turns it off entirely."""

CONTEXT_WINDOW_TOKENS = 1_000_000
CONTEXT_REFRESH_FRACTION = 0.8

_CONFIG_KEYS = (
    "foamd_url",
    "foamd_api_key",
    "anthropic_api_key",
    "llm_base_url",
    "model",
    "effort",
)


def preferences_path() -> Path:
    """The user's standing note, kept beside the credentials.

    Plain markdown, written by the user, relayed into every session's briefing in
    their own words. It is theirs: the harness never writes it, never edits it, and
    relays it verbatim."""
    return config_path().parent / "preferences.md"


def config_path() -> Path:
    """Where credentials are stored, per-user and outside any repository."""
    override = os.environ.get("OPENREYNOLDS_CONFIG")
    if override:
        return Path(override)
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "openreynolds" / "config.json"


@dataclass
class Config:
    foamd_url: str = ""
    foamd_api_key: str = ""
    anthropic_api_key: str = ""

    llm_base_url: str | None = None
    """Where the Anthropic client points. `None` means the API directly.

    TODO (future): the hosted service dropped its LLM proxy when it moved to
    bring-your-own-key. If a `/v1/llm` endpoint ever ships, set this to
    `<foamd_url>/v1/llm` and use the service key — a config change, not a code change.
    """

    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    max_tool_output: int = DEFAULT_MAX_TOOL_OUTPUT
    llm_timeout_s: float = DEFAULT_LLM_TIMEOUT_S
    mirror_interval_s: float = DEFAULT_MIRROR_INTERVAL_S
    narrate_every_s: float = DEFAULT_NARRATE_EVERY_S
    capture: bool = True
    """Whether transcripts are uploaded to the workspace service as the study runs.

    On by default, so a study is kept somewhere other than one laptop. Off with
    `--no-capture` for a session or `OPENREYNOLDS_CAPTURE=0` for an environment --
    a flag has to be remembered every time, and the place it is most likely to be
    forgotten is the scheduled run nobody is watching."""
    desk: bool = True
    """Whether the front-desk agent runs. It answers the user while the main agent is
    mid-turn -- the difference between a message heard in seconds and one that waits
    out a five-minute solve."""
    desk_model: str = DEFAULT_DESK_MODEL
    studies_dir: Path = field(default_factory=lambda: Path.cwd() / "studies")
    preferences: str = ""
    """The standing note from `preferences_path()`, or empty when there is none."""

    @classmethod
    def load(cls) -> Config:
        stored: dict[str, object] = {}
        path = config_path()
        if path.exists():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stored = {}

        def pick(env: str, key: str, default: str = "") -> str:
            return os.environ.get(env) or str(stored.get(key) or "") or default

        llm_base_url = pick("OPENREYNOLDS_LLM_BASE_URL", "llm_base_url") or None
        max_output = os.environ.get("OPENREYNOLDS_MAX_TOOL_OUTPUT")
        timeout = os.environ.get("OPENREYNOLDS_LLM_TIMEOUT_S")
        mirror_every = os.environ.get("OPENREYNOLDS_MIRROR_INTERVAL_S")
        narrate_every = os.environ.get("OPENREYNOLDS_NARRATE_EVERY_S")

        try:
            preferences = preferences_path().read_text(encoding="utf-8").strip()
        except OSError:
            preferences = ""

        def switched_off(env: str) -> bool:
            return (os.environ.get(env) or "").strip().lower() in ("0", "false", "no", "off")

        return cls(
            preferences=preferences,
            capture=not switched_off("OPENREYNOLDS_CAPTURE"),
            desk=not switched_off("OPENREYNOLDS_DESK"),
            desk_model=pick("OPENREYNOLDS_DESK_MODEL", "desk_model", DEFAULT_DESK_MODEL),
            foamd_url=pick("FOAMD_URL", "foamd_url").rstrip("/"),
            foamd_api_key=pick("FOAMD_API_KEY", "foamd_api_key"),
            anthropic_api_key=pick("ANTHROPIC_API_KEY", "anthropic_api_key"),
            llm_base_url=llm_base_url,
            model=pick("OPENREYNOLDS_MODEL", "model", DEFAULT_MODEL),
            effort=pick("OPENREYNOLDS_EFFORT", "effort", DEFAULT_EFFORT),
            max_tool_output=int(max_output) if max_output else DEFAULT_MAX_TOOL_OUTPUT,
            llm_timeout_s=float(timeout) if timeout else DEFAULT_LLM_TIMEOUT_S,
            mirror_interval_s=(
                float(mirror_every) if mirror_every else DEFAULT_MIRROR_INTERVAL_S
            ),
            narrate_every_s=(
                float(narrate_every) if narrate_every else DEFAULT_NARRATE_EVERY_S
            ),
        )

    def save(self) -> Path:
        """Write the credential fields back, readable only by this user."""
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: getattr(self, key) for key in _CONFIG_KEYS if getattr(self, key)}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if os.name != "nt":
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return path

    def missing(self) -> list[str]:
        """Which required settings are absent. LLM auth is bring-your-own-key."""
        gaps = []
        if not self.foamd_url:
            gaps.append("FOAMD_URL")
        if not self.foamd_api_key:
            gaps.append("FOAMD_API_KEY")
        if not self.anthropic_api_key and not self.llm_base_url:
            gaps.append("ANTHROPIC_API_KEY")
        return gaps
