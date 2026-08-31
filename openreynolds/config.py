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

from .llm.presets import FALLBACK_CONTEXT_WINDOW, preset_for

DEFAULT_FOAMD_URL = "https://api.tryreynolds.com"
"""Where the workspace service lives unless told otherwise. A key from
`openreynolds login` belongs to whichever service issued it, so the two are saved
together."""

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

DEFAULT_NARRATE_EVERY_S = 600.0
"""Seconds between mid-run progress wakes while jobs are being watched.

A long solve used to be a silent one: the model was woken only when the job
ended, so twenty minutes of "watching 1 job(s)" was all a person saw. At this
cadence the model is woken with progress facts (elapsed time, log growth, the
last lines) and can say where things stand. Each wake is a model turn and is
priced like one -- a whole context re-read, every time. At sixty seconds that
was the single largest cost of a study (thirty-odd wakes across one solve, for
nothing the progress bar was not already showing); ten minutes keeps the
narration and drops that bill by an order of magnitude. Zero turns it off."""

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
    "provider",
    "llm_api_key",
    "llm_base_url",
    "model",
    "desk_model",
    "effort",
    "context_window",
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
    foamd_url: str = DEFAULT_FOAMD_URL
    foamd_api_key: str = ""

    provider: str = "anthropic"
    """Whose model, as a preset name (`openreynolds config` lists them) or a bare API
    family, `anthropic` or `openai`, with `llm_base_url` saying where. Bring your own
    key is the default, and any vendor that speaks one of the two dialects works,
    local ones included. The one preset that needs no key of its own is `reynolds`:
    there the service fronts the model and meters it to the account, and
    `make_provider` derives the endpoint and the key from the service settings."""
    llm_api_key: str = ""
    """Older config files call this `anthropic_api_key`; `load()` still reads that."""

    llm_base_url: str | None = None
    """Where the model client points. `None` means the preset's endpoint, or the
    vendor's default for a bare family.

    The `reynolds` preset ignores whatever is here: the service's own `/v1/llm` is
    derived from `foamd_url` at run time rather than stored, so moving the service
    moves the model with it and a stale URL in a config file cannot outlive it.
    """
    context_window: int = 0
    """Tokens the model can hold in one thread; the loop refreshes at a fraction of it.
    Zero means "whatever the preset says", or a conservative default for a vendor the
    presets do not know."""

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

    def __post_init__(self) -> None:
        preset = preset_for(self.provider)
        # A preset's numbers apply when its endpoint is the one in use; an explicit
        # base URL pointing elsewhere is a vendor the table knows nothing about.
        at_preset = preset is not None and (not self.llm_base_url or self.llm_base_url == preset.base_url)
        if not self.context_window:
            self.context_window = preset.context_window if at_preset else FALLBACK_CONTEXT_WINDOW
        if preset is not None and preset.name != "anthropic":
            if self.model == DEFAULT_MODEL:
                self.model = preset.model
            if self.desk_model == DEFAULT_DESK_MODEL:
                self.desk_model = preset.desk_model

    def model_key_missing(self) -> str | None:
        """The name of the model-key setting that is absent, or `None` when the model
        can be reached: a key is present, an explicit endpoint stands in for one, or
        the preset (a local model) needs none."""
        preset = preset_for(self.provider)
        if preset is not None and preset.name == "reynolds":
            # The service key is the model key; `missing()` already asks for it.
            return None
        if self.llm_api_key or self.llm_base_url:
            return None
        if preset is not None and not preset.needs_key:
            return None
        return preset.key_env if preset is not None else "OPENREYNOLDS_LLM_API_KEY"

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

        provider = pick("OPENREYNOLDS_PROVIDER", "provider", "anthropic").strip().lower()
        preset = preset_for(provider)
        # The vendor's own variable name works too (ANTHROPIC_API_KEY, OPENAI_API_KEY...),
        # because that is what people already have exported.
        llm_api_key = (
            os.environ.get("OPENREYNOLDS_LLM_API_KEY")
            or (os.environ.get(preset.key_env) if preset else "")
            or str(stored.get("llm_api_key") or "")
            or str(stored.get("anthropic_api_key") or "")
        )
        llm_base_url = pick("OPENREYNOLDS_LLM_BASE_URL", "llm_base_url") or None
        window = pick("OPENREYNOLDS_CONTEXT_WINDOW", "context_window")
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
            desk_model=pick(
                "OPENREYNOLDS_DESK_MODEL", "desk_model",
                preset.desk_model if preset else DEFAULT_DESK_MODEL,
            ),
            foamd_url=pick("FOAMD_URL", "foamd_url", DEFAULT_FOAMD_URL).rstrip("/"),
            foamd_api_key=pick("FOAMD_API_KEY", "foamd_api_key"),
            provider=provider,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            context_window=int(window) if window else 0,
            model=pick("OPENREYNOLDS_MODEL", "model", preset.model if preset else DEFAULT_MODEL),
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
        key = self.model_key_missing()
        if key:
            gaps.append(key)
        return gaps
