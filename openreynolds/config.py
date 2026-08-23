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
    capture: bool = True
    studies_dir: Path = field(default_factory=lambda: Path.cwd() / "studies")

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

        return cls(
            foamd_url=pick("FOAMD_URL", "foamd_url").rstrip("/"),
            foamd_api_key=pick("FOAMD_API_KEY", "foamd_api_key"),
            anthropic_api_key=pick("ANTHROPIC_API_KEY", "anthropic_api_key"),
            llm_base_url=llm_base_url,
            model=pick("OPENREYNOLDS_MODEL", "model", DEFAULT_MODEL),
            effort=pick("OPENREYNOLDS_EFFORT", "effort", DEFAULT_EFFORT),
            max_tool_output=int(max_output) if max_output else DEFAULT_MAX_TOOL_OUTPUT,
            llm_timeout_s=float(timeout) if timeout else DEFAULT_LLM_TIMEOUT_S,
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
