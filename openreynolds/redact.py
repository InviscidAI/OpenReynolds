"""Credentials out of a trace, before it is written to the upload queue.

The privacy policy: "OpenReynolds removes known credentials and secret-shaped values
before writing a trace to its upload queue, and we remove them again before storing it."
This is the first pass. It knows two things the service's pass cannot: the values
themselves -- this machine's model key and service key, and every environment variable
whose name says it is a secret -- and so it removes those exactly, whatever shape they
have. Then the same patterns the service applies, for a credential this machine never
held (one pasted into a case file, printed by a tool).

The service keeps the same pattern list in its own `app/redact.py`; a pattern added to
one belongs in the other. Removing credentials does not remove all personal or
engineering information from a trace, and the policy says so.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

MARK = "[redacted]"

PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bof_(?:live|test)_[A-Za-z0-9]{16,}"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"(?i)(aws_secret_access_key\s*[=:]\s*)[A-Za-z0-9/+=]{30,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer|basic|token)?\s*)[^\s\"']{8,}"),
    re.compile(r"(?i)\b([A-Z0-9_]*(?:API_KEY|APIKEY|SECRET|TOKEN|PASSWORD|PASSWD)[A-Z0-9_]*"
               r"\s*[=:]\s*[\"']?)[^\s\"']{6,}"),
)

_SECRET_NAME = re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)$")

_SHORTEST_KNOWN = 8
"""A known value shorter than this is not removed by value: `KEY=1` in the environment
would otherwise blank every `1` in a trace. Patterns still apply to it."""


def known_values(extra: Iterable[str] = (), environ: dict[str, str] | None = None) -> list[str]:
    """The secrets this machine holds: `extra` (the configured keys) and the value of
    every environment variable named like a secret. Longest first, so a key that
    contains another is removed whole."""
    env = os.environ if environ is None else environ
    values = {v for v in extra if v}
    values |= {v for k, v in env.items() if _SECRET_NAME.search(k) and v}
    return sorted((v for v in values if len(v) >= _SHORTEST_KNOWN), key=len, reverse=True)


class Redactor:
    """`text` and `deep` with this machine's known values removed before the patterns."""

    def __init__(self, known: Iterable[str] = ()):
        self.known = [v for v in known if v]

    def text(self, value: str) -> str:
        for secret in self.known:
            if secret in value:
                value = value.replace(secret, MARK)
        for pattern in PATTERNS:
            if pattern.groups:
                value = pattern.sub(lambda m: m.group(1) + MARK, value)
            else:
                value = pattern.sub(MARK, value)
        return value

    def deep(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, (list, tuple)):
            return [self.deep(v) for v in value]
        if isinstance(value, dict):
            return {self.text(k) if isinstance(k, str) else k: self.deep(v) for k, v in value.items()}
        return value
