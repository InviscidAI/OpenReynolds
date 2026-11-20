"""Credentials out of a trace, before it is written to the upload queue.

The privacy policy: "OpenReynolds removes known credentials and secret-shaped values
before writing a trace to its upload queue, and we remove them again before storing it."
This is the first pass. It knows two things the service's pass cannot: the values
themselves -- this machine's model key and service key, and every environment variable
whose name says it is a secret -- and so it removes those exactly, whatever shape they
have. Then the same patterns the service applies, for a credential this machine never
held (one pasted into a case file, printed by a tool), and a string under a key that
names a secret (`{"password": "..."}`) whole.

Every pattern runs in time linear in the text: a run of token characters is matched
only from its first character, and every open-ended part has an upper bound
(`tests/test_sharing.py` times each against text built to defeat it).

The service keeps the same pattern list in its own `app/redact.py`; a pattern added to
one belongs in the other. Removing credentials does not remove all personal or
engineering information from a trace, and the policy says so.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

MARK = "[redacted]"

SECRET_NAME = (r"(?:api[_\-]?key|access[_\-]?key|private[_\-]?key|secret|token|password"
               r"|passwd|passphrase|credentials?|cookie|_pwd(?![a-z])|(?<![a-z])pass(?![a-z]))")
"""What a name says when what it names is a secret."""

_NAMED = re.compile(r"(?i)" + SECRET_NAME + r"|authorization")

_PEM_BEGIN = re.compile(r"-----BEGIN [A-Z ]{0,40}PRIVATE KEY(?: BLOCK)?-----")
_PEM_END = re.compile(r"-----END [A-Z ]{0,40}PRIVATE KEY(?: BLOCK)?-----")

PATTERNS: tuple[re.Pattern[str], ...] = (
    # Model providers: Anthropic (sk-ant-), OpenAI (sk-, sk-proj-), OpenRouter (sk-or-).
    re.compile(r"(?<![A-Za-z0-9_\-])sk-[A-Za-z0-9_\-]{16,4096}"),
    # This service's own keys.
    re.compile(r"(?<![A-Za-z0-9_])of_(?:live|test)_[A-Za-z0-9]{16,4096}"),
    # AWS access key ids, and a secret key named as one.
    re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])"),
    re.compile(r"(?i)(aws_secret_access_key[\"']?\s{0,8}[=:]\s{0,8}[\"']?)[A-Za-z0-9/+=]{30,256}"),
    # GitHub, Google, Slack, Hugging Face, Stripe.
    re.compile(r"(?<![A-Za-z0-9_])(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,4096}"),
    re.compile(r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{30,4096}"),
    re.compile(r"(?<![A-Za-z0-9_\-])AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"(?<![A-Za-z0-9\-])xox[abprs]-[A-Za-z0-9\-]{10,4096}"),
    re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{30,4096}"),
    re.compile(r"(?<![A-Za-z0-9_])(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,4096}"),
    # JWTs: three base64url parts, the first a JSON header.
    re.compile(r"(?<![A-Za-z0-9_\-])eyJ[A-Za-z0-9_\-]{8,8192}\.[A-Za-z0-9_\-]{8,8192}"
               r"\.[A-Za-z0-9_\-]{8,8192}"),
    # An Authorization header's credentials, whatever the scheme, quoted or not.
    re.compile(r"(?i)((?<![A-Za-z0-9_])authorization[\"']?\s{0,8}[:=]\s{0,8}[\"']?"
               r"(?:(?:bearer|basic|token|digest)\s{1,8})?)[^\s\"']{8,8192}"),
    # A Cookie or Set-Cookie header: the whole value.
    re.compile(r"(?i)((?<![A-Za-z0-9_\-])(?:set-)?cookie[\"']?\s{0,8}[:=]\s{0,8}[\"']?)"
               r"[^\r\n\"']{6,8192}"),
    # A URL's password: scheme://user:password@host.
    re.compile(r"(?i)((?<![A-Za-z0-9+.\-])[a-z][a-z0-9+.\-]{0,30}://[^\s:/@\"'<>]{1,256}:)"
               r"[^\s/\"'<>]{1,256}(?=@[^\s@/\"'<>]{1,256}(?:[/:?#\s\"'<>]|$))"),
    # curl -u user:password (and --user), quoted with a space in it, or not.
    re.compile(r"((?<!\S)(?:-u|--user)(?:\s{1,8}|=)\"[^\s:\"]{1,256}:)[^\"\r\n]{1,512}"),
    re.compile(r"((?<!\S)(?:-u|--user)(?:\s{1,8}|=)'[^\s:']{1,256}:)[^'\r\n]{1,512}"),
    re.compile(r"((?<!\S)(?:-u|--user)(?:\s{1,8}|=)[\"']?[^\s:\"']{1,256}:)[^\s\"']{1,512}"),
    # NAME=value, NAME: value, "name": "value" (and JSON escaped inside a string,
    # \"name\": \"value\"), where the name says secret. A quoted value is taken
    # whole, spaces and all.
    re.compile(r"(?i)((?<![A-Za-z0-9_\-])[A-Za-z0-9_\-]{0,40}?" + SECRET_NAME +
               r"[A-Za-z0-9_\-]{0,40}\\?[\"']?\s{0,8}[=:]\s{0,8})"
               r"(\\\"[^\"\\\r\n]{6,1024}\\\"|\"[^\"\r\n]{6,1024}\"|'[^'\r\n]{6,1024}'"
               r"|[^\s\"',;&\\]{6,4096})"),
    # A command-line flag that names a secret, its value after a space:
    # `docker login --password hunter2`. (`--password=x` is the pattern above.)
    re.compile(r"(?i)((?<!\S)--?[a-z0-9\-]{0,40}?" + SECRET_NAME + r"[a-z0-9\-]{0,40}\s{1,8})"
               r"(\"[^\"\r\n]{1,1024}\"|'[^'\r\n]{1,1024}'|(?!-)[^\s\"']{1,4096})"),
)


def _pem(value: str) -> str:
    """PEM private keys, each block whole. Not one regular expression: a block's body is
    unbounded, and `BEGIN.*?END` rescans the rest of the text for every BEGIN."""
    if "-----BEGIN " not in value:
        return value
    out: list[str] = []
    at = 0
    while True:
        begin = _PEM_BEGIN.search(value, at)
        if not begin:
            break
        end = _PEM_END.search(value, begin.end())
        out += [value[at:begin.start()], MARK]
        if not end:
            # Cut short (by `head`, or by a cut before this): the key runs to the end.
            at = len(value)
            break
        at = end.end()
    out.append(value[at:])
    return "".join(out)


def _keep_name(match: re.Match[str]) -> str:
    """The name and separator kept, the value marked; a quoted value keeps its quotes."""
    value = match.group(2) if match.re.groups >= 2 else ""
    quote = "\\\"" if value.startswith("\\\"") else value[0] if value[:1] in ("\"", "'") else ""
    return match.group(1) + quote + MARK + quote


_SECRET_ENV = re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|_PASS|_PWD|CREDENTIALS?)$")
"""`_PWD`, not `PWD`: that one is the shell's working directory."""

_SHORTEST_KNOWN = 8
"""A known value shorter than this is not removed by value: `KEY=1` in the environment
would otherwise blank every `1` in a trace. Patterns still apply to it."""


def known_values(extra: Iterable[str] = (), environ: dict[str, str] | None = None) -> list[str]:
    """The secrets this machine holds: `extra` (the configured keys) and the value of
    every environment variable named like a secret. Longest first, so a key that
    contains another is removed whole."""
    env = os.environ if environ is None else environ
    values = {v for v in extra if v}
    values |= {v for k, v in env.items() if _SECRET_ENV.search(k) and v}
    return sorted((v for v in values if len(v) >= _SHORTEST_KNOWN), key=len, reverse=True)


def secret_name(name: str) -> bool:
    """Whether a key in a JSON object names a secret (`{"api_key": "..."}`)."""
    return len(name) <= 80 and bool(_NAMED.search(name))


def _marks_whole(key: Any, value: Any) -> bool:
    """A value under a key that names a secret goes whole, whatever it is: a string, a
    number, a list, an object (`{"credentials": {"user": ..., "pass": ...}}`)."""
    return (isinstance(key, str) and secret_name(key) and value not in (None, "", [], {})
            and not isinstance(value, bool))


def _named_pair(value: dict) -> bool:
    """`{"name": "API_KEY", "value": "..."}`: a secret named by a sibling field, as
    environment lists and key-value settings are written."""
    name = value.get("name", value.get("key"))
    return isinstance(name, str) and secret_name(name) and "value" in value


class Redactor:
    """`text` and `deep` with this machine's known values removed before the patterns."""

    def __init__(self, known: Iterable[str] = ()):
        self.known = [v for v in known if v]

    def text(self, value: str) -> str:
        for secret in self.known:
            if secret in value:
                value = value.replace(secret, MARK)
        value = _pem(value)
        for pattern in PATTERNS:
            value = pattern.sub(_keep_name, value) if pattern.groups else pattern.sub(MARK, value)
        return value

    def deep(self, value: Any) -> Any:
        """`text` over every string, keys included; a string under a key that names a
        secret is marked whole, whatever it looks like."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, (list, tuple)):
            return [self.deep(v) for v in value]
        if isinstance(value, dict):
            pair = _named_pair(value)
            return {(self.text(k) if isinstance(k, str) else k):
                    (MARK if _marks_whole(k, v) or (pair and k == "value" and v is not None)
                     else self.deep(v))
                    for k, v in value.items()}
        return value
