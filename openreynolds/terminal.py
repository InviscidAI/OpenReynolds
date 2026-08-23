"""Making stdout survive what gets written to it.

A CFD conversation is full of Greek letters, superscripts and arrows, and stdout is not
always UTF-8 -- a redirect to a file on Windows lands on cp1252, where a single sigma
raises mid-stream and takes the session with it.

This lives on its own rather than inside the CLI because the test harness prints the
same text through the same kind of pipe and hit the same crash. One copy, so the two
cannot drift apart and be fixed separately.
"""

from __future__ import annotations

import sys

ATTEMPTS = (
    # UTF-8 first, so a redirected log keeps mu and y+ intact.
    {"encoding": "utf-8", "errors": "replace"},
    # Then replacement alone, so an undecodable character still cannot raise.
    {"errors": "replace"},
)


def tolerant_stdout() -> None:
    """Keep an undecodable character from killing the process."""
    for stream in (sys.stdout, sys.stderr):
        for attempt in ATTEMPTS:
            try:
                stream.reconfigure(**attempt)
                break
            except (AttributeError, ValueError, OSError, LookupError):
                continue
