"""A small POSIX shell, enough to RUN the two /proc probes against a made-up container.

The probes are the only part of the client that decides, from inside the workspace,
whether somebody else's work is in flight -- `cli._NEIGHBOUR_PROBE` before a shutdown
and `stopping._OWN_PROBE` before a kill. Both are shell text sent to a container, so
the fakes in the suite answered them with canned rows and never evaluated the `case`
patterns at all. That is how a probe that matched NOTHING in production for its whole
life passed every test it had: inside a Modal Sandbox `/work` is a symlink to
`/__modal/volumes/vo-<id>`, `/proc/<pid>/cwd` is a kernel magic link that answers with
the physical path, and the scripts were comparing that against the literal `/work`.

So this evaluates the real script text. It understands only the constructs the two
probes use and raises on anything else, deliberately: a probe that grows a construct
this does not know must fail loudly here rather than quietly stop being tested.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
from dataclasses import dataclass, field

PROBE_SHELL_PID = "9001"
"""The pid the probe's own shell believes it has (`$$`)."""

PROBE_HELPERS = (("9002", "cat"), ("9003", "readlink"))
"""What the probe spawns per process it looks at. They are real processes with real
working directories, and a probe that does not exclude them reports itself."""


@dataclass
class Process:
    pid: str
    comm: str
    cwd: str
    """As `/proc/<pid>/cwd` answers it: the PHYSICAL path, whatever the process typed
    to get there."""


@dataclass
class Machine:
    """A container with symlinked mounts and a process table.

    `links` maps a logical prefix to the physical one it resolves to, which is the one
    thing about a Modal Sandbox that broke both probes: `{"/work": "/__modal/volumes/
    vo-x"}`.
    """

    links: dict[str, str] = field(default_factory=dict)
    processes: list[Process] = field(default_factory=list)
    cwd: str = "/work"
    """Where the probe's own shell starts, i.e. where the exec route dropped it."""

    def resolve(self, path: str) -> str:
        """`readlink -f`: the physical spelling of a path, following the mounts."""
        for logical, physical in sorted(self.links.items(), key=lambda kv: -len(kv[0])):
            if path == logical:
                return physical
            if path.startswith(logical.rstrip("/") + "/"):
                return physical.rstrip("/") + path[len(logical.rstrip("/")):]
        return path

    def run(self, script: str) -> str:
        """The script's stdout, with the probe's own shell and helpers in the table."""
        return _Run(self, script).output()


_ASSIGN_LITERAL = re.compile(r"^(\w+)=((?!\$\().*)$")
_ASSIGN_RESOLVE = re.compile(
    r'^(\w+)=\$\(readlink -f "\$(\w+)" 2>/dev/null \|\| echo "\$(\w+)"\)$'
)
_CASE = re.compile(r'^case "\$w" in (.+?)\) (continue|printf .*?) ;; esac$')
_CD = re.compile(r"^cd (\S+)")
_SKIP_SELF = re.compile(r'^\[ "\$p" = "\$self" \] && continue$')
_IGNORED = re.compile(
    r'^(self=\$\$'  # the shell's own pid, which `PROBE_SHELL_PID` already stands for
    r'|p=\$\{d#/proc/\}'
    r'|c=\$\(cat "\$d/comm" 2>/dev/null\) \|\| continue'
    r'|w=\$\(readlink "\$d/cwd" 2>/dev/null\) \|\| continue'
    r'|for d in /proc/\[0-9\]\*; do'
    r'|done)$'
)


class _Run:
    """One evaluation. Split out so `Machine` stays the description of a container."""

    def __init__(self, machine: Machine, script: str):
        self.machine = machine
        self.vars: dict[str, str] = {"self": PROBE_SHELL_PID}
        self.arms: list[tuple[list[str], str]] = []
        self.skip_self = False
        self.cwd = machine.cwd
        for line in script.splitlines():
            self._read(line.strip())

    def _read(self, line: str) -> None:
        if not line or _IGNORED.match(line):
            return
        if _SKIP_SELF.match(line):
            self.skip_self = True
            return
        cd = _CD.match(line)
        if cd:
            self.cwd = cd.group(1)
            return
        resolved = _ASSIGN_RESOLVE.match(line)
        if resolved:
            name, source, fallback = resolved.groups()
            assert source == fallback, "the fallback must be the path it failed on"
            self.vars[name] = self.machine.resolve(self.vars[source])
            return
        literal = _ASSIGN_LITERAL.match(line)
        if literal:
            name, value = literal.groups()
            self.vars[name] = "".join(shlex.split(value)) if value else ""
            return
        case = _CASE.match(line)
        if case:
            patterns, action = case.groups()
            self.arms.append(([p for p in patterns.split("|")], action))
            return
        raise AssertionError(
            f"this shell does not know {line!r}; teach it, or the probe is untested"
        )

    def _expand(self, pattern: str) -> str:
        out = pattern.strip()
        for name, value in self.vars.items():
            out = out.replace(f'"${name}"', value).replace(f"${name}", value)
        return out.replace("'", "")

    def output(self) -> str:
        table = list(self.machine.processes) + [
            Process(PROBE_SHELL_PID, "bash", self.cwd),
            *(Process(pid, comm, self.cwd) for pid, comm in PROBE_HELPERS),
        ]
        lines = []
        for process in table:
            if self.skip_self and process.pid == PROBE_SHELL_PID:
                continue
            for patterns, action in self.arms:
                if not any(
                    fnmatch.fnmatchcase(process.cwd, self._expand(p)) for p in patterns
                ):
                    continue
                if action == "continue":
                    break
                lines.append(f"{process.pid} {process.comm}")
                break
        return "".join(line + "\n" for line in lines)
