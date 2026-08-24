"""Options that exist and do nothing.

Four separate times in this project something was defined, documented, listed in
`--help`, and never read: `--fresh` (so no persona run was ever isolated), `--turns`
(clipped by `min()` against a default, so asking for more silently gave less), the
loop's interject callback (so nothing typed mid-turn ever reached the model), and a
line telling the user where their files were.

Every one of them looked finished from the outside. None was caught by a test, because
a test that never exercises a flag cannot tell a flag from a comment. These are the
structural check: whatever is declared has to be read somewhere.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "openreynolds" / "cli.py"
USER_TEST = ROOT / "scripts" / "user_test.py"


def parsed(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def names_used(node: ast.AST) -> set[str]:
    """Every bare name and attribute read anywhere inside a node."""
    used = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            used.add(child.id)
        elif isinstance(child, ast.Attribute):
            used.add(child.attr)
    return used


def click_commands(module: ast.Module):
    """Each function carrying click decorators, with the parameters click will pass."""
    for node in module.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = [
            d
            for d in node.decorator_list
            if isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr in ("option", "argument", "command", "group")
        ]
        if decorators:
            params = [a.arg for a in node.args.args if a.arg != "ctx"]
            yield node.name, params, node


@pytest.mark.parametrize(
    "command,params,body",
    list(click_commands(parsed(CLI))),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_every_command_line_option_is_read(command, params, body):
    """A click option arrives as a parameter. A parameter nothing reads is an option
    that does nothing, and `--help` will still advertise it."""
    used = names_used(body)
    unread = [p for p in params if p not in used]
    assert not unread, f"{command}() never reads {', '.join(unread)}"


def argparse_dests(module: ast.Module) -> list[str]:
    """The destination name of every `parser.add_argument(...)` in a module."""
    dests = []
    for node in ast.walk(module):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        explicit = next(
            (kw.value.value for kw in node.keywords if kw.arg == "dest"),
            None,
        )
        if explicit:
            dests.append(explicit)
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
        longest = max((f for f in flags if f.startswith("--")), key=len, default=None)
        if longest:
            dests.append(longest.lstrip("-").replace("-", "_"))
    return dests


@pytest.mark.parametrize("dest", argparse_dests(parsed(USER_TEST)))
def test_every_harness_option_is_read(dest):
    """`--fresh` was defined, documented, shown in --help, and never read. Four
    persona runs shared a workspace because of it."""
    source = USER_TEST.read_text(encoding="utf-8")
    assert f"args.{dest}" in source, f"--{dest.replace('_', '-')} is declared and never read"


def test_the_check_would_have_caught_the_one_that_got_through():
    """A guard nobody has seen fail is a guard nobody should trust."""
    module = ast.parse(
        "def main():\n"
        "    parser.add_argument('--fresh', action='store_true')\n"
        "    parser.add_argument('--turns', type=int)\n"
    )
    assert argparse_dests(module) == ["fresh", "turns"]

    unwired = ast.parse("def cmd(path, study_id):\n    return path\n")
    name, params, body = next(
        (n.name, [a.arg for a in n.args.args], n)
        for n in unwired.body
        if isinstance(n, ast.FunctionDef)
    )
    assert [p for p in params if p not in names_used(body)] == ["study_id"]
