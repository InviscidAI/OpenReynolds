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
import re
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


# -- surfaces that nothing calls -----------------------------------------------


PACKAGE = ROOT / "openreynolds"


def protocol_methods(name: str) -> list[str]:
    module = parsed(PACKAGE / "view.py")
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
    raise AssertionError(f"no class named {name}")


def package_source(*skip: str) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.name not in skip
    )


@pytest.mark.parametrize("method", protocol_methods("View"))
def test_every_view_method_is_called_by_something(method):
    """A view method nothing calls is a thing the interface can draw and the session
    never asks for -- and both implementations still have to carry it."""
    callers = package_source("view.py", "tui.py")
    assert re.search(rf"\.view\.{method}\(|\bview\.{method}\(", callers), (
        f"View.{method} is declared and never called"
    )


def dataclass_fields(path: Path, class_name: str) -> list[str]:
    """Annotated attributes on a class, and not the annotated locals in its methods."""
    for node in ast.walk(parsed(path)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and not item.target.id.startswith("_")
            ]
    raise AssertionError(f"no class named {class_name}")


@pytest.mark.parametrize("field", dataclass_fields(PACKAGE / "config.py", "Config"))
def test_every_config_setting_is_read(field):
    """A setting nobody reads is a promise to the user that nothing keeps."""
    readers = package_source("config.py")
    assert re.search(rf"\bcfg\.{field}\b|\bconfig\.{field}\b|\bself\.cfg\.{field}\b", readers), (
        f"Config.{field} is declared and never read"
    )


@pytest.mark.parametrize("field", dataclass_fields(PACKAGE / "tools.py", "ToolContext"))
def test_every_tool_context_field_is_read(field):
    """`ToolContext.view` was added, wired into the tools, and then not passed in by
    the session for a whole release -- so the jobs panel never updated."""
    readers = package_source()
    assert re.search(rf"\bctx\.{field}\b", readers), (
        f"ToolContext.{field} is declared and never read"
    )


def backend_methods() -> list[str]:
    module = parsed(PACKAGE / "backend" / "base.py")
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == "Backend":
            return [
                n.name
                for n in node.body
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
            ]
    raise AssertionError("no Backend protocol")


@pytest.mark.parametrize("method", backend_methods())
def test_every_backend_method_is_used(method):
    """The protocol is the whole independence story: every method on it is a promise
    a future local backend has to keep. One nothing calls is a promise for nothing."""
    callers = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(PACKAGE.rglob("*.py"))
        if "backend" not in path.parts
    )
    assert re.search(rf"backend\.{method}\(", callers), (
        f"Backend.{method} is on the protocol and nothing above it calls it"
    )


def test_the_documents_do_not_carry_a_test_count():
    """A number that has to be edited by hand goes stale, and a stale count in a
    document about verification is worse than no count at all."""
    for document in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        text = document.read_text(encoding="utf-8")
        stale = re.findall(r"\b\d{3,4} (?:unit )?tests\b", text)
        assert not stale, f"{document.name} claims {stale}, which nothing keeps true"


# -- the scripts nobody runs until they matter ---------------------------------


SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_script_parses(script):
    """These only run when someone has credentials, which is the worst moment to
    discover a typo in one."""
    ast.parse(script.read_text(encoding="utf-8"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_script_says_how_to_run_it(script):
    doc = ast.get_docstring(ast.parse(script.read_text(encoding="utf-8")))
    assert doc, f"{script.name} has no docstring"
    if script.name == "personas.py":
        return  # imported, never invoked
    assert "python" in doc, f"{script.name} does not show how to run it"


def cli_command_names() -> list[str]:
    """Every subcommand click will register, from the decorators themselves."""
    names = []
    for node in parsed(CLI).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "command"
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
            ):
                names.append(decorator.args[0].value)
    return names


@pytest.mark.parametrize("command", cli_command_names())
def test_every_command_is_in_the_readme(command):
    """`studies` shipped undocumented. A command nobody is told about is a command
    nobody uses, which is the same outcome as not having written it."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"openreynolds {command}" in readme, (
        f"`openreynolds {command}` exists and the README never mentions it"
    )
