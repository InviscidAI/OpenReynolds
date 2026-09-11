"""The cell log: what the desk built, in the order it built it, as a script.

The accepted cells concatenated **are** the deliverable. Nothing here keeps a second
copy of the build beside the artifact, and nothing writes the artifact at the end to
satisfy a check -- `script()` is the cells, and the cells are what ran.

**The append invariant is not "the cell succeeded".** A cell can run perfectly in a
live session and fail in sequence: it reached for a binding from a cell that was never
accepted, or its effect depends on having been run once already. The invariant is that
*the concatenation still runs*, and it is enforced in two layers that catch different
things:

* here, on accept, a **static check** -- walk the cell's AST, collect the names it uses
  freely, and compare them against the names bound by the cells already accepted, plus
  imports and builtins. Cheap, immediate, and exact on the common case.
* at the finish, the **full replay** -- `cad/check.py` runs the concatenation from empty
  and compares what it makes against what is on disk. That is what catches
  non-idempotence, which no static analysis sees.

Replaying on every accept would be the obvious third option and is the wrong one: it is
O(n^2) against a script whose every run costs seconds to minutes, so twenty cells would
spend the whole wall-clock budget re-running themselves.

**A refused cell has already run.** It was sent to the kernel and it worked; refusing it
means the kernel now holds state the log does not. That divergence is correct -- the log
is truth, the kernel is convenience -- and dangerous if silent, because the desk would
go on building against a binding that will not exist at replay. So the refusal says so,
in as many words, and says what to do about it.

**Two things in a cell are kernel and not python**, and both are refused here rather
than at the finish, where the same failure arrives as a syntax error in a replay
minutes later with nothing to read. `!blockMesh` and `%matplotlib` are IPython; a name
IPython injects, `display` above all, is in the session and not in a file. Each gets
its own refusal naming the substitute, because the brief invites both idioms and the
reason they cannot be in the script is not that they are wrong.

**And one thing the check cannot read at all:** `from build123d import *` is how
build123d is written, and no static pass can say what a star import binds. Claiming
anyway would refuse `Box` and every other correct name. So under a star import the
check narrows to what it is still sure of -- a name some earlier cell bound and no
accepted cell did, which is never a name a module supplied -- which is why every cell
passes through `propose`, accepted or not.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass

SCRIPT_HEADER = '''"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""'''

REASONING_LINES = 12
"""How much of the desk's prose travels into the file above its cell. Enough for the
argument, not so much that the script is mostly commentary."""

REASONING_CHARS = 100
"""Where a single line of that prose is cut. A wrapped paragraph, not an essay."""

_BUILTINS = frozenset(dir(builtins)) | {
    "__name__", "__file__", "__doc__", "__builtins__", "__spec__", "__package__",
    "__loader__", "__debug__",
}
"""In scope in any module, so never a reason to refuse a cell."""

FROM_THE_KERNEL = frozenset({"display", "get_ipython", "In", "Out", "_ih", "_oh", "_dh"})
"""Names IPython injects and a plain `python3 build.py` does not have.

Refusing them is correct -- the script would raise `NameError` at replay -- but it is
worth its own sentence, because the difference between the session and the file is
exactly what this whole gate is about and `display(fig)` is a thing the brief invites.
"""

_MAGIC = ("!", "%")
"""How a line of IPython that is not python starts. `!blockMesh` and `%matplotlib` are
the kernel's conveniences; the script is run as a script, where they do not parse."""


@dataclass
class Cell:
    """One block the desk sent, with the words it sent it in."""

    source: str
    reasoning: str = ""
    """Its own prose, carried with the action rather than discarded."""
    accepted: bool = False


class CellLog:
    """The accepted cells, and the gate in front of them."""

    def __init__(self) -> None:
        self._cells: list[Cell] = []
        self._bound: set[str] = set()
        """Every name an accepted cell binds at module level. What the static check
        measures the next cell against."""
        self._proposed: set[str] = set()
        """Every name any cell bound, accepted or not. The difference between this and
        `_bound` is precisely the set of names that are live in the kernel and absent
        from the script, which is the divergence this whole file exists to police."""
        self._starred = False
        """Whether anything in scope did `from x import *`.

        It matters because it is the one construct that makes the static check unable to
        say what is bound -- and `from build123d import *` is how build123d is written,
        so pretending otherwise would refuse most correct cells. Under a star import the
        check narrows to what it can still be sure of: a name some earlier cell bound
        and no accepted cell did. A star import cannot have supplied that."""

    # -- the log ---------------------------------------------------------------

    def propose(self, source: str, reasoning: str = "") -> Cell:
        """A cell as sent, before anything is decided about it.

        Every cell passes through here, which is what lets the log know the names that
        exist in the session but not in the file.
        """
        cell = Cell(source=(source or "").strip("\n"), reasoning=(reasoning or "").strip())
        self._proposed |= bound_names(cell.source)
        return cell

    def accept(self, cell: Cell) -> str:
        """Append it, or return why it cannot be appended -- `""` when accepted.

        Same shape as `parse_action`'s `(cmd, complaint)`: the string is handed
        straight to the desk and is written for it to act on.
        """
        try:
            missing = free_names(cell.source, self._bound)
        except SyntaxError as exc:
            magic = _magic_lines(cell.source)
            if magic:
                return _magic_refusal(magic)
            return (
                f"That cell is not in the script: it does not parse ({exc.msg} on line "
                f"{exc.lineno}). The script is the accepted cells concatenated and run "
                "from empty, so a cell python cannot read cannot be in it. Re-emit it."
            )
        starred = self._starred or _stars(cell.source)
        if starred:
            missing = [name for name in missing
                       if name in self._proposed or name in FROM_THE_KERNEL]
        if missing:
            return _refusal(missing)
        cell.accepted = True
        self._cells.append(cell)
        self._bound |= bound_names(cell.source)
        self._starred = starred
        return ""

    def cells(self) -> list[Cell]:
        return list(self._cells)

    def bound(self) -> set[str]:
        """The names an accepted cell binds. Read by the desk's own reporting."""
        return set(self._bound)

    def script(self) -> str:
        """The concatenation: the deliverable, runnable top to bottom from empty."""
        if not self._cells:
            return ""
        parts = [SCRIPT_HEADER]
        for number, cell in enumerate(self._cells, 1):
            block = [f"# -- cell {number} " + "-" * max(4, 74 - len(str(number)))]
            block += _as_comment(cell.reasoning)
            block.append(cell.source)
            parts.append("\n".join(block))
        return "\n\n".join(parts) + "\n"


def _as_comment(reasoning: str) -> list[str]:
    lines: list[str] = []
    for line in (reasoning or "").splitlines():
        line = line.strip()
        if not line:
            continue
        lines.append("# " + line[:REASONING_CHARS])
        if len(lines) >= REASONING_LINES:
            break
    return lines


def _magic_lines(source: str) -> list[str]:
    """The `!` and `%` lines in a cell that does not parse.

    Only asked of a cell that already failed to parse, so a `%` at the start of a
    continuation line inside a string is not mistaken for a magic -- if the cell parses,
    the script runs, and there is nothing here to say.
    """
    return [line.strip() for line in (source or "").splitlines()
            if line.strip()[:1] in _MAGIC]


def _magic_refusal(magic: list[str]) -> str:
    """`!blockMesh` ran; it is still not python.

    Worth its own refusal rather than a parse error, because the brief hands the desk
    this idiom and the reason it cannot be in the file is not that it is wrong -- it is
    that the file is run as a file, by somebody who was not there, with no kernel.
    """
    shown = "`" + "`, `".join(magic[:3]) + "`"
    return (
        f"That cell ran, and what it did is done -- but it is not in the script, because "
        f"{shown} is IPython, not python. The script is run as `python3 build.py`, with "
        "no kernel under it, and a line starting with `!` or `%` is a syntax error "
        "there. Re-emit the cell with `subprocess.run([...], check=True)` in place of "
        "the `!` line -- that runs the same command here and still runs on replay."
    )


def _stars(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.ImportFrom)
               and any(alias.name == "*" for alias in node.names)
               for node in ast.walk(tree))


def _refusal(missing: list[str]) -> str:
    """What the desk is told when a cell ran and still cannot go in the script."""
    kernel_only = [name for name in missing if name in FROM_THE_KERNEL]
    if kernel_only and set(missing) == set(kernel_only):
        names = ", ".join(f"`{name}`" for name in kernel_only)
        return (
            f"That cell ran, because {names} is a name IPython puts in your session. It "
            "is not in the script: the script is run as `python3 build.py`, with no "
            f"kernel under it, and {names} does not exist there. Import what you need "
            "(`from IPython.display import display`) or say it another way -- a figure "
            "shows here by being the cell's last expression, and that costs the script "
            "nothing."
        )
    names = ", ".join(f"`{name}`" for name in missing)
    many = len(missing) > 1
    return (
        "That cell ran. Its result is in your kernel session and everything it bound is "
        "still live there, so you can keep measuring it. It is **not** in the script, "
        f"because it uses {'the names' if many else 'the name'} {names}, which no "
        f"accepted cell binds. The script is the accepted cells concatenated and re-run "
        f"from empty at the finish, and {'those names' if many else 'that name'} would "
        "not exist there -- usually because it came from a cell that errored or was "
        f"itself refused, and neither of those enters the log. Re-emit this cell with "
        f"{names} defined inside it, or send the cell that defines "
        f"{'them' if many else 'it'} first and have that one accepted."
    )


# -- the static check ----------------------------------------------------------


def bound_names(source: str) -> set[str]:
    """Every name this cell binds at module level, imports included."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return _block_bindings(tree.body)


def free_names(source: str, known: set[str] | frozenset[str]) -> list[str]:
    """The names this cell uses and nothing in scope binds, sorted.

    In scope: the names `known` carries (what accepted cells bound), everything this
    cell binds itself, every builtin, and every local of whatever function, class or
    comprehension the use sits inside. Anything else is a name that will not exist when
    the concatenation is run from empty.
    """
    tree = ast.parse(source)
    visible = set(known) | _block_bindings(tree.body)
    free: set[str] = set()
    for node in tree.body:
        _scan(node, visible, free)
    return sorted(free)


def _scan(node: ast.AST, visible: set[str], free: set[str]) -> None:
    """Walk one node, adding whatever it reads and nothing in `visible` binds.

    Each construct that opens a scope is named here rather than walked through, because
    a comprehension's target and a function's parameters are exactly the names that
    would otherwise be reported as missing from a perfectly good cell.
    """
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id not in visible \
                and node.id not in _BUILTINS:
            free.add(node.id)
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for decorator in node.decorator_list:
            _scan(decorator, visible, free)
        _scan_signature(node.args, visible, free)
        if node.returns is not None:
            _scan(node.returns, visible, free)
        inner = visible | _arg_names(node.args) | _block_bindings(node.body) | {node.name}
        for statement in node.body:
            _scan(statement, inner, free)
        return
    if isinstance(node, ast.Lambda):
        _scan_signature(node.args, visible, free)
        _scan(node.body, visible | _arg_names(node.args), free)
        return
    if isinstance(node, ast.ClassDef):
        for decorator in node.decorator_list:
            _scan(decorator, visible, free)
        for base in node.bases:
            _scan(base, visible, free)
        for keyword in node.keywords:
            _scan(keyword.value, visible, free)
        inner = visible | _block_bindings(node.body)
        for statement in node.body:
            _scan(statement, inner, free)
        return
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        inner = set(visible)
        for generator in node.generators:
            inner |= _target_names(generator.target)
        for generator in node.generators:
            _scan(generator.iter, inner, free)
            for condition in generator.ifs:
                _scan(condition, inner, free)
        if isinstance(node, ast.DictComp):
            _scan(node.key, inner, free)
            _scan(node.value, inner, free)
        else:
            _scan(node.elt, inner, free)
        return
    for child in ast.iter_child_nodes(node):
        _scan(child, visible, free)


def _scan_signature(args: ast.arguments, visible: set[str], free: set[str]) -> None:
    """Defaults and annotations are evaluated where the `def` is, not inside it."""
    for default in [*args.defaults, *[d for d in args.kw_defaults if d is not None]]:
        _scan(default, visible, free)
    for arg in _all_args(args):
        if arg.annotation is not None:
            _scan(arg.annotation, visible, free)


def _all_args(args: ast.arguments) -> list[ast.arg]:
    found = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            found.append(extra)
    return found


def _arg_names(args: ast.arguments) -> set[str]:
    return {arg.arg for arg in _all_args(args)}


def _block_bindings(body: list[ast.stmt]) -> set[str]:
    found: set[str] = set()
    for statement in body:
        found |= _statement_bindings(statement)
    return found


def _statement_bindings(node: ast.stmt) -> set[str]:
    """Names this statement binds in the scope it sits in.

    Nested function and class bodies are not descended into -- what they bind is theirs
    -- but every block that shares the enclosing scope is, because a name bound inside
    an `if` or a `for` is bound afterwards.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    found: set[str] = set()
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            if name != "*":
                # What a star import binds cannot be read off the source. `accept`
                # narrows what it claims instead of guessing at a name here.
                found.add(name)
        return found
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return set(node.names)
    if isinstance(node, ast.Assign):
        for target in node.targets:
            found |= _target_names(target)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        found |= _target_names(node.target)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        found |= _target_names(node.target)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                found |= _target_names(item.optional_vars)
    if isinstance(node, ast.Try):
        for handler in node.handlers:
            if handler.name:
                found.add(handler.name)
    found |= _walrus_names(node)
    found |= _match_names(node)
    for field in ("body", "orelse", "finalbody"):
        for child in getattr(node, field, None) or []:
            if isinstance(child, ast.stmt):
                found |= _statement_bindings(child)
    for handler in getattr(node, "handlers", None) or []:
        found |= _block_bindings(handler.body)
    for case in getattr(node, "cases", None) or []:
        found |= _block_bindings(case.body)
    return found


def _target_names(node: ast.AST) -> set[str]:
    """The plain names an assignment target binds.

    `a[i] = 1` and `a.b = 1` bind nothing: in both the `a` is a load, which is what
    makes walking for `Store` contexts the right question rather than a shortcut.
    """
    return {
        sub.id for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del))
    }


def _walrus_names(node: ast.AST) -> set[str]:
    return {
        sub.target.id for sub in ast.walk(node)
        if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name)
    }


def _match_names(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.MatchAs, ast.MatchStar)) and sub.name:
            found.add(sub.name)
        elif isinstance(sub, ast.MatchMapping) and sub.rest:
            found.add(sub.rest)
    return found
