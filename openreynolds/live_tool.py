"""`live_sim`: the agent's door to the live backend, and the one tool that runs here.

Beside `tools.py` rather than in it, for one reason that is not tidiness: `tools.py` is
imported by every session, every CLI command and the MCP server, and it must not import
Warp. Keeping the handler here means the only thing `tools.py` owes the live backend is
one dict and one name in `_HANDLERS`, and the Warp import stays behind
`openreynolds.live.__getattr__` where `tests/test_live_optional.py` can check it is
still behind it.

## Why this tool does not run in the workspace

Every tool in `tools.py` except `fetch` and `cad` runs a command in the study's
workspace. This one cannot, and `DECISIONS_LIVE.md` D6 is the record: the workspace
image has no Warp, no CUDA toolchain and no GPU, and putting them there would cost the
1.2 GB-compressed pull to every user who never asks for a live run -- the same argument
`toolbox/lbm.py`'s docstring makes about OpenLB and Palabos, with more force, since a
CUDA runtime is bigger than either.

So the shape is `cad`'s: a host-side tool, offered only when it can actually do
something, writing its figures into the study's own directory on the user's machine so
that the pictures are where every other picture in the study already is.

## Why it is withheld rather than present-and-refusing

`tools_for`'s docstring carries the general argument. The particular one here: the
`[live]` extra is a CUDA toolchain and a robotics simulator, and most people who
install OpenReynolds will never have it. A tool in the list that can only answer "not
installed" costs the model a call to find that out, every session, on every machine
without a GPU. `offered()` is the question, asked once per request by `tools_for`, and
the answer cannot change mid-session because nobody pip-installs Warp into a running
interpreter.

The refusal text still exists and is still returned by `handler`, because the tool list
is not the only way a call arrives: a replayed transcript, a hand-written harness or an
MCP client can dispatch a name the list did not offer, and the answer to that has to be
the sentence naming the install command rather than a traceback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

TOOL: dict[str, Any] = {
    "name": "live_sim",
    "description": (
        "Run one of the live backend's flow cases and get its measurements back. This "
        "runs on the user's own machine, in this process, and NOT in the study's "
        "workspace: it is a GPU lattice Boltzmann solver that the optional [live] "
        "extra installs next to the user's own GPU, and the workspace image has "
        "neither it nor a GPU. With a CUDA GPU a case takes seconds to a minute; with "
        "no GPU the same kernels run on the CPU and the same case takes minutes, so "
        "resolution and steps are worth leaving alone unless there is a reason to move "
        "them. Each case measures itself against a published or committed reference "
        "and comes back with every check's measured value, its target, whether it "
        "passed, and the paths of the figures it drew. The figures are written into "
        "this study's own directory on the user's machine, beside the study's other "
        "renders, so nothing has to be copied out to see them. case=list returns the "
        "catalogue: every case name with one line about it, and runs nothing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "case": {
                "type": "string",
                "description": (
                    "A case name from the catalogue, or `list` to get the catalogue "
                    "itself and run nothing."
                ),
            },
            "resolution": {
                "type": "integer",
                "description": (
                    "Cells across the domain's longest side. The case's own default is "
                    "chosen to be worth running on a CPU; the cost of a run is this "
                    "cubed times the steps."
                ),
            },
            "steps": {
                "type": "integer",
                "description": (
                    "Lattice steps to run. The case's default is what its reference "
                    "measurement needs; fewer answers sooner and measures a flow that "
                    "has not settled yet."
                ),
            },
            "collision": {
                "type": "string",
                "enum": ["trt", "bgk", "regularized"],
                "description": (
                    "The collision operator. trt is two relaxation times with the "
                    "magic parameter that puts a straight no-slip wall on the link "
                    "midpoint at any viscosity, and is what the graded cases default "
                    "to; regularized is stable at a higher Reynolds number and loses "
                    "that wall property; bgk is the single-relaxation-time baseline, "
                    "for showing the difference."
                ),
            },
            "fluid": {
                "type": "string",
                "enum": ["air", "water"],
                "description": (
                    "Which fluid's viscosity and density to use, both at 20 C. The "
                    "case's own choice is usually the one its reference was measured "
                    "in."
                ),
            },
            "note": {
                "type": "string",
                "description": (
                    "Optional, free text: what this run is being asked for. It is "
                    "recorded with the result and changes nothing about the run, so a "
                    "figure found later says what question it was drawn for."
                ),
            },
        },
        "required": ["case"],
    },
}
"""The tool as the model is offered it, in the shape of `tools.py::TOOLS`' entries.

`case` is the only required field, and `list` is a value of it rather than a second
tool, because a catalogue and a run are the same question asked with different
specificity and two tools would cost two descriptions in every request's prefix."""

OVERRIDES = ("resolution", "steps", "collision", "fluid")
"""The arguments that are the case's to override. `note` is not one: it is recorded,
and passing it to `run_case` would make every case's signature carry a field no case
reads. `case` is the selector, not an override."""


def offered() -> bool:
    """Whether `live_sim` goes in the tool list this session.

    One question: is the extra importable. Not whether there is a GPU -- Warp's CPU
    backend runs every kernel in this extension and the cases are sized to be worth
    running on it, so a machine with the extra and no GPU can do real work slowly,
    which is a different thing from a machine that cannot do it at all. The
    description says which one the caller is on.
    """
    from . import live

    return live.available()


def handler(ctx: Any, args: dict[str, Any]) -> str:
    """Run a case here and hand back the report `openreynolds live run` would print.

    Text, never an attachment. The figures are already on the user's machine -- this
    tool wrote them there -- and a tool result carrying a picture of a plot the model
    cannot act on is a megabyte in every request after it. The paths are in the report,
    `read_file` attaches one on request, and `on_render` has already told the interface
    to show them.
    """
    from . import live

    if not live.available():
        try:
            live.require()
        except ImportError as exc:
            return str(exc)

    case = str(args.get("case") or "").strip()
    if not case:
        return "nothing was run: no case was named; `case=list` lists what can be run"
    if case == "list":
        return "\n".join(live.describe_cases())

    # Below the guards above, so that an answerable call is answered without reading
    # the case catalogue at all -- and so that a refusal does not depend on the module
    # whose absence it may be reporting.
    from .live import cases as live_cases

    overrides = {key: args[key] for key in OVERRIDES if args.get(key) is not None}
    result = live.run_case(case, out=str(figures_dir(ctx)), **overrides)
    for figure in result.get("figures") or ():
        rendered(ctx, str(figure))
    return live_cases.report(result)


def figures_dir(ctx: Any) -> Path:
    """Where this run's figures go: the study's renders directory, else here.

    `Store.renders_dir` is the flat, obvious home for pictures that the mirror already
    copies every workspace render into, so a live figure lands in the one place the
    person and the interface both already look. The fallback is the working directory
    rather than a temporary one: a context with no store is a harness, a test or an MCP
    client, and a figure written somewhere the caller cannot name is a figure nobody
    sees.
    """
    store = getattr(ctx, "store", None)
    directory = getattr(store, "renders_dir", None)
    if directory is None:
        return Path.cwd()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def rendered(ctx: Any, path: str) -> None:
    """Tell the interface about a figure, the way `tools._read_image` does.

    Swallowing the failure is the point: `on_render` is a nudge to a panel, and a
    panel that raises must not cost the model a result that was already computed.
    """
    hook = getattr(ctx, "on_render", None)
    if hook is None:
        return
    try:
        hook(path)
    except Exception:  # noqa: BLE001 - a nudge may not cost the model its answer
        pass
