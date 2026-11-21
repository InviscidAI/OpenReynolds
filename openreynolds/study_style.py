"""How ambitious a study should be, and how much of it the person wants explained,
said in the person's own voice.

The agent never has a workflow imposed on it: the harness relays a standing note
(`Config.preferences`) at the start of every briefing, and what to do about it is the
model's call. Left alone, ambition wins -- a first "2D flow past an L-shape" became a
three-mesh transient refinement family with animation frames. So a study started
through an interface that offers a depth (the MCP servers' `depth`, `openreynolds
--depth`, the hosted app's study style) starts with one of these notes, and whatever
else the person wants the agent to know about how they like to work goes after it.
Nothing here is a system prompt; it is the person speaking.

`learn` and `beginner` are the same mechanism asked a different question. The roadmap
wanted a "learn mode" and a "beginner mode", and the shape that is NOT available is the
obvious one: a teaching checklist injected into the prompt, an explanation demanded
before each tool call, a gate that holds the solve until the person has answered a
question. That is the workflow injection `docs/design.md` section 1 forbids and
`tests/test_prompt.py` and `tests/test_briefing.py` reject. What a person can do is ask
for something, in their own words, at the start of a session -- which is what these two
are. They change what the person asked for and they gate nothing: `modes.py` remains the
only thing in the repository that holds a tool call, and it holds `job_start` and the
`checkpoint` question and nothing else.

Both are built on `quick`'s scope rather than `full`'s, because a run somebody is
reading along with is a run they want to see the whole of: a refinement family
explained at length is mostly explanation nobody reaches the end of.
"""

from __future__ import annotations

DEFAULT_STYLE = "quick"
NOTES_MAX = 2000

ALWAYS = (
    "\n\nWhatever the size of the study: before any solve, render the geometry and the "
    "mesh as images (the toolbox's geometry_view.py and render.py --scene mesh are there "
    "for this), look at those images yourself with read_file, run checkMesh, and tell me "
    "what you see and whether the mesh is fit for the question -- a solve on a mesh nobody "
    "looked at is wasted money. Leave the renders and analysis outputs in the workspace "
    "so they reach me."
)
"""Said with every style: look at the geometry and the mesh before solving. Still a
preference in the person's voice, not a harness rule -- the agent decides how."""

QUICK = (
    "Start with a quick look: one coarse mesh, a steady solve if the physics "
    "plausibly allows it, one figure and the headline numbers, using at most a few "
    "cores. Then tell me what a fuller study would involve and roughly what it would "
    "cost before doing any of it -- do not start refinement families, transient runs "
    "or animations unless I ask."
)
"""The scope half of `quick`, shared with the two teaching styles."""

LEARN = (
    "\n\nI am using this study to learn CFD, so I want to follow your reasoning rather "
    "than only read the answer. As you go, say what each choice is for and what the "
    "alternative was: why that solver and that turbulence model, what the boundary "
    "conditions you wrote mean physically, what the cell count and the near-wall "
    "spacing buy, what the residuals and the Courant number are telling you. When you "
    "write a dictionary, say what the entries in it do. When you read a number out of "
    "the run, say what would make it wrong and how you would tell. I would rather have "
    "the reasoning as you work than a lecture at the end, and I would rather you keep "
    "working than wait for me to reply to it."
)
"""`learn`: the person asks for the reasoning as prose beside the work.

The last sentence is load-bearing. A note asking to be taught reads as an invitation to
stop and check understanding, and a turn that ends on "does that make sense so far?"
ends the work -- the same failure `modes.AUTO_NO_QUESTIONS` was written for after a
full-auto session opened with a numbered list of questions and then meshed for seven
minutes while the list sat there (study 20260920-161908-c7ef)."""

BEGINNER = (
    "\n\nI am new to CFD and to OpenFOAM, so please write to me in plain language. "
    "Spell out a term the first time you use it -- y+, residual, Courant number, RANS, "
    "a patch, a cell zone -- in a few words, and keep using the plain words afterwards "
    "rather than the abbreviation. Tell me what the result means in the quantities I "
    "asked about, in the units I asked in, and tell me what it does not support: what "
    "this mesh and this model can and cannot be trusted for. If I have asked for "
    "something that does not make sense, say so and tell me what I probably meant. "
    "Keep going while you explain; I do not need to approve anything."
)
"""`beginner`: plain language, terms defined, and the limits of the answer stated.

"What it does not support" is the part worth asking for by name. The repository's own
record of how a validated case misses (`benchmarks/README.md`) is five ways in order of
how often they are blamed, and the two that dominate -- the reference value and the
reference's own uncertainty -- are both invisible to a reader who was handed a number
and no error bar."""

STYLES: dict[str, str] = {
    "quick": QUICK + ALWAYS,
    "full": (
        "Do a proper study: choose mesh resolution, steadiness and validation the way a "
        "careful engineer would, and say what you decided and why. Keep me posted on "
        "cost as you go." + ALWAYS
    ),
    "learn": QUICK + LEARN + ALWAYS,
    "beginner": QUICK + BEGINNER + ALWAYS,
}

TEACHING: tuple[str, ...] = ("learn", "beginner")
"""The styles that ask for something about the telling rather than about the scope.
Named so the MCP help text, the CLI option and the tests cannot drift apart on which
two they are."""


def normalize_style(value: str | None) -> str:
    """The style name as stored, or a ValueError naming the ones that exist."""
    name = str(value or DEFAULT_STYLE).strip().lower()
    if name not in STYLES:
        raise ValueError(f"style is one of {', '.join(STYLES)}")
    return name


def normalize_notes(value: str | None) -> str:
    """The free-text note, trimmed, or a ValueError past the cap."""
    text = str(value or "").strip()
    if len(text) > NOTES_MAX:
        raise ValueError(f"the note is at most {NOTES_MAX} characters")
    return text


def preferences_for(style: str, notes: str = "") -> str:
    """The standing note for a session: the style's text, then the person's own."""
    text = STYLES[normalize_style(style)]
    extra = (notes or "").strip()
    return f"{text}\n\n{extra}" if extra else text
