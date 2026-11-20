"""How ambitious a study should be, said in the person's own voice.

The agent never has a workflow imposed on it: the harness relays a standing note
(`Config.preferences`) at the start of every briefing, and what to do about it is the
model's call. Left alone, ambition wins -- a first "2D flow past an L-shape" became a
three-mesh transient refinement family with animation frames. So a study started
through an interface that offers a depth (the MCP servers' `depth`, the hosted app's
study style) starts with one of two notes, and whatever else the person wants the agent
to know about how they like to work goes after it. Nothing here is a system prompt; it
is the person speaking.
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

STYLES: dict[str, str] = {
    "quick": (
        "Start with a quick look: one coarse mesh, a steady solve if the physics "
        "plausibly allows it, one figure and the headline numbers, using at most a few "
        "cores. Then tell me what a fuller study would involve and roughly what it would "
        "cost before doing any of it — do not start refinement families, transient runs "
        "or animations unless I ask." + ALWAYS
    ),
    "full": (
        "Do a proper study: choose mesh resolution, steadiness and validation the way a "
        "careful engineer would, and say what you decided and why. Keep me posted on "
        "cost as you go." + ALWAYS
    ),
}


def normalize_style(value: str | None) -> str:
    """The style name as stored, or a ValueError naming the two that exist."""
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
