"""Two named levels: how far to take a study, and when to come back and ask.

There used to be one control over how ambitious a run is, and it was a blank page.
`preferences.md` is free text, it is empty by default, and when it is empty nothing is
said -- so ambition wins. A modest question becomes a refinement family with transient
runs and animation frames, and the person finds out afterwards. Anything else took
knowing to write a standing note, and knowing what to put in it.

Whatever people write in that file is nearly always saying two separable things: how
much work the study is worth, and when to be interrupted. A blob of prose bundles them,
so the useful combinations are hard to ask for -- "be thorough, but check with me
before real money" and "just the quick version, don't ask" are both reasonable and
neither was easy to say. Two axes, three named levels each.

Three, not two: two is a switch, and the middle position is the one most studies
actually want. Three, not seven: a list nobody understands is worse than a blank page.
The six names are distinct across both axes, which is what lets `/level thorough never`
route each word to its own axis without saying which is which.

**These carry intent, not a ceiling.** A spend or wall-clock cap the harness checked
would be a budget the model must reason about, which `docs/design.md` §1 rules out by
name, and the CLI has no cost figure to check against anyway -- only the four token
counts `/status` already reports. So a level says what the work is worth and the model
decides what that means, exactly as it decides what a standing note means.

The text of each level is written to survive `tests/test_prompt.py`'s imperative
patterns, because it is relayed into the briefing: it says what the person wants, never
what the model has to do.
"""

from __future__ import annotations

AMBITION = {
    "sketch": (
        "the quick version -- one mesh, a headline number, and an honest word about "
        "what it rests on. Refinement families, transient runs and validation are "
        "more than this question is worth."
    ),
    "standard": (
        "one solid answer to the question as asked, checked as far as that answer "
        "needs and no further."
    ),
    "thorough": (
        "mesh independence, transient where the physics wants it, and a comparison "
        "against published data or an independent check. This one is worth real work."
    ),
}
"""How much work the study is worth."""

CONSENT = {
    "early": (
        "a word before solver time is spent -- the plan, the mesh, the numbers about "
        "to run."
    ),
    "costly": (
        "a word before anything expensive -- a long transient, a large mesh, a "
        "family of runs. The cheap version of a thing does not need the call."
    ),
    "never": (
        "no interruptions; a run that goes the whole way and reports at the end."
    ),
}
"""When to come back and ask. Its own axis rather than folded into ambition, because
the two really are independent: thorough work that checks in about money and quick
work that does not are both things people want, and one blob of prose can say neither
of them cleanly."""

DEFAULT_AMBITION = "standard"
DEFAULT_CONSENT = "costly"
"""What an empty `preferences.md` means now.

It used to mean nothing was said, which is not neutral -- it is the ambitious end,
picked by default and discovered afterwards. A default that is written down is a
default someone can disagree with."""

AXES = {"ambition": AMBITION, "consent": CONSENT}


def normalise(value: str, axis: str) -> str:
    """A recognised level for `axis`, or that axis's default.

    Anything can end up in a config file or an environment variable, and a typo in
    `OPENREYNOLDS_AMBITION` should not be a session that fails to start. The effective
    value is what `openreynolds config` prints and what `/level` shows, so a value that
    was silently corrected is still visible in one command."""
    known = AXES[axis]
    picked = (value or "").strip().lower()
    if picked in known:
        return picked
    return DEFAULT_AMBITION if axis == "ambition" else DEFAULT_CONSENT


def axis_of(word: str) -> str | None:
    """Which axis a bare level name belongs to, or `None` if it is not one.

    The six names are distinct, so `/level thorough never` needs no more grammar
    than this."""
    picked = (word or "").strip().lower()
    for axis, known in AXES.items():
        if picked in known:
            return axis
    return None


def chosen(text: str) -> tuple[str, str, list[str]]:
    """Route the words of `/level thorough never` to their axes.

    Returns the ambition asked for, the consent asked for, and anything that could not
    be used. A line that is not wholly understood applies none of itself: `/level
    thorogh never` shows the menu rather than silently changing consent and dropping
    the part they cared about.

    Two words on the same axis are that same case, not a last-one-wins. `/level
    standard thorough` is somebody correcting themselves without clearing the line, and
    quietly taking the second half would leave them looking at a confirmation naming
    one word out of the two they typed."""
    words = (text or "").split()
    picked = {"ambition": "", "consent": ""}
    unusable = []
    for word in words:
        axis = axis_of(word)
        if axis is None or picked[axis]:
            unusable.append(word)
        else:
            picked[axis] = word.strip().lower()
    if unusable:
        return "", "", unusable
    return picked["ambition"], picked["consent"], []


def describe(axis: str, level: str) -> str:
    """One level as "ambition, `thorough`: mesh independence, ..."."""
    return f"{axis}, `{level}`: {AXES[axis][level]}"


def _opened(text: str) -> str:
    """Sentence case without touching the rest.

    `str.capitalize()` lowercases everything after the first character, which would
    turn the level names inside the sentence into something nobody typed."""
    return text[:1].upper() + text[1:]


def briefing_lines(ambition: str, consent: str) -> list[str]:
    """What the briefing says about the two levels.

    Facts about what the person chose, in the register the standing note is relayed
    in. What to do about them stays the model's call, which is what keeps this inside
    the contract -- `tests/test_briefing.py` checks every combination of the two.
    """
    return [
        "The user picked two levels from a menu this tool offers, which is how they "
        "say how far to take a study and when to hear from you.",
        _opened(describe("ambition", normalise(ambition, "ambition"))),
        _opened(describe("consent", normalise(consent, "consent"))),
    ]


def spoken(ambition: str = "", consent: str = "") -> str:
    """A mid-study change, in the user's own voice.

    `/level thorough` is the person saying what they now want, so it reaches the model
    the way anything else they type does. The menu text goes along with it because the
    word on its own is only a label until the thread has seen what it stands for."""
    picked = []
    if ambition:
        picked.append(describe("ambition", ambition))
    if consent:
        picked.append(describe("consent", consent))
    return "Changing what I picked for the rest of this study. " + " ".join(
        _opened(text) for text in picked
    )


def menu_lines(ambition: str, consent: str) -> list[str]:
    """The menu and where the session currently stands on it, answered locally."""
    lines = [f"ambition is {normalise(ambition, 'ambition')}, consent is {normalise(consent, 'consent')}"]
    for axis, known in AXES.items():
        lines.append(f"{axis}:")
        for level, text in known.items():
            lines.append(f"  {level:9} {text}")
    lines.append("/level <one or two of those words> changes them for this study")
    return lines
