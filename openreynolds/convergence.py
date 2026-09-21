"""What a residual that stopped falling means, and the words for it.

The complaint that produced this file, from the owner, after five solving studies in
three days (2026-09-19 to 09-21, workspace 35c9f018): "In each solve I just wanted a
quick look, not a mesh independence study, why is it that every time it kept telling
me it couldn't converge??? Is the solver like not useful??? Nothing converges is it?"
And, mid-investigation: "if it's never gonna converge it doesn't mean it's physically
inaccurate! But the way it's phrased it's always so negative. There are times when the
case just cannot work, but obviously this is not one of them!"

Read back with the operator tool, not one of the five had diverged. Every mesh was
checkMesh OK (max non-orthogonality 30-38, skewness 0.6-1.9). What the person was told:

- 20260919-041432-e0b6, a backward-facing step at Re_S 800 under simpleFoam: the Ux
  residual sat at 2.2e-2..3.2e-2 from iteration 250 to iteration 30,000, on two meshes,
  at relaxation down to p 0.2 / U 0.4, first and second order -- eight solver runs to
  establish the plateau. The flow is past the 2D Hopf bifurcation, so that plateau IS
  the physics. Headline to the person: "at Re_S = 800 there is no steady solution to
  converge to"; README column "converged?" answered "no (res. floor 1.5e-2)" three
  times; the figure titled "SIMPLE snapshot (flow is unsteady: no steady state)".
- 20260920-155504-4379, "Inviscid AI" text under simpleFoam k-omega SST, 106,716
  cells, Re 6.7e4, 2000 iterations in 157 s: Ux 1.2e-2..2.0e-2, Uy 3.6e-2..5.1e-2, p
  3.0e-2..3.9e-2 from iteration 200 to 2000, flat. The wake sheds; the field was fine
  and the pictures were made from it. Told: "**Honesty:** the run did **not**
  converge -- residuals plateau at ~1e-2 ... The fields are a frozen pseudo-transient
  snapshot ... treat them as +/-25%"; README "NOT converged to a steady state".
- 20260920-161908-c7ef, the same text under pimpleFoam, 0 -> 0.8 s in 22 min, per-step
  residuals Ux 5e-5, p 4e-2 (normal for a transient): "take Cd as +/-20-30%, not
  converged. One mesh, no independence study", and "Mesh independence -- 3 meshes ...
  ~1.5 h" offered first among the follow-ups. Nobody had asked for one.
- 20260921-033356-076b, the text under pimpleFoam at Re 6.7e3, stopped by the person
  at t = 1.48 s ("just plot whats there already"), no bounding, Co max 2.4: the gif
  was delivered and then "Cd ... **Not converged and not trustworthy.**", "Strouhal
  number: **I will not quote one.**", figure titled "NOT statistically converged".
- 20260921-033019-e1b4, pisoFoam laminar at Re 200: delivered cleanly; the one
  mention was "no grid-refinement check, so the numbers above carry no convergence bar".

Three senses of "converge" -- a steady residual, a grid, a time average -- landed on
the person as one message, four times in five studies: nothing converges. The
diagnosis is accurate in every case and wrong in its framing: a steady solver's
residual levelling off on a shedding wake is the residual reporting the flow, and the
levelled-off field is exactly the picture a first look is for. The only sentence the
system prompt had on the subject named "an unconverged solve" as the first thing to
confess to, and nothing anywhere told the model what a stall is, or that a quick look
may take the stalled field and say what the flow is doing.

So this module holds the line, in words, in two places the model reads at the right
moment: the briefing (`SOLVE_NOTE`, every session) and the launch note of a steady
solver (`STEADY_LAUNCH_NOTE`, `tools._solve_shape`). `toolbox/log_digest.py` reads a
finished log the same way (`residual_shape`, `how_it_ended`): the words depend on why
the residuals stopped falling. Nothing here changes a solver setting or a tolerance:
the runs were right; the reports were not.
"""

from __future__ import annotations

SOLVE_NOTE = (
    "A residual that stops falling is not one kind of thing, and the person hears the "
    "difference. A steady solver on a flow that is unsteady -- a bluff body, a shedding "
    "wake, a separated jet -- levels off and sits there, because the residual is then "
    "reporting the flow rather than the numerics; a plateau around 1e-4..1e-3 on a "
    "first look is a field fit to show. Neither is a failure, and the words for one -- "
    "could not converge, did not converge, failed to converge -- do not describe them: "
    "the field is physically meaningful, so the report says what the flow is doing and, "
    "if the residuals come up at all, why they levelled off, in one clause and in neutral "
    "terms. Failure language is for a run that cannot work -- a residual that climbs, a "
    "floating point exception, a field the solver keeps bounding, a mesh checkMesh "
    "rejects -- and then it says what failed and what would fix it. Said the first way: "
    "\"The wake behind the letters sheds, so the steady solve levelled off at 3e-3 from "
    "iteration 800 and this is the field at iteration 2000; a transient would give the "
    "shedding frequency.\" Or: \"Residuals sat at 2e-4 from iteration 1500 on, which is "
    "fine for this look; the pressure drop settled at 41 Pa.\" Said the second way: \"The "
    "run diverged at iteration 37: Uy climbed from 1e-3 to 40 and the pressure solve threw "
    "a floating point exception, so there is no field to show. The inlet meets a wall in a "
    "cell of near-zero volume; remeshing that corner is the fix.\" A first look is one "
    "mesh, one run and the picture; a transient window or a mesh study is a next step to "
    "offer, not to start unasked."
)
"""What the briefing says about reading a solve, in the harness's voice.

Two kinds of sentence, each with the number in it, so the person learns what the flow
did and the residual is a clause rather than a headline. The failure words appear
once, as the words this is not: `tests/test_convergence.py` holds them to that."""

STEADY_LAUNCH_NOTE = (
    "steady solver: on a flow that is unsteady (a bluff body, a shedding wake) its "
    "residuals level off rather than fall, and the levelled-off field is a snapshot "
    "worth showing, not a failed run"
)
"""The clause `job_start` adds when the solver it launched is a steady one. Said at
the launch because the next thing the model reads is that solver's residuals, and the
plateau reads as a failure to a model that was not told to expect one."""

FAILURE_WORDS = ("could not converge", "did not converge", "failed to converge")
"""The three phrasings the person heard as "nothing converges". Kept as data so the
tests can check the guidance names them exactly once, as the words it rules out."""


def is_steady_solver(executable: str) -> bool:
    """Whether a solver is a steady one, from its name: `simpleFoam`, `rhoSimpleFoam`,
    `buoyantSimpleFoam`, `SRFSimpleFoam`, `porousSimpleFoam` and their kin all carry
    SIMPLE in the name and iterate towards a steady state. `foamRun` with a
    `steadyState` ddt scheme cannot be told from its name, and is not claimed."""
    return "simple" in (executable or "").lower()
