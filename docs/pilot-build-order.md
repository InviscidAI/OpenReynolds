# Pilot build order

The pilot answers four things: what a publishable clip costs in instance-hours, what fraction of
autonomous runs produce something postable, how much of subject → clip runs unattended, and whether
the format guesses in [`content-plan.md`](content-plan.md) survive contact with an audience.

**The ordering principle is that the riskiest unknown goes early, and it is not the APIs.** The APIs
are known work. Whether the agent can autonomously produce a clip good enough to stop a scroll is not
known at all, and nothing downstream matters if it cannot.

The exception is the render seam, which comes first only because the manual spike is impossible
without it — `render.py` cannot currently emit a frame sequence at all, so "render frames by hand"
has nothing to run. Keep it to exactly what the spike needs.

---

## Phase 1 — the render seam

**Goal:** frames out of OpenFOAM, reliably, with a locked colour scale. The only change inside
OpenReynolds, and it is small.

`toolbox/render.py` today opens **one** time value via `open_case(case, time)`, writes one PNG per
field, hardcodes `window_size=(1100, 800)` and `cmap="viridis"` in both `render_field` and
`render_mesh`, always adds a scalar bar and caption, and passes no `clim` to `add_mesh` — so pyvista
rescales colours to each render.

Additions:

- **Frame sequences.** Iterate the reader's `time_values` instead of choosing one. Note `open_case`
  builds a fresh `OpenFOAMReader` per call; a sequence wants one reader reused across
  `set_active_time_value`, so this is a small refactor rather than a loop around the existing
  function.
- **`--clim LOW HIGH`.** Explicit colour limits, passed through to `add_mesh`. **This is the
  load-bearing change.** Auto-scaling flickers frame to frame in video, and across two cases it
  silently invalidates the comparison — the result looks fine and means nothing. The identical-scale
  rule from the content plan is enforced here or nowhere.
- **`--range FIELD`.** Report a field's min/max across every written time without rendering, so the
  joint range can be computed before either case is drawn. **Take the range on the slice, not the
  volume** — a volume maximum that never appears on the cut would widen the scale and flatten
  everything actually visible.
- **`--size W H`.** Vertical output; 1100×800 is the wrong shape for every target platform.
- **`--cmap`.** Jet for social, viridis stays the default for engineering use.
- **`--bare`.** Suppress the scalar bar and caption, since the compositor draws its own shared bar
  across both panels.

**Sequencing constraint this creates:** solve A → solve B → compute joint range → render both. The
render step cannot start until both solves finish. Worth knowing before designing any orchestration.

**Testing note.** `tests/test_toolbox.py` only parses `render.py` — pyvista lives in the container,
so there is no import-level test to extend. Anything guarding the new behaviour is either an
AST-level assertion about the declared flags, or a real test that runs on the instance. The `--clim`
contract is the one worth guarding, because it carries a correctness claim rather than a cosmetic
one, and because a silent regression there produces comparisons that look convincing and are wrong.

**Scope discipline:** frame-sequence rendering with explicit limits is a defensible CFD feature that
any user animating a transient wants. Question text, brand marks and A/B layout are content-specific
and do not belong in the shipped product. That seam is the same one the architecture already draws —
the toolbox is offered, never imposed.

---

## Phase 0 — one clip, by hand

Numbered zero because it is the spike, and it runs as soon as Phase 1 renders frames.

**Goal:** prove a clip can exist. No new abstractions, no scripts that will be thrown away.

Subject: **window fan, blowing in vs blowing out.** Chosen over the stronger fan-jet subject because
it is a genuine two-case A/B, so it exercises the whole path — matched setup, joint colour scale,
composite — and the geometry is a box with two openings. The strongest subject is worth spending
once the pipeline exists, not while it is being discovered.

1. Drive OpenReynolds interactively. Two cases, identical mesh and schemes, differing only in the
   fan boundary condition.
2. `--range` both cases, take the union, `--frames --clim` both.
3. Composite, label and encode by hand, however works.
4. Look at it. Would this stop a scroll?

**Exit criterion:** one clip that passes the five format criteria and that you would actually post.
If it takes a week of fiddling, that is the finding — record what the fiddling was, because it is
the specification for Phase 2.

**Do not** write a reusable compositor during this phase. The point is to find out what it needs to
do.

---

## Phase 2 — the compositor

**Goal:** two frame sequences in, one platform-ready MP4 out. Lives outside OpenReynolds.

- Two panels joined directly, no outer box, orientation chosen by geometry
- One shared colorbar spanning the full width
- Word labels ("fan blowing out", never `case_02_outflow`)
- Question text, clear of the bottom 40% and TikTok's icon rail
- Inviscid AI mark, bottom-left
- Audio track attached — a muted Reel is explicitly demoted
- Encode per platform from one master

ffmpeg plus a compositing step is sufficient. Resist making this clever; it runs a handful of times
per week at most.

---

## Phase 3 — five subjects, hand-posted

**Goal:** the numbers the cadence decision actually turns on.

Run subjects 1–5 from the content plan end to end. Post by hand.

Record per subject: instance-hours to a publishable clip · whether the run converged and looked good
without intervention · which steps needed a human and why · wall-clock from subject to posted.

**Still no APIs.** Ten hand-posts cost minutes each. Instagram's Content Publishing API needs a
Business account, a linked Facebook Page and app review for publishing permissions — days to weeks
of process, and it is pure overhead at this volume. It also automates a guess: you cannot codify a
workflow you have not performed.

This phase closes E6 and starts E1, E3, E4 and E5 — write the predicted rank for each subject
*before* posting it, per E5.

---

## Phase 4 — orchestration

**Goal:** automate what Phase 3 proved repetitive. Not before.

By this point the actual shape is known: which decisions the agent made well, which needed a human,
where runs failed. Build to that, not to a guess. Likely candidates:

- Subject → two matched case setups (the "identical except X" constraint is a real test of the agent)
- Solve both, poll, compute joint range, render
- Composite and encode

Keep the human in subject selection until E5 says otherwise. If the stake ranking does not predict
performance, curation is a lottery and the strategy shifts to volume — which changes what is worth
automating.

---

## Phase 5 — APIs, only if cadence demands

Revisit only once Phase 3 has produced a real cost-per-clip and a cadence has been chosen. If the
answer is two clips a week, hand-posting is permanently fine and this phase never happens.

If it is needed, expect: Instagram Graph API (Business account, linked Page, app review), and note
that everything else in the plan is a cross-post of the same asset — TikTok, X and Reddit numbers do
not count toward the scoreboard, so their posting can stay manual indefinitely.

---

## What is deliberately not in this plan

- **No API integration on the critical path.** It is the best-understood work and the least urgent.
- **No orchestration before the manual path works.** Automating an undiscovered process.
- **No content-specific code inside OpenReynolds.** Frames are a product feature; brand marks are not.
- **No second subject before the first clip is good.** If Phase 0 produces something mediocre, the
  answer is to fix the render, not to run more cases.
- **No compositor in Phase 1.** The render seam stops at frames on disk. Everything about how two
  panels are joined is discovered by doing it once by hand.
