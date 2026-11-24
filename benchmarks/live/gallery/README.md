# The live-sync gallery

Pictures of what `openreynolds[live]` measures, drawn from the recorded runs in
`benchmarks/live/results/` and from stated lines of [`PROGRESS_LIVE.md`](../../../PROGRESS_LIVE.md).

    python benchmarks/live/gallery.py            # redraw everything into this directory
    python benchmarks/live/gallery.py --check    # verify what is here already

Nothing here is solved or re-rendered. Every number traces either to a `result.json`
field, which the caption names, or to a line of `PROGRESS_LIVE.md`, which the caption
quotes with its line number -- `gallery.py::_quote` locates that line at draw time and
the script fails if the claim has moved, so a citation printed on a figure was true when
the figure was made. **Points from the two sources are drawn with different markers**:
filled for measured, hollow or triangular for prose-only.

**Four of the five figures are pictures of a failure.** That is not a framing choice.
This repository's rule is that a figure which flatters is worse than no figure, and the
honest state of this work is 56 of 64 graded checks passing with five named cases
outside their tolerance, one milestone frozen on a measured defect and one not started.

---

## `scorecard.png` -- where the project actually stands

**What it shows.** Every graded check in every recorded validation run, as a bar whose
length is the measurement normalised by that check's own tolerance. The black line at
1.0 is the bound: a bar past it is a check the grader failed. Grouped by milestone,
coloured green for pass and red for fail.

**The number.** **56 of 64 graded checks pass, 8 fail**, over 17 validation runs -- the
same count, the same runs and the same milestone map as `PROGRESS_LIVE.md`'s summary
table, because the rows are read through `summarise.py::rows()` rather than re-derived.
The failures, worst first: `crosscheck` Strouhal against Williamson **215x** its bound
and against the OpenFOAM run **212x**, `crosscheck` mean Cd **5.11x**, `cylinder`
observed order of convergence **3.37x**, `cylinder` Cd_max **2.38x**, `sphere` Cd at
Re 10 **1.13x**, `slosh-n240` first-mode period **1.07x**, `dam-n1300` depth at the dam
**1.05x**. 40 checks have a numeric tolerance and get a bar; 24 are invariants
(`exactly 0`, `True`, `> 0`, `finite`) and get a verdict with no bar, because drawing a
zero-length bar for a third of the suite would read as a third of the suite being
perfect.

**What it does NOT show.** The open questions, or any work that is not graded.
Diagnostic runs -- the resolution sweeps, the inlet-distance probes, the step-count
sweeps, the M6 reproduction attempts -- are excluded, as they are from the summary
table, because counting them as tests inflates the denominator. Rows that are reported
and not graded are excluded too. **M6 contributes no rows at all**: its draft did not
reproduce between identical runs at the same commit, the +1.9% once reported was
withdrawn, and the milestone is frozen on a measured defect (the interface thickens by
45% within ten steps of the hull being released). M7 is not started. A scorecard with
M6 missing is not a picture of a finished project. M4 contributes no rows either -- the
performance numbers are rates, not graded checks, and they are in `PROGRESS_LIVE.md`.

---

## `convergence-dam-break.png` -- converging, and not to Ritter

**What it shows.** Three grids at a constant refinement ratio. Left: the depth at the
dam as a multiple of Ritter's `4 h0/9`, against the +/- 15% band the case grades on and
against the Richardson limit. Right: the depth profile's RMS, which passes at every grid.

**The numbers.** Measured, from `dam-n325`, `dam-n650`, `dam-n1300` (`at_dam` /
`exact_at_dam`): **1.2450, 1.1837, 1.1576**, and profile RMS **9.02%, 7.55%, 6.53%** of
h0. From the prose (`PROGRESS_LIVE.md:529-530`): observed order **1.232** with a limit of
**1.1382**, and 0.527 with a limit of 4.22%. All three runs are at commit `e5b1698`,
clean, on an H100 80GB.

**What it does NOT show.** A converged dam break, and not a passing one either: all
three grids fail the depth-at-the-dam check, and `dam-n1300` is a **FAIL** row in the
summary table at 25.7244 mm against 22.2222. The dashed limit is where a three-point fit
says the scheme is going; it is not a run and no grid has reached it. The argument that
the surviving 13.8% is physics rather than error -- Ritter is inviscid and hydrostatic,
`Re_eff` is 37/101/202 on these three grids -- is in the prose and is not plotted,
because a plot cannot carry it.

---

## `convergence-sloshing.png` -- converged, five per cent above linear theory

**What it shows.** Four grids of the first-mode period against linear theory, the +/- 5%
band, the extrapolated limit, and the resolution at which the case stops working.

**The numbers.** Measured, from `slosh4-n80`, `slosh3-n120`, `slosh4-n180`,
`slosh3-n240` (`period` / `theory`): **0.7469, 1.0728, 1.0563, 1.0529**. The divergence
is measured too: `slosh3-n480` records `lattice finite` = **False** with a mass drift of
**-1.64e8** over 5,008,707 cell conversions. From the prose
(`PROGRESS_LIVE.md:493`): observed order **3.5**, limit **1.0510**. The convergence
family is all at commit `fcf2417`, clean, H100 80GB.

**What it does NOT show.** A case that passes -- the graded run (`slosh-n240`,
0.439358 s against 0.41709 +/- 5%) is a **FAIL** -- or an explanation of the five per
cent. The usable window is 120 to 240 cells and nothing wider: 80 cells is
under-resolved by 25% and 480 diverges on a Reynolds-number ceiling, so the convergence
can be followed over a factor of two and no further. Damping, finite amplitude, the
depth, gravity, the estimator and resolution are each ruled out with a number in the
prose; what is left is unexplained, and it is the one M5 number with no account of
itself. Note also that the graded row in the summary table is a **different run** of the
same case at the same resolution (commit `e5b1698`, 1.0534 of theory); this figure plots
the convergence family, which was run at one commit so that the four grids are
comparable with each other.

---

## `convergence-cavity-float32.png` -- the error grows with the run length

**What it shows.** The Re 1000 cavity's u RMS against Ghia, as a function of how long
the run was. The error rises monotonically with run length and crosses the 2% tolerance.

**The numbers.** Measured, from `cavity-re1000`, `cav-steps-600000`,
`cav-steps-1200000`, `cav-steps-2400000` (`u_rms`, with `stopped.steps` on the x axis):
**0.24%, 0.39%, 0.91%, 1.96%** at 180k / 600k / 1.2M / 2.4M steps. From the prose and
**not on disk** (`PROGRESS_LIVE.md:237`): the fifth point, 3,492,000 steps at **2.88%**,
drawn hollow. The slope, about **0.86% per million steps**, and the crossing at about
2.45 M steps are prose too (`PROGRESS_LIVE.md:239`); no fitted line is drawn, because
anchoring one would decide the crossing rather than report it.

**What it does NOT show.** A converged cavity. The graded run is the **shortest** of the
five and the agreement is best there, which is the opposite of the usual reading that a
longer run is a better one -- erosion starts immediately and there is no plateau to sit
on. The purple triangle is **not** the prose's fifth point: it is `cav-re1000-bgk`, a
different collision operator run to the same step count (2.762%), included only because
it is the nearest recorded neighbour to a point that has no `result.json` in this
repository. Four of the five runs on this plot (`cav-steps-*`, `cav-re1000-bgk`) carry
**no provenance block at all**, so the commit they ran at is unrecorded.

---

## `cylinder-mach-and-inlet.png` -- the +11.7%, localised two ways

**What it shows.** Left: Cd_max error against the Schaefer-Turek 2D-2 interval as the
Mach number is halved, with the two zero-Mach extrapolations. Right: the same error as
the inlet is moved upstream.

**The numbers.** Measured: Ma 0.064 gives Cd_max 3.6088 = **+11.73%**
(`cylinder/result.json`, the 32-cells-per-diameter row of `resolutions`); +4 diameters
of inlet gives 3.3083 = **+2.42%** and +10 diameters gives 3.3408 = **+3.43%**
(`cyl-upstream-0.4`, `cyl-upstream-1.0`). From the prose and not on disk: Ma 0.032 gives
3.4613 = **+7.16%** (`PROGRESS_LIVE.md:285`), and the zero-Mach limits **+2.59%** linear
/ **+5.64%** quadratic (`PROGRESS_LIVE.md:290`).

**Provenance warning, on the figure as well as here:** both inlet-distance runs were
taken on a **dirty working tree** -- commit `7dde521` with one uncommitted file -- so
their sha does not pin what ran. `cylinder` itself is `2e60227`, clean.

**What it does NOT show.** A passing cylinder. The graded case runs the paper's own
2.2 x 0.41 channel and **fails** at +11.9% on Cd_max, with an observed order of
convergence of 0.2966 against a `>= 1.0` target -- a second failure this figure does not
plot. A run in a longer channel is not Schaefer-Turek 2D-2 and is not reported as it.
`Cl_max` moves the **other** way under the extended domain, +10% to -15%, so the longer
channel is not uniformly better and the lift is not claimed either way. And two Mach
points cannot establish whether the error is linear or quadratic in Ma, which is exactly
what decides whether this case can ever pass in its own domain -- the third point at
Ma 0.016 that would settle it is described in the prose as running and has no
`result.json` here.

---

## `field-dam-break.gif` -- the solver's own field, moving

**What it shows.** 60 frames of |u| through the 1300-cell dam break, as
`figures.animate` wrote them during the run that produced `dam-n1300/result.json`:
the column collapses, the surge runs the length of the tank and the front reflects.
Copied, not re-rendered -- it came off an H100 and this gallery has no solver.

**The number beside it.** That run's graded checks: depth at the dam 25.7244 mm against
Ritter's 22.2222 (**FAIL**, +15.8%), profile RMS 6.53% of h0 (PASS, bound 15%), mass
drift -2.48e-4 of the initial mass (PASS, bound 1e-3), zero fluid cells touching gas
cells (PASS), lattice finite (PASS). Commit `e5b1698`, clean.

**What it does NOT show.** Agreement with Ritter -- this is the picture of the run that
fails the depth check, and the animation cannot show that. Its **frame 0 is a single
flat colour**, which `--check` reports: the tank is at rest, every cell has |u| = 0, and
a one-colour frame is the correct picture of that. The colour scale is fixed across the
sequence from the first few frames, so late frames are comparable with early ones.

**A discrepancy found while picking this file, and not fixed here.** Every `*.gif` under
`benchmarks/live/results/` was read with both PIL and imageio. **Only the dam break is
animated.** `live-cylinder-*.gif`, `live-falling-sphere.gif` and `live-quadrotor-*.gif`
each contain exactly one frame, so they are stills with a `.gif` extension -- which means
`PROGRESS_LIVE.md`'s M4 row, "the cylinder, the falling sphere and the quadrotor
animate", is not what is on disk. Nothing under `openreynolds/` is touched by this
gallery and the machine it was assembled on has no CUDA device, so this is reported
rather than repaired.

---

## `isaacsim-freesurface.gif` and `isaacsim-submarine.gif` -- RTX, inside Kit

**What they show.** The two demos rendered by Isaac Sim 6.2.0.0's RTX path on a Modal
L40S: a 144 x 64 x 64 dam break with the water as a heightfield mesh rebuilt each frame
(40 frames), and the submarine swimming and then taking a dive pulse, with its wake
drawn as a point instancer over the decimated velocity field (44 frames). Copied from
`results/isaacsim/`.

**The numbers.** From `freesurface-l40s.json`: 144-cell lattice, 3,920 steps over 40
frames, 0.5162 simulated seconds, water volume holding at 0.002351 m3 against an initial
0.0023525 (**0.06%** over the run). From `submarine-l40s.json`: a 480 x 160 x 200 lattice
= **15.36 M nodes**, 23,184 solid cells in the hull against an analytic volume of
2.8798e-6 m3 (voxelised 2.898e-6, **+0.6%**), 5,236 steps over 10.437 simulated seconds.
The coupling numbers these runs support -- drag against thrust `|Fx|/T` = 0.815 and
momentum balance 0.879 over 347 wrench samples -- are in `PROGRESS_LIVE.md`.

**What they do NOT show.** A validated result. These are demonstrations that the
extension runs, couples and renders inside a live Kit process; neither has a published
reference and neither contributes a row to the scorecard. "It rendered" is itself a
measured claim here and not an assumption -- `--check` reports a per-channel standard
deviation and a unique colour count for every frame, because a whole-array standard
deviation scored 51.32 on a flat teal frame whose true colour count was 1. The submarine
GIF's median frame is the dimmest thing in this directory (channel sigma 28.5) and that
is the scene, not an encoding failure.

---

## How a figure here is checked

`--check` reads every file back and prints, per file: size, colour mode, frame count,
the **median** frame's per-channel standard deviation and unique colour count, and the
index of any frame that is uniform. It exits non-zero if a median frame is flat
(sigma < 1.0 or fewer than 16 colours) or if a PNG was written RGBA -- an alpha channel
makes a thumbnail composite to a black rectangle, which is why `figures.ensure_rgb`
exists and why every PNG here goes through it.

The worst-frame rule was tried first and called the dam break broken, on its
legitimately uniform frame 0. A check that cannot tell a tank at rest from a dead
encoder is a check that gets switched off, so the uniform-frame count is printed beside
the verdict instead of being folded into it.
