# OpenFOAM field notes

Distilled from a longer architecture document (`openfoam-agent-architecture.md`, in this
directory). These are observations, not a procedure — nothing here is a step you owe anyone.

---

## The cost asymmetry

Mistakes are cheap while you are authoring files and expensive once you have committed
compute to them.

| What you are about to do | What a mistake costs |
|---|---|
| Write a dictionary | seconds |
| `snappyHexMesh` build | 20 minutes to hours |
| Production solve | hours to days |
| Report a wrong number | the whole study |

Everything below is a consequence of that table. The cheapest checks — `foamDictionary`
parsing a file, `foamToC` telling you what this build actually compiled in, a one-iteration
solver probe on a trivial mesh, an arithmetic cell-count estimate — cost less than a minute
and sit directly upstream of the expensive things.

## What a transient run's wall clock is actually made of

Three numbers set it, and they are not equally large. Measured on a 2D laminar cylinder
at Re=100, one cell in z, `pimpleFoam`:

**Cores.** A solver started plainly occupies one, however many the container has.
`decomposePar` then `mpirun -np N solver -parallel` occupies N. What the extra ranks
return tracks cells *per rank*, not rank count: at 33.6k cells, 4 ranks ran 2.9x, 8 ranks
3.6x, 16 ranks 4.6x; at 5.8k cells, 8 ranks returned nothing 4 had not already. Below
roughly two thousand cells a rank the halo costs more than the arithmetic saved.

**`endTime`.** This is the one that dwarfs the others, and on a bluff body it is mostly
spent waiting for symmetry to break. From an unperturbed symmetric initial field the
instability grows out of round-off: |Cl| was 1.3e-4 at t=35 and only 0.25 by t=90 — some
fifteen shedding periods bought nothing but exponential growth from noise. The same case
started with a small transverse component in the internal field, `internalField uniform
(1 0.05 0)`, reached saturated amplitude by t≈55. Identical physics and Strouhal number
(0.169 measured against 0.164 published); less than half the solver time. An `endTime` of
200-plus is usually the cost of an unperturbed start rather than a requirement of the flow.

**Write frequency.** `writeInterval 0.1` against `endTime 250` is 2,500 write times. On a
40k-cell case that is several gigabytes and thousands of directories, which is also what
the mirror and the file tree have to walk. Frames for an animation can be subsampled from
sparse writes; they rarely need every one.

One thing that looks like a fourth lever and is not: raising `maxCo`. At `nOuterCorrectors
1` the solver is in PISO mode and `maxCo 5` diverged outright (Cl to 1e42). Adding outer
correctors bought stability at `maxCo 2` and was still *slower* to the same solver time
than plain PISO under Co<1, because each step then costs three. The timestep is not where
this class of case is won.

## Verifying against the installation rather than from memory

There is no complete static grammar for OpenFOAM dictionaries. It is distributed across the
C++ classes that consume each dict, differs by fork and version, and grows with every
compiled library. The grammar does exist in one authoritative place: the binaries in front
of you.

- `foamDictionary -expand <file>` resolves includes and variables. Good for reading; do not
  write the expanded form back, since it flattens exactly what you wanted to keep.
- `foamToC -functionObjects` / `foamToC -table <name>` enumerate the selection tables a
  build has — schemes, boundary-condition types, turbulence models, function objects —
  and the answer is specific to that build *and* to the `libs` entries in the case.
  **It is not installed in the hosted image**, so there the tutorials and the solver's own
  rejection messages are what is left.
- A deliberately invalid token in a selection slot produces a fatal error listing the valid
  options. It works everywhere, but it is fatal by construction (one enumeration per run)
  and serially dependent (the error only reaches the slot you care about if everything
  constructed before it is already valid). One `foamToC` pass beats a chain of these.
- Cloning `system/ constant/ 0/` onto a trivial blockMesh whose patch names match, then
  running the solver for one iteration, catches missing keys, invalid enums, dimension
  mismatches and failed model construction in seconds. It does not work on cases that need
  zones created by snappy, cyclicAMI pairing, `setFields`, or nonuniform internal fields
  sized to the real mesh — on those it is silent rather than wrong.
- After the mesh exists, one iteration on the *real* mesh has no false-alarm case at all,
  and it still runs before the expensive solve.

## Before a snappy build

Each of these takes seconds against the STL and dictionaries alone, and each catches
something you would otherwise discover an hour later:

- **Look at the surface** — `geometry_view.py` in this directory draws every STL from four
  fixed views with the facet edges and the bounding-box ticks on, and prints open edges,
  non-manifold edges, connected bodies, extents and face-area range. Wrong units, a stray
  solid, a hole, and geometry that does not sit where you thought it did are all obvious
  in the picture and invisible in a directory listing.
- **`locationInMesh`** — is the point actually inside the fluid region, not inside a wall,
  and not so close to a surface that it lands in a cell that gets removed? This is the
  classic cause of an empty or inverted mesh.
- **Region names** — every name in `refinementSurfaces`, `refinementRegions`, `layers` and
  `geometry` should resolve to a real solid or patch name in the STL. Snappy silently
  ignores entries it cannot resolve, and you get an unrefined surface with no error.
- **Cell count** — `cells_estimate.py` in this directory. A level that implies hundreds of
  millions of cells, or a mesh too coarse to resolve the region you care about, is visible
  before the build rather than after it.
- **Layer feasibility** — total layer thickness (first layer × expansion^n) against the
  local cell size after refinement. An incompatible spec produces collapse, not an error.
- **`snappyHexMesh -dry-run`** where the build offers it.

Predicted-versus-realized cell count is itself a signal afterwards: a large divergence
usually means a leak.

`preflight.py` in this directory runs the first, third and fourth of those as checks
alongside eight more, and answers each with what it measured, what that means and a
repair rather than with a log; `first_look.py` puts the geometry, the whole mesh, a
close-up on whatever was refined, the named patches and the counts on one contact sheet,
which is one `read_file` rather than five.

## Before a solve

The mesh being built is not the same question as the case being runnable, and the
second one has its own short list, all of it answerable in seconds against files that
already exist: does every patch in the mesh have an entry in every field, does a 2D
case have its `empty` patch declared in both places, does the viscosity match the
Reynolds number somebody has in mind, is the timestep going to put the Courant number
somewhere sensible, and is there disk for the write cadence. `preflight.py` asks all of
those and reports `ok`/`warn`/`fail`/`skipped` per check with a suggested repair -- the
repair being a suggestion, since it does not know what the study is for. A one-iteration
solver probe on a copy of the case is in there too, which is the cheapest way to find
out that the solver will start at all.

## Failure signatures

**Geometry** — not watertight; units or scale wrong; inverted normals; self-intersections;
multi-solid or region-name mismatch.

**Mesh** — high non-orthogonality; high skewness; negative volumes; layer collapse; a
snappy leak (realized cell count far outside the volume estimate); refinement not realized;
snap failure at sharp features; invalid `locationInMesh`; unresolved region reference.

**Case** — patch/field mismatch; parse error; solver and model incompatible; unstable
boundary-condition pairing; bad initialization.

**Run** — residual divergence; Courant explosion; bounding spam; timestep collapse;
residuals plateauing high; oscillating quantity of interest; growing continuity error;
crash or floating-point exception; wall time or disk exhausted.

Reading a long solver or mesher log by grepping it several times is slower than it
looks: each `grep` is a round trip to the instance, and one live session spent a
full minute on three passes over the same 1,700-line `log.snappy`.
`log_digest.py` in the toolbox turns a solver log into a residual plot, a
last-iteration table and the continuity/bounding summary in one call, and
`mesh_digest.py` does the same for `checkMesh`.

**Verify** — quantity of interest never plateaued; grid-convergence index too high or
non-monotone; mass imbalance; result outside its expected band; dimension mismatch;
definitional mismatch (right units, right plumbing, measuring a different quantity).

On layer collapse specifically: widespread collapse usually responds to fewer layers or a
gentler expansion ratio; collapse localized at features responds to lower `minThickness` and
a higher `featureAngle`; curvature-driven collapse responds to one more level of local
surface refinement. Partial coverage away from the region you care about is often simply
acceptable — worth saying so out loud rather than chasing it.

## When `mpirun` will not start

The signature is PMIx, not MPI:

```
[modal:35552] PMIX ERROR: ... The PMIx server's listener thread failed to start
```

and, if you go looking, `/sys/class/net` is missing and `ip addr` shows nothing.
It reads like a broken MPI install. It is not. The instance runs with egress
restricted to loopback, and PMIx's default `ds12` component wants to enumerate
network interfaces before it will open its rendezvous socket. With nothing to
enumerate it gives up, `mpirun` dies, and a parallel decomposition that was going
to use every core runs on one.

`PMIX_MCA_gds=hash` selects the component that needs no interfaces and works under
the same policy. Instances now start with it set, along with
`OMPI_ALLOW_RUN_AS_ROOT` and `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM` (the container is
root, and OpenMPI refuses root without both). So `mpirun -np 4 …` should simply
work; if it does not, that env var is the first thing to check and
`export PMIX_MCA_gds=hash` the first thing to try.

This cost two studies an afternoon between them on 2026-08-27. One spent five
minutes probing `ompi_info`, `/sys`, and interface indices, concluded "MPI is
unusable in this container", and ran an eight-core instance serially for an hour.
The other found the hint in OpenMPI's own error text — it suggests `gds=hash` if
you read far enough down — and had four ranks a minute later. The difference was
entirely in how far down the error each of them read.

A related habit worth having: an OpenFOAM or OpenMPI error that suggests a setting
is usually suggesting the right one. The tail of the message is where the remedy
lives, and `tail -30` of a log often skips it.

## Traps that produce a confident wrong number

**Kinematic versus dynamic pressure.** In incompressible OpenFOAM, `p` has dimensions m²/s²
— it is pressure divided by density. For air the difference between that and a pressure in
pascals is a factor of 1.204, which hides inside any tolerance band wide enough to be
useful. For water it is 998 and any band catches it. A tolerance band is not the mechanism
that saves you here; checking dimensions is. The field header states the dimensions, so
there is no need to assume them.

**Definitional mismatch.** Two instruments can have identical dimensions, agree with each
other to a fraction of a percent, and measure a different physical quantity than the
reference they are being compared against. A published elbow loss coefficient is defined on
*total* pressure loss; differencing area-averaged *static* pressure across the elbow gives a
different number, and after a sharp bend the outlet velocity profile is still recovering at
20 diameters, so the kinetic-energy correction differs between the two stations. Worth
saying, for any number reported, what quantity it is, on what surfaces, and with what
convention.

**Millimetres versus metres.** CAD exports in millimetres constantly. A bounding box three
orders of magnitude off the expected characteristic length is the tell.

**A steady solver on genuinely unsteady flow.** This is the dangerous one, because the
obvious detector misses it. The visible version is a solver struggling in public: residuals
plateau high and the quantity of interest oscillates. The dangerous version is the opposite
— first-order upwinding, heavy under-relaxation and steady SIMPLE together damp the physical
instability, residuals fall several orders, the quantity sits perfectly flat, and the answer
is a well-converged wrong number with no oscillation and no elevated residuals at all.

Two probes work where passive detection does not. First, start a URANS run from the
converged steady field *plus a small deliberate perturbation* in the expected separation
region, and fit a growth rate to the envelope rather than thresholding amplitude: a
perturbation that decays is real evidence of stability, and a perturbation that grows is a
positive detection while the amplitude is still invisible. Started from an unperturbed
steady field, a bluff body typically needs tens of shedding cycles before amplitude shows,
so a short window returns a false negative with a certificate attached. Second, vary the
numerics along an axis that actually has headroom — if the baseline is already second-order,
"rerun on second-order schemes" varies nothing and reports a reassuring zero. If relaxing
under-relaxation makes a marginally stable steady solve diverge, that is not a numerics
failure to be patched; it is among the strongest evidence available that the steady
formulation was holding an unsteady flow together by numerical damping.

## Reading a mesh's quality

Warn-tier metrics couple to numerics rather than being pass/fail: high non-orthogonality
wants non-orthogonal correctors and limited gradients; high skewness wants limited
divergence schemes. Accepting a warn-tier mesh and *not* adjusting the schemes is the
combination that bites.

The numbers summarize; a picture localizes. `render.py --scene mesh` draws fixed-camera
cuts through the built mesh, and `read_file` on the PNG hands the picture back to look
at — refinement bands that did not reach the surface, layers that collapsed on one
patch, a snap that shredded a sharp feature, all show up in a cut and vanish into a
max/average in `checkMesh`. The same holds for geometry before the build
(`geometry_view.py`) and for fields after the solve. Renders left in the workspace are
mirrored to the user's machine within moments, so the picture that convinced you is
also the picture they see.

y⁺ is not a 30–100 band to be tested over a histogram, **provided the wall treatment is a
blended one**. That qualifier is the whole of it, and it was missing here for a while:
`nutUSpaldingWallFunction` blends through the viscous sublayer, the buffer layer and the
log layer and really is y⁺-insensitive; `nutkWallFunction` is the plain high-Re
Launder–Spalding form and is valid to roughly y⁺ 300, above which skin friction comes from
an extrapolation nothing supports. A generated case writes Spalding on every model, so the
insensitivity holds — but a case that came from somewhere else may not, and the note used
to reassure exactly where the exposure was. Check `0/nut` before believing it.

With a blended treatment, every real geometry still has stagnation points and separation
lines where y⁺ → 0 no matter how good the mesh is. The useful question is what fraction of
the wetted area on the surfaces that matter sits in the wall treatment's valid range.

Grid convergence: Richardson extrapolation presumes monotone convergence, and real triplets
often do not deliver it. If the sign of the change reverses across three grids, the honest
output is the spread across the ladder as an interval, labelled oscillatory — not a
grid-convergence percentage that presumes the thing it is missing.

## Per-class starting points

Solver names are ESI. These are places to start, not defaults to defend.

| Class | Solver | Turbulence | Domain | Note |
|---|---|---|---|---|
| internal duct, steady | `simpleFoam` | `kOmegaSST` | fit + development lengths, ~10D in / 20D out | usually well behaved |
| external aero, steady | `simpleFoam` | `kOmegaSST` | ~10D upstream, 20D downstream, 12D sides (an aerofoil wants 50-100c) | vary the schemes; bluff or post-stall bodies are shedding candidates |
| indoor / buoyant | `buoyantSimpleFoam` | `kEpsilon` | room plus plenum | |
| transient shedding | `pimpleFoam` | `kOmegaSST` | per class | a short-window check before the long run pays for itself |

## When a steady solve will not converge

Residuals that fall a few orders and then sit flat are the commonest shape of "not
converging", and the plateau has a small number of causes worth telling apart before
anything is re-run harder:

- **A limit cycle.** Separated regions, bluff bodies and sharp corners shed; steady
  SIMPLE damps the shedding into a small residual oscillation, typically plateauing
  around 1e-4 to 1e-5. The residual is then reporting the flow, not the numerics.
  The useful instrument is the quantity of interest, not the residual: a force, flux
  or pressure-drop monitor over the last few hundred iterations that is flat to
  within its own noise is usually a usable answer with an honest caveat; one that
  drifts or oscillates at growing amplitude is the unsteady-flow trap described
  above, and a steady formulation is the wrong tool for it.
- **Numerics fighting the mesh.** A warn-tier mesh on default schemes plateaus or
  bounds. Non-orthogonality is corrected in two places, and the gradient limiter is
  neither of them: `nonOrthogonalCorrectors` in the `SIMPLE`/`PIMPLE` block, and the
  `corrected`/`limited corrected 0.33` forms on `laplacianSchemes` and `snGradSchemes`.
  `cellLimited Gauss linear 1` limits gradients, which helps a skewed or a stretched
  mesh and does nothing for non-orthogonality -- pointing at it sends you to the wrong
  dictionary entry. Skewness wants limited divergence schemes (`bounded Gauss
  linearUpwind`, `limitedLinear`). Accepting the mesh and not adjusting the schemes is
  the combination that bites.
- **Under-relaxation.** The classic stable pair is `p 0.3 / U 0.7`. Lowering buys
  stability at a linear cost in iterations; `SIMPLEC` (`consistent yes;` in
  `SIMPLE`) takes **no pressure relaxation at all** -- `p 1.0`, which is what you get by
  leaving `p` out of the `fields` block entirely, as the generated `fvSolution` does --
  with `U 0.9`. Relaxing pressure under SIMPLEC undoes the consistency the scheme is
  named for and slows it toward plain SIMPLE. If a solve only holds together far below the classic
  values, that is evidence about the case — the mesh, the boundary conditions or
  genuine unsteadiness — not a knob to keep turning.
- **Boundary conditions that cannot be jointly satisfied.** A fixed-value inlet
  facing a fixed-value outlet, `zeroGradient` pressure everywhere with no reference
  (`pRefCell`/`pRefValue` on closed domains), backflow at an outlet with no
  `inletOutlet` — each shows up as residuals that refuse to fall from the first
  hundred iterations rather than plateauing later.
- **A bad start.** `potentialFoam` before the segregated solver, or a first-order
  run continued (`startFrom latestTime`) on second-order schemes, routinely turns a
  diverging case into a converging one for seconds of extra cost.

Turbulence quantities (`k`, `omega`, `epsilon`) plateauing one to two orders above
`U` and `p` is normal near walls and rarely worth chasing on its own.

## When the shape is analytic, the reference is checkable

A hull, an aerofoil, a sphere, an Ahmed body: where the geometry comes from a formula
rather than a scan, its invariants come from the same formula. Volume, wetted area, block
coefficient, frontal area are all closed-form, so a stated reference value and the geometry
in hand check each other for nothing.

The incident. A Wigley hull's displacement was carried in three documents as 0.0600 m3,
alongside a block coefficient of 4/9. The form
`y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)` integrates to `B*(2L/3)*(2T/3)` = 0.075 m3, which *is*
Cb = 4/9; 0.0600 m3 would be Cb = 0.356. The two numbers had sat next to each other in one
line of one file, contradicting each other, through every reading of it.

It surfaced two rounds later and sideways. A free-surface run ballasted to the wrong
displacement floated 18 mm high while trying to measure a 7.5 mm sinkage -- the artefact
more than twice the signal -- and the discrepancy was read for two rounds as a *meshing*
defect, with a mesh rebuild commissioned to fix it, because the reference was trusted and
the measurement was not. What settled it was three instruments agreeing: holding the body
fixed with the flow at rest and integrating the vertical force on it, which weighs the
meshed hull directly; clipping the surface at the waterline and integrating it; and the
closed form. All three are minutes of work and none needs a solve.

Two generalisations worth carrying:

- Where two stated properties of one object are related by a formula, they are a free
  cross-check on each other, and a pair that disagrees says one of them is wrong without
  saying which. That is still most of the way there.
- **An error that does not scale with the quantity it corrupts is not a physics error.** A
  sinkage wrong by a roughly constant offset while sinkage itself varies five-fold across
  the speed range is a datum or a reference problem; a physics or resolution error grows
  and shrinks with the thing it is spoiling. The shape of the error narrows the search
  before any of it is explained.

A wrong constant does not announce itself. It turns into a plausible, detailed, entirely
wrong story about something else, and the story survives review because every step in it is
sound.

## A ladder of reduced cases, and the rule that makes one worth building

A ladder is a short sequence of cases run before the one that was asked for, each adding
exactly one piece of physics to the one below it, and each with an expected answer that is
known before it runs. A rung that fails localises the fault to the single thing that rung
introduced. A rung that passes is evidence that survives into the next session, which is
more than can be said for a hypothesis.

The rule that separates a ladder from a set of smaller unknowns: **each rung's answer comes
from outside CFD.** Hydrostatics, a symmetry argument, a closed-form integral, a measured
correlation with a citation on it. Archimedes gives the vertical force on a hull at a known
draught. Kelvin's 1887 wedge is 19.47 degrees for any disturbance over deep water,
independent of speed and of hull form. ITTC-57 gives a flat-plate friction line.
Hagen-Poiseuille gives a centreline speed exactly twice the bulk mean. A fluid at rest under
gravity has a uniform `p_rgh` and an interface that does not move, by the definition of
hydrostatic equilibrium. If a rung's expected value could only come from another solve, it
is the same unknown at a lower price and it does not belong on the ladder.

The incident. A Wigley free-surface case was built from the `DTCHullMoving` tutorial and
then diverged from it in four places at once: the internal field started at rest instead of
at tow speed, the inlet was ramped instead of held constant, the outlet became `inletOutlet`
instead of `outletPhaseMeanVelocity`, and the patch values of `alpha` were hand-written
instead of set by `setFields`' `boxToFace` pass. None of the four was tested on its own. The
run died at the inlet, six mechanisms were proposed for it, and all six were falsified --
four of them by the people who had proposed them. The acceptance number was never measured.
The fault was present in a still tank with no hull and no motion, which is a two-minute run
that nobody made, because the case had gone straight to its full configuration and there was
nothing cheaper to fall back to.

Worth noticing that the tutorial itself was not the problem. A tutorial is a point in
configuration space known to work, and four simultaneous steps away from it is four
untested changes however good the starting point was. For a genuinely new case there is no
tutorial at all, and then a manufactured known-good point is the only one available.

The last rung on any ladder is the case as asked, and it is the one rung with no
independent answer -- which is the reason the others are underneath it. When it disagrees
with a published value, the disagreement is already narrowed to what that rung added.

`ladder.py` reads a case and prints one: the class it appears to be with the evidence for
that reading, then the rungs with their known answers, tolerances and costs, and `--rung n`
with the dictionary edits that would build a particular one. `--record n` writes a rung's
outcome into the study manifest on the volume, and results recorded there survive the
session and the sandbox, which is how a solver choice established in round 1 of a study
was lost to round 2 -- the evidence lived only in a transcript the fresh thread could not
see. It edits no case, runs nothing, and exits 0 whatever it says. Skipping every rung is a legitimate choice and the
report says so: a case adapted from a worked tutorial already sits on a known-good point,
and the ONERA M6 that succeeded this week succeeded by taking HiSA's own worked M6 case
almost unchanged -- one reference, one adaptation, one run, no ladder needed.

## Locating beats explaining

On the Wigley free-surface case, eight hypotheses about a divergence were proposed and all
eight were falsified, over five rounds and two days. What settled it in one step was not a
better mechanism but a measurement with three parts: *where* the extremum sat (the first
cell row at the inlet, x = -4.4375), *which field moved first* (p_rgh, at the first write,
before omega and before U), and *whether the value survived a change of mesh* (-4723 Pa on
two domains with completely different cell sizes, agreeing to four significant figures). A
quantity invariant to the mesh is set by the boundary specification, not by the grid, and
that one sentence ended what five rounds of argued mechanisms had not. The asymmetry is
worth keeping: a mechanism that explains every symptom is still a hypothesis, because a
wrong mechanism explains them too; a location is a measurement, and the fault has to live
where the measurement is. No second hypothesis is worth much until a localisation has
falsified the first.

`locate.py` makes the three measurements in one call: the last written fields with each
extremum's cell, its coordinates, the patch its boundary faces belong to and its distance
to the domain boundary; the solver log read for the field whose residual or bounding
warning degrades first, with the time it starts and the Courant tail; and
`--compare <other-case>` for the mesh-invariance check against a differently meshed
sibling. It reads and prints, edits nothing, and exits 0 whatever it finds. What it
deliberately never prints is a mechanism.

## Where rendering happens

Stills render on the instance, next to the data. A field render reads hundreds of
megabytes of mesh and fields to make a 100 KB PNG, so moving the data to the
renderer is the wrong direction; pyvista + OSMesa are in the image for exactly
this, and every picture left on disk mirrors to the user's machine within moments.

Video is the opposite case. Encoding needs no case data -- only the frames -- and
the image ships no encoder (no ffmpeg), so assembling a video on the instance
fails. The route that works: write the frames with a fixed camera into a directory
whose name ends in `_frames` (or is called `frames/`), and that is the whole of it.
The frames mirror home as they are written, and the harness assembles them into a
gif on the user's machine on its own -- no `fetch`, and nobody has to be told the
pictures exist. Because the frames come home one at a time, the harness will
assemble a partial gif from the frames-so-far while the rest still render, which is
what someone means by "a gif of what you have so far." `animate.py` in the toolbox
writes frames this way and is built to run as a job (rendering fifty frames outlives
a synchronous `bash` call's ceiling). The same logic fits any artifact: work that
needs the data runs where the data is; work that only needs the results comes home.

Stills are delivered the same way -- every `.png` written anywhere in the case
lands in the user's flat `renders/` folder as it mirrors, so a render is something
the user has, not something they have to be handed. There is nothing to remember to
send.

`results.py` is the standard set of those pictures by preset -- `external-flow-2d`,
`duct-flow`, `transient-wake`, `vehicle-aero`, `mesh-validation` -- so the pressure,
velocity, vorticity, streamline, residual and force plots come out of one call instead
of a render script written again per study. `gallery.py` turns everything that has been
registered into one self-contained page and one contact sheet.

## Long solves

The container has a 24-hour ceiling. A job that hits it ends with
`end_reason: sandbox_expired`, and the volume is untouched — the case, every write time, and
the log are all still on disk. OpenFOAM restarts natively from where it left off:

```
startFrom       latestTime;
```

Which makes `writeControl` and `writeInterval` worth a moment's thought before a long run
rather than after one. Intervals sparse enough not to fill the disk, frequent enough that an
interrupted solve resumes somewhere useful, and `purgeWrite` set with the times you would
actually want to look at later in mind.

### The ceiling is not the only way a run is interrupted

The 24-hour limit is the *advertised* end. In practice a container can also be reclaimed by
the platform at any moment, and on a bad afternoon that happens every ten to fifty minutes.
The recovery is identical -- `startFrom latestTime` and the volume is untouched -- so the
question is not whether a long run survives but how much each interruption costs.

Two numbers set that, and they are both chosen before the run starts:

- **`writeInterval` is the amount of work an interruption can destroy.** A solve reaching
  iteration 895 with its last write at 850 loses 45 iterations, every time, and if the
  interruptions come faster than the interval the run makes no net progress at all. Sparse
  enough not to fill the disk, frequent enough that the loss is smaller than the gap between
  interruptions.
- **The larger cost is usually not the lost iterations.** It is the turn spent noticing the
  job died and starting it again, which is minutes of wall clock and a paid model turn for
  what is mechanically a one-line restart. A solve launched inside a loop restarts itself:

```sh
until mpirun -np 4 interFoam -parallel >> log.run 2>&1; do
    echo "$(date -u +%T) solver exited non-zero; resuming from latestTime" >> log.run
    sleep 5
done
```

  With `startFrom latestTime` in `controlDict` each pass picks up the last write by itself.
  The loop cannot outlive the container it runs in, so it does not help when the whole
  sandbox goes; what it removes is every *in-sandbox* failure, which is the more common kind.

A run designed this way is also easier to report honestly, because "stopped at iteration N
with the last write at M" is a fact about the platform rather than a gap in the work.

## An analytical answer is sometimes the better instrument

For a bounded set of problems there are published correlations that are cheaper, better
characterized, and often more accurate than a RANS solve: pipe and duct pressure drop
(Darcy–Weisbach with minor-loss coefficients, Idelchik, Crane TP-410), fan and system
curves, simple heat-exchanger duties, air-change and mixing-time estimates, hover thrust and
disc loading. Where one applies, it is worth computing — as the answer, or as the number the
simulation should land near. Reasons to simulate anyway are real and easy to state: spatial
field information, geometry off the correlation's envelope, evidence that needs a
distribution rather than a bulk number, a design sweep that runs out of correlation
coverage.

## What a study leaves behind for the next session

A session can end in the middle of a solve: the 24 h Sandbox ceiling, the idle reaper, a
preemption, a closed window. Everything under `/work` is on the Volume and survives that;
what does not survive is the reasoning -- which case was the real one, what had already
been rendered, whether checkMesh had been run and what it said. Re-deriving that from the
files costs a session's first twenty minutes.

`study_state.py` writes two plain files under `<study>/.reynolds/` for exactly that gap: a
manifest saying what each artifact is *for* (`mesh-full`, `vorticity`, `residuals`, ...)
and a phase table saying how far the study got. `study_run.py` reconciles that table
against what is actually on disk, so a study advanced by hand between sessions is read
from the evidence rather than from the record. `progress_report.py` answers "where is it?"
from the same two files plus the solver log, and `gallery.py --final` prints the path of
the newest artifact of each kind.

## Waiting on a job without paying for it

A running solve invites the reflex of pacing with `sleep` inside `bash` -- `sleep 200;
grep "^Time" log` -- and it is a trap twice over. A synchronous command is capped
(300 s), so a long sleep is cut off and reads back as an error rather than a wait; and
even a short one burns the cap on doing nothing. The tool built for this is
`job_check` with `wait_s`: it holds its answer until the job ends or the wait runs out,
up to 300 s, and that wait is the harness's own -- it does not count against any command
timeout, and it ends early the moment the user says something. Calling it again is free.
So a wait is `job_check(job_id, wait_s=300)`, not `sleep` in a shell. The harness is
also watching every running job on its own and can show progress without being asked, so
polling by hand mostly is not needed at all.

## Steady or transient is a physics decision, not a default

A steady solver (simpleFoam) looks for a time-independent state, and for a great many
flows there is one. But a bluff body above its shedding onset -- a cylinder past
Re ~ 50, a square or an L or any vehicle-like shape in the hundreds and up -- has no
steady state: the wake sheds vortices forever. A steady RANS run on such a case does not
fail loudly; it converges to a damped, symmetric, time-averaged-looking answer that
quietly omits the shedding, the unsteady loads, and the Strouhal number -- often the
very things the study was about. The tell is a lift/side-force coefficient that wants to
oscillate and a residual that plateaus instead of falling. Deciding steady vs transient
from the regime before solving, and saying which was chosen and why, saves the redo that
otherwise arrives when someone looks at the result and asks for "the transient one." A
transient run (pimpleFoam) costs more but is the honest instrument for a shedding flow;
a steady run is right for an attached, genuinely steady one.

## Choosing a solver class from the deliverable

Steady or transient is one axis. Pressure-based or density-based is a second one, and it
is decided by a different question: not how the flow behaves in time, but whether the
answer is a **discontinuity**.

A pressure-based solver -- `simpleFoam`, `rhoSimpleFoam`, the `pimple` family -- reaches a
transonic state through a pressure equation. Keeping that equation stable at M ~ 0.85 needs
`cellLimited` gradients and an upwinded pressure flux, and those are the same terms that
spread a compression. The front arrives eight or ten cells wide, and it arrives that wide
on a fine mesh too, because the width comes from the scheme rather than from the cells. A
density-based solver -- `rhoCentralFoam`, `sonicFoam` -- reconstructs the flux itself with
a limiter, and a shock lands in about three cells and stays there.

The incident. A 1.79M-cell ONERA M6 wing, refinement box driven through the whole
supersonic pocket rather than just the skin, farfield at sixteen chords, converged: 3228
iterations, drag flat to six significant figures. No shock at any of the seven span
stations, lift 22% low, drag 50% high. Two rounds went into the mesh and the farfield, and
neither was ever the problem; the deliverable was unreachable from the first line of
`fvSchemes`. An earlier session on the same case had written down that a density-based
solver was the way to resolve a lambda shock, and the session that needed it a day later
did not have that note.

What "cannot resolve it" looks like from inside: a smooth, monotonic compression ramp at
*every* station, with the suction plateau sitting at roughly half the measured value. That
is not the signature of a coarse mesh, and refining does not move it. A coarse mesh gives a
shock in the wrong *place*, or a wobbly one; a smearing scheme gives no shock at all.

Two scheme traps sit underneath the solver choice, and both survive a report that says
"less dissipative schemes were used":

- Upgrading `div(phi,U)` to `linearUpwind` while `div(phid,p)` stays `Gauss upwind` changes
  the momentum equation and leaves the term that actually carries the compression at first
  order.
- `cellLimited Gauss linear 1` clips hardest exactly at an extremum, and a shock is an
  extremum, so a fully limited gradient reverts to first order at the one place the answer
  lives. A coefficient nearer 0.33, once the start-up transient is past, keeps the limiter
  where it is needed and lets the front alone.

Worth separating, before meshing, whether the deliverable is a *number* or a *shape*. A
coefficient can come out inside a published band while the structure that is supposed to
produce it is absent, which is the one case where agreement is not evidence. A shape has a
topology, a position at every station, and a place where the topology changes, and none of
those are reachable by accident. `preflight.py --resolve shock` reads the solver and the
schemes and answers in about a second, before anything is meshed; it suggests and does not
refuse, and there are good reasons to run a smearing scheme knowingly -- for the loads, for
a starting field, to debug a mesh -- so long as the report says which of the two it has.

## The shock-capturing solver, and how to tell where it is

The section above ends at "reach for a density-based solver" and leaves open which one.
The image ships `rhoCentralFoam` and `sonicFoam`, both explicit, and -- on newer images
-- **HiSA 1.13.4** (`gitlab.com/hisa/hisa`), an *implicit* density-based OpenFOAM solver:
AUSM+up flux, `wVanLeer` reconstruction, dual-time stepping with local time-stepping,
GMRES under an LU-SGS preconditioner. It was first compiled onto a workspace volume on
2026-08-30, on the day the ONERA M6 replication described above needed it; it is baked
into the image now, at the OpenFOAM site directories
(`$WM_PROJECT_DIR/site/2512/platforms/linux64GccDPInt32Opt/{bin,lib}`), which the
sourced `bashrc` puts on `PATH` and `LD_LIBRARY_PATH` on its own. On such an instance
`which hisa` answers after any ordinary sourced shell, no exports are needed, plain
`mpirun -np N hisa -parallel` works, and the solver survives instance deletion because
the image is not the volume.

An instance running an older image may instead carry the 2026-08-30 build on its
*volume*, which is a fact about that instance and not about OpenReynolds -- deleted and
rebuilt, it has no HiSA and no sign that it ever did. That build is invisible to a plain
shell and answerable in a second:

    export WM_PROJECT_USER_DIR=/work/OpenFOAM/user-v2512
    export FOAM_USER_APPBIN=$WM_PROJECT_USER_DIR/platforms/linux64GccDPInt32Opt/bin
    export FOAM_USER_LIBBIN=$WM_PROJECT_USER_DIR/platforms/linux64GccDPInt32Opt/lib
    export LD_LIBRARY_PATH=$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH
    which hisa            # or: ls -l $FOAM_USER_APPBIN/hisa

`hisa_env.py` in this directory is the whole check written down: it looks in the image's
site directories first and at the volume second, says which of the two it found, runs
`ldd` over the binary so a half-broken build reads as broken rather than as missing,
prints the export lines when they are needed and says so when they are not, and prints
the path to the bundled example. It reports and changes nothing.

A *volume* build needs those four lines in every shell -- OpenFOAM's own `bashrc` does
not set `WM_PROJECT_USER_DIR` to a path outside `$HOME`, so a solver that runs
interactively will still fail to launch from a job unless the job's own command exports
them. `mpirun` needs them forwarded explicitly: `mpirun -x WM_PROJECT_USER_DIR -x
FOAM_USER_APPBIN -x FOAM_USER_LIBBIN -x LD_LIBRARY_PATH -np N hisa -parallel`. Without
the `-x` the ranks start in a clean environment and die on a missing `libhisa*.so`,
which reads like a solver crash and is not one. None of this applies to the image build,
which lives on paths the sourced environment already carries.

HiSA's own worked ONERA M6 case ships with either build: at
`/opt/hisa/examples/oneraM6/simulation` in the image, or at
`/work/hisa/hisa/examples/oneraM6/simulation` beside a volume build's source tree
(`hisa_env.py` prints whichever it found). That case is the fastest route to a
correct dictionary set by a wide margin: HiSA's `fvSchemes`, `fvSolution` and
`thermophysicalProperties` do not look like a `rhoSimpleFoam` case's, several of the
entries are HiSA's own, and the introspection utilities that would let you discover them
(`foamToC`, `foamInfo`) are not in this image. Copying the example's `system/` and
`constant/` and changing the freestream is minutes; deriving them from a pressure-based
case is hours of rejected tokens.

**Implicit or explicit is a cost question, not an accuracy one.** `rhoCentralFoam` is
already in the image and resolves shocks perfectly well, and it was still the wrong
instrument here, for a reason worth carrying to any layered mesh: an explicit solver's
timestep is set by the *smallest* cell in the domain, not by the cell you care about. On
the M6 mesh the shock cells were 4.7 mm and the first prism layer was 165 um, a factor of
28, so the acoustic CFL limit came from the boundary layer and the run needed on the
order of 92,000 timesteps where the implicit solver needed about 6,000 pseudo-steps. The
same mesh, the same physics, an order of magnitude apart in money. A layered mesh
punishes an explicit solver in proportion to the layer thickness ratio, and that ratio is
usually chosen for y+ reasons that have nothing to do with the shock.

What it produced, on the same 1.79M-cell mesh the pressure-based run had already failed
on: the lambda structure resolved -- two distinct compressions at eta 0.44, 0.65 and 0.80
merging into a single front by eta 0.90 -- shock positions within 0.06c of AGARD AR-138
at every station, and CD within 0.3% of a five-code published band. 112 minutes and
$4.10. The pressure-based attempt that preceded it had cost two rounds of meshing and
produced no shock at all.

One number from that post-mortem is worth stating precisely, because it is easy to
inflate. First-order upwind carries a numerical diffusivity of about u*h/2, which at
269 m/s and a 4.7 mm cell is 0.63 m^2/s against a physical viscosity of 1.48e-5. That
ratio is not an effective Reynolds number for the case -- momentum was nominally second
order, and the bound applies locally, at the extrema where the limiter clipped and at the
pressure flux that stayed upwind. It is the right size to explain a smeared shock and the
wrong thing to quote as a global statement about the solve.

## Keeping a study 2D

A "2D" case in OpenFOAM is a 3D mesh one cell thick in the third direction with the two
faces normal to it patched `empty`. The pieces that make it actually 2D, all of which
have to agree: one cell in z in `blockMesh` (or a single-layer extrude), a front and a
back patch both of type `empty` in `constant/polyMesh/boundary`, and no z-velocity
anywhere in the setup. `checkMesh` reporting the empty patches and a cell count that is
the 2D count (not multiplied by a z-resolution) is the confirmation; a geometry or mesh
render viewed edge-on confirms it to the eye before any solver time is spent. The common
way it goes wrong is a mesh built with several cells in z, or side patches left as
`patch`/`wall` instead of `empty` -- which runs, slowly, as a thin 3D case and reads as
3D to anyone looking at the result. When the request says 2D, the geometry render is the
cheap place to catch it, not the finished solve.

Two things in the toolbox already know this pairing. `case_gen.py` writes 2D cases with
one cell in z and a single `frontAndBack` patch declared `empty` in the dictionary and in
every field, so the disagreement has no way to arise; `preflight.py`'s `empty` check reads
the mesh -- or the `blockMeshDict`, when the mesh has not been built -- against every
field in `0/` and says which file disagrees with which.

## Files the person sends up, and PDFs in particular

A hosted session's uploads land in the study's own `uploads/` directory, and the
person's message names each path when it arrives. A PDF is worth a special word
because its raw bytes read as noise: by the time the message arrives, the service
has already unpacked it into a directory named after the file, holding one PNG per
page named `page-01.png`, `page-02.png` (two digits, a dash) and the text layer as
`text.md` with per-page headings. `read_file` on one of those PNGs returns the page
as a picture, so a drawing, a title block or a plot can be looked at directly; the
message's own note names the exact files. When a name does not answer, `ls` the
directory rather than guessing variants -- the one time this went wrong, a session
guessed `page_1.png` for `page-01.png` and then spent ten minutes on the road below.

The road below: the workspace has no route to the internet. `pip install`, `apt
install`, `git clone` and every other fetch hang or fail there by design, always,
and nothing is gained by waiting on one or retrying. What the image ships is what
there is -- and it ships a lot (OpenFOAM v2512, cfMesh, HiSA, ParaView's pvpython,
Python with numpy and pyvista, and `pdftoppm`/`pdftotext` from poppler for
re-rendering a drawing at higher resolution than the unpacked 2400 px when small
dimension text needs it: `pdftoppm -r 300 -png plan.pdf out`). If a tool truly is
not there, the honest move is to say so and work with what is, the way any solve
here already does.
