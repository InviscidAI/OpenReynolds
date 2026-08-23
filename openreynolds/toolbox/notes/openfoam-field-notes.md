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

**Verify** — quantity of interest never plateaued; grid-convergence index too high or
non-monotone; mass imbalance; result outside its expected band; dimension mismatch;
definitional mismatch (right units, right plumbing, measuring a different quantity).

On layer collapse specifically: widespread collapse usually responds to fewer layers or a
gentler expansion ratio; collapse localized at features responds to lower `minThickness` and
a higher `featureAngle`; curvature-driven collapse responds to one more level of local
surface refinement. Partial coverage away from the region you care about is often simply
acceptable — worth saying so out loud rather than chasing it.

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

y⁺ is not a 30–100 band to be tested over a histogram. Modern `kOmegaSST` in OpenFOAM uses
blended wall treatments that are deliberately y⁺-insensitive across the buffer layer, and
every real geometry has stagnation points and separation lines where y⁺ → 0 no matter how
good the mesh is. The useful question is what fraction of the wetted area on the surfaces
that matter sits in the wall treatment's valid range.

Grid convergence: Richardson extrapolation presumes monotone convergence, and real triplets
often do not deliver it. If the sign of the change reverses across three grids, the honest
output is the spread across the ladder as an interval, labelled oscillatory — not a
grid-convergence percentage that presumes the thing it is missing.

## Per-class starting points

Solver names are ESI. These are places to start, not defaults to defend.

| Class | Solver | Turbulence | Domain | Note |
|---|---|---|---|---|
| internal duct, steady | `simpleFoam` | `kOmegaSST` | fit + development lengths, ~10D in / 20D out | usually well behaved |
| external aero, steady | `simpleFoam` | `kOmegaSST` | ~10D upstream, 20D downstream, 10D sides | vary the schemes; bluff or post-stall bodies are shedding candidates |
| indoor / buoyant | `buoyantSimpleFoam` | `kEpsilon` | room plus plenum | |
| transient shedding | `pimpleFoam` | `kOmegaSST` | per class | a short-window check before the long run pays for itself |

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
