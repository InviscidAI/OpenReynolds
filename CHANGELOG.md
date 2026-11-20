# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Three `case_gen.py` flags a design study needs and a hand-written case does not.**
  `--yplus` writes the `yPlus` function object at `writeTime`, so the wall spacing a
  turbulence model assumed can be read off the case rather than asserted; `--dp` writes
  two `surfaceFieldValue` objects -- `areaAverage` of `p` on the first inlet and the
  first outlet, every time step, into `postProcessing/pInlet/` and
  `postProcessing/pOutlet/` -- because a pressure drop is the number an internal flow is
  judged by and it is not a force on a body, so `forceCoeffs` does not carry it and no
  amount of reading the log recovers it; and `--periodic A,B` couples two patches as a
  cyclic pair in `constant/polyMesh/boundary` itself and drives the cell at a fixed bulk
  velocity with a `meanVelocityForce` source in `constant/fvOptions`, writing the pitch
  it measured off the two patches' centres to `periodic.json`, so one stage of a
  many-stage device is measured without paying for the other stages and the end
  fittings. `cyclic` when the two patches have the same face count, else `cyclicAMI`;
  `--periodic-kind` overrides. The three function objects are assembled as lines beside
  the ones already there (`case_gen.force_coeffs`), so a case with no body, no drop and
  no y+ still gets no empty `functions {}`. Split out of #54 at review; the consumers of
  `periodic.json` and the two pressure records arrive with `score.py` in that PR.

- **Every case this repository can run is in the tutorial index, with the scale each
  reference is normalised on.** The index listed three of them. `benchmarks/tutorials/`
  carried the heated pipe, the buoyant cavity and the backward-facing step, while the
  lattice Boltzmann channel and cavity, the three SPH cases, the thermal-stress bar, the
  panel method, the plane duct solved in the `journey` job and the four graded cases in
  `felix/` and `moving_mesh/` were reachable only by finding the directory they live in.
  `benchmarks/tutorials/README.md` is now the index of all of them, sorted by what it
  costs to run one: tier 1 needs nothing but this package and no account, tier 2 needs an
  OpenFOAM, and tier 3 costs a workspace and a model turn and is run by nothing
  automatically. Every row carries the command, the reference, **the length and velocity
  scales that reference is normalised on**, and the agreement a real run measured; rows
  with no verdict say so in words rather than leaving the cell empty. The normalisation
  basis is in every row because that is where this repository's two wrong reference values
  lived -- Gartling's 6.1 is in channel heights and had been read as step heights, a factor
  of two that reached a customer-facing page, and de Vahl Davis's 4.519 is his cavity
  average and not his hot-wall 4.509 -- and both errors flattered the result, which is why
  neither was noticed.
- **The seven templates have a tutorial.** `benchmarks/tutorials/templates.md` is the
  tier 2 page: what each of `duct2d.py`, `body_in_box.py`, `free_surface.py`,
  `rotating.py`, `porous.py` and `thermal_stress.py` builds, and **what is known about the
  answer before the solver starts.** Two of them end in an exact comparison CI already
  runs -- the plane duct against Poiseuille in `tests/test_journey_duct.py` and the bar
  against `-E alpha T` in `tests/test_journey_thermal_stress.py`. The porous slab's
  coefficient route is exact by construction and is checked in both directions (the fit
  run backwards against its own four points, worst 7.6e-5%; the independent Ergun
  reduction recovering the same `d` and `f` to 2.7e-4%), and its **solve is recorded as
  ungraded**, because nothing here has solved it and quoted a drop against the 503.34 Pa
  the template prints. The dam break and the frozen rotor have no reference and are
  written that way, with the two conservation statements that are still exact for a
  rotating case -- the impeller and vessel-wall torques equal and opposite, and
  power = torque x omega -- and the sphere's exact Stokes `Cd = 24/Re` listed as a gap
  rather than quoted with no measurement beside it.
- **Two more rows in the benchmark index**: the plane duct against the exact Poiseuille
  expressions, and the porous slab's coefficients. Both were graded in CI already and
  neither was in `benchmarks/README.md`.
- **A row for `templates/thermal_stress.py` in the toolbox index**, which had it in the
  prose and not in the table, so it was the one template a session reading the table could
  not find.
- **Fluid-structure interaction, as a case generator and an instrument rather than a
  feature.** `benchmarks/moving_mesh/` had graded a spring-mounted cylinder in September
  and nothing in the toolbox could set one up or read one back, so the groundwork was a
  benchmark nobody could reuse. Two scripts close that:
  - `openreynolds/toolbox/templates/viv_cylinder.py` writes the two `pimpleFoam` cases a
    flow-induced vibration study actually is -- `fixed/` with the body bolted still for
    the Strouhal number, `released/` on `dynamicMotionSolverFvMesh` +
    `sixDoFRigidBodyMotion` with a linear spring and zero structural damping -- off two
    edited numbers, the diameter and the reduced velocity. Mass comes from the mass ratio
    and the displaced fluid mass, stiffness from U\*, free stream from the Reynolds
    number. The mesh is the O-grid plus H-grid `blockMeshDict` of the graded 2026-09-12
    run, reproduced token for token: 898 tokens each, with the 42 wall-circle coordinates
    agreeing to 1e-11 m (that run rounded its dedup key to 10 decimals). 240 cells around
    the wall, 2.5% blockage, 51,150 cells, which is the mesh that measured St = 0.166141
    at Re = 100 against Williamson's 0.164 for parallel shedding, +1.3%.
  - It ships **tightly coupled** -- `moveMeshOuterCorrectors yes`, three outer correctors,
    `accelerationRelaxation 1.0` -- because the cheaper scheme is what held a limit cycle
    13% too wide on that run, and the comments say so where the settings are.
  - `openreynolds/toolbox/viv.py` reads a finished case back: A/D, the response frequency
    against both the structure's own and the frequency the same mesh shed at with the body
    still, lock-in decided rather than asserted, the net fluid work per whole cycle, and a
    three-panel figure. On the two committed runs in `tests/data/moving_mesh/` it
    reproduces `benchmarks/moving_mesh/grade.py` digit for digit -- A/D 0.640193 and
    0.567028, work -10.3734 to -10.0933% and +0.0437 to +0.1756% of the body's energy per
    cycle -- and `tests/test_toolbox_viv.py` is what holds it there.
  - It also prints the natural frequency with the added mass in it, which nothing here
    did. A circular cylinder's added mass is exactly its own displaced mass (Lamb,
    *Hydrodynamics*, 6th ed., art. 68), so at m\* = 10 the still-fluid natural frequency
    is 0.9535 f_n, and the tight run's measured f/f_n of 0.9697 is **1.7% above** the
    frequency it would ring at in water rather than 3% below resonance. That is a 5%
    reading error on a quantity these cases are reported to a percent on.
  - The amplitude is printed and not graded, and `benchmarks/moving_mesh/README.md` now
    says why more precisely than it did (see Fixed).
- **Particle tracking, as post-processing over a solved field, graded against a published
  capture threshold.** `openreynolds/toolbox/tracks.py` integrates particle and tracer
  paths through a finished Eulerian field with RK4, vectorised over the seed set: drag
  (Stokes or Schiller-Naumann) plus buoyancy-corrected gravity, `--stokes 0` for massless
  tracers, JSON out with every seed's fate and a PNG of the paths. It reads a real case
  through the same reader `render.py` uses.
  - Post-processing rather than a parcel-cloud generator, stated as a choice: `DPMFoam`,
    `DPMDyMFoam`, `MPPICFoam` and `MPPICDyMFoam` are in `$FOAM_APPBIN` on the instance and
    `tutorials/lagrangian/` carries four more, measured out of
    `studies/20260823-213712-babc/messages.jsonl`, but the exact v2512 spelling of the
    uncoupled kinematic solver could not be verified from outside the image, and a
    generator naming a binary nobody has run is a generator that fails on the instance.
    For a dilute, one-way-coupled cloud the physics is the same either way.
  - `benchmarks/particles/` grades it, and like `lbm/` and `sph/` it **solves** in the
    test job rather than reading a committed result. Three cases: velocity relaxation
    into a uniform stream and terminal velocity under gravity, both exact closed forms
    (1.9e-12 of U, 2.2e-14); a tracer orbit in solid-body rotation that must close, 5.1e-8
    of the radius with an observed order of 4.00; and the critical Stokes number for
    inertial capture on a circular cylinder in inviscid flow, **0.1262909 against 1/8,
    +1.03%**.
  - That 1/8 is the factor-of-two trap this repository has already paid for twice. The
    source read is Robinson, Warren, Turner & Sear, arXiv:2310.03474v2 section III.3,
    which states St_c = 1/4 in a convention whose U is the local stagnation-point speed
    (twice the free stream) and whose length is the radius, and which says in the same
    paragraph that Langmuir and Blodgett quote 1/8 because they use U_inf. The convention
    `tau_p U_inf / R` is written into the threshold's key, the printed line, the README and
    a test. Taylor (1940) and Langmuir & Blodgett (1946) could not be read and are named
    as unread.
  - The residual +1% is diagnosed, not quoted: the threshold is an infinite-time statement,
    so a finite stopping rule can only measure it high, and the measurement walks
    +5.671 -> +3.152 -> +2.000 -> +1.375 -> +1.033% as the stopping speed tightens from
    1e-6 to 1e-14 of U, then stops at the double-precision floor (1e-16 measures the same
    0.1262909). A factor of five in the time step moves it 0.12% and the release point
    -3 to -20 radii moves it 0.01%, so it is neither discretisation nor the far field. The
    grader fails a measurement *below* 1/8 as a bug rather than passing it as a better
    answer.

### Fixed

- **`benchmarks/moving_mesh/README.md` quoted an amplitude a paper does not state.** Its
  nearest reference for the VIV amplitude, Menon & Mittal (arXiv:2006.11649, published as
  *J. Fluid Mech.* **907**, A37, 2021), was re-read at source and gives no A/D number in
  its text at all -- the value at U\* = 5 is readable only off its Figure 5(a), so "about
  0.56" was a figure read presented as a citation and now says so. The mass-ratio note was
  right and is now exact: the paper's definition is `m* = 2m/(rho D^2)` verbatim, so its
  m\* = 10 is 6.37 on displaced mass against this case's 10, a 57% difference in body mass
  between two papers both saying "m\* = 10". Placzek, Sigrist & Hamdouni could not be
  re-read (HAL and the publisher both refuse an automated request) and is now marked as
  carried over unverified rather than reading as a checked figure. Nothing is graded
  against either, which is the only reason the first error cost nothing.

- **Learn mode and beginner mode, as a depth the person chooses.** `study_style.py` gains
  `learn` and `beginner` beside `quick` and `full`, and `openreynolds --depth` makes all
  four reachable from the terminal, where only the MCP servers and the hosted app could
  ask for one before. Both new styles are built on `quick`'s scope -- a run somebody is
  reading along with is a run they want to see the whole of -- with the telling added on
  top: `learn` asks for the reasoning behind each choice as the work goes, `beginner` for
  plain language, the terms spelled out, and what the answer does not support.
  - Shape first, because this is the roadmap item most easily built wrong. A teaching
    mode that injected a checklist into the prompt, demanded an explanation before each
    tool call or held the solve until the person answered would be the workflow injection
    `docs/design.md` section 1 forbids. These are text in the person's voice, relayed
    verbatim by the briefing like any standing note, and they gate nothing: `modes.py`
    stays the only thing in the repository that holds a tool call.
  - Each note ends by saying not to wait. A request to be taught reads as an invitation
    to stop and check understanding, and a turn that ends on a question ends the work --
    the failure `modes.AUTO_NO_QUESTIONS` was written for after a full-auto session opened
    with a numbered list of questions and then meshed for seven minutes while the list sat
    there (study 20260920-161908-c7ef).
  - Nothing about billing changes. The service sets a study's ceiling from a style of its
    own and reports it back (`ceiling.style_of` only ever sees `extra["style"]` from the
    service), so the two new words are not a third billing vocabulary and
    `ceiling.NORMAL_CENTS` keeps the service's two.
- **`toolbox/amr.py`: adaptive mesh refinement, which on this image is a dictionary.**
  OpenFOAM refines a running mesh through `dynamicRefineFvMesh` named in
  `constant/dynamicMeshDict`, and nothing here wrote one: `case_gen.py` dresses a mesh and
  stops there. `write` writes it; `check` reads a case back. Every fact in the script was
  read from the OpenFOAM v2512 tag on 2026-11-21 rather than from a tutorial copy.
  - The nine entries that have no default are the deliverable. `correctFluxes` and
    `dumpLevel` are read in `readDict()`, and `refineInterval`, `field`,
    `lowerRefineLevel`, `upperRefineLevel`, `nBufferLayers`, `maxRefinement` and
    `maxCells` in `updateTopology()`, each with `get<>`, so a dictionary short of one
    fatals at the first update. Only `unrefineLevel` has a default (`GREAT`, which never
    unrefines), so left out the mesh only grows, and the dictionary says so where it is
    left out. `correctFluxes` being mandatory is the one nobody expects.
  - Refused here rather than at the first update: the three fatal ranges
    (`refineInterval` >= 1, `maxCells` > 0, `maxRefinement` > 0, whose own FatalError
    misnames itself as `maxCells` and is quoted with that correction), and
    `lowerRefineLevel >= upperRefineLevel`, which OpenFOAM accepts -- the band is empty,
    nothing is ever marked, and the run pays for the topology machinery and refines
    nothing.
  - `check` also reports `maxCells` against the mesh's own cell count where a polyMesh
    header carries one, and every `surfaceScalarField` on disk that `correctFluxes` does
    not name, each of which gets `WarningInFunction "Cannot find surfaceScalarField"` and
    an unmapped flux at exit 0.
  - **No accuracy and no saving is claimed.** The honest check on refinement is
    self-consistency -- a refined run agreeing with a uniformly fine one, for fewer cells
    -- and no such pair has been run on this image, so the script names the instruments
    that would make the measurement (`grid.py`, `mesh_digest.py`) and states that nobody
    has. `preflight.py`'s reading of `dynamicRefineFvMesh` as `ran=False` is unchanged.
  - Two limits are stated rather than checked: non-hex cells are protected from
    refinement by `hexRef8` (`bitSet protectedCell_`, "Protected cells (usually since not
    hexes)"), so AMR refines the hexes in a mesh and leaves the rest; and whether the type
    is registered in the build stays `preflight.py --only probe`'s question.
- **Polyhedral meshing, as the same dictionary and a different binary.** `cfmesh.py case`
  takes `--mesher pMesh`. `pMesh` builds a `voronoiMeshGenerator`, whose own description
  is "takes a triangulated surface and generates a body-fitted polyhedral mesh", and it
  reads `IOobject("meshDict", runTime_.system(), ..., MUST_READ)` -- the same
  `system/meshDict` this script already wrote, handed to the same `checkMeshDict`,
  `meshOctreeCreator`, `refineBoundaryLayers` and `renameBoundaryPatches`. So the
  dictionary needed no change and only the printed route's mesher and log name do. It had
  been probed by `where` and unreachable since the script was written.
  - What is not claimed: no polyhedral mesh has been run on this image and measured, so
    the Wigley hull's 100.0% layer coverage and 1.21 mm first cell stay attached to
    `cartesianMesh`, and the new note says they do not transfer.
  - The other route is recorded beside it: `polyDualMesh <featureAngle>` takes the dual of
    a mesh that exists, one cell per point, which also means the boundary layer you had is
    gone. It is in the ESI v2512 tree at
    `applications/utilities/mesh/manipulation/polyDualMesh/polyDualMeshApp.C`.
- **`toolbox/scratch.py resume`: which time an interrupted run can restart from.** The
  promise the script already made -- "a preempted run resumes from `latestTime` with
  nothing lost" -- has one silent hole: `checkpoint` completes a mid-write time directory
  on the next pass, and a preemption is the pass that never comes, so the newest time on
  the Volume is whatever `rsync` had reached and `startFrom latestTime` resolves to it by
  name. `resume` compares that directory with the one before it, checks every file for the
  end divider OpenFOAM writes last, counts a time only where every rank has it, and names
  the newest time that is whole plus the directory to move aside. It reads and prints; the
  two edits it names are left to whoever is reading.
- **The journey a user takes, solved in CI and checked against Poiseuille.** Every test
  in the repository was either a unit test against a fixture or a tool run against a
  study somebody had already finished, so a seam between two stages was invisible to all
  of them. `tests/test_journey_duct.py` walks the chain from an empty directory: the
  `duct2d.py` template builds a 0.10 x 0.02 m plane duct with gmsh and `gmshToFoam`,
  `case_gen.py` reads that mesh and writes `0/`, `system/` and `constant/` for a steady
  laminar solve at Re 10, `simpleFoam` solves it, and then `probe.py`, `balance.py`,
  `wall.py`, `log_digest.py`, `results.py` and `report.py` read it back. It runs in a
  new `journey` job inside the workspace image -- 1 min 40 s of runner time, of which
  the walk itself is 7.3 s -- and unlike the advisory `image` job it gates.
  - The reference is exact rather than published, which is what makes the assertions
    physics and not exit codes: developed plane Poiseuille flow is three lines of algebra
    off the steady Navier-Stokes equations, and `u_mean` is 1 m/s because the inlet is a
    uniform 1 m/s `fixedValue` and mass is conserved. Measured against it: `u_max` 1.4863
    against 1.5 exactly, `u_max/u_mean` 1.532 against 1.5, the developed `(dp/dx)_kin`
    -123.5 against -120 m/s^2, the wall shear the measured y+ implies 1.08 and 1.10
    against 6 nu u_mean / H = 1.2 m^2/s^2, and the throughflow 3.93e-5 against
    U x H x thickness = 4e-5 m^3/s.
  - Asserted at the seams rather than at the end: the mesh is one cell thick with its z
    faces `empty`; `nu` on disk is `U L / Re`; `p`'s dimensions are `m^2 s^-2` and not
    pascals; the solver's own continuity closes to -9e-08 of a 4e-05 throughflow and
    `balance.py`'s boundary integral to 0.84%; `wall.py`'s `nu` came out of the case's
    own dictionary rather than a default; and `report.py`'s key numbers are the residuals
    `log_digest` reads out of the solver's own log, number for number.
  - Two of this test's own reference numbers were wrong on its first CI run and are now
    measured instead of assumed. `CELL_SIZE_M` in the template is a background size
    handed to gmsh, not a division count, and the quad recombination does not honour it
    as one: the run came out with 1000 cells whose first wall row is 1.491 mm on a 2 mm
    request, so the first-cell height is read off `constant/polyMesh` with
    `layer_report.py`. And the solver's continuity error and `balance.py`'s closure are
    two different quantities, asserted separately -- conflating them is how a reading
    error gets called a continuity failure.
- **An index of every validated case, three tutorial cases, and two reference values that
  turned out to be wrong.** `benchmarks/README.md` lists each case in the repository whose
  answer is known from outside a solver, with its source, the agreement the real run
  achieved, and where the remaining difference comes from.
  `benchmarks/tutorials/` adds three cheap cases as runnable prompts: a heated pipe
  against the exact Graetz value, a buoyant cavity against de Vahl Davis, and a
  backward-facing step. Every number was recomputed from the run's own final field and
  mesh rather than copied from its summary, and `tests/test_benchmarks_index.py` holds the
  prose to the graders' own constants so the two cannot drift.
  - **de Vahl Davis's 4.519 is his cavity average, not his hot-wall Nusselt number.** The
    hot-wall value in his Table V is 4.509. The misattribution is near-universal in
    secondary sources and worth 0.22%, against a paper whose own stated accuracy at
    Ra 1e5 is 0.3%, and it flatters the result, which is why it survives. The cavity run
    recomputes to 4.5335, so +0.54% on 4.509 rather than +0.32% on 4.519.
  - **Gartling's reattachment length of 6.1 is in channel heights, not step heights**, so
    12.2 step heights. The step case here was built with the step height as the unit
    length where Gartling uses the channel height: one mistake, two consequences. It ran
    at Re 1600 where the benchmark is 800, and its 6.42 step heights were compared with
    6.1 as though the units matched. In one unit it is about 47% short, not 5% long.
    `openreynolds/toolbox/reattach.py` and `claims.py` still carry the older reading in
    their docstrings. The case is documented as a units failure rather than as a passing
    validation, and Gartling (1990) is paywalled and was not read, which is said rather
    than smoothed over.
  - **On four of the eleven cases the reference's own uncertainty is larger than the
    agreement being claimed**, and the index now prints it beside the number. Ghia's
    Re 1000 centreline extrema sit 1.46% to 2.20% from Botella & Peyret's spectral
    solution, so agreeing with him to 0.45% is agreeing to inside his error. The 0.164
    Strouhal reference is the value at Re 100 of Williamson's parallel-shedding fit
    (Williamson 1989, *JFM* **206**, eq. 6, reproduced as Fig. 7(a) of the 1996 review,
    where the number itself is never printed), and it is 0.1643, while Roshko (1954) and
    Williamson & Brown (1998) both give 0.1671 at the same Reynolds number: the
    correlations disagree by 1.7% against a result reported as 1.31% high. The SUBOFF
    resistance is located to Liu & Huang (1998) Table 14 p. 23, configuration 1, where
    the bare hull is 87.4 N and the appended hull at the same speed is 102.3 N, and its
    total resistance coefficient is marked as derived because that table publishes none.

- **`advise.py`: which solver, and how many ranks, both read out of the case.** The
  toolbox classifies a case (`ladder.classify`) and then lists this image's solvers for
  that family with what each one is, the condition that makes it the one, and whether it
  is on `$PATH` on the machine asking. The compressible three and the pick at a given
  `--mach` come from `compressible.py`, so the comparison and the dictionary writer
  cannot come to disagree; with no Mach number given, nothing is picked, because a Mach
  number is not in a case's files. For the rank count it reports cells -- measured off
  the mesh, or predicted by `cells_estimate.py` before the build -- cells per rank at
  each count up to the cores there are, and what this case's own solver logs measured,
  split per run because a restarted case leaves two or three runs in one log file.
  - No scaling curve is asserted, because there is not one to assert. Across the 103
    case directories in the local studies, exactly one case had been run at more than
    one rank count, and in that one (study `20260830-164816-eee0`) the mesh and the
    schemes changed between the rank counts as well. So a case run at one rank count is
    told it was run at one rank count and gets no speedup figure.
  - The one measured claim it does make is the one that datum supports: in that study a
    1,793,894-cell case was re-decomposed 12 ways on a container whose `nproc` had
    answered `4`, `mpirun` did not refuse, and the 12-rank run's median 1.45 s/iteration
    landed at the bottom of the 1.45-1.77 s band its four 4-rank stages had already
    measured.
- **Real-site data, and an atmospheric boundary layer inlet built from it.** The
  workspace has no outbound network, so the fetching happens on your machine and the
  workspace reads a file.
  - `openreynolds site --lat <deg> --lon <deg>` writes one self-describing
    `site.json`: hourly wind at 10 m and 100 m from Open-Meteo's ERA5 archive as a
    12-sector rose with per-sector statistics, a prevailing sector and a two-height
    shear fit; OpenStreetMap building footprints from Overpass with each height and
    where that height came from; and a Copernicus DEM GLO-90 elevation grid. No API
    key for any of the three. Every block names the URL, the query, the retrieval
    time, its units and its licence, and the wind block records the reanalysis grid
    cell it actually answered for and how far that is from the coordinates asked for.
    A source that fails leaves its sentence in place of its block rather than taking
    the other two down. No request carries a name, an address or an account id.
  - `toolbox/site_abl.py` turns the wind into an inlet, by writing the dictionary
    OpenFOAM's own `atmBoundaryLayerInlet*` conditions read rather than
    re-implementing the profiles: `0/include/ABLConditions`, the inlet entry in
    whichever of `0/U`, `0/k`, `0/epsilon` and `0/omega` are there, the roughness
    wall functions on a named ground patch, and the profile as a table, a figure and
    a `--json` sidecar. `z0` has no default and it refuses without one, printing the
    Davenport classes. Measured against a real solver: one `simpleFoam` iteration,
    the written inlet values read back, agreeing with the script's own arithmetic to
    4.4e-12 at worst across U, k, epsilon and omega.
  - `toolbox/site_stl.py` turns the footprints and the terrain into STL in the site
    file's own frame, ear-clipping the roofs so a concave footprint comes out as its
    own shape, and counting what it leaves out instead of extruding it to a default.

- **A study record you can ask about: projects, tags, accounting, and running one
  again.** The record on disk carries a `project`, `tags`, the wall-clock time its
  sessions took, the tokens they sent by class, when it last ended, and the study it
  was cloned from. A record written before any of this loads with the defaults, as it
  always did for a key it did not know.
  - `openreynolds studies` takes `--project`, `--tag` (repeatable, and every tag given
    has to be on the study), `--text` over the title and the project, and
    `--state new|finished|unknown`. Its `--json` keeps every key it answered and adds
    the new ones.
  - `openreynolds dashboard` puts every study on one screen, newest first, with its
    state, its jobs, its time and its tokens. It reads `./studies` and starts nothing:
    a study whose job rows were last written `running` is reported `unknown`, because
    settling it means asking the workspace service, and a listing does not start a
    billed machine to draw a table.
  - `openreynolds label` files a study under a project and tags it; `/project` and
    `/tag` do the same from inside a session, answered by the harness with no model
    call.
  - `openreynolds clone` copies a past study's setup and local files into a new study
    without its conversation. `openreynolds rerun` is a clone plus the first thing you
    said in the old study, written in word for word and left unanswered, so an
    ordinary resume answers it. A study whose transcript holds nothing you said is
    refused by name rather than reconstructed.
  - Cost is reported as wall-clock time and tokens for every study, and as a dollar
    figure only where your own key paid your own vendor and that model's rate is
    known. A study on Reynolds' model is never priced in money here, as the service's
    own spending cap is not.

### Fixed

- **A GPU solve refused at the study's spending ceiling was told to the model and to
  nobody else.** `felix run` is itself a workspace job, so when foamd pauses a study at
  80% of its ceiling the refusal comes back as the job's own output rather than as the
  `BackendError` that `tools._handled` turns into the person's question
  (`openreynolds/tools.py:832`). The model read `409 study_paused`, had no way to raise
  anything, and the study sat there: #79. A finished Felix solve's output is now read
  for the refusal, and a study the service still reports as paused is put to the person
  through the same `Approver` every other spending question goes through. Yes raises the
  ceiling; no stops the study with a plain message, and nothing more is sent to the
  model until they speak. A ceiling raised meanwhile on the study page is appended to
  the facts the model receives, so it does not read the old 409 as still current.
  - **The refusal is read as an object, from anywhere in the output.** `felix` catches
    the error and prints `{"state": "refused", "status": 409, "error": "study_paused"}`
    pretty-printed over several lines, and nothing puts it last or makes the exit code
    say so: `felix run cavity; echo done` prints a line after it, and `felix run cavity
    2>&1 | tee run.log` without pipefail exits 0. So the whole window is scanned for the
    object and the exit code is not consulted (`openreynolds/felix_progress.py:315`). A
    reading of the ERROR prose was tried first and withdrawn: foamd prints the spend
    rounded to the cent, so 759.6 cents reads back as 760 and the comparison against the
    live row stopped matching once the spend stopped rising. `state` has to be
    `refused`, because `rejected` is felixd's own word for a solve it turns down after
    creating it, a failed mesh check for instance, and not foamd refusing the study.
  - **`bash` looks at its own output too.** A refusal answers within seconds, so `bash
    felix run <case>` usually finishes inside the synchronous window and is never
    promoted to a job, leaving no job record for anything else to look at
    (`openreynolds/tools.py:891`). Gated on the command being a `felix run|continue`, as
    the job path is, so a `cat` or a `grep` of an old log cannot ask.
  - **Each solve is accounted for once, whatever came of it**
    (`openreynolds/ceiling.py:222`). An earlier version left every failure that was not
    a refusal eligible for another look, which cost a status call, a 256 KB log read and
    a study GET before every model turn, once per crashed solve in the study. A pause
    that could not be confirmed on the one look is not lost by that: the pause belongs
    to the study, so the next call that meets it asks.
  - **Studies saved before this change load as already looked at**
    (`openreynolds/store.py:48`). The flag defaults True and is set False when a job is
    recorded, so resuming an older study does not re-examine every failed solve in it,
    and cannot ask about a refusal the person dealt with days ago.
  - **A one-shot run stopped at a solve's refusal reports `failed`.** The stop broke
    `_run_one_shot`'s loop and fell through to `return "ok"`, so a scheduled run that
    stopped without doing the work exited 0, while the same pause met at a model call
    already reported `failed`. Both go through `_not_completed` now, and
    `test_one_shot_returns_failed_when_the_person_stops_at_a_job_pause` holds it.

- **A rate limit ended the run that hit it.** `_MAX_ATTEMPTS` is five and the ordinary
  backoff is `min(2**attempt, 30)` seconds (`backend/hosted.py:145`, `_retry_delay`), so
  a 429 that keeps coming back raised `rate_limited` at the caller after about fifteen
  seconds of waiting. That is sized for a burst, not for a sustained overload -- several
  workspaces polling one account's key is one, and it lasts as long as they are all up.
  On 2026-09-27 three multi-hour runs ended inside eight minutes of each other that way,
  on ordinary bookkeeping calls that had no answer to `rate_limited` but to die. A 429
  now has its own patience (`_THROTTLE_ATTEMPTS`), its own cap on one wait
  (`_THROTTLE_MAX_DELAY_S`), full jitter rather than a fixed delay several clients would
  all wake from together, and -- the point -- it does not spend the ordinary attempt
  budget. 503 and 500/502/504 keep theirs on purpose: a 429 always clears because the
  window always rolls, while a workspace that cannot boot answers 503 for ever, and
  waiting that out two hundred times turns a five-attempt failure into a half-hour hang.
  `BackendError.retry_after` carries the service's `Retry-After` through to the caller,
  since `extra` carries the error body and the wait only ever arrives in a header. Split
  out of #54 at review.

- **A study on a local workspace uploaded nothing at all.** Capture is on by default and
  `--no-capture` is how it is turned off, but `cli.session` leaves the workspace
  service's client `None` for a workspace it does not reserve, and the capture block
  read that client -- so `--local` and `OPENREYNOLDS_LOCAL=native` turned capture off as
  a side effect, without saying so, at the switch that is not about capture. The
  transcript, the renders delivery surfaces, the end-of-study results payload and the
  case bundle (definition, notes, solver logs, `postProcessing/` tables) of every local
  study stayed on the one laptop. A signed-in local study now opens its row and uploads
  all four exactly as a hosted study does; a machine with no service key uploads
  nothing, as before.
  - The row is opened on a thread rather than in line, because a local study is the one
    that may have no network at all and `FoamdClient` spends up to about a minute (five
    attempts, 1+2+4+8 s apart) finding that out. `Capture.deferred` already holds what is
    said in the meantime and posts it in order once the row exists.
  - A session that ends while that attempt is still in flight waits
    `capture._OPENING_WAIT_S` (2 s) for it and then says that its transcript stays on
    this machine. Before, the close abandoned a row that was one POST away from
    existing: nothing posted and nothing said.
  - The service's own listing of a workspace copy is not offered for these studies --
    their files are on this machine, and `backend.study_id` stays unset -- and the row
    carries no `instance_id`, which foamd's `create_study` already allows.
- **An earlier time directory now reaches the mirror whole or not at all.** A live
  cycle brings earlier written times home behind the newest one, and it copied them
  straight into `files/`, so a round trip that failed partway left a time holding half
  its fields: one the viewer lists and cannot draw (#103). `mirror._pull_time` fetches
  each one into a staging directory beside `files/` instead, and renames its files into
  place only once every file asked for has landed. A partial copy is a directory nothing
  else walks -- the files pane, the viewer and the S3 sync all read `store.files_dir` --
  and it is dropped.
  - Deleting what had arrived once a later file failed was tried first and could not be
    made right. It took back the fresh copies of a time that was *already* whole, which
    is the common case since a restarted solve rewrites times that are already here; a
    file a round trip simply left out raised nothing for it to react to, so the
    half-copied state survived it (`_extract_tar_gz` passes over a non-regular member
    silently, `backend/hosted.py:1505`); and an `openreynolds pull` running alongside a
    background cycle had the file it had just reported mirrored unlinked under it.
  - A time that does not arrive whole is set aside for the session, as before, so one
    that will not copy does not hold every older time behind it; its skip now names the
    time, as the other earlier-time skips do. `bytes_pulled` counts a time only once it
    is in the mirror, and a failed earlier-time batch is no longer taken apart to find
    the file it failed on -- that copy is being dropped either way.

- **A steady solver with a refining mesh is no longer failed as a motion case.**
  `preflight.py`'s moving-mesh check answered `simpleFoam` plus any `dynamicFvMesh` with
  "a motion solver attached to one has nothing to integrate" and sent the reader to
  `pimpleFoam`. Refinement is a topology change and not motion -- no point moves -- and
  `simpleFoam` at v2512 includes `createDynamicFvMesh.H` and calls
  `mesh.controlledUpdate()` once per `simple.loop()`, read from the tag on 2026-11-21. So
  a steady solve whose mesh refines as its residuals fall is reported as the legitimate
  pairing it is, and `case_gen.py`'s steady case can carry AMR with no change of solver.
- **The backward-facing step was never 4% from Gartling's benchmark, and the toolbox
  said it was in four places.** `reattach.py`, `claims.py`,
  `toolbox/notes/openfoam-field-notes.md` and `tests/test_toolbox_reattach.py` all told
  the story of the study that measured one reattachment length twice and believed the
  wrong script, and all four corroborated the surviving number as "4% above Gartling's
  6.1". That comparison was not available to it. **Gartling's 6.1 is in units of the
  downstream channel height**, which at an expansion ratio of 2 is 12.2 step heights,
  while the study's 6.32 and 7.5667 are both in step heights. In one unit they are about
  48% and 38% short.
  - The same wrong unit length set the case up at **Re 1600** on the channel height
    where Gartling's 800 is on that same length, so the run was never solving the
    benchmark's flow either. One mistake, two independent consequences, and both of them
    flattered the result.
  - The incident the docstrings exist for is unchanged and is now sharper: the study's
    two scripts disagreed with each other by **16%**, in the same unit, and *neither*
    could have been checked against the benchmark. What makes 6.32 the right answer is
    which crossing of its own list it is, not that a benchmark appeared to bless it. The
    `DEFAULT_TOL` percentages for figure against answer (15%) and figure against the
    other script (16%) are unaffected, being comparisons between two of the study's own
    numbers in one unit, which is the only kind `claims.py` makes.
  - Specimen strings that exercise the number parser are deliberately untouched, including
    the study's delivered text in `tests/test_toolbox_claims.py` and the sentence
    `LOOK_AHEAD` was measured off. Those are evidence of how people write, and the
    physics claim inside them is not the parser's business.
  - The normalisation is settled from MFIX-Exa's reproduction, which states `H = 1`,
    "where H is the expanded channel height", and matches its own extrapolated 6.097 to
    Gartling's 6.10 with no conversion, plus a published table listing him at 12.20 and
    11.26 step heights (2 x 6.10 and 2 x 5.63, the latter being the upper-wall bubble
    length 10.48 - 4.85). **Gartling (1990) itself is paywalled and was not read**, which
    the docstrings now say rather than imply away. Issue #205 tracks closing that and
    re-running the case at the benchmark's actual Reynolds number.

## [0.4.2] - 2026-11-18

### Changed

- **Signing in with no model key of its own uses Reynolds' model.** A machine with no
  provider, model, endpoint or model key chosen (any vendor's key variable counts), is set to
  the `reynolds` preset at `openreynolds login`, metered to the account, and is told
  how to switch to its own key. A machine that has a key or chose a provider keeps it,
  and is now told `openreynolds config --provider reynolds` is the other way.

### Fixed

- **`openreynolds login --browser` waits through a brief outage.** A 5xx, a 429 or a
  dropped connection while the code waited for approval ended the sign-in, so the
  page then said no terminal was waiting for the code. It now keeps asking until the
  code is approved, refused or expired, and says when the service stopped answering. A
  429 doubles the wait between polls. A key the service handed over on a reply that was
  lost is named as that, with where to revoke it.

## [0.4.1] - 2026-11-17

### Added

- **`openreynolds logout`.** Revokes this machine's service key on the service, then
  removes it from the config and deletes any traces queued to share, unsent. If the
  service cannot be reached nothing changes, unless `--here-only` says to remove the
  key from this machine anyway. A key set in `FOAMD_API_KEY` is left alone, and the
  command says the machine is still signed in through it.

### Fixed

- The release workflow now creates the GitHub release for a tag. It read the tag from
  a job it did not wait on, got nothing, and took the latest release for this one.
  An empty tag now fails the job.

## [0.4.0] - 2026-11-17

### Added

- **A workspace on your own machine: `--local`.** The hosted workspace's image
  (`ghcr.io/inviscidai/reynolds-workspace:2512`: OpenFOAM ESI v2512, HiSA, gmsh,
  build123d, pyvista) runs in Docker here, with `~/.openreynolds/work` mounted as
  `/work`, as your user and with no network. No account is needed, only a model key;
  `--study <id> --local` resumes a local study, and jobs outlive the session.
  `OPENREYNOLDS_LOCAL=native` uses an OpenFOAM installed on the machine instead, with
  no sandbox, and now says so at the start when it finds none. Felix stays cloud-only.
- **`openreynolds mcp`: Reynolds as an MCP server.** Claude Code or any MCP client can
  start studies, watch them, answer their questions and fetch their results through
  the same nine tools the hosted Reynolds MCP server offers (`openreynolds/mcp/core.py`
  defines them for both). Studies run in this process, on a Reynolds workspace or with
  `--local` on this machine's Docker. `claude mcp add reynolds -- openreynolds mcp`. The
  `mcp` package is now a standard dependency, so it works from pip, uvx, pipx and npm
  installs alike; the `[mcp]` extra is kept, empty, for older instructions.
- **A signed-in machine shares a trace of each study.** While OpenReynolds is signed in
  (`openreynolds login`), it sends the service a trace of each study after it ends --
  usage (tools and whether they worked, models, tokens, timings, workspace kind and
  solver, OS and versions) and study content (messages, reasoning, tool inputs and
  outputs, each cut at 20,000 characters) -- whether the workspace is the service's or
  this machine's. Credentials it knows, environment variables named like secrets and
  the shapes of common keys are removed before a trace is queued (`redact.py`), and
  again by the service. It says so in one line at sign-in (password or `--browser`),
  or at the next study on a machine signed in before. `openreynolds sharing
  off|on|status` and `OPENREYNOLDS_SHARING=off` switch it per machine, and `off`
  discards the queue; nothing is recorded or sent without a service key. Traces can be
  deleted from the web app's Account page. `doctor` shows whether sharing is on. What
  is kept and for how long: https://tryreynolds.com/privacy.
- **A new hosted study gets its own workspace (#92).** Studies no longer share one
  instance's `/work`; a resumed study goes back to its own.

- **A study's spending ceiling is the person's question.** When the hosted service
  pauses a study at 80% of its ceiling (`409 study_paused`), on a workspace call or on
  a model call, the person is asked in any mode: yes doubles the ceiling (`POST
  /v1/studies/{id}/cap`) and the paused call runs again, no stops the study with a
  final message and asks the model nothing more. Before the first solve the person is
  told how far the study may go before it asks; in ask-before-compute mode that is
  part of the job's question. See docs/modes.md.

- **Felix mode: a second solver.** `--solver auto|openfoam|felix` (or
  `OPENREYNOLDS_SOLVER`). With `auto`, the default, the solver is chosen from the
  person's first message by a request of its own, before the agent's first turn (see
  *Changed*), and the workspace -- of that solver's kind -- is reserved and started,
  and the platform's study opened with that solver, only once it has been chosen. A Felix study gets its own system prompt (pointing at `/opt/felix/docs`), no
  `cad`, `mesh_review` or OpenFOAM toolbox, and a `job_start` that runs `felix run`
  without the OpenFOAM restart guard and trapFpe check. The solver is fixed for the
  whole study and recorded in `session.json`; a resume keeps it. The OpenFOAM prompt
  is unchanged byte for byte.

### Changed

- The `openai-responses` preset defaults to `gpt-6.1-sol`, not `gpt-5.6-sol`, and compacts before
  272K tokens. `gpt-5.6-sol`'s promotional price ends in November 2026, and on the CAD desk
  `gpt-6.1-sol` had 22 of 26 cases survive the mesh vet against 16 of 26. A model set in your
  config stays as it is.
- **A study's pause speaks in normal studies, never dollars.** Plans are sold as
  studies, so what a study has used is said as a multiple of a normal one (200 cents
  quick, 500 full, the service's `times_normal`), and no hosted cost reaches the
  person anywhere: the pause asks "This study has used 2x the usage of a normal full
  study. Do you want to continue?", yes doubles the ceiling (a typed dollar amount is
  no longer an answer), and the stop message, the notice after a raise, the line
  before the first solve and a paused error's text all speak in multiples. The
  default ceilings are the service's new 500 and 1250 cents. `job_start` no longer
  takes `estimate_usd`, the Felix status line no longer shows `cost_usd`, and both
  prompts say no cost in money is told to the user.

- **The service's pricing errors keep their fields.** `BackendError.extra` (and a
  model call's `ProviderError.body`) carry everything a 4xx body says beyond `error`
  and `message` -- the tier and `upgrade_to` of a 403 `plan_limit`, the period and
  overage of a 402 `allowance_exhausted`, the spend and ceiling of a 409
  `study_paused` -- and the error's text names the plan and the tier to upgrade to.
  A spent allowance names a renewal day only from the 402's `renews_at`, so a work
  trial, a plan set to cancel and a past-due account are no longer told it renews
  (an older service without `renews_at` falls back to a `period_end` still ahead);
  the postpaid mode is called pay-as-you-go overage and the prepaid one credits.

- **Workspace calls name their study.** Once a session knows its study, every call
  to the hosted service carries `X-Study-Id` -- sandbox start, exec, files and jobs, a
  Felix solve's job among them -- so the service tags that spend to the study and its
  pause is exact; a resumed study names itself on the start too. A raise of the
  ceiling the service answers `paused: true` (its own count still past 80%) is asked
  again at once instead of retrying the paused call. The Felix prompt and `job_start`
  name `402 allowance_exhausted`, `403 plan_limit` and `409 study_paused`, and a
  model call refused with `allowance_exhausted` or `plan_limit` says what that means
  (`budget_exhausted`, from an older service, still blocks as a 402).

- **A session's system prompt and tool list no longer change mid-conversation.** A study
  whose solver is `auto` used to start the agent on a solver-choosing prompt with only
  `choose_solver`, and swap both once it was called; through the service's Bedrock
  proxy the next request was a 400 (a thinking block is bound to the prompt and tools
  it was made under) and the turn ended with nothing said. The solver is now chosen
  from the person's first message by a request of its own, before the agent's first
  turn (`desk.route`); the agent starts on that solver's prompt and tools, and the
  choice and its reason are told to the person and in the briefing. A routing request
  that fails leaves the study on OpenFOAM, said so. `choose_solver`,
  `SOLVER_CHOICE_PROMPT` and the swap are gone.
- **`/btw <question>` is a side question.** Answered by a fork of the agent's last
  request -- same model, prompt, tools, settings and messages, one message appended, so
  it reads the conversation from the prompt cache -- and never added to the
  conversation. It may run `read_file`, a non-waiting `job_check` and `send_to_agent`,
  which passes the person's own words to the agent at its next step; any other call is
  not run. `send_to_agent` is in every tool list for that reason. Bare `/btw` is still
  `/status`; the "By the way, no need to stop" framing is gone.
- **The front desk is gone.** Nothing answers a mid-turn message in the agent's place.
  A message typed mid-turn gets a receipt at once (queued, what the agent is doing and
  for how long, `/btw` for a side question), held waits ask after the person every
  second rather than every five, and the "what's happening now" line is built from the
  tracker's job facts with no model call. `desk_model` is still read and switched, for
  the configurations and the hosted app that set it.
- **`/mode` no longer changes the tool list.** `checkpoint` is in every mode's list and
  answers outside structured mode that checkpoints are structured mode's; the switch
  reaches the model as a harness note, as it already did. The system prompt never
  carried the mode. Structured mode still holds `job_start` until a plan is approved.
- **Reasoning blocks follow what they are bound to.** Measured with the binding checked
  strictly: a thinking block is bound to the system prompt, the tools and the messages
  before it (an edited earlier message is the same 400; effort and model changes are
  not). Image eviction and `drop_images` rewrite an earlier message, so the turns after
  it lose their thinking blocks; a `/model` switch that rebuilds the client strips them
  even when the model id is unchanged.
- **A failed model request is in the transcript.** A turn a provider error ended leaves
  an `event` row saying the request failed and why, not only a live notice.

### Fixed

- **End stops the workspace before bundling the case (#93),** and the bundle stops at
  its budget; a bundle over budget keeps the record in part instead of dropping it
  (#95).
- **A SOCKS proxy can start the model client (#90).** `httpx2[socks]` is a dependency.
- **A live study's times reach the viewer within a cycle (#98),** the newest first,
  before the backfill of earlier ones (#102).
- **Every model client a session builds names the study (#105),** so the service can
  count its usage against that study.
- **A cut stream is retried, and a transient model failure no longer ends the turn
  (#87, #74).**
- **Ending a session keeps a job's real ending (#85),** and a workspace start that fails
  says so (#83).
- **The solver-could-not-be-chosen line is printed once,** not twice, in a plain session.
- **The `--no-capture` help said it sends nothing to the platform.** It keeps the
  study's transcript on this machine; traces have their own switch. The help, README
  and `doctor` now say so.
- **The trial text in `login` and the README said a company address needs no card.**
  Both trials take a card and renew unless cancelled.
- **The end-of-session hint offered `openreynolds pull --all`,** a flag that does not
  exist; it is `openreynolds pull`, and a local study is not offered it at all. A
  session on a native local workspace no longer says it joined a running workspace.
- **The MCP server reported an empty version;** it reports the package's.

- **A piped command fails when any stage of it does (#48).** `blockMesh | tail -40`
  reported exit_code 0 from a FOAM FATAL ERROR because `tail` succeeded. `bash` and
  `job_start` now run a command that has a pipe in it under `set -o pipefail`; one
  without a pipe goes as written. The cost is SIGPIPE: a stage cut off by `head` or
  `grep -q` now exits 141, so `if cat long.log | grep -q OK` can take its else branch.
  The tool descriptions say so, and `bash` and `job_check` say what a 141 from a piped
  command is.

- **End copies the study to durable storage before saying its files are kept (#51).** A
  hosted End leaves the workspace up for the reaper, and the service copied a workspace
  to S3 only while jobs ran and when it was stopped, so a report written in the last
  turn sat on the volume alone for minutes. End now asks the service to copy the study's
  own directory (`POST /v1/instances/{id}/checkpoint`) and waits up to 90 s. It records
  the copy's `taken_at` and manifest in `session.json` under `checkpoint`. A copy that
  fails or times out is said plainly and recorded as an `error`, never assumed.

## [0.3.2] - 2026-10-05

### Fixed

- **Patch-normal measurement keeps the PyVista surface extractor it was measured
  with.** PyVista 0.49 warns that `extract_surface()` will change its default from
  `dataset_surface` to automatic selection, which chooses a different algorithm for
  most inputs. The patch table relies on stable face ordering to compute a mean normal,
  so it now selects `dataset_surface` explicitly instead of changing with a dependency
  update. This also removes the repeated warning from the toolbox tests. The `algorithm`
  keyword is new in PyVista 0.47, so that is now the floor. (#46)
- **A `bash` command the workspace moved to a job no longer comes back as `exit_code:
  0`; the call waits the job out for the rest of its `timeout_s`, and a command still
  running when that is up says so in its first line.** The hosted service caps a
  synchronous exec at 120 s (`FOAMD_SYNC_EXEC_MAX_S`); a command still running there,
  whose caller asked for longer, is started again from scratch as a detached job and the
  answer is `{exit_code: null, promoted: true, job_id, output, note}`
  (`OpenFoam_Instance/app/execs.py`, `spec/openapi.yaml`). `HostedBackend.exec` turned
  that into `ExecResult(exit_code=0, output=note, job_id=...)`, and `_bash` printed the
  zero first and the note about the job second. Measured in production on 2026-09-21
  (study 20260921-033019-e1b4, workspace 35c9f018): `bash sleep 240; ls -R .../mesh |
  head -40` at 03:37:36, `bash sleep 200; ... grep "^Time =" log.pisoFoam | tail -1` at
  03:48:23 and `bash sleep 180; ... tail -3 log.recon` at 04:04:22, each asked with
  `timeout_s=300`, each answered after the 120 s window, and each transcript row reads
  `-> exit_code: 0` for a command that was still running (jobs 808edf10, 5432459d,
  e839ac57). The model read three commands that had finished with no output: it
  `job_check`ed the first once, two seconds in (a wait answered at 0 s by the put-back
  line #41 fixed), and none of them again, and paced the rest of the study with `bash
  sleep` calls of 60-115 s. Nothing on the control plane read the jobs either -- a row was only
  reconciled by a `GET /jobs/{id}` or by the reaper once the workspace had gone idle --
  so the three rows read `running` for 46, 35 and 20 minutes after their commands had
  ended, until the 04:27 sweep closed them off the store. foamd #55 fixed the service's
  half (the promoted job is watched and its row closed with the real exit code); this is
  the agent's. `ExecResult` carries the truth: `exit_code=None`, a new `promoted` flag,
  `job_id`, and `output` as what the synchronous run had produced (empty from today's
  service; carried in case that changes), with the wrapper's `stderr` through -- never
  `exit_code=0`. `EXEC_SYNC_WINDOW_S = 120` joins `EXEC_MAX_TIMEOUT_S` on the protocol,
  because the tool states the number to the model. `_bash` on a promoted answer records
  the job as the session's (`store.record_job`, the way `job_start` does) and follows it
  for what is left of the caller's own `timeout_s` -- the same held wait `job_check
  wait_s` makes (`_hold`, now the one loop behind both), ended by the same things: the
  job ending, the time running out, the person writing (`ToolContext.on_wait_input`,
  #41) or leaving (`on_leaving`, #43). A job that ends in time is the command's result:
  `exit_code: <the job's>` first, a note naming the job and the two spans (`moved to
  job <id> at 120s ... waited 130s more`), the job's status line and its log as the
  output. One still running when the time is up, or when the wait was cut, leads with
  the fact -- `still running as job <id> (promoted after 120 s; 180 s waited)` -- then
  the mechanism, `job_check <id>` as what follows it and "not a command to run again",
  the wait's own note when the person wrote or left (the written one says `job_check`,
  not "call bash again": `bash` again would run the command a third time), the job's
  status line and the log so far. Never `exit_code: 0`. The record's `log_offset` is
  where this left the log, so the model's `job_check <id>` continues rather than
  repeats; a job handed back still running stays `running` in the record, so the watch
  loop polls it and wakes the model when it ends -- the poll that never happened. A
  status or log that could not be read is a note under the first line, which stands.
  The `bash` description says that a command still running at 120 s may become a job
  the call waits on and what the two answers look like, as a fact about the workspace
  (the local backend never promotes: it runs a command to its `timeout_s`, and nothing
  there changed). The mesh desk's step observation, which asked for 240 s and read the
  same shape, says `no exit code yet: running as job <id>` where it would now have said
  `exit None`. Unchanged: a `timeout_s` at or under 120 (the service never promotes
  it), `-1` and its note, truncation, `[took Ns]`, the workspace `stderr` note, `idle`.
  Tests: the hosted backend on the service's promoted answer (direct, through the
  edge's 303, with `stderr` and output through, the other three shapes untouched);
  `_bash` with `timeout_s=300` on a promotion under a hand-turned clock -- the job ends
  with rc 0 and a log (`exit_code: 0`, the log, the job named, `[took 130s]`, the record
  closed), ends with rc 1 (`exit_code: 1`), outlives the caller's timeout (the first
  line, `job_check <id>`, the log so far, exactly 300 s held, the record left running),
  is cut by the person writing and by the person leaving (the same notes the other waits
  carry), has no time left after the exec, cannot be read; the first run's output kept
  apart from the job's log; a `timeout_s` within the window byte-identical to before;
  the description's words and register.
- **The workspace is listed by the service, running or stopped; the `find` over the
  exec channel is the fallback, and a cut output can no longer invent a file.**
  `Browser.tree` listed a workspace with `find ... | sort | cut | head -n 4001` over
  the exec channel, and the hosted workspace's daemon caps a command's output at 64 KB
  and cuts it mid-line (`OpenFoam_Instance/daemon/settings.py` `EXEC_OUTPUT_CAP_BYTES`,
  `daemon/runner.py` `out[:cap]`; the answer carries `truncated: true`, which
  `HostedBackend.exec` mapped onto `ExecResult.truncated` and `Browser.tree` ignored).
  Measured in production on 2026-09-21 (study 20260921-033356-076b, workspace
  35c9f018): the listing is ~1,532 rows at ~83 bytes each, ~127 KB, against the 64 KB
  cap -- so the 4,000-entry cap never bound anything, the byte cap did, at ~850 rows.
  The partial last row, `f\t1528155\t<mtime>\t/work/20260921-033356-076b` -- the row
  for `mesh/zoom.png`, 1,528,155 bytes, cut right after the study id -- has four
  well-formed fields, and `_parse` accepted it: an `Entry` with that path,
  `is_dir: False` and `size: 1.5 MB` -- a file at the study root. Everything that
  followed was measured too: the page's tree collapsed to one unexpandable "remote
  1.5M" leaf (ui #31 now hardens the page); the mirror asked the service to archive
  that "file" every cycle (`tar?mode=pack&paths=/work/20260921-033356-076b` -> 502/504
  each 20 s from 04:07 to 04:16), plus 404s for the other paths a cut produced (`/r`,
  `/run/processor`, `/run/p`); and the live tree during a run held ~850 of the
  study's 1,532 rows -- since #36 the order is breadth-first, so the cut lands in
  `run/processors4/<time>/`, but everything past it was invisible to the live pane and
  to the mirror until the workspace was stopped. Meanwhile #38/#39 had given the
  hosted backend `list_stored` -- `GET /v1/instances/{id}/files?list=1`, recursive,
  5,000 entries -- which lists any path under `/work` from the live daemon while the
  workspace is up and from the service's copy once it is stopped, breadth-first,
  capped at 5,000 with its own `truncated`, never starting a machine and never
  counting as use of one (foamd #53) -- and `Browser.tree` used it only when a poll
  said the workspace was idle. Now `tree()` asks `list_stored` first, whatever state
  the workspace is in and whether or not the call is a background poll; the answer is
  the listing, cut to the depth and to 4,000 entries breadth-first, the service's
  `truncated` carried. No `find`, no exec channel, no byte cap, no partial line, and
  one path for both states. The walk runs only when `list_stored` answers None -- a
  local backend, a workspace still coming up (`PendingBackend.list_stored` is None
  until the machine is there, and the walk then waits for it as every call did), a
  service without the route, a failed request -- and then exactly as before: a poll,
  then work only when nothing is running and somebody is asking; a background walk
  that finds nothing running still raises `workspace_idle` and starts nothing. The
  walk itself is made safe for the local backend and for the fallback: when the
  backend says the output was cut, the tail after the last newline is dropped
  unparsed, and the listing is marked truncated with a notice that names the output
  cap (a listing of 850 rows told it was "capped at 4,000 entries" would contradict
  what is in front of the reader); and `_parse` refuses a row whose path is the
  listed root or is not under it -- the walk is `-mindepth 1`, so such a row can only
  be a cut or noise. The listing request is bounded at 60 s, the minute the `find`
  had (`FoamdClient.request` with no timeout is no timeout at all, and the mirror
  asks this every twenty seconds). The route's `mtime` are whole seconds where
  `find`'s carried a fraction; `Entry.mtime` stays a float and nothing that reads it
  is finer than two machines' clocks. Behaviour changed for the background cycles of
  a session whose workspace is stopped: they now read the service's copy each cycle
  -- one bounded request, nothing started, nothing pulled when nothing has changed --
  where before they waited out the idle workspace. Unchanged: a backend without
  `list_stored` (a test's, an embedder's) lists as it always did; the 4,000-entry cap
  and its notice; the poll-first order of the walk.
- **End pressed mid-turn ends the turn at its next safe point, then the session -- the
  way a Stop button is expected to work.** The web's End button sends `/exit` into the
  session's inbox (the runner's `relay.py`); typed, `/exit` and `/quit` are the same
  command, and an EOF (the interface's ctrl+C, a closed stdin) is the same leaving. Met
  by the loop's drain between tool calls (`cli._typed_while_working`), all of them were
  put back for whoever waits at the prompt -- and honoured there, when the turn ended on
  its own. In production on 2026-09-21 (study 20260921-033019-e1b4) the person pressed
  End at 03:33:40 into a turn that was waiting on the mesh desk; the turn went on for
  another thirty-seven minutes (to 04:10:46: eighteen `bash sleep 60..240` calls, eight
  held waits, a fifteen-minute `pisoFoam` solve on four ranks launched at 03:44:36, a
  reconstruction, a 300-frame animation and a gif, all for a person who had left) and
  the session ended 27 s after the turn did, at 04:11:13. (#41 fixed what the put-back
  line did to the waits in that same turn -- `mesh_wait` and `job_check` answering at
  0 s with nothing to deliver -- and deliberately left the End semantics for this
  item.) Now the drain tells the loop (`Loop.leave`) when what it meets is a `/exit`, a
  `/quit` or an EOF, and still puts the line back as before; from then on the turn ends
  at its next safe point. A held `job_check` or `mesh_wait` asks the loop a second
  question beside "did the person write?" (`ToolContext.on_leaving`, wired to
  `Loop.leaving`) and returns within a poll: `[waited 12s] [the person ended the
  session, so this answered early; the session is closing down]`. A call already
  running when End was pressed -- a `bash` command, a `job_start` -- is not cut: each is
  bounded by its own deadline, and its result is recorded when it comes back. A call the
  model asked for in the same batch that had not started is not started, and is answered
  `This call did not run: the person ended the session.` (the shape `settle` gives an
  interrupted turn's calls, so the thread stays whole). After that batch's results the
  model is not asked again -- no further tool call, no reply -- and the transcript
  carries one line in the harness's voice: `The person ended the session while this
  turn was running, so the turn stops here: the tool calls already in flight finished,
  no further call was made, and the model was not asked again.` The session loop
  (`cli._run_interactive`) returns straight to the close-down -- the final sync, the
  capture, `_close_down`, unchanged -- without another prompt. A mesh desk still
  building is told at that moment rather than at the close-down: a background run's
  budgets are zeroed (`DeskRun.abandon`) so it stops at its next command instead of
  spending model calls for as long as the call in flight takes to come back, and the
  session's own desk is stopped the same way, so a foreground `mesh` (`wait: true`)
  holding the turn ends at its next lap rather than at its fifteen-minute budget. What
  a typed line does mid-turn is otherwise exactly as #41 left it: words for the model
  cut a wait once and ride with that batch's results; `/status` and its kind are
  answered on the spot and cut nothing. A prompt-time `/exit` -- typed when the model is
  waiting for input -- is unchanged. Tests drive the real loop, drain and wiring: End
  (`/exit`, `/quit`, EOF) pressed during a held `job_check` and a held `mesh_wait` ends
  the wait within a poll, the model is not called again, the last message carries the
  one line, the desk's budgets are zero and the line is still in the reader; words
  typed in the same breath as End ride with the result and the note says the person
  left; End pressed during a `bash` lets it finish, marks the next call in the batch as
  not run and makes no further call; the session loop returns without a second prompt;
  `/exit` typed to an open question (partial mode) declines the call and ends the turn
  there; a whole session through `drive` ends with the ordinary close-down and the
  workspace put down; a prompt-time `/exit`, `/quit` and EOF are as before; a real
  `Mesher` run told by `Loop.leave` ends at its next lap; #41's typed-words and
  `/status` tests pass unchanged. Two warm-start tests that queued a `/exit` behind
  their first message as the way to end afterwards now end on the reader's EOF at the
  prompt instead: a `/exit` in the inbox at the first tool call is, correctly, the
  person leaving. `docs/session-commands.md` says what `/exit` does mid-turn.
- **A steady solver's residuals levelling off on an unsteady flow is no longer reported
  to the person as a run that "did not converge".** The owner, after five solving
  studies in three days: "In each solve I just wanted a quick look, not a mesh
  independence study, why is it that every time it kept telling me it couldn't
  converge??? ... Nothing converges is it?" -- and, once the transcripts were read
  back to him: "if it's never gonna converge it doesn't mean it's physically
  inaccurate! But the way it's phrased it's always so negative." Read with the
  operator tool (`OpenFoam_Instance/scripts/study_log.py`), not one of the five had
  diverged, and every mesh was checkMesh OK (max non-orthogonality 30-38, skewness
  0.6-1.9). What happened, and what was said: (a) 20260919-041432-e0b6, a laminar
  backward-facing step at Re_S = 800 under `simpleFoam` -- the Ux residual sat at
  2.2e-2..3.2e-2 from iteration 250 to 30,000, on two meshes (9k, 23k cells), at
  relaxation down to p 0.2 / U 0.4, first- and second-order, eight solver runs in all;
  the flow is past the 2D Hopf bifurcation, so the plateau is the physics. Headline:
  "**The headline, honestly: at Re_S = 800 there is no steady solution to converge
  to.**"; README column "converged?" answered "no (res. floor 1.5e-2)" three times.
  (b) 20260920-155504-4379, the "Inviscid AI" text under `simpleFoam` k-omega SST,
  106,716 cells, Re 6.7e4, 2000 iterations in 157 s -- Ux 1.2e-2..2.0e-2, Uy
  3.6e-2..5.1e-2, p 3.0e-2..3.9e-2 from iteration 200 to 2000, flat, no bounding; the
  pictures were made from that field. Told: "**Honesty:** the run did **not**
  converge -- residuals plateau at ~1e-2 ... The fields are a frozen pseudo-transient
  snapshot ... treat them as +/-25%." (c) 20260920-161908-c7ef, the same text under
  `pimpleFoam`, 0 -> 0.8 s in 22 min, per-step residuals Ux 5e-5 / p 4e-2 (a
  transient's normal shape): "take Cd as +/-20-30%, not converged. One mesh, no
  independence study", with "Mesh independence -- 3 meshes ... ~1.5 h" first among the
  offered follow-ups; nobody had asked for one. (d) 20260921-033356-076b, `pimpleFoam`
  at Re 6.7e3, stopped by the person at t = 1.48 s ("just plot whats there already"),
  no bounding, Co max 2.4: the gif was delivered, then "Cd ... **Not converged and not
  trustworthy.**", "Strouhal number: **I will not quote one.**", a figure titled "NOT
  statistically converged". (e) 20260921-033019-e1b4, `pisoFoam` laminar at Re 200:
  clean, one line about "no grid-refinement check". Three senses of "converge" -- a
  steady residual, a grid, a time average -- reached the person as one message, four
  times in five. Every diagnosis was accurate; the framing turned the physics into a
  confession. Where it came from: the system prompt's one sentence on the subject
  named "an unconverged solve" as the first thing to be honest about (the "Honesty:"
  headers in (a), (b) and (c) are that sentence answered); nothing told the model what
  a stall is, that a bluff body under a steady solver stalls by nature, or what a
  quick look may take from a stalled field; no residual tolerance in the harness was
  involved (none of the five cases used `case_gen.py`'s `residualControl`) and no
  mesh-quality gate sent anything back to remeshing. The change is guidance, in the
  places the model reads at the right moment, plus the one summariser that had the
  negative wording built in: `openreynolds/convergence.py` holds the line -- a stall on
  an unsteady flow or a plateau around 1e-4..1e-3 on a first look is a physically
  meaningful field, reported as what the flow is doing with the residual in one neutral
  clause; failure words are for a residual that climbs, a floating point exception, a
  field the solver keeps bounding or a mesh checkMesh rejects, with what failed and what
  would fix it; a transient window or a mesh study is offered, not started unasked --
  with an example sentence of each kind, and the briefing carries it in every session
  (`cli._situation_brief`). The system prompt's honesty sentence now names "a run still
  moving when its number was read" as "a fact about it and not a verdict on the run".
  `job_start` on a steady solver (`simpleFoam` and its kin, told by name) adds one
  clause to the launch note: its residuals level off rather than fall on an unsteady
  flow, and the levelled-off field is a snapshot worth showing, not a failed run.
  `toolbox/log_digest.py` read every run that reached its end without the solver's own
  "converged" message as "ran to the end of controlDict without reporting convergence"
  -- the same words for a plateau on a shedding wake as for a residual climbing towards
  an exception; it now reads the residual series (`residual_shape`: diverging, still
  falling, levelled, or too short to say) and says which, e.g. "residuals: levelled off
  (Ux ~1.5e-02, p ~3.0e-02) from about step 200 and stayed there -- a plateau, not a
  divergence", against "residuals: climbing (Ux best 1.6e-05, last 1.8e+01) -- this run
  is diverging, not converging slowly, and the last field is not one to show"; the
  climbing/blow-up thresholds are the ones `preflight.py`'s residual gate already used,
  now shared. No solver default changed: the runs were right. Tests pin the guidance
  text (the failure words appear once, as the words ruled out; no imperative workflow
  language), the briefing and launch note carrying it, and the digest's reading of a
  steady stall, a divergence, a still-falling series and a run sitting at its floor.
- **A held `mesh_wait` or `job_check` ends early only for words the model is about to
  read, and says so.** The wait asked the reader's `pending()` -- is anything in the
  queue -- while the loop's drain between tool calls (`cli._typed_while_working`) does
  not take everything out of it: `/exit` is put back for whoever waits at the prompt,
  an EOF likewise, and a `/status` is answered without a word reaching the model. In
  production on 2026-09-21 (study 20260921-033019-e1b4) a put-back line sat in the
  queue for the rest of a forty-minute turn: `mesh_wait` returned three times in nine
  seconds -- 03:33:40 after 126 s, then 03:33:41 and 03:33:45 with `[waited 0s] [the
  user said something, so this answered early]` -- and nothing the person had said
  followed, because there was nothing to deliver. The model concluded the tool did not
  wait, switched to `bash sleep 60..240` and paced the remaining thirty-seven minutes
  with it (eighteen sleeps between 03:35 and 04:09; the eight waits it still tried,
  four of them `job_check(wait_s=...)`, all answered in 0 s the same way). The same
  prompt on the same model, started beside it sixteen seconds after that first cut
  (20260921-033356-076b), held its `job_check(wait_s=120..300)` waits in full
  throughout, and every line the person typed reached the model. The
  question is the loop's to answer now (`Loop.heard`, wired as
  `ToolContext.on_wait_input`): it drains the inbox the way the loop does between tool
  calls -- commands answered on the spot, words for the model kept for this batch's
  results -- and answers whether anything is kept. So a wait ends for a message that is
  in the same user message as the result that ended, once (delivered, it cannot end the
  next wait too), and a line that is nobody's to deliver to the model never ends one.
  The result says what happened and what comes next: `[the person wrote, so this
  answered early; their words follow this result. Answer them, then call mesh_wait again
  -- the desk is still building]` (`job_check` says the same of the job). `job_check`
  also decided its note by asking the question a second time after its loop, which
  with a draining answer would have dropped the note; it remembers instead. What a
  typed `/exit` does is unchanged: it is honoured when the turn ends, as before.
- **`mesh_look.py --out` is relative to where you run it, like every other toolbox
  script, and the report prints the absolute path it wrote.** A relative `--out` (and
  `--json`) was joined to the case; `render.py`, `results.py`, `showcase.py`,
  `geometry_view.py` and `animate.py` all read one against the working directory. In
  production (study 20260920-161908-c7ef, 16:26:34 UTC) `cd /work/<study> && python3
  /work/.toolbox/mesh_look.py mesh --out look.png` wrote `mesh/look.png` and printed
  `picture: look.png`; four seconds later `read_file /work/<study>/look.png` answered
  `not_found (404)`, and three turns went on finding the file -- for a command the
  harness itself suggests in those words (`mesh_look.py <case> --out look.png`, in the
  desk's result and the resume briefing). Now `--out` and `--json` resolve against the
  working directory, the default without `--out` is still `<case>/look.png`, and the
  `picture:` line is the absolute path. The JSON's `render` keeps its shape -- relative
  to the case when the picture is inside it, which is what `mesher/check.py` reads and
  joins to the case's paths -- and a `render_abs` beside it is absolute either way. The
  mesh desk (`check.py`: `mesh_look.py . --out renders/mesh_look.png`, run in the case)
  and the templates (`. --out look.png`, run in the case) name the case as `.`, so
  nothing changes for them. `--help`, the script's docstring, `ENVIRONMENT.md` and the
  toolbox `README.md` say where the file goes.
- **The workspace listing is breadth-first, so the cap falls in the solver's bulk, not
  on the pictures.** `Browser.tree` ran `find | head -n 4001`, and `find` walks
  depth-first in directory order. In a transient study (20260920-161908-c7ef, in
  production) the walk went down `run/processors4/` first and spent the whole cap on
  per-time field files; `renders/shedding.gif`, `README.md`, `make_gif.py` and the 201
  animation frames -- all written, looked at and described to the person -- were past
  the cap in every listing, so the live mirror never saw them and never brought them
  home, and the finished study's page showed `run/` and nothing else. The listing now
  prints each entry's depth, sorts on it (stably) and strips it again before the cap:
  every shallow entry precedes any deep one, and the cut, when there is one, lands
  among the deepest. The cap and its notice are unchanged.
- **The close-down of a session whose workspace has already been stopped no longer
  starts a machine to list it.** The final sync begins with a listing (`Browser.tree`,
  a `find` over the exec channel), and a foreground exec on a hosted workspace the
  service had stopped lazy-starts a new one to run it. In production on 2026-09-21
  (issue #37, item 1) an idle-timed-out session's close-down started a c7i.2xlarge at
  02:24:32 -- the workspace had been reaped at 01:31 -- listed the files, and the
  machine then sat until the reaper took it down again at 02:42: eighteen minutes of
  instance for one listing. The service keeps a copy of the workspace, written at every
  checkpoint and stop, and once the workspace is stopped it already serves `get_file`
  and `get_tree` from it; `GET /v1/instances/{id}/files?list=1` lists it. So a
  foreground listing is now asked as a poll first (`background=True`, which never
  starts anything and does not count as use of the workspace); when the poll finds
  nothing running, the listing is read from the copy (`Backend.list_stored`, which
  `HostedBackend` answers through that route, cut to the depth asked, capped and
  ordered as the walk's is); and the machine is started only when there is no copy
  to read -- which is what a foreground listing always did, and is still right when
  somebody is working. The first cut asked the study-scoped route
  (`GET /v1/studies/{id}/workspace`) and still started a machine for a study whose
  home is the workspace root itself -- that route refuses `/work` -- measured on
  production the same morning (09:58 SGT, a pool adoption for one `find`); the
  instance-scoped route has no such refusal and needs no study id. The session still
  tells the workspace which study it serves (`Backend.study_id`, set when the study's
  row is opened or resumed, and carried through the `PendingBackend` stand-in).
  Unchanged: a running workspace's
  own listing is used and the copy is never asked; the background cycles still wait
  out a stopped workspace rather than read it every twenty seconds; a local backend,
  and a study with no row on the platform, list exactly as before.

### Changed

- **Reynolds' model is GPT or Claude, and the model picks the API.** The `reynolds`
  preset was Claude 5 over the Messages API at `{service}/v1/llm`. The service now
  meters four models for a study, two per family: `gpt-6.1-sol` (the default) and
  `gpt-6-astra` over the Responses API at `{service}/v1/llm/v1` (the client appends
  `/responses`), and `claude-sonnet-5-5` and `claude-opus-5-5` over the Messages API at
  `{service}/v1/llm` as before -- all on the service key as the model key, whatever
  model key the config still holds. `make_provider` picks the adapter from the model,
  not the preset (a `claude-` id is the Messages API), and each place that builds a
  client builds it for the model it will ask: the loop, the CAD desk, its reviewer and
  the front desk. The front desk speaks the main model's family -- `gpt-6-luna` for GPT,
  `claude-haiku-4-5` for Claude -- and a `/model` between families on `reynolds`
  rebuilds the client, moves the desk, and replays the thread through neutral blocks,
  dropping the other family's reasoning. A refusal is retried on the other model of the
  same family, never across. The window is the model's: 1,000,000 for the Claude 5.5
  models, and 272,000 for GPT -- not those models' own, but where they start billing at
  a long-context rate the service does not price -- so a GPT thread is compacted before
  it gets there. The CAD desk builds with the main model. On GPT that is measured:
  `gpt-6.1-sol`'s first corpus (`docs/cad-buildup/sweeps/core-gpt-6.1-sol-20261002-075149-9301`)
  had 22 of 26 survive the vet, against 16 of 26 for gpt-5.6-sol. Neither Claude 5.5
  model has been swept on the desk (Sonnet 5 finished 0 of 16 in its 30 steps). A
  config file or a study that names `claude-sonnet-5` or `claude-opus-5` under
  `reynolds` loads, or resumes, onto `claude-sonnet-5-5` or `claude-opus-5-5`, and any
  other id the service does not meter onto `gpt-6.1-sol`, and says so.

  The Messages adapter now sends Claude 5.5 only what it accepts. Thinking is adaptive
  with `display: "summarized"` and the effort is sent on every turn, because Opus 5.5
  cannot turn thinking off and defaults to `medium`; even on an endpoint that refused
  the extras, a 5.5 model keeps its effort. The front desk's short answers ask a 5.5
  model for `low` effort -- with `thinking: {type: "between_tools"}` on Sonnet 5.5, and
  room for the thinking it cannot skip on Opus 5.5 -- rather than `thinking: disabled`,
  which both refuse. No path sends a forced `tool_choice`, sampling parameters or a
  prefill. Responses usage now counts a reported cache write as its own class, and list
  prices for all seven models are in `llm/presets.py` (Claude's at Anthropic's list;
  Bedrock bills at partner prices that may differ).

- **The mesh desk builds in the background.** A `mesh` call used to run the desk on
  the loop's own thread and hold it for the whole build. Measured in production
  (study 20260920-161908-c7ef): one call held the agent for 402 s and some twenty-five
  model calls, during which the main agent answered nothing, every line the person
  typed was drained into the desk by its `interject` -- the same inbox the loop reads
  between its own tool calls -- and on the web page each typed line sat marked as
  pending until the desk consumed it, which read as "sending a message takes minutes".
  Now `mesh` starts the desk on a daemon thread (`openreynolds/mesher/background.py`,
  `DeskRun`) and returns at once with where it is working; the conversation carries
  on; the desk's steps show in the transcript as before; and its end wakes the model
  through the same watch loop a job's end does (`watch.Wake("desk")`), the person's
  typed lines still winning. Two tools join it: `mesh_note` passes a remark to the
  running desk, which reads it at its next command -- the desk's ears are its own now,
  and the session's inbox is the main agent's again -- and `mesh_wait` holds for the
  result, bounded like `job_check`'s wait and ending early when the person types. One
  desk runs at a time; a second `mesh` call answers with where the first has got to.
  `mesh` with `wait: true` is the old shape, whole. A `-p` run waits for a live desk
  as it waits for a job. `session.json` records a live run, so a session resumed
  after its process ended mid-build is told so once, with where the desk's work is.
  All three tools are absent when the desk is not configured, as `mesh` was.

## [0.3.1] - 2026-09-17

### Fixed

- **A session that joins a running workspace no longer leaves it up for nobody.**
  Joining used to decide the close-down on its own: a joined session always left the
  workspace running, so the next `openreynolds` found it still up, printed "joining
  the workspace already running", and left it up again -- relaunching the CLI kept an
  idle container alive (and billing) indefinitely, with no session on the web to
  explain it. A joined session now asks the same questions as one that started the
  workspace -- running job rows, and the probe for another session's processes -- and
  stops it when nobody is working on it. A sibling session that is live but idle at
  that moment starts a fresh container on its next call; the volume is untouched.
  Studies with no directory of their own still leave a joined workspace up.

## [0.3.0] - 2026-09-17

### Added

- **Three modes: full auto, ask before compute, structured.** `--mode`,
  `OPENREYNOLDS_MODE`, the config file's `mode`, or `/mode <name>` mid-session
  (`openreynolds/modes.py`). `auto` is the default and is today's behaviour byte for
  byte: the briefing is unchanged and no tool call is ever held. `partial` puts every
  `job_start` and `mesh` call to the person before it runs (`/yes`, `/no [reason]`,
  `/all`), and a declined call goes back to the model as an error result carrying the
  person's reason verbatim. `structured` offers a `checkpoint` tool that shows the
  person a summary and what comes next and waits for their answer, and holds
  `job_start` and `mesh` until a checkpoint has been approved since the mode began.
  Its stage names are the guided pipeline's own phases (`toolbox/study_state.py`).
  The question and its answer live in `openreynolds/approval.py`; the one place a call
  is held is `Loop._consult`. A resumed study keeps its mode unless one is given.
  `-p` with a non-auto mode is a usage error (exit `2`), because nobody is there to
  answer.
- **Change the model or effort mid-study.** `/model` shows the provider, model, effort,
  mode and the models this provider is known to have; `/model <model>`,
  `/model <provider>:<model>` and `/model <provider>` switch
  (`openreynolds/switch.py`). A candidate is probed, image included, before it is
  accepted, and applied only between turns. Earlier thinking blocks are dropped when
  the model changes, and a thread too big for the new model's window (known per model
  where it differs from the preset's, as for `claude-haiku-4-5`) is refreshed on the
  current model first. `/effort low|medium|high` applies from the next request, and
  `--effort` sets it at the start. The mesh desk and the front desk follow the switch.
  A resumed study carries on on the model it was last running: `session.json` now
  records the provider and the endpoint beside the model, and `--study <id>` restores
  the record unless `--model` or `OPENREYNOLDS_MODEL` names a model, or this machine
  cannot serve it -- in which case the configured model runs and the session says why.
  Serving it means a key for the recorded provider here, switching provider if the
  record says another one (it brings that provider's key, endpoint, window and desk
  model with it), and on the configured provider the same endpoint as well, because a
  gateway or router in front of one family answers to other vendors' ids and a pair
  restored across endpoints fails a turn later with the vendor's 400. A study that
  recorded no provider or no endpoint is refused rather than guessed at, and starts as
  it did before. A refused restore does not rewrite the study's record: the run that
  cannot serve the pair is the last one that should forget it, and the next resume
  where the key is set carries on there. The record is the local `session.json`, so a
  study resumed where it has never run starts on the configured model, and
  `OPENREYNOLDS_PROVIDER` on its own does not hold a resume to that provider.
- **`/help` with topics.** `/help commands`, `/help modes`, `/help model`,
  `/help tools` and `/help keys`. Every command lives in one registry,
  `commands.COMMANDS`, which the parser, the help text, the terminal's completion and
  the hosted app's suggestion list are all read from.
- **Completion in the interface.** Typing `/` in the prompt opens a list of matching
  commands above it with a grey completion in the prompt itself; Up and Down move,
  Tab takes the suggestion, Esc closes the list. After `/mode `, `/model `, `/effort `
  and `/help ` the list offers that command's choices. The session bar shows model,
  effort, provider and mode.
- **New stream events.** `approval`, `approval_done`, `model` and `mode` on
  `--output-format stream-json`. A question is answered by sending `/yes`, `/no ...` or
  `/all` as an ordinary `user` line.
- **The moving-mesh benchmarks are in the repository.** `benchmarks/moving_mesh/`
  carries the two prompts run end to end on 2026-09-12 -- a `cyclicAMI` sliding
  interface through a circular-Couette annulus, and a cylinder on a spring free to move
  at Re = 100 -- and `grade.py`, which grades a rerun the same way. What was graded
  lived outside this repository, so nobody could reproduce it. The grader reads only
  raw solver output and never the run's own analysis, because grading those two by hand
  found four summary numbers that disagreed with their own files and one analysis
  script that read a torque at t = 13 and called it t = 20, having never opened the
  restart directory. Restart directories are stitched and each cut at the start of the
  next, columns are found by header name, and every physical constant comes off a case
  file or the grade stops and names the flag that supplies it. Two checks are the
  findings themselves: the interface's torque imbalance, which conservation fixes at
  exactly zero and which the solver's own `sum(weights)` cannot see, and the VIV energy
  audit. The known result now carries its grid convergence (three meshes per case,
  observed order 1.93 conformal and 0.79 with the interface) and the separation that
  matters for a rerun: at a fixed time step, four times the cells moves the interface's
  torque error from 0.2174% to 0.2205%, so the quantity to converge is `omega*dt` and
  not the cell count. The two nearest published amplitudes are named with the
  parameters that differ, and A/D is still reported without a verdict.
- **The moving-mesh field note carries those two runs.** "When the mesh moves" had been
  written from the Wigley free-surface hull alone. It now also records the cell
  diffusion number `nu*dt/dr^2` as what a diffusion-dominated refinement is limited by
  (stable at 3.1 on two correctors, divergent at 6.25 and 12.5, stable to 11.5 on
  four), that holding Courant fixed under refinement doubles it every level, that a
  sliding interface's error is first order in the angular slide per step, that the two
  wall torques' imbalance is a free check the `sum(weights)` health check misses
  (0.91% against a conformal control's 0.011% on a case whose weights were perfect to
  43 ppm), and the energy audit that moved a VIV amplitude from 0.640 to 0.567 after
  every conventional convergence test had passed it.
- **The mesh desk's finish check asks whether anything changed.** Asked to make a
  channel 30 mm tall instead of 20, the desk rebuilt it, ran its own look, reported a
  clean rebuild -- and the channel was still 20 mm, with identical bounds and an
  identical cell count. Every clause the check had passed, because after a no-op there
  is a valid mesh of the right rough size and nothing compared what was asked against
  what changed.
- **One tag publishes both registries.** `publish.yml` sends the Python package to PyPI
  and the launcher to npm from the same tag, by trusted publishing on both sides, and
  refuses before publishing anything if the tag, `__version__` and
  `launcher/package.json` disagree or if the tag falls outside the launcher's
  `PYTHON_PACKAGE_SPEC`. 0.2.0 reached PyPI while npm still shipped a launcher pinned
  `>=0.1,<0.2`, which does not half-ship a release, it mis-ships one: every
  `npm install -g openreynolds` keeps installing the old version and `upgrade` keeps
  agreeing it is current. `workflow_dispatch` catches one registry up on its own.

### Fixed

- **A Sandbox that cycled under a read is waited out instead of failing the tool
  call.** The hosted Sandbox goes away under load -- 9 of 68 tool calls in one live
  study and 56 of 244 in another came back `409 sandbox_gone`, in bursts, three of the
  four starting within 90 seconds of an 8-rank solve being launched (F-58). Nothing was
  lost, because `/work` is a persistent Volume and a job restarts from the latest time,
  but every one of them reached the model as a failed call and one run spent about 16
  minutes improvising recovery and relaunched the same solve stage five times. The
  service now answers that 409 with `Retry-After: 5`, and `FoamdClient.request` asks
  again once for a method that only reads (a stat, a file, a listing, a job status),
  honouring the header and capping what it can add at 15 seconds. 409 stays out of
  `_RETRY_STATUSES`: that set is keyed on the status alone and would cover
  `POST .../jobs` too, and a retried job start once produced five duplicate running
  jobs. `job_start` needs an idempotency key before it can be sent twice.
- **A file read back mid-write is no longer handed over as though it were whole.**
  `read_file` stats the path, asks for exactly that many bytes and is given exactly
  that many, so the short-read guard could never fire: a `postProcessing` forces file,
  a `.dat`, a `.csv` or a log read while the solver is still appending came back
  looking complete, and a file cut short is a number the model will happily average
  (F-64). Pictures were covered when a half-written PNG ended two sessions; the text
  the conclusions are drawn from was not. The path is stated a second time after the read -- one
  round trip on a path that already makes several -- read once more if the size moved,
  and if it is still moving the answer says so and names the three sizes rather than
  presenting a snapshot as the file.
- **A named model is no longer swapped for a preset's default.** A provider named on
  its own still arrives at its preset's model, but `OPENREYNOLDS_PROVIDER=reynolds`
  with `OPENREYNOLDS_MODEL=claude-opus-5` -- one of the two models that service meters,
  and what the hosted app sends when someone picks Opus -- loaded as Sonnet, on new
  sessions as well as resumes, and `/model claude-opus-5` on `reynolds` was undone the
  same way. What was asked for is re-asserted after the preset fills its blanks
  (`Config.load`, `switch.candidate`), for the desk model too.
- **Piped output stops folding long lines.** `rich` folds at the width it is given,
  however wide, so a message longer than `PIPED_WIDTH` arrived at a pipe with a newline
  in the middle of it -- a broken path for any program or agent reading the CLI through
  one, and the last of Windows CI's red. `soft_wrap` on the non-terminal console.
- **`job_kill`'s docstring stops promising a `killed` status for a kill that did
  nothing.** The service no longer marks a job killed whether or not the signal reached
  anything (F-63): it records a real ending for a job that had already ended, and
  refuses with `409 kill_not_delivered` for one that may still be running. Callers
  already treated a `BackendError` as a failed kill, so only the promise changed.
- **The dev extra installs `pyvista`.** Six toolbox tests read measurements through
  `mesh_look.py` and got empty values back instead of a loud `ModuleNotFoundError`.
  The symlink test also measured the fixture rather than the link, and failed on a
  `disk.py` that was already correct (#28, #29).
- **The README's images load, and the eighth tool is the one that exists.** All five
  figures had been 404 since tryreynolds.com moved them into a hashed asset bundle, so
  the repository front page, the PyPI page and the npm page had shown broken images
  through a release of each. The eighth tool has been `mesh`, not `geometry`, since
  2026-09-07, and `fetch` copies files out of the workspace rather than reading the
  open web.
- **The README says how to get in and what it costs.** Sign-in was described as email
  and password only; an account created with Google has no password at all, and
  `login --browser` is the way in for one. A company email address starts the account
  with $10 of credit, once, which the CLI already printed and the README did not
  mention. The hosted app's mesh editor and `benchmarks/moving_mesh/` are named, and
  `doctor --json` and `studies --json` are documented where a program looks for them
  rather than in the last clause of a table row.

### Changed

- **The free-will contract is amended, not dropped.** `docs/design.md` section 1, the
  system prompt and the README's Security section now say what is true: no approvals
  and no enforced ordering in the default mode, and exactly what the person chose to
  have gated in the other two.
- **`fetch` is described correctly in the README.** It copies files from the workspace
  to your machine; the README had said it read the open web.

## [0.2.0] - 2026-09-12

### Added

- **A fourth interface: `--output-format stream-json`.** A program can now drive a
  session and read it as data. `openreynolds/jsonview.py` implements the whole `View`
  protocol as NDJSON on stdout, one JSON object a line, locked and flushed per line so a
  reader never sees half a record; `JsonReader` is the other half for anything that
  wants to consume it in Python. `studies --json` and `doctor --json` answer the same
  way. The seam already carried three interfaces (the terminal, the hosted app, the
  quiet mode), so this is a fourth implementation rather than a new pathway, and
  stdout purity is held by a test: nothing but NDJSON may reach it, which is why
  `images.suppress()` exists (a graphics escape sequence in the middle of the stream
  leaves a strict reader resynchronising inside a base64 blob, permanently).
- **Trace rows carry identity.** Every tool call now records its `tool_use_id` and a
  `result` of `{ok, bytes}`, so a transcript can be joined call-to-answer by a program
  instead of by eye.
- **The mesh is allowed to move.** `mesh_look.py` reports cell and face zones, the
  motion ladder in the field notes covers `solidBodyMotion`, AMI and morphing meshes,
  preflight carries motion rules, and a "When the mesh moves" note is drawn entirely
  from what the free-surface Wigley run actually paid for.

### Fixed

- **A picture the model will not accept no longer costs the session.** Two long runs
  died on `400 invalid_request_error: Could not process image`, both immediately after
  the agent redrew a figure and read it back, one 27 minutes in and one 2 h 23 m in.
  Two defects had to line up. `attachment` base64-encoded whatever it was handed, and
  base64 of half a PNG is well-formed base64, so nothing between the disk and the API
  could tell: `images.incomplete` now checks that a file carries the marker saying it
  ENDED (PNG's IEND, JPEG's EOI, GIF's trailer, WebP's RIFF length) and a partial
  render comes back as a sentence saying it is not ready. And a 400 ended the run,
  which was right about waiting and wrong about repair, because the bad bytes sit in a
  thread this process owns: `Loop.drop_images` replaces every image with a note and the
  turn is retried once.
- **A resumed session says which workspace it joined.** `acquire()` took `existing[0]`
  from an unordered listing. With one instance per account that was not a choice; with
  several it attaches a resume to whichever row the service returned first, and the
  failure does not look like a wrong choice, it looks like a workspace that lost its
  files. The listing is ordered most-recently-active first, by the same key the
  service's own repair step uses, and the join notice says how many workspaces the
  account holds and that `--instance` picks another.
- **`/work` is resolved before it is compared.** It is a symlink to
  `/__modal/volumes/vo-<id>`, and `/proc/<pid>/cwd` yields the physical path, so any
  probe that string-matched a cwd against the literal `/work` matched nothing in
  production. That had silently disabled the escaped-solver probe for its whole life.
  `stopping.py` and `toolbox/disk.py` resolve the root first.
- **A leaving session stops killing its neighbour's work.** A session that joins a
  workspace somebody else started no longer sweeps or stops it on the way out.


### Added

- The mesh desk (`openreynolds/mesher/`): geometry and meshing are now one small agent
  with one tool. It gets a shape in words and a case directory, and works the way a
  person at a terminal does -- one fenced ```bash block a message, run on the instance
  that already has OpenFOAM, gmsh, build123d and pyvista on it, with the output pasted
  back. Two things the harness owns rather than the model: any PNG a command writes
  comes back attached to that command's output, so seeing is not something to remember
  to arrange; and `echo MESH_DONE` does not end the run -- `mesher/check.py` runs
  `mesh_look.py` on the machine and hands the run back with the reasons unless
  `constant/polyMesh` exists, `checkMesh` passes, at least two patches carry names
  somebody chose (not `patch0` out of gmshToFoam), an `Allmesh` or build script is
  there to rebuild it, and a picture was drawn. Budgets are a turn count and a wall
  clock, and a run that hits one is still checked, because an unexamined mesh is the
  failure this desk exists to end. Three files and a brief, in place of a spec
  language, a compiler, a linter, a claims engine and a shape library.
- `toolbox/mesh_look.py`: any meshed case drawn and measured in one call, straight from
  `constant/polyMesh` -- no solve, no fields, no time directory needed. One PNG with
  **one colour per patch** (2D: the patches and the cells, with the `empty` front and
  back left out so the others are visible; 3D: the patches with the flow box drawn faint
  and a cut on each axis), and under it cells, faces, points, the bounding box, whether
  the mesh is one cell thick, `checkMesh`'s verdict with maximum non-orthogonality,
  skewness and aspect ratio, and for every patch its type, face count, area, centre and
  mean unit normal. The normal is the point: it says which end is the inlet without
  anybody guessing from where the patch sits, which is the classification mistake that
  silently mislabelled every L, U and elbow.
- The hosted app's **Mesh** panel (`ui`): pick a case, press Look, and see the same
  picture and the same table the desk works from, with a box to ask the session for a
  change in words. It replaces the CAD-spec editor, whose spec language went with the
  toolbox that read it.

### Added

- `toolbox/disk.py` -- what is filling the workspace and what of it is safe to delete
  (F-56). There is one disk: an account is capped at one instance, a new study joins the
  existing one, and every study ever run is a sibling directory under `/work` sharing a
  single 20 GB quota, with no retention policy and no owner. Measured 2026-09-08 at
  30.8 GB used against that quota, with a live 24 MB study unable to write; 24.5 GB of it
  was regenerable scratch. `report` ranks by size and says how much of each directory
  could be produced again; `prune` removes only that and never an input, a log, a result,
  a figure or an `Allrun`. Dry run unless `--apply`, refuses to guess which study, and
  never deletes a whole study directory.
- `casebundle.py` -- the case itself, captured at session end. The capture plane has
  always carried the conversation and never the work: measured over 215 production
  studies, the transcript is there for essentially all of them, an artifact for 40%, a
  results payload for 12%, and the case definition, mesh and solver logs for **none**. So
  every mesh and every `Allrun` this product has made lived in exactly one place, on that
  shared disk. The bundle is built from the local mirror (no extra network) in tiers,
  irreplaceable first -- definition, record, notes, geometry, then mesh and latest fields,
  which a resume can rebuild and re-solve. `definition + record` is about 4 MB a study, so
  the part that cannot be reproduced always fits. Nothing is dropped silently: the report
  and the `MANIFEST.json` inside the archive both name what was left out and why.
- `toolbox/cfmesh.py` -- cfMesh reachable from the toolbox, with the four traps that each
  cost a round encoded so they cannot be walked into (F-53). snappy's layer coverage
  erodes 88.6% -> 66.7% over fifty iterations on a thin hull while cfMesh covers 100% of
  the same wall at its defaults, and cfMesh has been prebuilt in the image all along.
  What it costs is stated rather than glossed: no y+ control.
- `toolbox/reattach.py` and `toolbox/claims.py` -- one blessed reattachment implementation,
  and the number a figure was drawn from stamped into the figure (F-36). A study once
  delivered a figure annotated 7.57 beside an answer that said 6.4 and reconciled neither.

### Fixed

- A redirect on a request carrying a body is no longer followed blind (F-47). httpx
  re-issues a 301/302/303 on a POST as a **GET with no body**, so the blanket
  `follow_redirects=True` added for F-45 made any redirect on an upload a body-losing
  event. 307/308 and bodyless requests are followed; a 303 is followed and *remembered*,
  so a 4xx collected through a hop the body did not travel on is treated as ambiguous
  rather than as a fact about the request; 301/302 on a body-carrying request are refused
  and named. And `put_tree` still holds the archive it built, so on a `400 not a valid
  tar.gz archive` it opens those bytes and, if they are a readable gzipped tar, says so:
  the bytes that arrived are not the bytes that were sent. The message used to point at
  the wrong party, and the wrong party is the only one that can prove it.
- `describe_job` no longer labels every `killed_by` "matched kill_on line". That was right
  while a `kill_on` regex was the only writer; foamd now writes the field on every
  terminal state that is a kill, and only one of six producers is a log line. The old
  label would have told the model a vanished container was a regex match.
- A session no longer stops a workspace with somebody else's work still running on it
  (F-46). Teardown asked only whether another session had the instance up when this one
  joined it, which is the wrong question about a detached job: jobs outlive sessions by
  design, and a job started outside any session leaves no flag anywhere, so the next
  session to start the instance also owned it and stopped it on the way out. A detached
  rendering job died that way with six of its eight steps done and its two animation
  passes holding empty output directories. `Backend.active_jobs()` asks the service what
  is actually running -- one read of the job rows, which starts no container -- and the
  workspace is left up, and named, when anything is. A listing that fails answers
  "nothing" and says so, because a workspace left up on an unanswerable question bills
  until a reaper notices.
- The capture warning says what it lost (F-44). `capture dropped 1 item(s)` reported a
  count and nothing else, and the only question a person has on reading it is *what*: a
  dropped message leaves a gap in the transcript on the web, a dropped artifact leaves a
  picture that exists on the laptop and nowhere else. Up to six items are now named
  (`artifact mesh_z.png`, `message seq 41 (assistant)`); past that the kinds are counted
  and the first is named, because the middle of a long outage is not readable.
- A word in a comment is no longer a finish. The mesh desk's `MESH_DONE` was matched
  anywhere in a bash block, and the brief hands the desk that token -- so a step whose
  comment said `# will echo MESH_DONE once checkMesh passes` had its actual command
  never run, and got the finish check's refusal as the answer to something it had not
  asked. Found by reading the new code adversarially rather than by a run failing.
- A picture is sent to the desk once per version of the file. A command that merely
  mentioned a `.png` -- a `cat` of the build script, an `ls` -- used to have that file
  attached to its output as though it had just been drawn.
- A model call that fails with an overloaded or gateway status is retried once. A desk
  five minutes into a mesh cannot resume; the next call starts a clean thread.
- A user turn carrying a picture reaches an OpenAI-family model. The provider's renderer
  handled an image inside a tool result and dropped a bare `image` block in a user
  message, which is exactly the shape the mesh desk sends every render in: a model was
  shown nothing and told it had been shown a picture.
- The workspace mirror no longer stalls a session for minutes (F-54). The desk's uploaded
  `Allrun`, a file nothing on the instance ever opens, made foamd's path probe time out
  and answer 400, and the mirror read the 400 as one unresolvable file, split every batch
  into single requests, and repeated that every cycle; a 27-second finish step waited five
  minutes behind it. The mirror now remembers a path the service refused and names it,
  splits a failed batch once to find the culprit, and stands aside while a tool call is in
  flight (`mirror.Gate`, held by the loop around each call); each cycle reports its round
  trips and seconds. Uploaded files keep their source mtime instead of 1970, so a file the
  instance never opened is not mistaken for one never written. foamd's side (a timed-out
  probe answered 504 with its paths) shipped as foamd `9b52d97`.

### Changed

- The previous geometry and meshing stack is **gone**, not extended: the
  `openreynolds/geometry/` package (a sketch API, a compiler, a linter, a claims engine,
  a measurement kernel, a fitness table, a preview renderer, a shape library -- 12,400
  lines), `toolbox/mesh2d.py`, `toolbox/cad_gen.py` and `toolbox/snappy_gen.py`, their
  tests and fixtures, the `geometry` tool, the `geometry` install extra, and the gmsh +
  matplotlib + X/GL layer the hosted runner carried so the old desk could draw in its own
  process. Every mesh generator went with it: the mesh desk writes the geometry itself,
  in whatever gmsh, build123d, blockMesh, snappyHexMesh or cfMesh script the shape
  actually calls for, and leaves that script in the case so a person can change a number
  and re-run it. What is kept from all of it is the one lesson that paid: render and
  measure, never assert.
- `toolbox/case_gen.py` dresses a mesh instead of building one. Its template half went
  with the rest -- a dozen parameterised shapes, an O-grid tiler and a blockMeshDict
  writer, and with them the habit of adding a thirteenth shape whenever a new case did
  not fit. What is left reads the mesh in front of it (patch names and types from
  `constant/polyMesh/boundary`, cells, bounding box and smallest cell from `checkMesh`,
  each patch's area, centre and normal from `mesh_look.py`) and writes `0/`, `system/`
  and `constant/` onto it. The three things that used to come from the template are now
  measurements: the inlet's direction is its own normal with the side the fluid is on
  probed off the mesh (the bounding-box rule it replaces is exactly backwards on a
  U-duct, and worse on anything concave); the characteristic length is the inlet's
  hydraulic diameter, 2w for a plane passage; and a wall patch that reaches at most one
  of the domain's axes is a body in the flow rather than the flow's boundary, which is a
  hundredfold in the free-stream turbulence. Every patch gets a role from what you name,
  then from the mesh's own constraint types, then from its name -- and a name that says
  nothing is a wall, which is the safe reading. Measured: the L-duct the mesh desk built,
  dressed by this, solved 20 iterations of simpleFoam with residuals falling.

### Added

- Each generated case carries an `Allmesh` script: blockMesh, feature extraction,
  snappyHexMesh, `topoSet` where there is a rotating zone, and checkMesh in one
  call, with one log per stage and a digest at the end -- replacing the fifty-odd
  round trips a 3D setup was taking. It stops on the first failing stage with that
  stage's log, and treats checkMesh as the diagnostic it is rather than as a build
  step, since a layered snappy mesh routinely reports concave cells.
- `Allmesh` fails loudly when snappyHexMesh refined **nothing**. A background cell
  wider than the body's cross-section leaves the surface crossing no cell edges, so
  snappy meshes an empty box and exits 0 saying "Finished meshing without any
  errors" -- and the empty box then solves cleanly. The generator now sizes the
  background cell so it cannot happen, and the script checks anyway.

### Fixed

- An unfollowed redirect says so. Modal's edge answers any request open past 150 s
  with a bodyless `303` to the real result; `FoamdClient` has followed those since
  `a228cbf`, but a `3xx` is under 400, so a client built without redirect-following
  handed one back as a success and it died three frames later in the JSON decoder as
  `bad_response (303): the body was empty` -- the most frequent tool error in the
  transcripts, and a sentence that named the symptom while hiding the cause. Both the
  request path and the decoder now raise `redirect_not_followed`, which says which
  switch is off. The four sign-in helpers, which built their own clients, were that
  exact mistake still in the codebase; they follow redirects too. `EXEC_MAX_TIMEOUT_S`
  stays at 300: the cap governs the command, the 150 s governs one HTTP request, and a
  redirect-following client runs a 250 s exec to completion.
- `locationInMesh` survives a body that overhangs the domain. A box cut
  deliberately INSIDE the geometry -- tubes trimmed by their own end planes so they
  span the bank the way a correlation assumes -- puts the body's bound below the
  domain's, and stepping a fraction of the way towards it walked out through the
  floor. snappy then rejects the point while printing a bounding box that does not
  contain it, which reads as a domain problem and is not one.
- `forceCoeffs` states `rhoInf` on compressible cases too. It is the reference
  density the coefficients are normalised by whichever solver is running, and
  without it a thermal run aborts on its first write -- after the mesh is built and
  the case decomposed.
- A symmetry plane gets its own patch. `symmetryPlane` is a constraint requiring
  coplanar faces, so a centreline and a waterline in one patch is rejected by
  blockMesh; and `--far symmetry` now types the mesh patches to match the fields,
  which otherwise fails at `decomposePar` with "attempt to cast type patch to type
  symmetryPlane".
- `locationInMesh` no longer lands on a cell face. Half of an even number of cells
  is exactly a face, and snappy rejects such a point while printing a bounding box
  that plainly contains it.
- Thermal cases use plain SIMPLE rather than SIMPLEC. Pairing SIMPLEC with a 0.3
  pressure factor is conservative on both counts and converges too slowly for a
  fixed iteration budget, which on a heat-transfer case means the wall heat flux --
  the entire answer -- has not settled.
- `case_gen.py` had no `symmetry` role. Every 0/ writer now answers one, `p`
  included -- it was the omission that mattered, because a `symmetryPlane` patch
  carrying a `zeroGradient` pressure is rejected by OpenFOAM outright. The
  constraint is written as `symmetry` rather than `symmetryPlane`: the latter
  requires its faces to be coplanar and so cannot carry the two opposite walls of a
  tunnel as one patch.
- A long `exec` returning a bare 303 with an empty body (F-45). `FoamdClient` now
  follows redirects, so a command that outlives the edge's patience returns its
  result instead of raising. Measured: `sleep 200` now returns rc=0 in 203.9 s
  where it previously raised `bad_response (303)`. This was expensive -- re-reading
  round 2's timings, **27 of 114 minutes (24%) went to backend errors, more wall
  clock than all the solving (18 min, 16%)**, concentrated in one study that lost
  45% of its runtime to it.

- Every field picture came out a quarter turn round. `view_vector` was given the slice
  normal and no up vector, so VTK chose one, and for a z-normal slice it chose +x: a
  cylinder wake in a left-to-right flow rendered as though the flow went upward. Found
  by looking at a render from a live run rather than by reading the code -- the physics
  was right and the orientation was not, which is the worse of the two, because a wrong
  number gets checked and a wrong orientation gets believed. `results.py` and
  `animate.py` now name the up vector; `animate.py` reuses the first frame's camera for
  the rest, so one wrong frame was a wrong animation.
- The colour bar sat across the bottom of the picture, its tick labels over the flow.
  It is under the picture now, in `results.py` and `animate.py`.
- `first_look.py`'s patch panel was a coloured slab on any 2D case: the `empty` patches
  are most of the boundary by area (8,476 faces against 384 for everything else in the
  live run), and from an iso camera they covered every patch the panel exists to name.
  They are left out and marked `[hidden]` in the legend.
- `preflight.py` failed a healthy case. Its cell-count check ran the snappyHexMesh
  estimator -- a uniform background mesh refined around an STL -- against a case with no
  `snappyHexMeshDict`, read a 12-block graded O-grid as 105 cells against 4,238 real
  ones, and advised checking `refinementRegions` the case does not have. Where blockMesh
  builds the mesh alone the count is the product of the block divisions, which is exact,
  so that is what it compares now.
- `render.py` pointed its camera with `plotter.camera_position = "z"`, whose string
  form takes a view plane ("xy", "xz") and not an axis letter, so every field and mesh
  render came out of a camera that was not aimed where the caption said. A study found
  this, worked out the fix and applied it to its own copy under `/work/.toolbox/` --
  the copy that is overwritten from the distribution at the start of the next session.
  Fixed in the distribution, where it survives the sync.

### Changed

- A refusal from the model service is no longer retried. A 402, a 401 or a bad model
  id is the API answering a question about the account or the request, and it answers
  the same way a minute later; 408/409/425/429 and every 5xx are still retried as
  before. When the answer is a refusal the session says so plainly, on the page as well
  as the console, and stops sending until you say something -- job endings are still
  recorded in the thread for whenever it resumes, progress chatter is not. Found in a
  live study: the account budget ran out thirty-five minutes in, and the harness spent
  the next twenty-six minutes making ninety refused calls while the person typed
  "whats going on?" and got nothing.
- The system prompt no longer says the container has 8 cores (the default shape is 4)
  or that `mpirun` fails without `OMPI_ALLOW_RUN_AS_ROOT`. The workspace service now
  starts every instance with the environment OpenMPI needs, so `mpirun` works without
  arranging anything; `nproc` is what reports the core count.

### Added

- Eight toolbox scripts, and one file they all agree on. The toolbox was six small
  readers; the studies kept rewriting the same four hundred lines around them --
  a blockMeshDict generator, a "render every field" script, a plotting script, a
  frames-to-gif script -- from scratch, per study, with the same mistakes in each.
  What is offered now (still offered, never imposed):
  - `study_state.py` -- the contract the rest share. A manifest under
    `<study>/.reynolds/` saying what every artifact is *for* (`mesh-full`,
    `vorticity`, `residuals`, ...) and a phase table saying how far the study got.
    Plain text on the Volume, so a session that ends mid-solve leaves an answer
    behind rather than a directory to re-derive.
  - `case_gen.py` -- runnable cases from a template and a few numbers: 2D external
    flow (circle, square, L, a vehicle, an imported profile), Y/T/Z/F/M ducts, sharp,
    mitred and rounded bends; mesh-only, steady or transient; a moving ground and
    rotating wheels. It solves `nu = U*L/Re` and says so, and every 2D case it writes
    is one cell thick with a single `empty` patch declared in the mesh *and* in every
    field -- the pairing that otherwise stops the solver on the first time step.
  - `first_look.py` -- geometry, the whole mesh, a close-up on whatever the mesh was
    refined around, the named patches and the counts, composed into one contact sheet.
    One `read_file` instead of five round trips, before a core-hour is spent.
  - `preflight.py` -- the cheap questions asked before the expensive run: patch names
    against the fields, `empty` on a 2D case, STL scale, Re against nu, predicted
    cells, checkMesh, a one-iteration probe, Courant, residual divergence, force units,
    free disk. Each answered as a finding with what was measured, what it means and a
    repair -- a suggestion, not a gate.
  - `results.py` -- a finished case as the standard set of pictures and plots in one
    call, by preset, rather than a render command per view.
  - `animate.py` -- extended: velocity, pressure, vorticity, turbulence fields and
    streamlines; colour limits and camera fixed across the sequence; labels burned in;
    a sidecar naming the intended container and fps for the machine that has an
    encoder; and resume, so re-running during a long solve renders only what is new.
  - `progress_report.py` -- one answer to "what is happening?": the phase, the time
    reached against the target, the residual trend, the Courant number, frame progress,
    an ETA that says what it was estimated from, and the paths of the pictures that
    already exist.
  - `study_run.py` -- the pipeline geometry -> preview -> mesh -> checkMesh -> probe ->
    solve -> reconstruct -> render -> animate -> report, resumable: it reconciles the
    recorded phase table against what is actually on disk and picks up at the first
    thing not done. A failing phase stops it and is recorded; what to do about the
    failure stays the agent's call.
  - `gallery.py` -- the manifest as a self-contained `gallery.html` (images embedded,
    no network of any kind in it), a contact sheet, and `--final`, the paths of the
    latest artifact of each kind.
- The frames an animation is rendered from now say what they were rendered *for*, and
  the harness reads it. `animate.py` writes a `frames.json` beside them naming the
  container and the frame rate; `delivery.py` (the automatic assembly during a session)
  and `openreynolds video` both take the output name and the fps from it, falling back
  to a 10 fps gif only for a directory that never declared one. Before this,
  `--format webp --fps 24` rendered fifty correct frames and the harness silently made a
  10 fps gif -- the pictures crossed the gap between the instance and the laptop and the
  intent did not, and `webp` was unreachable by any path.
- Field notes: "When `mpirun` will not start" -- the PMIx signature, why an instance
  with no outbound network produces it, and `PMIX_MCA_gds=hash`. Two studies lost an
  afternoon between them to this; one ran eight cores serially for an hour rather than
  read to the end of the error that named the fix.
- Field notes: a pointer to `log_digest.py` and `mesh_digest.py` where the failure
  signatures are read. Grepping a long log several times is three round trips to the
  instance for what one call answers -- a live session spent a minute on three passes
  over the same 1,700-line `log.snappy`.

## [0.1.0] - 2026-08-27

The first release: the tool-use loop over a hosted OpenFOAM workspace, seven tools, the
terminal interface, watch mode with factual wakes, the local study mirror, capture,
stopping that verifies, the toolbox and the field notes -- and, since the first cut
on 2026-08-24:

### Added

- An npm launcher (`launcher/`, published as `openreynolds`): `npm install -g
  openreynolds` finds or installs `uv`, installs the Python package from PyPI once,
  and runs it. No Python source ships in the npm tarball.
- `openreynolds login`: sign in from the terminal by approving a short code in the
  browser; the service key lands in the config file. The service has a default address
  now (`https://api.tryreynolds.com`), so a fresh install needs no URL.
- Bring your own model, not just your own key: `OPENREYNOLDS_PROVIDER` /
  `openreynolds config --provider` pick Anthropic, OpenAI, Z.ai, DeepSeek, Moonshot,
  MiniMax, OpenRouter or a local Ollama, or any endpoint speaking the Messages API or
  Chat Completions (`openreynolds/llm/`). `ANTHROPIC_API_KEY` still works; the general
  name is `OPENREYNOLDS_LLM_API_KEY`, and older config files are read as before.
- A live mirror: the study's files are synced to `./studies/<id>/files/` in the
  background for the whole session, and a render the model just looked at arrives at
  once. The interface's files pane reads from it.
- A progress bar for real compute -- solver time against `endTime`, residuals, Courant
  number, snappyHexMesh phase -- shown only while a solve, mesh, decomposition or sync
  is running.
- A front desk: a second, cheap agent that answers the user within seconds while the
  main agent is mid-turn, and writes the plain-language "now" line. It is read-only
  and cannot steer the agent. `OPENREYNOLDS_DESK=0` turns it off.
- Render delivery: every image the instance writes lands in a flat
  `./studies/<id>/renders/` folder, frame directories are assembled into gifs locally,
  and the interface has a renders tab. `openreynolds renders` lists them.
- `openreynolds video` encodes a frame set on the user's machine (ffmpeg, or imageio as
  a fallback); the instance never carries an encoder.
- `openreynolds push` carries a local file or directory up to the study's workspace.
- `job_check` takes `wait_s`, holding the answer until the job ends or the user types.
- `python -m openreynolds` as an entry point beside the console script.
- `cli.session(..., interface=)`: a seam through which another interface (a web page,
  a test harness) can drive a session using the same `View` the terminal uses.
- `-p` runs exit `1` when the model API would not complete a turn and `2` when
  `--max-wait` ran out with a job still running, so a script can tell the two apart
  from a run that finished.
- `OPENREYNOLDS_CAPTURE=0` switches transcript upload off from the environment; the
  README says up front that transcripts go to the workspace service by default.
- Packaging metadata: project URLs, authorship, keywords, classifiers, a PEP 639
  licence expression, the version read from the package, `video` and `toolbox`
  extras, and an sdist that leaves study mirrors and engineering scaffolding behind.
  CI runs on Windows and Python 3.11 as well.
- `SECURITY.md`, `CONTRIBUTING.md` and this changelog.

### Changed

- Turn-end syncs no longer block the session thread; the mirror is poked instead, so a
  message typed during a long solve is read at once.
- After two model-API failures in a row the session says plainly what is happening and
  how to resume, instead of repeating that the thread is intact.
- The thread sheds the pixels of images the model has already looked at, keeping the
  path and description, so long render-and-look sessions stay a size the API accepts.
- Field notes: steady-versus-transient, the 2D recipe, `job_check wait_s` in place of
  `sleep`, rendering next to the data, and what a steady solve that will not converge
  usually means.
- The design document moved to `docs/design.md`.
- `--rotate y:90` turns an uploaded surface before meshing, because the inlet is
  always -x and geometry does not arrive aligned to anybody's wind tunnel. The case
  is written with the surface that was actually meshed, not the original file --
  copying the original would leave the body somewhere the domain is not, which
  snappy reports as an empty mesh rather than as a mismatch.
- Every case prints its projected area down each axis, and an `--mrf` case objects
  when the axis it was given is not the axis of largest projection: looking down a
  rotor's shaft you see its blade planform, so that projection IS the disc.
  Spinning a propeller about a line lying in its own disc gives a plausible thrust
  beside a torque an order of magnitude too large, which reads as a mesh problem
  and is not one.
- Thermal cases carry an `outletTemperature` function object. Every heat-transfer
  coefficient worth comparing with a correlation is defined on the **log-mean**
  temperature difference, and without the outlet bulk temperature the obvious thing
  to reach for is `T_wall - T_inlet` -- 9% of the answer on a real bundle, in the
  pessimistic direction, with every term in it right except the one nobody writes
  down.
- The MRF zone is sized to **contain** the rotor by default. A cylinder taken from
  the disc radius but a quarter of it thick is thinner than the blades are long
  axially, so the tips sit outside the rotating frame and feel no rotation.
- `--medial-ratio`, `--medial-angle` and `--layer-iter` expose the three controls
  that decide whether a layer stack is BUILT rather than how thick it is, and
  `maxThicknessToMedialRatio` defaults to 0.6 rather than the tutorial 0.3, which
  assumes a chunky body. Worth recording what they cannot fix: the binding
  constraint on a thin hull turned out to be the ratio of first layer to surface
  cell. snappy builds 1:26 and 1:32 happily and refuses 1:118, and no combination
  of these controls changed that.

### Fixed

- `openreynolds doctor` opened a real study on the platform every run to check that
  capture worked. It now makes a read-only call and changes nothing.
- A duplicate copy of the architecture notes at the repository root is gone; the one
  under the toolbox notes is the one that ships.

[Unreleased]: https://github.com/InviscidAI/OpenReynolds/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/InviscidAI/OpenReynolds/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/InviscidAI/OpenReynolds/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/InviscidAI/OpenReynolds/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/InviscidAI/OpenReynolds/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/InviscidAI/OpenReynolds/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/InviscidAI/OpenReynolds/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/InviscidAI/OpenReynolds/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/InviscidAI/OpenReynolds/releases/tag/v0.1.0
