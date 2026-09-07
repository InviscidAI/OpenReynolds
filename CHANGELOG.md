# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The geometry segment is a package, `openreynolds/geometry/`, and the desk authors in
  Python instead of JSON (`qa-runs/geometry-design/DESIGN.md`, Phase 1). The old grammar
  made overlap the default outcome: a turtle path plus a `repeat` with its own pitch asks
  the model for arithmetic it does not do reliably, and the shape that came back had four
  loops growing through each other. A script says what it means instead. `Row(bypass,
  count=4, gap=2)` derives its pitch from the footprint the tool measures and refuses the
  row with the two numbers when they do not fit; `Bypass(wall=main.top, at_x=14,
  leave_angle=25, outer_radius=6, return_angle=45)` is solved in closed form and lands on
  the wall it names; ports are declared by intent (`s.inlet = duct.left`) and resolved
  against the built face, never by a bounding box. The lap ends in a print-back the model
  reads: the features it built, the leg table, the lint findings with coordinates, the
  measurements, and the compliance table.
- A shape is judged against the request's own words. The desk's first lap writes claims --
  every number with a tolerance, every clause a named predicate (`returns_against_flow`,
  `shallow_angle`), the counts, where the inlet and outlet belong -- before it draws
  anything, so the geometry is measured against the sentence and not against itself. It
  cannot commit with a lint error or a failing claim; where a request cannot be met it
  replies `COMMIT disagrees: <clause>` and the person sees which clause. Three invariants
  are pinned by tests: no reference without a golden, no commit without a clean lint and a
  compliance table, no mesh result without a fitness table.
- A library of reference shapes (`geometry/library/`): the Tesla valve, a serpentine and a
  T-junction, each parameterised code with a rendered golden and golden measurements. No
  entry reaches the model until a person has looked at its render and approved it; a shape
  described in a docstring and never drawn is how the wrong valve was copied into four
  runs. All three are unapproved and invisible until then.

### Fixed

- A mesh-only case names an application in `controlDict`. The 3D desk's finish ran Allmesh
  on a case whose dictionary said `application ;` (a mesh-only study has no solver) and
  gmshToFoam refused it; the 2026-09-07 penne re-measure spent twenty turns fixing that
  line by hand after the desk had drawn the right solid in 34 seconds. `snappy_gen`'s
  writer names `simpleFoam`, as `case_gen`'s already did.
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

- Geometry is checked by machine before it is a case (`qa-runs/PLAN-geometry-revamp.md`,
  Phase 0). A "correct" Tesla valve had passed every count -- four islands, an inlet, an
  outlet, Mesh OK -- with its four loops overlapping each other by a quarter of their
  area and a notch at every junction. `mesh2d.py` now refuses copies of a `repeat` that
  overlap or touch (the overlap measured), a `to` leg that lands off the body, and a
  passage with other than one inlet and one outlet; it warns, with coordinates and red
  crosses on the preview, about short edges (a third of the narrowest channel, not a
  thousandth of the span) and a channel end read as a wall; it prints a leg table for
  every channel (each leg's start, end and absolute heading, each arc's radius and
  sweep, where a leg lands) so a request's angles and radii are read off numbers; a
  `to` leg and a channel given `from` are cut flush at the wall (the joint disks that
  made the notches are gone); a `near:x,y` patch rule names an end when two ends sit on
  one line. The worked Tesla example is gone from the docstring: a shape written by hand
  is a shape nobody measured, and it was copied into a wrong valve by every author that
  read it. `cad_gen.py --internal` refuses a passage whose automatic reading gives other
  than one inlet and one outlet, or an inlet and outlet that differ threefold in area
  (an L duct's side wall taken for its end, both ends of a U taken for inlets, an elbow
  with no outlet -- each was written as a case before); `--inlet` / `--outlet` name the
  ends (`x:min`, `y:0.08`, `near:x,y,z`). It also takes `mesh2d.py`'s spec envelope
  (`{"ops": [...], "scale": 0.001}`) and draws its preview with matplotlib where pyvista
  is not importable, which is the runner the geometry desk draws in -- there, every 3D
  lap had failed. The desk reasons at its own effort (`geometry_effort`,
  `OPENREYNOLDS_GEOMETRY_EFFORT`, default high; the hosted app runs the main loop at
  medium and authored wrong shapes at it), replies in up to 16k tokens, is told to write
  the request's checkable claims down first and to judge the leg table against them, sees
  the picture of a refused spec, and finishes the job: after the case is shipped it runs
  Allmesh, checkMesh and the mesh render on the instance in one exec and returns the
  checkMesh digest and the mesh picture with the outline, so the main agent has nothing
  left to discover. Measured on six prompts through the desk (`qa-runs/RESULTS-phase0.md`):
  the first correct Tesla valve from any run (leave 20°, outer radius 6, return against the
  flow, loops apart) in 4 turns and 5 minutes, and an L duct, a U duct and a serpentine
  each right in 4 to 5 turns. What the runs found and what changed after them: a `near`
  rule picks the nearest edge and the nearest surface rather than one within a tolerance
  (gmsh's box round a circle is off its centre, and a cylinder named `near` came out as
  `walls`); the desk's laps get ten minutes and eight laps (a lap at high effort is a
  minute); the 3D spec is stated to be the solid only, the envelope carries `domain`
  (internal|external) and `inlet`/`outlet`, and a hollow external body is refused as the
  fluid drawn instead of the solid (the 3D penne run drew the box, and the report's "1
  enclosed void" went unrefused for six laps).

### Added

- An eighth tool, `geometry`, and the desk behind it (`openreynolds/geometry.py`).
  The same model that one-shots a Tesla-valve mesh in a chat app took 66 turns and
  29 minutes in this harness for the wrong shape, and with `mesh2d.py` in the
  toolbox still spent four preview laps of its own turns getting the shape right:
  every write-and-look was a remote round trip. The desk runs those laps where the
  model is -- gmsh and the preview in the runner's own process, one to two seconds a
  lap -- under its own brief (spec, picture, measured report, revise, commit; six
  laps; never a solver), then ships the finished case to the workspace and hands
  the main agent one result: the picture, the measurements, where the case is. The
  main prompt gains a descriptive sentence and nothing that says when to use it.
  `pip install openreynolds[geometry]` (gmsh, matplotlib); without it the tool says
  so and the toolbox scripts do the same from a hand-written spec.
  `OPENREYNOLDS_GEOMETRY_MODEL` / `geometry_model` pick its model; the default is
  the main one, and its tokens land in the same totals as the main loop's.
- `toolbox/mesh2d.py`: any closed 2D outline -- primitives, booleans, constant-width
  passages swept along a centreline -- to a one-cell-thick all-hex OpenFOAM case in
  one call, drawn and measured (extent, area, enclosed loops, per-patch edge counts
  and lengths) before it is meshed. `cad_gen.py`'s 3D path prints the same kind of
  measured report (surfaces, shells, enclosed voids, wetted area) off the B-rep.
- `toolbox/snappy_gen.py`: complete, runnable cases around an uploaded surface.
  `case_gen.py` writes a body-fitted blockMesh for shapes it can draw; this writes
  the other half -- a background box, snappyHexMesh cut to an STL, boundary layers
  sized to a stated `--y-plus`, and the dictionaries for a steady, transient,
  thermal (`buoyantSimpleFoam`) or MRF run. Three things in it are load-bearing:
  the reference area is **measured off the STL** rather than typed in (wetted area
  is the triangle sum, frontal area the rasterised silhouette, which is right for a
  body with a hollow or a second part behind the first); a `--symmetry` plane that
  **bisects** the body scales that area with it while one that merely bounds the
  flow -- a waterline closing a hull -- does not; and the first layer follows from
  y+ with the factor of two the centroid definition implies. `--wall-speed` exists
  because a propeller tip at 4000 rpm sees 50 m/s while its tunnel sees 7.
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

[Unreleased]: https://github.com/InviscidAI/OpenReynolds/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/InviscidAI/OpenReynolds/releases/tag/v0.1.0
