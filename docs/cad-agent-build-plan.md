# CAD agent — build plan (revised 2026-09-10)

The CAD stage of the Inviscid AI OSS simulation workflow agent.

This revises the original plan against the tree at `376ec4c`. Every change is traceable
to a numbered entry in `cad-plan-decisions-2026-09-10.md`, cited inline as (#n). The
research grounding of the original survives intact -- all five external citations were
verified -- and most of what changed is its description of this codebase, which was
written against a stack deleted three days earlier.

## Sections

0. Why this is not the stack that came out
1. Scope and non-goals
2. Architecture and boundaries
3. Geometry layer
4. Meshing
5. Checks
6. Design iteration loop
7. What is actually new

Not planned here: **sequencing**, decided at the start of implementation. **Open risks**,
where they are constraints rather than decisions -- OCCT's boolean robustness on
thin-walled geometry, and model quality under BYOK -- are stated where they bear on a
choice. The high-order SEM track is out of scope, and hand-emitted all-hex meshes have no
design yet.

---

## Section 0 — Why this is not the stack that came out

On 2026-09-07, `b6ac498` deleted `openreynolds/geometry/`: a sketch DSL, a compiler, a
measurement kernel, a claims engine, a linter, a fitness table, a preview renderer and a
three-shape library, behind a ninth `geometry` tool. 63,943 lines out, 2,075 in. It is
close enough to what this plan proposes that the difference has to be stated first.

**What failed, from its author.** Not authoring. Two things:

*Context.* The chat-app version of the same model did the job better -- "it didnt have all
the CFD context and irrelevant tool context. just focused on the CAD task and did it well."

*An ethos that preferred model reasoning to computation.* "the principal rule earlier was
llms are faster then solving -- that led to significant overthinking and fucking up within
the CAD."

The symptom: "you ask it to make a tesla valve / it gives circles connected by a line /
making edits was hell." It was not abandoned cheaply -- "I burned almost a billion tokens
trying to fix it. any fix made 10 more problems."

**What this plan does differently.**

- No house spec language. The model writes build123d directly, a real library it saw in
  training, with no facade in front of it (#13). `sketch.py` was the facade, and "the chat
  versions were much better at the job" is the evidence against repeating it.
- One narrowly-briefed desk, not a flat namespace spanning geometry, meshing and case
  setup (#5). The flat namespace is the diagnosed cause restated as a design.
- No gate DAG. One finish check and named instruments (#4).

**What it keeps.** The lesson the deleted stack paid for and the CHANGELOG records:
*render and measure, never assert.*

**Still unverified.** Whether the old stack was measured to fail or replaced on judgement
-- no comparison eval, no retrospective, and its `DESIGN.md` and `qa-runs/` were never
committed. That is a pattern, not an accident; see §7.

---

## Section 1 — Scope and non-goals

v1 is **a replacement for `openreynolds/mesher/`**: one desk that does CAD and meshing as
one job, and can be handed a customer STEP (#5). Not a geometry stage feeding a separate
pipeline -- geometry and meshing are already one desk, and the STL-to-case half the
original plan wrote its contract against does not exist (§4).

It ships inside the existing agent and CLI, so we own the loop and the stopping criterion.
Model access is BYOK, and already is: nine presets across two API dialects.

### Two capabilities

**STEP prep.** Ingest customer STEP/IGES, repair, defeature, extract the fluid domain by
boolean subtraction with capping, tag boundary faces using geometric selectors re-derived
on each run, and export per-patch STL. Imports are multi-solid from the outset -- selecting
among solids, booleans between them, and interference detection are in scope (#14).

**Authoring.** Full parametric part design in build123d, the script as the source of truth
and the only thing edited. This is not new ground: the current desk already writes its
geometry as a script and is gated on leaving it behind. What is new is doing it against
imported geometry.

Both capabilities stay, because what failed before was the spec language and the context,
not authoring (#1).

### Non-goals

**The agent does not change an import's design.** Prep operations that remove or cut
geometry are expected; redesigning the part is not. This is a distinction of intent, not
of operation type, so it cannot be enforced by the type system and lives in the desk's
brief -- **not** in the frozen system prompt, which `tests/test_prompt.py` holds free of
mandated workflow (#4).

**No reverse-engineering an import into a parametric spec.** A research problem, not an
engineering one. Point2CAD recovers a fitted B-rep from a segmented point cloud, which is
adjacent evidence rather than direct -- the sentence should say so, or cite something that
addresses feature-program recovery (#15).

### Where v1 is exposed

Prep has an unambiguous verifier: the geometry meshes or it does not. Authoring does not.
A cold plate can pass every check and still have the channel in the wrong place -- the
J-Geo ~ 0.35 against J-Sem ~ 0.8 gap across all eleven models in P3D-Bench, with no
automated check behind it. That gap is verified and the numbers hold. Authoring is the
part of v1 that cannot be checked into correctness; §5 says what is done about it.

---

## Section 2 — Architecture and boundaries

### One desk, replacing the current one

The desk gets a request in words, optionally a path to a CAD file, and a case directory.
It returns a mesh, the script that rebuilds it, a picture and the measurements. It is a
**replacement** for `mesher/`, not an extension: the brief and the finish check are
written for CAD-and-mesh as one job rather than grown from a mesh brief with CAD bolted on
(#5). That is also how the previous stack was handled -- "gone, not extended".

Because CAD and meshing live in one desk, there is no cross-stage handoff to specify and
no failure-classification protocol to design. The original plan spent a subsection
accepting a round trip between two stages; merged, that channel is internal and free.

**What the replacement must inherit**, because each was paid for in real runs: any PNG a
command writes comes back attached; the finish token is a request to be checked, never a
result; a budget-exhausted run is still checked, "because an unexamined mesh is the failure
this desk exists to end"; unreachable is not failure (`83fb3a9`); the user's verbatim words
outrank the caller's paraphrase, and a mid-run remark reaches the desk while it works;
patch names come from the surfaces as they are created, never from a bounding box; metres,
always; coarse first; say what you did not check. The full list is in (#5).

### Kernel

OCCT, and the argument is narrower than a general preference. It is effectively the only
usable open-source STEP reader, so the dependency is fixed by ingest alone. Given it is
present, the question is how far downstream the B-rep survives, and two prep features
settle it. Fillet removal is a face-level operation -- identify the cylindrical and
toroidal faces, delete, extend and reintersect the neighbours -- which OCCT's defeaturing
API does directly; on a tessellated model it would first require re-segmenting the mesh
into primitives, an open research problem. Face tagging has the same shape: B-rep faces are
natural named units, whereas on a mesh they must be recovered by normal and connectivity
heuristics.

So B-rep is retained to export, and tessellation happens only at per-patch STL write.

**One probe owed before this argument is relied on** (#13): confirm defeaturing is
reachable on the image -- `BRepAlgoAPI_Defeaturing` via OCP if build123d does not surface
it. Known present in `gmsh.model.occ`: `fillet`, `chamfer`, `cut`, `fuse`, `fragment`,
`removeAllDuplicates`, and `healShapes`, which is the repair primitive §1 needs.

**Where it runs — deferred** (#2). The loop is not in the workspace container: it holds a
backend that speaks to the instance, and `tests/test_negative_obligation.py` forbids
anything above that protocol from knowing the transport exists. Three architectures are in
play -- kernel in the loop's process, a long-lived kernel process on the instance, or a
fresh script per call -- and the choice is taken later, against something running. v1 is
developed against `LocalBackend` first.

*The guardrail that keeps it deferrable:* even locally, geometry executes through the
backend, never by importing a CAD kernel into the loop's process. On one machine those two
are indistinguishable, which is exactly how the deleted stack acquired an in-process kernel
and the X/GL layer that went with it.

**Held in reserve, not built.** The fluid-domain boolean is a large subtraction against
thin-walled fin geometry, OCCT's known weak case. If that becomes a real failure source,
that one operation can run on tessellated input via Manifold (Apache-2.0), which propagates
face properties so tagging survives the boolean. Nothing here should foreclose it.

### Licensing

The repository is MIT. We distribute no GPL binaries through any channel: the package ships
via PyPI and its dependencies are distributed by that index, not by us. gmsh is not even a
declared dependency -- it appears nowhere in `pyproject.toml`, and lives only on the
workspace image. The hosted platform is SaaS and plain GPL has no network-use clause.

| Dependency | License | Notes |
|---|---|---|
| build123d | Apache-2.0 | Pure Python, on the image, on OCP. |
| OCP / `cadquery-ocp` | Apache-2.0 wrapper; bundles compiled OCCT | The Apache tag covers the wrapper only; it does not relicense the bundled OCCT. |
| OCCT (via OCP) | **LGPL-2.1-only WITH `OCCT-exception-1.0`** | No "or later" is granted. The exception permits distributing object code of a work that uses the library under terms of our choice, given prominent notice. |
| gmsh | GPL-2.0-or-later **with a linking exception** | Used **in-process** (`import gmsh`), as an undeclared, image-only dependency. The exception lets gmsh link OCC/Netgen/METIS/ParaView; it does not let a third party link gmsh into non-GPL code. Not distributing it is what makes this fine. |
| Manifold | Apache-2.0 | Not a dependency unless the fallback above is built. |

An earlier draft said gmsh was "invoked as subprocesses, keeping the GPL chain outside our
code. The only thing that would break this is pulling a GPL library in-process later." That
is false -- it is imported in-process, deliberately, with a test blessing it -- and a plan
defending a correct conclusion with a premise the first `grep` refutes will have the
question re-opened by whoever checks (#7).

**The edge.** Shipping anything to the client -- a desktop build, a downloadable bundle,
wasm -- is distribution again, and so is handing out the workspace image. At that point the
LGPL-2.1 notice, the Open CASCADE attribution and a written offer for source return, and
gmsh's GPL becomes live rather than dormant.

---

## Section 3 — Geometry layer

### Base library

**build123d in algebra mode, written directly.** Not CadQuery, not builder mode, and not
behind a facade. The same OCCT kernel sits under all of them, so this is a choice about the
surface the model writes against.

The reason is locality of reference. In algebra mode every operation names its operands
(`y = x + sphere`), so a line reads without replaying what came before, and there is no
hidden state for the model to track. Builder mode consumes pending faces and edges from
earlier lines and inherits locations from enclosing contexts, so `extrude(amount=5)` cannot
be understood in isolation -- the same failure class as CadQuery's fluent chaining, and the
source of the Undefined-Reference and Parameter classes in P3D-Bench.

Note that "stateless", build123d's own word, is inaccurate. `x` is a binding that persists,
and boolean order matters irreducibly: difference does not commute and mixed compositions
do not associate. Every CAD language encodes that ordering somewhere -- algebra mode in
named bindings, OpenSCAD in tree nesting, builder mode in accumulated context. The axis
worth designing around is **explicit named state versus global ambient state**, not
imperative versus declarative.

Two further advantages are specific to a harness rather than a human. Named bindings are
inspection points: any binding can be measured, tessellated or rendered on demand. And they
make repair local -- a failed boolean names its operands, so the model can be told which
line to re-emit, where a builder-context failure surfaces as a traceback from inside a
context manager with nothing named.

**No facade** (#13). The original lean was a facade for prep and direct algebra mode for
authoring. Rejected on §0: a house surface over the library is what `sketch.py` was, and it
produced "circles connected by a line". What replaces the facade's benefits is an
**introspected exact-signature API reference** generated from the library and kept in
context -- cheap, worth scheduling early, and cited from a vendor blog with n=1, so worth
doing and not worth quoting as evidence (#15).

### State: the script is the artifact

Cells concatenated **are** a script. The accepted-cell log is not a record kept beside a
deliverable; it is the deliverable, runnable top to bottom (#3). Nothing maintains two
copies, and the question of whether the desk also owes an `Allmesh`-style rebuild script
does not arise -- it already has one, which is what the current desk's finish check
demands.

**The append invariant is not "the cell succeeded".** A cell can run correctly in a live
session and fail in sequence: it referenced a binding from a cell that was never accepted,
or it is not idempotent under replay. The invariant is **the concatenation still runs**.

So cell authoring stops being good practice and becomes the correctness condition:
parameters as named constants rather than literals at their use sites, no dependence on
anything outside an accepted cell, no cell whose effect requires having been run once
already, and the model's reasoning carried with the action it takes.

**The finish check that follows.** Re-run the concatenated script from empty and confirm it
reproduces the accepted geometry. This is the geometry analogue of a rule the desk already
enforces: "`Allmesh` is the thing you actually ran, not a file written at the end to satisfy
somebody [...] if a step in it failed while the mesh came out anyway, fix the step before
you finish."

**Memory, and the cheaper answer to it.** Imported chassis STEP files are large and
tessellated copies accumulate. Rather than explicit release, let an expensive
import-and-repair stage checkpoint its B-rep to a STEP file on the volume and have the rest
of the script load that. It stays inside the script -- a cache the script writes and reads,
not a second source of truth -- and it is the move `toolbox/scratch.py` already makes
against network-filesystem cost.

### Deferred to implementation

**Solver-aided placement.** AIDL (arXiv:2502.09819) offloads spatial reasoning to a
geometric constraint solver, targeting the Parameter class -- 55% of CadQuery failures in
P3D-Bench, right primitives at wrong coordinates -- that no encoding choice fixes. The
citation is verified; one wording fix, that the solver is SolveSpace-based and OCCT is the
boolean and export kernel behind it (#15).

Filed as an experiment in the original. §0 raises its standing: "llms are faster then
solving" is named as a cause of the previous failure, which is local evidence for
preferring a solver where a solver can settle a placement. That reading is terse and should
be confirmed with its author before it carries weight (#0).

---

## Section 4 — Meshing

Not a handoff. The desk meshes.

**Two premise corrections to the original.** There is no STL-to-case half to hand off to:
no `snappyHexMeshDict` writer exists anywhere in the repo -- every hit for
`snappyHexMeshDict`, `refinementSurfaces` and `castellatedMesh` is a read -- and
`case_gen.py` "builds no geometry: that is the `mesh` tool's job", dressing a polyMesh that
already exists. And geometry and meshing are already one desk. The real chain is
**desk -> case_gen**.

### snappyHexMesh is first-class

Tetrahedra pay a large tax in finite-volume discretisation: inherent non-orthogonality,
poorer gradient reconstruction, more cells for the same accuracy, and boundary layers
needing prism extrusion that is fragile on concave walls. A hex-dominant mesh is what the
schemes want (#5a).

The existing brief's preference for gmsh-OCC body-fitted, and the field note about snappy
"rediscovering edges the CAD already knew", are arguments about **time to get a mesh**, not
about the quality of the answer. gmsh-OCC stays available and is right for a quick shape
check and for cases where a body-fitted tet mesh genuinely suffices. It is not the default.

**cfMesh is the other hex-dominant option**, and better where layers matter: snappy's layer
stage inserts prisms then deletes them where a quality metric fails -- on one hull it
reached 88.6% coverage on its first iteration, eroded to 66.7% by its fiftieth, printed
66.7% and exited 0 -- where cfMesh's `cartesianMesh` extrudes a sheet and optimises
afterwards, and covered 100.0% of the same wall.

### What the export must guarantee

**Per-patch files, exhaustive and disjoint.** One STL per named patch, every face of the
exported domain in exactly one patch. Unassigned faces land silently in a default patch and
take whatever boundary condition it carries -- a wrong answer rather than an error --
and double-assigned faces produce overlapping surfaces that snap inconsistently. Note the
existing converter emits **one merged file with no patch concept at all**, so this is new.

**Closure of the union, not of the parts.** Individual patch files are open surfaces; their
union must be a closed manifold with no free edges and matching shared edges.

**Consistent outward normals**, one coordinate frame, no per-file transforms. snappy uses
normal orientation for inside/outside, so an inverted patch turns a solid into a void.

**Units declared explicitly.** STEP is usually millimetres, OpenFOAM works in metres. The
converter already refuses a file that declares no unit with no `--unit` given, "a factor of
1000 on every length in the study" -- that refusal surfaces to the caller, who asks the
user, rather than being guessed at (#12).

**Tessellation against intended cell size.** `clmax` is the primary and mandatory control
and bounds edge length, which is what ties tessellation to `dx_surface`; the epilog's
`clmax ~= 0.5 * dx_surface` is the rule. `MeshSizeFromCurvature` (default 20) is the
edges-per-2pi control and the one that matters on curved walls. Maximum deviation is **not**
exposed and must not be assumed: `StlLinearDeflection` is named in the converter's
docstring explicitly as unused, and a standing decision blocks building on it until two
measurements land -- whether it governs OCC tessellation on our path at all, and how it
interacts with curvature. Those run at the start of the export work (#6).

**`locationInMesh` verified, not chosen.** The point must classify as strictly inside the
fluid domain by a point-in-solid test, with margin from any surface. A classic silent
failure, trivially checkable, and **nothing validates it today** -- the point-in-solid ray
cast exists only as a design note.

**Retained solid regions (CHT) are reachable but unassisted.** The desk's finish check is
region-aware -- it discovers `constant/*/polyMesh`, falls back to `constant/polyMesh`, runs
`checkMesh -region` per region and requires all of them to pass (#17). That removes the one
place the agent's freedom was actually blocked: today `check.py` refuses a multi-region mesh
however correctly it was built, reporting "nothing has been meshed yet" about a mesh that
exists.

Everything else about a CHT case stays the agent's own work with bash, as §1 intends. It
gets no leverage from the toolbox while doing it -- `case_gen.py` is incompressible-only,
and it, `preflight.py`, `first_look.py`, `layer_report.py` and `locate.py` all hardcode the
singular mesh path.

One correction to the original: `occ.fragment` is described there as "already in the stack".
It appears in no code at all, and the converter performs no boolean of any kind. What is in
the stack is gmsh's OCC kernel generally; imprinting conformal interfaces with it is work to
be done, not a capability to be called.

---

## Section 5 — Checks

v1 runs without a human reviewing geometry before the case runs, so the checks have to be
sufficient on their own.

### One finish check, and instruments the desk is told about

**No phase ladder** (#4). The original put gates at phase transitions. That ordering breaks
in six places, two of them inside the plan itself: authoring drops the first phase; the
mesh-failure return is a back edge; defeaturing decisions need a cell size that arrives from
the mesh side; repair recurs mid-prep because the fluid boolean fails on thin fins;
multi-solid work interleaves tagging and repair across regions; and `occ.fragment` for CHT
interfaces can only run after the region plan exists.

What survives is the real argument -- cheap checks before expensive operations -- as advice
in the brief rather than a ladder the desk is held to. This also costs no test changes:
`mesher/check.py` already runs a hard gate and passes the free-will contract's three tests,
because a desk's own finish check is not the harness enforcing a workflow. "This desk is
allowed to be told what to do."

**Instruments must be named in the brief, by name and by what they answer.** An instrument
nobody is told about is unused: `geometry_view.py` was used **zero times across every
persona run**, while `read_file` on a PNG was used constantly.

### The finish check

Floor, inherited from the current desk: mesh present and non-empty, `checkMesh` passes, at
least two patches carrying names somebody chose, a rebuild script present, a picture drawn,
scale matching the request's stated dimensions, and unreachable reported as unreachable
rather than as a verdict.

Added for CAD: closure of the exported union with no free edges, manifoldness, consistent
outward normals, exhaustive and disjoint patch coverage, self-intersection, a validated
`locationInMesh`, and the concatenated script re-running clean from empty (#3).

Deliberately **not** a cheap coarse mesh as a proxy for validity -- a coarse mesh can ignore
small features entirely, so it would pass on exactly the cases where defeaturing was
supposed to have resolved something. False confidence is worse than no test.

### Reporting

Failures are reported in the register `preflight.py`, `locate.py` and `cfmesh.py` already
share: `Finding(check, status, measured, meaning, repair)` with statuses
`fail/warn/ok/skipped` (#11). This inherits thresholds rather than reinventing them --
`NON_ORTHO_WARN/FAIL = 70/85`, `SKEWNESS_WARN/FAIL = 4/10`, `ASPECT_WARN = 1000`,
`MILLIMETRE_SUSPICION = 100.0`, `SCALE_FACTOR = 100.0` -- and is an upgrade on the current
desk, which emits prose with no status, no measured value and no repair.

Not adopted: the GEO-01..05 / M-01..11 codes from `notes/openfoam-agent-architecture.md`.
Real and well-formed, but from a document `design.md` demotes to optional capabilities, and
worth their indirection only alongside the attempt-graph retrieval they were designed for,
which does not exist.

`preflight.surface_topology()` already supplies open edges, non-manifold edges,
same-direction (inverted-normal) edges and degenerate triangles, after welding vertices --
"without the weld every edge looks open". Genuinely missing: self-intersection,
`surfaceCheck` invocation, and `locationInMesh` validation.

**One definition worth keeping**, because nothing in the repo has it: local width is well
posed as twice the inscribed-sphere radius, a routine SDF computation on the tessellation
needing no face naming; the same field on the solid gives minimum wall thickness. Width
*along* the flow needs a centreline and is left alone.

### When the desk cannot finish

It reports up and asks no one (#10). It returns `ok`, its reasons and its `stopped` state
to the caller; the main agent -- conversational, and already permitted to ask -- decides
whether to retry with a changed request or put the question to the user. Nothing new is
built, the desk never blocks on a human mid-run, and §1's autonomy holds.

The original called this "the existing goal-directed iteration mechanism". No such name
exists in the code: the mechanism is `check.verify()` -> `Check.as_refusal()` -> the desk's
continue-loop, under the rule "a failed check is handed back as work rather than reported as
success". The ask-the-user half did not exist and is not being built (#9, #10).

### Acceptance

**The eight benchmark prompts the current desk was accepted on are v1's final gate** (#8),
run against the replacement. Running them also settles the experiment `5b588c3` opened and
left open -- whether "a slow non-deterministic natural-language sub-agent is a worse
interface than the bash the caller already has" -- so the replacement carries the
`OPENREYNOLDS_MESH_TOOL=0` toggle that makes the comparison measurable.

**The gate has a prerequisite that is not satisfied:** the eight prompts are not in the
repository, and neither are the T-numbered acceptance runs. See §7.

**No golden-file regression suite**, and that is deliberate: CAD preparation is not
deterministic -- a Tesla valve has many valid representations -- so golden-file testing is
impossible. The original called this "no agent-level eval suite", which is wrong: a live
persona harness exists (`scripts/user_test.py`, `scripts/personas.py`), `a4-acceptance.md`
is a written acceptance evaluation, and the desk's own finish check is a continuous
invariant grader that costs nothing because it is the gate rather than a second suite.

### Reviewer agent — deferred

A reviewer inspecting renders and measurements would target the residual failure: geometry
that passes every numeric check and is visibly wrong. There is no deterministic check for
whether a design is the right design.

Against it: latency, and under BYOK it is the customer's model reviewing its own work --
the self-confirmation problem CADSmith addressed by making the judge deliberately stronger
than the coder. A reviewer that mostly returns "looks fine" is worse than none.

**Closing condition:** a one-off manual pass over authoring output the checks accepted.
Outputs that pass everything and are visibly wrong define the reviewer's job and give its
false-negative rate; if there are none, it is not needed. Scheduled after everything else.

---

## Section 6 — Design iteration loop

The design -> simulate -> modify loop comes cheaply, because the desk is already invoked as
a tool and can be re-invoked with modifications derived from simulation results. The
subagent facility the original doubted exists -- two of them do, and the replacement takes
one of their slots (#9).

**State across invocations is the script.** A live process is process state rather than
conversation state and does not survive a dormant period while the solver runs; the
concatenated script does, and re-running it restores the geometry. Nothing new is built for
the loop -- it is the same artifact reproducibility and crash recovery already require (#3).

**The read-only constraint shapes what a modification can be.** Imported geometry cannot be
redesigned, so a simulation-driven modification to an imported part means re-running prep
with different parameters. Genuine design modification applies to authored parts, where it
means re-running the script with changed named constants. This makes §3's cell authoring
requirements load-bearing for the loop rather than merely good practice.

---

## Section 7 — What is actually new

Most of what the original plan described exists. Gathered from the decisions, the work that
does not:

- an explicit `geometry` path parameter on the tool, refused early when unreadable (#12)
- STEP/IGES ingest, repair via `healShapes`, defeaturing, multi-solid selection (#13, #14)
- face tagging, and per-patch STL export that is exhaustive and disjoint (#5a)
- a `snappyHexMeshDict` writer -- nothing in the repo writes one (#5a)
- `locationInMesh` validation by point-in-solid test (#5a)
- a self-intersection check (#11)
- the finish check rewritten for CAD-and-mesh, reporting in the Finding register (#4, #11)
- a brief written for the whole job, inheriting §2's list

### Prerequisites and owed measurements

- **Recover the eight benchmark prompts and the T-numbered acceptance runs, and commit
  them** (#8). Without them the final gate is not an executable instruction. This is the
  third time evidence cited in a commit message lives outside the tree, after the deleted
  stack's `DESIGN.md` and `qa-runs/`. A gate whose test set exists only in a chat history
  is not a gate.
- Source a real multi-solid STEP and commit it as a fixture (#14). There is no STEP or IGES
  anywhere in the repo or in any of the 42 recorded studies.
- Re-run the unit-declaration behaviour on the pinned gmsh 4.15.2 / OCC 7.8.1 image; the
  original probe ran on 4.12.1 / OCC 7.6.3 (#14).
- Confirm defeaturing is reachable from build123d or OCP on the image (#13).
- The two deflection measurements, at the start of the export work (#6).
- Confirm the reading of "llms are faster then solving" with its author (#0).

### Costs of the replacement, priced

Recorded in full at (#18); the three most forgettable:

- **The frozen system prompt has 53 characters of headroom.** It is 5,947 against a 6,000
  cap, and the `mesh` bullet that must be rewritten to mention a CAD file is 297. Editing
  it costs a one-time global prefix-cache miss.
- **Capture would silently drop the new artifacts.** A rebuild script not named literally
  `build.py` falls through every branch of `casebundle.classify()` and does not travel at
  all; per-patch STLs under `constant/triSurface/` land in the tier that always travels and
  is assumed small. And the mirror drops `.step`/`.stl` before capture can see them, since
  `mirror.KEEP_SUFFIXES` has no CAD suffixes and `casebundle` reads from the mirror.
- **`mesh_look.py` is a three-way contract** between the desk's check, `case_gen.py` and
  the hosted UI's Mesh panel, which lives in a separate repo. Its `--json` keys should not
  be touched.

Four test files are directly coupled -- `test_mesher.py`, `test_mesher_check.py`,
`test_toolbox_mesh_look.py`, and the mesh-specific cases in `test_tools.py`; collect them
for the current figure rather than trusting one written here. Adding a `geometry`
property to the tool is safe; *renaming* the tool breaks an assertion that the eight names
are exactly the sorted eight. Config keys have no back-compat mechanism, so renaming
`OPENREYNOLDS_MESHER_*` silently breaks existing environments.

### One claim in §5 that `main` does not support

"The existing corpus of OpenFOAM cases stays what it is -- reference material the agent
consults at runtime" is not true on `main` (#19): `corpus.py` and `search.py` are
documented in the toolbox index in detail but exist only on the unmerged `pr22-corpus`
branch. Either merge it or drop the claim. The toolbox index test checks only that every
script is named in the index, never that every name is a script, which is why it slipped.
