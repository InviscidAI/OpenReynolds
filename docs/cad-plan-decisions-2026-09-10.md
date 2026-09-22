# CAD agent build plan — review decisions, 2026-09-10

Decisions taken while reviewing `cadagentbuildplan.md` (the geometry half of the
agent) against the tree at `376ec4c`. Each entry: what the plan claims, what the code
says, the decision, and where it differs from how the plan filed it.

The review that produced this queue found the plan's research grounding sound (all
five external citations verified) and its description of *this* codebase substantially
out of date: it is written against the geometry stack deleted in `b6ac498` on
2026-09-07 (63,943 lines out, 2,075 in).

Nothing here has been implemented.

---

## Queue


*(Entry 0 records the diagnosis the queue rests on.)*
| # | Decision | Status |
|---|---|---|
| 1 | Scope: both capabilities stay; rebuttal is context, not authoring | **taken** |
| 2 | Kernel location: deferred; develop on the local backend first | **deferred** |
| 3 | The accepted-cell log is the script; no second artifact | **taken** |
| 4 | One finish check + named instruments; no phase ladder | **taken** |
| 5 | One CAD+mesh desk, replacing `mesher/`; no handoff to design | **taken** |
| 5a | Amendment: snappyHexMesh is first-class, not gmsh-OCC tets | **taken** |
| 6 | `clmax` primary; C2 measurements at export-work start | **taken** |
| 7 | Licensing conclusion stands; reasoning and three rows corrected | **taken** |
| 8 | Eight benchmark prompts as final gate; prompts must be recovered | **taken** |
| 9 | §6 corrected; base-class question left to build time | **taken** |
| 10 | Desk reports up; the main agent owns asking | **taken** |
| 11 | Report in preflight's Finding register; inherit its thresholds | **taken** |
| 12 | Explicit `geometry` path parameter; no discovery fallback | **taken** |
| 13 | build123d algebra mode, written directly; no facade | **taken** |
| 14 | Multi-solid supported from the outset; no gating probe | **taken** |
| 15 | Citation corrections (four wording fixes; all five real) | **taken** |
| 16 | `README.md:250` corrected when the replacement lands | **deferred** |
| 17 | Finish check accepts a multi-region mesh | **taken** |
| 18 | Blast radius of the replacement, costed | **taken** |
| 18a | Capture, tiering, the rename, and the prompt bullet | **taken** |
| 19 | Corpus discrepancy handed off to issue #27 | **filed** |
| 20 | Main red since 2026-09-07; handed off to issue #28 | **filed** |

---
## 0. The diagnosis behind `b6ac498`, recorded because the repo does not hold it

The geometry segment was deleted without a retrospective; `docs/` never mentions it,
and the `DESIGN.md` and `qa-runs/` its modules cite were never committed. The reason it
went was supplied by the segment's author in conversation on 2026-09-10, and several
decisions below rest on it, so it is recorded here rather than left in a chat log.

**The symptom.** "you ask it to make a tesla valve / it gives circles connected by a
line / making edits was hell."

**Why replaced rather than repaired.** "I burned almost a billion tokens trying to fix
it -- no, way more. any fix made 10 more problems."

**The structural cause, in the author's words.** Two things, and neither is authoring:

1. *Context pollution.* The chat-app version of the same model did the job better:
   "bc I think the env was faster, it was less bogged down by over thinking and there
   was a fast rate of 'oh I fucked up doing this, lemme fix that'. **it didnt have all
   the CFD context and irrelevant tool context. just focused on the CAD task and did it
   well.**"

2. *An ethos that preferred model reasoning to computation.* "the reynolds v2 has the
   ethos of check twice slice once, bc the principal rule earlier was **llms are faster
   then solving** -- that led to significant overthinking and fucking up within the
   CAD. also lossy context."

**What this licenses, and what it forbids.** It does **not** say authoring cannot be
done; it says a house spec language inside a general CFD agent's context could not do
it, while a focused model with a clean context and a real library could. So:

- a narrow, separately-briefed CAD agent with its own context is the *indicated* shape,
  not merely a permitted one (bears on #4, #9);
- a shared flat tool namespace across geometry, meshing and case setup is the diagnosed
  failure restated as a design (bears on #2);
- a house facade over the library repeats the DSL mistake; "the chat versions were much
  better" is evidence for writing the real library directly (bears on #13);
- where a solver can settle a placement, preferring the model to reason it out is the
  named error (bears on the plan's §3 "solver-aided placement", filed there as an
  experiment; this is local evidence that it is more than one).

**Caveat on the last point.** "llms are faster then solving" is terse and is being read
here as *a constraint solver beats model spatial reasoning*. That reading should be
confirmed with its author before it carries weight in #13.

---

## 1. Scope: both capabilities stay; the rebuttal is context, not authoring

**What the plan claims.** Two capabilities: STEP prep, and from-scratch parametric
authoring (§1). The document does not mention the stack removed three days earlier.

**What the code says.** `b6ac498` (2026-09-07) deleted `openreynolds/geometry/` --
12,400 lines behind a ninth `geometry` tool: a sketch DSL, a compiler, a measurement
kernel, a claims engine judging built shapes against claims written first, a linter, a
fitness table, a preview renderer, a three-shape library. 63,943 lines out, 2,075 in.
Authoring now lives in `mesher/`, which writes `build.py` in build123d or gmsh and is
gated on leaving it behind, editable.

**Decision.** Both capabilities stay. The plan gains a short section up front stating
what was removed and why, drawn from #0, and the rebuttal it makes is:

- the old stack failed as a **house spec language inside a polluted context**, and this
  plan proposes neither -- build123d is a real library the model saw in training, and
  the stage is separately briefed;
- the two features it shares with the old stack are the two the deletion singled out --
  an in-process kernel and a curated surface. Both are still open, queued at #2 and
  #13. This rebuttal is only as good as those two settle: if either lands where the old
  stack stood, the section above has to be rewritten rather than merely cited;
- what it keeps is the lesson that paid: render and measure, never assert.

**Reversal of an earlier decision.** This entry first read "STEP prep only, authoring
cut to a non-goal", taken on the reviewer's recommendation before #0 was available.
That recommendation was wrong: it treated authoring as the thing that failed, when the
author's diagnosis names context and the spec language. Recorded rather than
overwritten, because the earlier reasoning is the one a reader would otherwise repeat.

**Still true.** STEP ingest overlaps the deleted stack nowhere -- it authored 2D
passages from a DSL and never read foreign CAD. That half needs no defence at all.

---
## 2. Kernel location deferred; build against the local backend first

**What the plan claims.** §2: "One process, one loop [...] the geometry kernel lives in
the same process as the loop driving it", with geometry, meshing and case setup in one
flat tool namespace.

**What the code says.** The loop is not in the workspace container. `cli.py` builds the
`Loop` in the local process and hands it a `HostedBackend` + `FoamdClient` speaking HTTP
to the instance; `backend/base.py` is explicit that "nothing above this interface may
know whether it is talking to a container over the network or to a local OpenFOAM
install", and `tests/test_negative_obligation.py` enforces it. So "in process" means the
loop's process -- a laptop or a SaaS server -- and never the machine the geometry,
the mesh and the case live on.

**An internal contradiction in the plan, recorded because it has to be resolved either
way.** §2 puts the kernel in the loop's process; §3 puts geometry state in "a persistent
IPython kernel", which is by construction a separate process the loop talks to. Both
cannot hold. Three architectures are actually in play:

| | Kernel lives | State between calls | New dependency |
|---|---|---|---|
| A | the loop's process (laptop / server) | in memory | OCC + build123d, locally |
| B | a long-lived process on the instance | in memory | `ipykernel` in the image |
| C | a fresh script per call, on the instance | files on `/work` | none |

**Decision.** **Deferred.** v1 is developed fully against `LocalBackend` first -- the
same posture `backend/local.py` already describes for working on the product ("`subprocess`
where the hosted backend has HTTP, a directory where it has a Volume [...] Same protocol,
same semantics, no network and no bill"). The A/B/C choice is taken later, against
something that runs.

This is deferrable precisely because the Backend protocol exists: work that goes through
it stays portable to hosted, and the choice above stays open at no cost.

**The trap, and it is the historical one.** On one machine, "in the loop's process" and
"on the workspace machine" are indistinguishable -- so developing locally will silently
become architecture A unless that is guarded. That is how the deleted stack acquired an
in-process kernel and, with it, "the gmsh + matplotlib + X/GL layer the hosted runner
carried so the old desk could draw in its own process" (`b6ac498`).

**The guardrail this decision carries.** Even locally, geometry executes through
`Backend.exec`, never by importing a CAD kernel into the loop's process. If that holds,
A/B/C is a late and cheap decision; if it slips, the deferral has quietly chosen A.

**Not deferred.** The flat-namespace half of §2 is a separate question, and entry 0
bears on it directly; it is queued at #4 with the gates.

---
## 3. The accepted-cell log is the script; there is no second artifact

**What the plan claims.** §3: geometry state in a live IPython kernel, with "an
append-only log of accepted cells" beside it covering three needs -- the reproducible
artifact, the parametric spec for authored parts, and crash recovery.

**What was decided, and it is simpler than the plan or the review had it.** Cells
concatenated *are* a script. The log is not a record of the session kept alongside a
deliverable; it is the deliverable, runnable top to bottom. Nothing maintains two copies
of the same thing, and the question "does the geometry stage also owe an `Allmesh`-style
rebuild script?" does not arise -- it already has one.

The review had framed this as a choice between a cell log and a `build.py`. That was a
false alternative and is recorded as such.

**What it sharpens.** The plan's append rule is "a cell is appended only after it
succeeds and any invariant we care about holds". Succeeding is not sufficient: a cell can
run correctly in a live session and still fail in sequence -- it referenced a binding from
a cell that was never accepted, or it is not idempotent under replay. The invariant that
actually matters is **the concatenation still runs**.

So §3's cell-authoring requirements stop being good practice and become the correctness
condition for the artifact: named constants rather than literals at their use sites, no
dependence on anything not in an accepted cell, and no cell whose effect depends on
having been run once already.

**The finish gate that follows from it**, and it is the geometry analogue of a rule the
mesh desk already enforces (`mesher/brief.py`): "`Allmesh` is the thing you actually ran,
not a file written at the end to satisfy somebody: build the mesh *through* it, and if a
step in it failed while the mesh came out anyway [...] fix the step before you finish."

For geometry: the finish check re-runs the concatenated script from empty and confirms it
reproduces the geometry that was accepted. A script that does not reproduce it is the
failure, whatever the session showed.

**Does not prejudge #2.** Whether those cells execute in a long-lived kernel and are
logged as they are accepted, or are appended and re-run per call, is the deferred
architecture question. The artifact is one file either way.

**Carried forward as an optimisation, not an artifact.** §3's memory concern ("imported
chassis STEP files are large and tessellated copies accumulate") has a cheaper answer
than explicit release under replay: let an expensive import-and-repair stage checkpoint
its B-rep to a STEP file on the volume, and have the later part of the script load that
instead of re-importing. It stays *inside* the script -- a cache the script writes and
reads, not a second source of truth -- and it is the same move `toolbox/scratch.py`
already makes against network-filesystem cost.

---
## 4. One finish check and named instruments; the phase ladder is advice, not structure

**What the plan claims.** §5 puts gates at phase transitions -- prep running
import/repair -> B-rep prep -> tessellation/export -- "because each gate checks what the
next phase requires". §1 puts the read-only constraint in "the tool contracts and the
prompt".

**What the code says.** `docs/design.md` §1 forbids the harness enforcing "any ordering
of actions, phases, or 'check X before Y'" and injecting "step-by-step instructions,
checklists, or mandated workflows", and states "there is no gate DAG, no state machine".
Three tests hold it; `tests/test_prompt.py` greps the system prompt for `phase \d`,
`workflow`, `required to`, `you must`.

**The collision is narrower than it looks.** `mesher/` already runs a hard gate and
passes all three, because the gate is a desk's own finish check rather than anything in
the main loop or the frozen prompt. `mesher/__init__.py`: "This desk is allowed to be
told what to do." So gates inside a geometry desk cost no test changes, and §1's
constraint belongs in the desk's brief, never in `prompt.py`.

**Decision.** One non-negotiable finish check, plus instruments named in the brief and
run whenever the desk judges. No enforced ordering.

**Why the ladder does not hold -- it breaks in six places, two of them in the plan.**

1. Authoring drops the first phase; §5 says so. Two structures before anything else.
2. §5's classified mesh failures return from after export into prep. That back edge
   makes it a loop with a preferred order, not a ladder.
3. Which fillets to remove depends on the intended cell size, and §4 has deviation chosen
   against cell size -- information that arrives from the mesh side, after tessellation.
4. Repair recurs mid-prep: §2 concedes the fluid-domain boolean on thin-walled fins is
   OCCT's known weak case, so invalidity surfaces during the subtraction, not at import.
5. On a multi-solid chassis, interference detection surfaces repair in one region while
   another is already tagged; phases assume the whole model advances together.
6. `occ.fragment` for conformal CHT interfaces can only run once the region and
   tessellation plan exists, which is after the phase that would own it.

**What survives from §5.** Its actual argument -- cheap checks before expensive
operations -- is right nearly always, and goes in the brief as advice. What it cannot be
is a ladder the desk is held to.

**The failure this decision has to avoid.** An instrument nobody is told about is an
instrument nobody uses: `docs/found-by-using-it.md` records `geometry_view.py` used
**zero times across every persona run**, while `read_file` on a PNG was used constantly.
So the instruments are named in the brief, by name and by what they answer, the way
`mesher/brief.py` names `mesh_look.py`.

---
## 5. One desk does CAD and mesh, and it replaces `mesher/` rather than extending it

**What the plan claims.** §1: "v1 does not own meshing, case setup, solving, or
postprocessing. It produces geometry and hands off." §4 specifies that handoff as
per-patch STL against "the existing STL -> running case half".

**Two premise corrections.**

*There is no STL -> case half.* No `snappyHexMeshDict` writer exists anywhere: every hit
for `snappyHexMeshDict`, `refinementSurfaces` and `castellatedMesh` is a read
(`cells_estimate.py`, `ladder.py`, `preflight.py`, `study_run.py`). `case_gen.py` "builds
no geometry: that is the `mesh` tool's job" -- it dresses a polyMesh that already exists.
The real chain is mesh desk -> `case_gen`, and nothing in it consumes per-patch STL.

*Geometry and meshing are already one desk.* CHANGELOG: "The mesh desk
(`openreynolds/mesher/`): **geometry and meshing are now one small agent with one tool**."
Its brief already has it writing `build.py` in gmsh or build123d and picking its own
mesher. A separate geometry stage would re-introduce a split that was closed deliberately.

**The plan argues for the merge without noticing.** §5 accepts a round trip -- "a bad
mesh can require a geometry revision, so the boundary is a channel rather than a wall
[...] This couples two stages across a round trip, which is more coupling than a
self-contained gate would need. Accepted" -- and that whole subsection is the price of a
split that does not currently exist. Merged, the channel is internal and costs nothing,
and there is no failure classification protocol to design (#11 shrinks accordingly).

**Decision.** One desk does CAD and mesh, taking words *and optionally a STEP/IGES file*,
and returning a mesh, the script that rebuilds it, and a picture. It is a **replacement
for `openreynolds/mesher/`, not an update to it** -- the brief and the finish check are
written for CAD-and-mesh as one job, rather than grown from a mesh brief with CAD bolted
on. This follows the repo's own precedent, where the previous stack was "gone, not
extended".

**What v1 becomes.** Not "build a geometry stage" but "the desk that ingests CAD" --
STEP/IGES intake, repair, defeature, tagging, a file input, and the checks that are
genuinely missing. Smaller than the plan as filed, and it is close to what is actually
absent.

**The inheritance list -- what a replacement must carry over, because each was paid for.**
A rewrite that re-learns these is a worse desk than the one it replaced.

- Any PNG a command writes comes back attached, so "seeing is not something to remember
  to arrange".
- The finish token is a request to be checked, never a result: `echo DONE` runs the check
  and hands the run back with reasons.
- The finish criteria as a floor: mesh present and non-empty, `checkMesh` passes, at least
  two patches carry names somebody chose (not `patch0`/`defaultFaces`), a rebuild script
  is present, a picture was drawn, and the scale matches the request's stated dimensions.
- Unreachable is not failure. `83fb3a9`: "A mesh nobody could reach is not a mesh that is
  not there" -- the check asks twice, ten seconds apart, and says `unreachable` rather
  than describing a mesh it never saw.
- A run that exhausts its budget is still checked, "because an unexamined mesh is the
  failure this desk exists to end".
- The user's own verbatim words outrank the caller's paraphrase, and a mid-run remark
  reaches the desk while it works.
- Patch names come from the surfaces as they are created, never from a bounding box.
- Metres, always, with the bounds in the look report as the check.
- Coarse first; measure the properties the request actually named; say what you did not
  check.
- Image eviction, and one fenced command per message against a step timeout.

**Two flags this raises.**

*Budgets.* `MAX_STEPS = 30` and `MAX_SECONDS = 900` are sized for meshing alone. STEP
repair on a chassis plus meshing will not reliably fit; the replacement needs its own
numbers, and probably a larger budget when a file is passed.

*An experiment in flight.* `5b588c3` added `OPENREYNOLDS_MESH_TOOL=0` specifically to
settle whether a "slow non-deterministic natural-language sub-agent is a worse interface
than the bash the caller already has", by running the same prompt both ways. That
experiment is unresolved, and replacing the desk changes what it measures. Either settle
it first on the current desk, or carry the toggle into the replacement and state that the
comparison restarts.

---
## 5a. Amendment: snappyHexMesh is the first-class mesher, not gmsh-OCC tets

Decision 5 said the desk would mesh the B-rep body-fitted by default and treat per-patch
STL as a secondary export. That half is **overturned**.

**Why.** Tetrahedra pay a large tax in finite-volume discretisation -- inherent
non-orthogonality, poorer gradient reconstruction, more cells for the same accuracy, and
boundary layers that need prism extrusion which is fragile on concave walls (the field
notes concede this: "A prism stack thicker than the surface cell extrudes into itself on
any concave wall and the mesher runs without end rather than failing"). A hex-dominant
mesh is what the schemes want.

**Where the earlier reasoning went wrong.** `mesher/brief.py` ("gmsh-OCC, body-fitted, no
STL and no snappyHexMesh") and the field note about snappy "rediscovering edges the CAD
already knew" are both arguments about **time to get a mesh**, not about the quality of
the answer it produces. Decision 5 read them as doctrine about accuracy. They are not.

**What this restores.** §4 is substantially right, and more of it survives than the review
first credited:

- per-patch STL, exhaustive and disjoint, is a **core deliverable**, not an optional
  export;
- tessellation deviation against intended cell size is on the **primary** path (see #6);
- `locationInMesh` verified by point-in-solid test becomes **essential**, not nice to have
  -- it is a snappy concept and nothing validates it today;
- consistent outward normals in one frame, and closure of the union rather than of the
  parts, are load-bearing for castellation.

**What it adds to the work list.** The missing `snappyHexMeshDict` writer is no longer
avoidable -- nothing in the repo writes one, and the first-class path needs it.

**cfMesh stays in scope as the other hex-dominant option.** `toolbox/cfmesh.py` records
that snappy's layer stage "inserts prisms and then deletes them where a quality metric
fails -- on one hull it reached 88.6% coverage on its first iteration and eroded to 66.7%
by its fiftieth, then printed 66.7% and exited 0", where cfMesh's `cartesianMesh`
"extrudes a sheet and optimises afterwards, so nothing can un-extrude a face, and it
covered 100.0% of the same wall". Where layers matter, cfMesh is the better first try.

**Unchanged from decision 5.** One desk does CAD and mesh; it replaces `mesher/` rather
than extending it; there is no geometry-to-mesh handoff to design, because both live in
the same desk. gmsh-OCC remains available and is the right tool for a quick shape check
and for cases where a body-fitted tet mesh genuinely suffices -- it is simply not the
default.

---
## 6. `clmax` stays primary; C2's measurements run when the export work starts

**What the plan claims.** §4: "Tessellation deviation chosen against intended cell size,
not a fixed default [...] Angular deflection is the control that matters on curved walls.
(Believed to exist already as maximum deviation and edges-per-2pi.)"

**What exists.** `--clmax` (mandatory; `cad_convert` refuses without it, and its epilog
says `clmax ~= 0.5 * dx_surface`), `--clmin`, `--curvature` -> `Mesh.MeshSizeFromCurvature`
default 20 -- which *is* the edges-per-2pi control and the one that matters on curved
walls -- and `--tolerance`, which is B-rep entity identity, not deviation. Maximum
deviation is **not** exposed: `Mesh.StlLinearDeflection` is named in the docstring
explicitly as unused. The sagitta is computed and *reported* as a consequence of `clmax`,
never used as a control.

**Correction to §4, now.** "Believed to exist as maximum deviation and edges-per-2pi" is
half right: edges-per-2pi exists, maximum deviation does not. And deflection is not the
control that matters on curved walls in the sense §4 means -- per C2, it "bounds how far a
facet departs from the surface -- on a flat face that is satisfied by arbitrarily large
triangles, which is geometrically exact and still a bad input", because snappyHexMesh uses
the triangles for surface refinement and proximity detection and `surfaceFeatureExtract`
works off them. `clmax` bounds edge length, which is what ties tessellation to
`dx_surface`. §4 should name `clmax` as primary and `MeshSizeFromCurvature` as the
curvature control.

**Decision.** `clmax` stays primary and mandatory. C2's two blocked measurements --
whether `Mesh.StlLinearDeflection` governs OCC tessellation on the path we use or only
STL import, and how it interacts with `MeshSizeFromCurvature` -- are **run at the start of
implementation of the STL export work**, not as a prerequisite of the plan and not left as
an open-ended follow-up. The fixture already carries a curved boss to measure against.
Until they land, C2 stands unchanged: "Do not build on it before then -- a knob that
silently does nothing is worse than one that is not there."

**Why it is scheduled there rather than now.** Amendment 5a puts STL export on the primary
path, so the answer matters; but it is a control-surface question that only bites once
something is exporting, and settling it earlier buys nothing that cannot be bought then.

**Not adopted: deriving `clmax` from the cell size.** Tempting, since §4 wants deviation
chosen against intended cell size and the desk will know `dx_surface`. Left alone because
`cad_convert` refuses without `--clmax` deliberately -- one of "two refusals, both times
where a wrong answer looks right" -- and deriving it silently removes that refusal. If it
is ever derived, it must still be printed and attributable.

---
## 7. The licensing conclusion stands; its stated reason does not

**What the plan claims.** §2: "OpenFOAM, snappyHexMesh and gmsh are GPL but are invoked
as subprocesses, keeping the GPL chain outside our code. The only thing that would break
this is pulling a GPL library in-process later."

**What the code says.** There is no gmsh subprocess anywhere. `cad_convert.py:286` and
`:488`, and both files in `toolbox/templates/`, do `import gmsh` -- the ctypes binding
onto `libgmsh`, in the same address space -- and `tests/test_toolbox.py:364` allow-lists
it on purpose ("gmsh joined this set when `pip install gmsh` went into the image: the
module, not only the command, and OCC-enabled"). The only shell-level GPL name is
`gmshToFoam`, an OpenFOAM utility, which is correctly a subprocess.

**Decision.** The conclusion is right and the reasoning is replaced.

*The correct reason:* we distribute no GPL binaries through any channel. The package ships
via PyPI (and npm where relevant) and its dependencies are distributed by those indexes,
not by us -- the same argument §2 already makes for OCCT via `cadquery-ocp`, and stronger
here, because **gmsh is not a declared dependency at all**: it appears nowhere in
`pyproject.toml`, neither in `dependencies` nor in the `toolbox` extra. It is present only
on the workspace image, built and run elsewhere. The hosted platform is SaaS and plain GPL
has no network-use clause, so running it there triggers nothing either.

*The sentence that must go:* "The only thing that would break this is pulling a GPL
library in-process later." That already happened, deliberately and with a test blessing
it. A plan defending a correct conclusion with a premise the first `grep` refutes will
have the whole question re-opened by the first person who checks.

**Table corrections.**

| Row | Was | Is |
|---|---|---|
| OCCT | `LGPL-2.1-or-later` with `OCCT-exception-1.0` | **`LGPL-2.1-only WITH OCCT-exception-1.0`** -- OCCT grants no "or later"; the error runs the wrong way, since "or-later" would imply an upgrade path to LGPL-3 that is not offered |
| Manifold | "to be confirmed if adopted" | **Apache-2.0**, resolvable now |
| gmsh | "Already in the stack; subprocess invocation." | **GPL-2.0-or-later with a linking exception** (Netgen, METIS, OpenCASCADE, ParaView) -- and used **in-process**, as an undeclared, image-only dependency. The linking exception permits *gmsh* to link those libraries; it does not permit a third party to link gmsh into non-GPL code, so it is not what makes this fine. Not distributing it is. |

**The edge, restated.** §2 already names it correctly and it now covers one more artifact:
shipping anything to the client -- a desktop build, a downloadable bundle, wasm in the
browser -- is distribution again. So is handing out the workspace image itself, if that
ever happens. At that point the LGPL-2.1 notice, the Open CASCADE attribution and a
written offer for source all return, and gmsh's GPL becomes a live question rather than a
dormant one.

**Not adopted: changing the posture.** Moving `cad_convert` off the gmsh module onto the
CLI would remove the in-process linkage, but costs exactly what C1 pinned the wheel to
obtain -- entity lists, per-entity bounding boxes, `getMass`, curvature and declared units,
"the intended way to decide the tessellation". Not worth paying for a problem that does
not exist.

---
## 8. The eight benchmark prompts become v1's final acceptance gate

**What the plan claims.** §5: "No agent-level eval suite. A deliberate omission [...] The
agent's end-to-end behaviour is not regression-tested, and that is an accepted gap."

**What the code says.** An agent-level eval already exists and is live: `scripts/user_test.py`
with `scripts/personas.py` drives real sessions from outside using prose-only personas
(`engineer`, `controller`, `shifting`, `novice`), wipes the workspace between them, and
emits a verdict and transcript; `tests/test_user_test.py` unit-tests the harness;
`docs/found-by-using-it.md` is the findings document derived from it, and
`docs/a4-acceptance.md` is a written acceptance evaluation of a full free-form run. It is
not in CI because it needs credentials. Separately, `mesher/check.py` is an invariant-based
grader that runs every session for free, which is the alternative §5 dismisses as too
expensive -- it is cheap precisely because it is the finish gate rather than a second suite.

**Decision.** The eight benchmark prompts the mesh desk was accepted on become the **final
acceptance gate for v1**, run against the replacement desk. §5's claim is corrected to say
what is actually omitted -- golden-file regression, for the reason it already gives, which
is sound -- rather than that no eval exists.

Running the same eight also settles the experiment `5b588c3` opened, which is still open:
whether "a slow non-deterministic natural-language sub-agent is a worse interface than the
bash the caller already has". `OPENREYNOLDS_MESH_TOOL=0` exists so the same prompt can be
run both ways; a replacement desk should carry that toggle (see #5).

**A dependency this gate has, and it is not satisfied today.** The eight prompts are not in
the repository. `5b588c3` cites them -- "eight benchmark prompts went through that tool and
came back with a correct mesh every time" -- and `83fb3a9` cites acceptance runs `T09`,
`T10` and `T06`. Neither set is committed; grepping `T0[0-9]`, `benchmark` and `tesla`
across `scripts/`, `openreynolds/mesher/` and `docs/` finds nothing.

This is the third instance of evidence cited in a commit message living outside the tree,
after the deleted stack's `DESIGN.md` and `qa-runs/`. A gate whose test set exists only in
someone's chat history is not a gate.

**So the gate carries a prerequisite:** recover the eight prompts and the T-numbered
acceptance runs, and commit them -- as fixtures under `scripts/` or `tests/data/`, with
what each one is checking. Until that is done, "run the same eight" is not an executable
instruction.

---
## 9. §6's subagent claim is corrected; the base-class question is left to build time

**What the plan claims.** §6: "There is likely no dedicated subagent mechanism to build
on, so 'subagent-shaped' describes the call structure rather than an existing facility."

**What the code says.** Two subagents exist and one of them is exactly the shape §6
describes wanting.

*`openreynolds/mesher/`* -- its own `while True` loop (`agent.py:150-248`), its own message
list built locally rather than the main thread's, its own system prompt via `brief.py`,
its own budgets (`MAX_STEPS = 30`, `MAX_SECONDS = 900`), its own verified finish, and
`tools=[]` on the provider call because its only tool is a fenced bash block parsed out of
prose. Invoked from the main tool namespace as `mesh` (`tools.py:279-321`, handler
`_mesh`), and *conditionally present*: `tools_for()` strips it when `ctx.mesher is None`,
so the tool is absent rather than present-and-refusing.

*`openreynolds/desk.py`* -- the `Concierge`, read-only, with its own provider.

They share `llm/make_provider`, `Provider.stream` and the `images` module. They share no
base class: the loop body, image eviction (`KEEP_LIVE_IMAGES = 5` against `KEEP_IMAGES = 2`)
and token accounting are written twice, the last bridged by hand through `ctx.on_tokens`.

**Decision.** §6 is corrected to state the above. The base-class question is **left to
build time**, decided by whoever writes the replacement on the evidence of how much
actually duplicates -- not settled here.

Note that decision 5 makes this smaller than it looks: the replacement takes `mesher/`'s
slot rather than adding a third desk, so the choice is between two hand-rolled
implementations and two sharing a base, not three of anything.

**The extension points, recorded so the question can be answered later without re-deriving
them.**

*Reusable as-is:* `make_provider(cfg)` and the provider-agnostic
`stream(model, system, messages, tools, effort, max_tokens, listener)`; the free functions
`parse_action`, `_is_finish`, `_summary`, `_observe`, `_evict`, `_clip`, `_add`, none of
which are mesh-specific except the finish token; the `ToolContext.<desk>` slot plus
`tools_for()`'s conditional-tool trick; `ctx.on_tokens` for token bridging; `_said()` /
`interject` / `remark_message` for reading the transcript and taking a mid-run remark.

*Per-desk configuration already has a precedent:* `mesher_model`, `mesher_effort`,
`mesher_max_steps`, `mesher_max_seconds`, `mesh_tool` in `config.py`, alongside `model` and
`desk_model`. A fourth slot costs nothing.

*Mesh-specific and therefore rewritten:* `MESHER_SYSTEM`, the `MESH_DONE` token,
`check.LOOK` and the whole finish rule set, `_has_mesh()`'s mesher-name list, `NUDGE`,
`MeshResult`'s fields and `mesh_text()`'s rendering, and `_case_name()`'s default.

---
## 10. The desk reports up; the main agent decides whether to ask

**What the plan claims.** §5: "Gate failures feed the existing goal-directed iteration
mechanism rather than a parallel construct built for geometry. A goal that appears
unachievable requires the agent to stop and ask the user before restating it. That
confirmation step is what separates a faulty goal from a hard geometry failure."

**What the code says.** The word `goal` appears zero times in `openreynolds/**.py`, and
`stopping.py` -- the obvious candidate by name -- is about killing solver processes whose
MPI ranks outlived their wrapper. The mechanism §5 means is real but is called something
else: `check.verify()` -> `Check.as_refusal()` -> the desk's continue-loop, whose text is
"The check did not pass, so the run is not finished: {reasons}. Fix it and say done again
when it is right", under the rule "a failed check is handed back as work rather than
reported as success". It is scoped to one desk call.

The second half -- stopping to ask the user -- **does not exist anywhere**. What exists is
one-way: `interject` lets the user speak *in* mid-run and the desk reports remarks back;
`NUDGE_AT_STEP = 12` self-corrects the model rather than asking anyone; budget exhaustion
is reported honestly to the caller ("it ran out of steps before it was finished, so this is
where it got to"). The main agent is only *told* it may ask, in a prompt sentence ("This is
a conversation. Ask the user whenever you want their input"), which is permission, not
plumbing.

**A tension in the plan to resolve while correcting this.** §1 says "It runs fully
autonomously -- no human reviews the geometry before the case runs"; §5 says a goal that
looks unachievable "requires the agent to stop and ask the user". The plan should say which
it means.

**Decision.** The desk reports up and asks no one. It returns `ok`, its reasons, and its
`stopped` state to the caller, exactly as `mesher/` does today; the main agent -- which is
conversational, and already permitted to ask -- decides whether to retry with a changed
request or put the question to the user.

Nothing new is built. §1's autonomy claim survives intact, because the desk never blocks on
a human mid-run, and §5's confirmation step still happens where the conversation already
is.

**What §5 should say instead.** Not "the existing goal-directed iteration mechanism", which
names nothing, but: gate failures are handed back to the desk as work by its own finish
check, and a desk that exhausts its budget returns what it has with what it could not do --
where the caller, not the desk, owns the decision to ask.

---
## 11. The desk reports in preflight's Finding register

**What the plan claims.** §5 defers this: "Concrete geometric checks are defined by
mirroring what the OpenFOAM side already does, which buys consistency across the two
halves rather than inventing a second vocabulary [...] Which failure modes fall on which
side is settled by reading the OpenFOAM side, not specified here."

**Mostly dissolved by decision 5.** With CAD and mesh in one desk there is no cross-stage
return channel to classify for, and §5's "mesh failures come back classified" protocol is
not needed. What remains is narrower: what vocabulary the desk uses to say what is wrong.

**What reading the OpenFOAM side actually turns up.** Two candidates, and the plan's
instinct to mirror rather than invent is right.

*A written taxonomy*, in `toolbox/notes/openfoam-agent-architecture.md`: GEO-01 not
watertight, GEO-02 units/scale, GEO-03 inverted normals, GEO-04 self-intersections, GEO-05
multi-solid / region-name mismatch, against M-01..M-11 on the mesh side. Not adopted: that
document is explicitly demoted by `docs/design.md` -- "None of that control structure
survives, deliberately (§1). What did survive is demoted to optional capabilities" -- and
its codes are only worth their indirection alongside the attempt-graph retrieval they were
designed for, which does not exist.

*The live house register*, in `preflight.py` and reused by `locate.py` and `cfmesh.py`:
`Finding(check, status, measured, meaning, repair)` with statuses `fail/warn/ok/skipped`
and named checks (`geometry`, `patches`, `empty`, `units`, `cells`, `checkmesh`, `probe`,
`courant`, `residuals`, `disk`).

**Decision.** The desk reports in the Finding register. Each failure names what was
measured, what it means, and a repair -- and inherits the thresholds rather than
reinventing them: `NON_ORTHO_WARN/FAIL = 70/85`, `SKEWNESS_WARN/FAIL = 4/10`,
`ASPECT_WARN = 1000`, `MILLIMETRE_SUSPICION = 100.0`, `SCALE_FACTOR = 100.0`,
`TOPOLOGY_TRIANGLE_LIMIT = 400_000`.

This is an upgrade on today: `mesher/check.py` emits prose strings with no status, no
measured value and no repair field.

**What it inherits for free.** `preflight.surface_topology()` already computes open edges
(fail: "the surface is not closed, so snappyHexMesh cannot tell inside from outside and
castellation leaks out through the hole"), non-manifold edges (warn), same-direction
"flipped" edges meaning inverted normals (fail, repair `surfaceOrient`), and degenerate
triangles -- after welding vertices, "without the weld every edge looks open".
`scale_diagnosis()` covers the millimetre-versus-metre ladder. Amendment 5a's list of what
is genuinely missing stands: self-intersection, `surfaceCheck` invocation, and
`locationInMesh` validation.

**One definition worth keeping from §5 as written**, because nothing in the repo has it:
local width is well posed as twice the inscribed-sphere radius, a routine SDF computation
on the tessellation needing no face naming; the same field taken on the solid gives minimum
wall thickness. Width *along* the flow needs a centreline and is left alone.

---
## 12. The tool takes an explicit geometry path

**What the code says.** The `mesh` tool's schema has exactly two properties: `request`
(the shape in words) and `case` (a directory name). There is no way to hand it a file.
A CAD file can reach the volume -- `openreynolds push` uploads one, and it is the only
direction that exists, since "the sync only runs one way [...] the agent's tools reach the
instance, not your disk" -- and the main agent can `bash` and `read_file` against it. But
nothing can point the desk at it. For a plan whose first capability is ingesting customer
STEP, this is the gap that matters most, and it is small.

**Decision.** The replacement tool gains an explicit `geometry` property: an absolute path
under the workspace root to a `.step`/`.stp`/`.iges`/`.igs` file. It is checked for
existence and readability before the run starts, and a missing or unreadable file is
refused immediately rather than discovered on step nine.

**Why not discovery.** Globbing the case directory needs no schema change and is ambiguous
the moment there is more than one file -- and it is the same class of error the desk
already refuses elsewhere: "Never assign a patch by asking where a face sits in the
bounding box -- on any bend or U-turn that labels the wrong end and says nothing." Choosing
a file by where it happens to sit is that mistake with a different subject. A fallback to
discovery was considered and rejected on the grounds that the fallback is the ambiguous
path and would be the one exercised.

**How the two refusals surface.** `cad_convert` refuses in exactly two places, "both times
where a wrong answer looks right": no `--clmax`, and a file that declares no unit with no
`--unit` -- the latter because getting it wrong is "a factor of 1000 on every length in the
study". A customer STEP with no declared unit is a live case, not a corner. Under decision
10 the desk does not guess and does not block: it reports up with the reason, and the main
agent asks the user which unit the file is in.

**Follow-on, not decided here.** Whether the hosted GUI needs an upload affordance of its
own, or whether `openreynolds push` is sufficient for v1.

---
## 13. build123d in algebra mode, written directly; no facade

**Decision.** The model writes **build123d, algebra mode, directly**. No house facade over
it, and no spec language.

**Why build123d.** Every operation names its operands, so there is no hidden state for the
model to track and a line can be read without replaying what came before. §3 makes this
argument well and it is adopted as written -- including its own correction, that
"stateless" is the wrong word for it: `x` is a binding that persists and boolean order
matters irreducibly, so the axis that actually matters is **explicit named state versus
global ambient state**, not imperative versus declarative. Builder mode's pending faces and
inherited locations, and CadQuery's fluent chaining, are the same ambient-state failure
class.

**Why no facade.** §3's lean was "facade for prep and direct algebra mode for authoring".
Rejected on entry 0: the chat-app version of the same model, writing a real library with no
house surface in front of it, beat a 12,400-line stack whose centre was `sketch.py` -- a
house authoring API. "The chat versions were much better at the job." A facade is that
surface again, and its failure mode is already recorded: "you ask it to make a tesla valve
/ it gives circles connected by a line".

What replaces the facade's benefits: an **introspected exact-signature API reference** kept
in context, generated from the library rather than hand-written. §3 already files this as a
cheap win worth scheduling early, and its cited result -- code errors on a full run falling
from 40 to 33 -- is a vendor blog with n=1, so it is worth doing and not worth quoting as
evidence (see #15).

**Kernel availability, corrected.** OCP is present on the image, transitively:
`ENVIRONMENT.md` describes build123d as "parametric CAD in readable Python on the same
OpenCASCADE kernel (via OCP)". So full OCCT bindings are reachable, and an earlier note in
this review suggesting otherwise was wrong.

**The pipeline this settles.** build123d authors and prepares the B-rep; `export_step`
writes it -- "the B-rep gmsh imports with `merge`/`importShapes` and meshes"; and per
amendment 5a the primary meshing path is per-patch STL into snappyHexMesh, with gmsh-OCC
body-fitted and cfMesh available where they fit. Tagging happens on build123d faces, which
is where "name the patches where you create the surfaces" lands for this kernel.

**One probe still owed, cheap, and load-bearing for §2.** §2's central argument for
retaining B-rep to export is that "Fillet removal is a face-level operation [...] that
OCCT's defeaturing API does directly". True of OCCT; unconfirmed for what is reachable
here. Confirm on the image that defeaturing is callable -- `BRepAlgoAPI_Defeaturing` via
OCP if build123d does not surface it -- before §2's argument is relied on. Known present in
`gmsh.model.occ`: `fillet`, `chamfer`, `cut`, `fuse`, `fragment`, `removeAllDuplicates`,
and `healShapes`, which is the repair primitive §1 needs (`fixDegenerated`, `fixSmallEdges`,
`fixSmallFaces`, `sewFaces`).

**Usage evidence, deliberately not leaned on.** build123d appears in zero of 42 recorded
studies -- but so does gmsh, and every one of those studies predates the mesh desk (latest
2026-09-02; the desk landed 09-07). The corpus measures the world before this desk existed
and says nothing about either library's fitness.

---
## 14. Multi-solid is supported from the outset; no gating probe

**What the plan claims.** §1: "Imported files are typically multi-solid (server chassis,
heatsinks), so selecting among solids, booleans between them, and interference detection
are in scope."

**What the record says.** `triage-decisions-2026-09-05.md` §9 established that gmsh reads
STEP and IGES as B-rep, then stated the limit plainly: "Not tested: STEP from a real CAD
system -- assemblies, multiple solids, degenerate faces. The probe's file was written by
OCC itself, the friendly case. A cheap follow-up, and a Docker image was left locally to
run it in."

**Decision.** Build for multi-solid from the start; no probe gates the plan.
`importShapes` returns a list of entities regardless, so a single solid is the n=1 case of
the general one, and supporting the general one is not materially more work than special-
casing it.

**The risk this does not retire, stated so it is not mistaken for retired.** The untested
thing was never "is multi-solid more work" -- it is whether output from a *real CAD system*
imports cleanly at all: degenerate faces, what `healShapes` in fact repairs, how assembly
structure arrives, and whether units are declared per component. Supporting multi-solid
does not answer any of that; it moves the discovery from before implementation to during
it. That is an acceptable trade and a deliberate one.

**Two cheap things that follow.**

*Get a real file into the fixtures early.* There is no STEP or IGES anywhere in the
repository or in any of the 42 recorded studies. A real chassis or heatsink assembly should
be sourced and committed as a fixture in the first implementation step, so the hard cases
are met on purpose rather than by a customer.

*The unit-declaration re-run is still owed, independently.* C1 asked for it explicitly:
"Re-run on the final image, not the probe container: the unit-declaration behaviour the
refusal text describes. It was measured on 7.8.1, so it should hold -- but it is exactly
the class of thing that moves between OCC versions." The original probe ran on gmsh 4.12.1
/ OCC 7.6.3; the image is pinned at 4.15.2 / OCC 7.8.1. This is unaffected by the decision
above and remains outstanding.

---
## 15. Citation corrections

All five external citations were checked and all five are real. The research grounding is
the strongest part of the plan; four entries need wording fixes rather than removal.
Licensing rows are handled in #7.

**AIDL -- arXiv:2502.09819.** Verified: "A Solver-Aided Hierarchical Language for LLM-Driven
CAD Design". The ID matches the paper §3 describes, and "AIDL's published experiments are
2D only" is accurate to the text ("For our experiments, we perform LLM-driven 2D CAD
generations with AIDL"). One wording fix: the constraint solver is SolveSpace-based
(iterated Newton); OCCT is the boolean and STEP-export kernel behind it. §3 currently reads
as though OCCT is the solver.

**P3D-Bench.** Verified in full. J-Geo and J-Sem are real metrics on [0,1]; Assembly-3D
gives J-Sem ~0.79-0.84 against J-Geo ~0.34-0.37, matching "J-Geo ~ 0.35 against J-Sem ~
0.8"; the main table carries 11 general-purpose models, so "all eleven models" is right;
and the failure taxonomy is Parameter / Geometry / Undefined-Reference / Syntax with
Parameter at 55% of CadQuery failures. No change.

**GrandpaCAD.** Verified verbatim -- an introspected exact-signature reference dropped code
errors on a full run from 40 to 33. Two caveats the plan should carry: it is a vendor blog
post with n=1, not a peer-reviewed result; and the same post has build123d **losing to
OpenSCAD overall** at roughly 0.4 errors per generation. Cited as a reason to do the work
-- which is cheap and sensible either way -- and not as evidence for it. Decision 13 already
assumes this framing.

**CADSmith.** Verified exactly: a stronger model (Opus) as Judge against the coder (Sonnet),
specifically "to avoid the self-confirmation bias inherent in single-model refinement".
§5's use of it is precise. No change.

**Point2CAD.** Real -- CVPR 2024, reverse-engineering CAD models from 3D point clouds -- but
asked to carry a wider claim than it makes. Its input is a *segmented point cloud* and its
output a fitted B-rep of surfaces, edges and corners: it is evidence that recovering
structure from unstructured geometry is hard research, not evidence about recovering a
*parametric feature program* from a B-rep, which is what §1's non-goal is about. Either
narrow the sentence or add a citation that addresses the feature-program problem directly.

---
## 16. `README.md:250` is corrected when the replacement desk lands

**What is wrong.** `README.md:250` still carries a module-table row for `geometry/` -- "the
desk behind the `geometry` tool [...] laps of build -> draw -> measure -> revise in-process"
-- describing the package deleted in `b6ac498`. `mesher/`, which replaced it, appears
nowhere in the README: grepping `mesher` and "mesh desk" across it returns nothing. So the
table is wrong in both directions at once, describing a module that does not exist and
omitting one that does.

`docs/found-by-using-it.md` names this failure mode directly: "prose that says something
the code stopped doing [...] It is the document someone reads first, and it will be
believed."

**Decision.** Corrected once, when the replacement desk from #5 lands, rather than now --
the row would otherwise be written twice, and the second version is the one that will be
true. The cost is accepted knowingly: the document a new reader opens first stays wrong
until then.

**Recorded so it is not lost.** This is the only stale `geometry/` reference in the tree --
`docs/design.md` does not mention it, and no other file does. Whoever writes the
replacement owns this row.

---

## Where the queue stands

Sixteen entries plus entry 0 and amendment 5a. Two things are deferred rather than settled:
kernel location (#2, pending a local build) and the subagent base class (#9, pending the
rewrite).

**What v1 became over the course of this queue.** Not "a geometry stage feeding an existing
STL -> case pipeline", which is what the plan describes and which does not exist in that
form, but: **a replacement for `mesher/` that does CAD and mesh as one desk, ingesting
STEP.** Smaller than the plan as filed, and closer to what is actually missing.

**The work that is genuinely new**, gathered from the entries above:

- an explicit `geometry` path parameter on the tool, refused early when unreadable (#12)
- STEP/IGES ingest, repair via `healShapes`, defeaturing, multi-solid selection (#13, #14)
- face tagging, and per-patch STL export that is exhaustive and disjoint (#5a)
- a `snappyHexMeshDict` writer -- nothing in the repo writes one (#5a)
- `locationInMesh` validation by point-in-solid test (#5a)
- a self-intersection check (#11)
- the finish check rewritten for CAD-and-mesh, reporting in preflight's register (#4, #11)
- a brief written for the whole job, inheriting the list in #5

**Prerequisites and owed measurements.**

- recover the eight benchmark prompts and the T-numbered acceptance runs, and commit them
  (#8) -- without them the final gate is not executable
- source a real multi-solid STEP and commit it as a fixture (#14)
- re-run the unit-declaration behaviour on the pinned 4.15.2 / OCC 7.8.1 image (#14)
- confirm defeaturing is reachable from build123d or OCP on the image (#13)
- C2's two deflection measurements, at the start of the export work (#6)
- confirm the reading of "llms are faster then solving" with its author (#0)
- settle or carry forward the `OPENREYNOLDS_MESH_TOOL=0` experiment (#5, #8)

---

# Second pass, same day

Three findings the first pass missed, plus one thing found in passing.

## 17. The desk's finish check accepts a multi-region mesh

**What the plan claims.** §1: "Conjugate heat transfer is reachable by retaining solid
regions rather than discarding everything but the fluid; the downstream pipeline supports
this through the freedom the agent has, not through an explicit CHT mode."

**A correction to this review's first pass.** That first pass called the claim refuted. It
is not. The agent has `bash`, `write_file` and root on an image with OpenFOAM 2512, every
utility on `PATH` including `splitMeshRegions` and `chtMultiRegionFoam`, and a populated
`$FOAM_TUTORIALS` -- the CHT audit itself found studies where the agent had grepped
`tutorials/heatTransfer/`. Cloning a tutorial, writing `regionProperties` and
`thermophysicalProperties` by hand, splitting regions and solving is entirely within the
freedom it already has. The claim holds.

**What survives, and it is narrower and more serious.** The freedom is real everywhere
except in one place -- the place this plan is building.

*Where it holds.* Case setup and solve. The agent can do it by hand. It gets no leverage
from the toolbox while doing so: `case_gen.py`, `preflight.py`, `mesh_look.py`,
`first_look.py`, `layer_report.py` and `locate.py` all hardcode singular
`constant/polyMesh`, and `case_gen.py` is incompressible-only -- no `0/T`, no
`thermophysicalProperties`, `grep -ci thermophysical` returns 0. So it is slower and more
error-prone than single-region work, but nothing forbids it.

*Where it does not hold.* `mesher/check.py` fails on a missing `constant/polyMesh`, so the
desk **cannot return a multi-region mesh however correctly it built one**. That is not
absent support; it is a gate rejecting a valid result. And it reports "nothing has been
meshed yet" about a mesh that exists -- a wrong diagnosis, not merely an unhelpful one.
`checkMesh -region` is invoked nowhere: `grep -rn "\-region "` over the package returns 0.

**Decision.** The replacement's finish check is region-aware. It discovers
`constant/*/polyMesh`, falling back to `constant/polyMesh`; runs `checkMesh -region <name>`
per region; and requires every region to pass, with the named-patch criterion applied per
region. Small, and it removes the only place the agent's freedom is actually blocked.

CHT case setup remains the agent's own work with bash, exactly as §1 says. The desk simply
stops refusing the result.

**Not adopted, for now: region-aware instruments.** Teaching `mesh_look.py` to report per
region would let the desk and the hosted UI *see* a CHT mesh. Left out because its `--json`
keys are a three-way contract with `case_gen.py` and the hosted UI's Mesh panel, which
lives in a separate repository (#18) -- changing the payload shape is a cross-repo change
and should be decided on its own, not as a side effect of this.

**Still true, and worth stating in the plan.** A CHT case is reachable but unassisted:
nothing in the toolbox helps build, dress or check one, and `occ.fragment` -- named in §4 as
"already in the stack" -- appears in no code at all. The converter performs no boolean of
any kind. "Already in the stack" is true only of gmsh's OCC kernel in general, and §4
should say that rather than implying an existing capability.

## 18. The blast radius of the replacement, costed

Recorded because decision 5 named the replacement without pricing it. Ordered by how
likely each is to be forgotten.

**The system prompt has 53 characters of headroom.** `len(SYSTEM_PROMPT) == 5947` against
`tests/test_prompt.py`'s `< 6000`. The existing `mesh` bullet is 297 characters, so a
rewrite that mentions a CAD file has 350 to work in, or must buy space elsewhere. Editing
it costs a one-time global prefix-cache miss, not a per-session one -- the prompt "sits at
the front of the cached prefix and any change invalidates the whole conversation's cache".
The same test bans the bare word `workflow`, `step \d`, `phase \d` and eleven other
patterns, forbids literal braces, and pins `v2512`, `pyvista`, `24 hours`,
`sandbox_expired`, `latestTime` and `/work` as facts that must stay present.

**Capture would silently drop the new artifacts.** In `casebundle.classify()` the
`constant/` test runs *before* the geometry-suffix test, so per-patch STLs under
`constant/triSurface/` land in the `definition` tier -- tier 0, the one that always travels
even when oversized, and which `tests/test_casebundle.py` assumes is small (~4 MB/study). A
chassis exported as twenty tagged STLs inflates exactly the tier the design guarantees will
survive. Worse: **a rebuild script not literally named `build.py` is dropped entirely** --
`DEFINITION_NAMES` is a fixed set, and a `cad.py` or `geometry.py` falls through every
branch of `classify()` and returns `None`. Either extend the set or pin the desk to the
name.

**The mirror drops CAD files before capture can see them.** `mirror.KEEP_SUFFIXES` has no
`.step`, `.stp`, `.iges`, `.igs` or `.stl`. On the default policy a `.step` outside
`system/`, `constant/`, `postProcessing/` or `0*` is skipped as "not an image, a report, a
log or a case dictionary", and a >2 MB STL under `constant/` is dropped as field data.
Since `casebundle.build()` reads from the local mirror, whatever the mirror drops cannot be
bundled. Add the CAD suffixes. (`disk.py prune` is safe by construction -- it is an
allowlist of regenerables and a `.step` matches none of them.)

**Config has no back-compat mechanism.** `mesher_model`, `mesher_effort`,
`mesher_max_steps`, `mesher_max_seconds` and `mesh_tool` are env-or-JSON only -- none are
in `_CONFIG_KEYS`, so `Config.save()` never persists them. Renaming them silently breaks
existing environments; there is no alias mechanism except the one hand-rolled for
`anthropic_api_key`. And `tests/test_wiring.py` structurally requires every new config
field to be *read* somewhere, so a field added and not wired fails a test.

**Four test files are directly coupled**, and the count is left to `pytest --co` rather
than written here, where it would go stale. `tests/test_mesher.py` pins the one-bash-block
protocol, finish-token semantics, PNG attach and evict, budget exhaustion still running the
check, and mid-run remarks. `tests/test_mesher_check.py` pins each finish criterion
including `SCALE_SLACK` and the twice-asked `unreachable` path. `tests/test_tools.py`'s
mesh-specific cases include one asserting the tool names are exactly the sorted eight -- so
*adding* a `geometry` property is fine and a *rename* is not.
`tests/test_toolbox_mesh_look.py` pins the JSON payload the finish check reads.

**`mesh_look.py` is a three-way contract and should not be touched.** Its `--json` keys are
consumed by the desk's check, by `case_gen.py`, and by the hosted UI's Mesh panel -- which
lives in a **separate repo** (`ui`; there is no UI code in this tree, and `launcher/` is a
three-file npm shim). `tests/test_mirror.py` puts it plainly: "the web viewer reads its mesh
out of the mirror and the page 404s without it."

**Smaller, still real.** `cli.py` imports `mesher` directly, constructs it, and renders
per-step progress from a `Step` shape (`cmd`, `exit_code`, `seconds`, `output`, `image`);
`mesher/__init__.py` exports 13 names; `ToolContext.on_tokens` exists solely so the desk's
spend lands in session totals; `tools.py`'s docstring says "The eighth, `mesh`, delegates to
the mesh desk (`mesher/`)"; and `toolbox/README.md`'s `case_gen.py` row says "It builds no
geometry: that is the `mesh` tool's job" -- both stale on a rename. Any new toolbox script,
such as the `snappyHexMeshDict` writer, must be added to the toolbox index or
`tests/test_toolbox.py` fails.

## 18a. The four calls inside the blast radius

**Settled without needing a choice.** Add `.step`, `.stp`, `.iges`, `.igs` and `.stl` to
`mirror.KEEP_SUFFIXES` -- the mirror drops them today, and `casebundle.build()` reads from
the mirror, so capture cannot bundle what never arrived. And leave `mesh_look.py`'s `--json`
keys alone: they are a three-way contract between the desk's check, `case_gen.py` and the
hosted UI's Mesh panel, which lives in a separate repository.

**The rebuild script: extend `DEFINITION_NAMES`.** Rather than pinning the desk to the
literal name `build.py`, the captured set grows to admit the names a CAD desk would
naturally choose. More permissive, and it carries a known risk -- the desk invents a name
nobody listed, and the script silently does not travel.

*Mitigation, and it closes the risk properly:* the finish check already requires a rebuild
script. Make it require one **that capture will actually take** -- i.e. check the name
against `DEFINITION_NAMES` rather than against a separate list. Then a name nobody listed
fails the run loudly instead of losing the artifact quietly, which is the same discipline
`b6ac498` applied to `Allmesh`: "if a step in it failed while the mesh came out anyway, fix
the step before you finish."

**Per-patch STLs: reorder `classify()`.** Test the geometry suffix before the `constant/`
membership test, so exported STLs land in the `geometry` tier where they are budgeted and
can be reported as skipped. Today `constant/` wins, putting them in `definition` -- tier 0,
which travels even when oversized and which `tests/test_casebundle.py` assumes stays around
4 MB a study. A chassis exported as twenty tagged STLs would inflate exactly the tier the
design guarantees will survive.

**The tool is renamed to match the job.** It is a CAD-and-mesh desk, and `mesh` understates
it. This is the most expensive of the four and the cost is accepted:

- `tests/test_tools.py` asserts the tool names are exactly the sorted eight -- update the
  literal list and keep it sorted, since the new name changes its position;
- the frozen system prompt's bullet is rewritten (below);
- `tools.py`'s module docstring says "The eighth, `mesh`, delegates to the mesh desk
  (`mesher/`)";
- `cli.py` imports and constructs `mesher` by name, and renders per-step progress from its
  `Step` shape;
- `toolbox/README.md`'s `case_gen.py` row says "It builds no geometry: that is the `mesh`
  tool's job";
- **environment variables need aliases.** `OPENREYNOLDS_MESHER_MODEL`, `_EFFORT`,
  `_MAX_STEPS`, `_MAX_SECONDS` and `OPENREYNOLDS_MESH_TOOL` are env-or-JSON only -- none are
  in `_CONFIG_KEYS`, so `Config.save()` never persists them, and there is no alias mechanism
  except the one hand-rolled for `anthropic_api_key`. Renaming without aliases silently
  breaks every existing environment. Note `OPENREYNOLDS_MESH_TOOL` also carries the
  unresolved experiment from `5b588c3` (#5, #8).

**The prompt bullet is rewritten inside the existing cap.** The prompt is 5,947 characters
against a 6,000 limit, and the bullet to be replaced is 297 -- so roughly 350 to work in.
That is enough to state the fact: what the tool takes (a shape in words, or a CAD file on
the volume), what it returns, and what stays with the caller.

*Why the cap is kept.* It is not a cost or context constraint -- 5,947 characters is about
1,565 tokens, small by any standard, and the prompt sits at the front of a cached prefix so
its length is nearly free. The guard is a proxy for the free-will contract: the sibling test
bans explicit workflow words, but procedure accumulates without them, one helpful
clarification at a time, and the length cap is what catches that drift. Its own docstring is
the argument: "Roughly one page. A long prompt is where procedure accumulates."

*The distinction that makes this fit.* The prompt's charter is facts about what exists,
never procedure. "It takes a shape in words, or a CAD file on the volume" is a fact, and
facts earn their space.

*Recorded for the next time it binds.* The cap has a measured cost in the other direction:
`c2f4241` added the environment card because a session "discovered the sealed network by
failing `pip install shapely` and rediscovered the recombine-for-empty-patches 2D recipe
from scratch", and that addition had to be "paid for by trimming restated/rhetorical lines
elsewhere". When the cap next binds, raise it deliberately with a written reason -- which
`design.md` permits, "change the tests together with the code rather than working around
them" -- rather than compressing facts into ambiguity.

## 19. The corpus discrepancy is handed off, not fixed here

**What was found.** `openreynolds/toolbox/README.md` carries detailed rows for `corpus.py`
and `search.py` -- "556 tutorials in under a second", `retrievals.jsonl`, `took <path>` --
and neither file exists on `main`. They live on `pr22-corpus`: four commits, ~5,600
insertions including ~2,800 lines of tests, 74 commits behind `main`, touching `prompt.py`
and `cli.py` which have both moved since.

**And it is worse than a documentation problem.** The frozen system prompt,
`openreynolds/prompt.py:35`, tells every session: "The rest of the volume holds other
studies' work, **searchable alongside the tutorials**." A session that takes it at its word
goes looking for a tool that is not there.

This is precisely the class of failure `tests/test_prompt.py` already guards against --
"The A4 run wasted a detour on foamToC, which the prompt claimed was there" -- and it
recurred because that guard checks the literal `foamToC` strings only.
`tests/test_toolbox.py::test_the_toolbox_index_names_every_script_in_it` missed the
documentation half for the mirrored reason: it checks that every script is named in the
index, never that every name is a script.

**Decision.** Out of scope for this review. Filed as
[InviscidAI/OpenReynolds#27](https://github.com/InviscidAI/OpenReynolds/issues/27), stating
both directions -- merge the rebased branch, or retract the two claims on `main` -- without
proposing one, and noting the two cheap guards that would catch a recurrence.

**The guards are deliberately left alone.** Making the index test bidirectional and
generalising the prompt test were considered and not taken here; they are noted in the
issue for whoever owns it.

**One consequence for this plan.** §5's sentence -- "The existing corpus of OpenFOAM cases
stays what it is -- reference material the agent consults at runtime" -- is unsupported on
`main` and stays flagged in the revised plan until the issue resolves either way.

## 20. Main is red, and has been since 2026-09-07 — filed

Found while establishing a test baseline; unrelated to this plan. `main` has failed CI on
every push for at least five consecutive runs. The run on `376ec4c` reports 7 failed
against 2042 passed; the same commit fails 1 locally. Two independent causes.

*A wrong test, shipped in HEAD.* `test_a_symlink_is_not_followed_and_not_counted_twice` was
added in `376ec4c` and has never been green. `disk.py` is correct -- summing the fixture
gives 31,524 bytes with no contribution from the symlink, which is the failing number
exactly, so the link is not being followed. The assertion compares the whole study against
ten times one time directory (20,000) and the fixture exceeds it. The test does not measure
what its docstring claims; comparing a scan before the symlink against one after would.

*CI installs an extra without `pyvista`.* `tests.yml` runs `pip install -e ".[dev]"`, and
`dev` carries numpy, matplotlib and pandas but not pyvista, which lives only in `toolbox`.
Six tests fail in CI and pass locally -- two in `test_toolbox_cad_convert.py` directly, and
four in `test_toolbox_case_gen.py` indirectly, because `case_gen` reads its measurements
through `mesh_look.py` and gets empty values rather than an import error. The `dev` extra's
own comment states the intent it misses: "the toolbox tests exercise the same parsing
locally, so CI needs them too".

**Decision.** Out of scope, same handling as #19. Filed as
[InviscidAI/OpenReynolds#28](https://github.com/InviscidAI/OpenReynolds/issues/28) with the
arithmetic, both diagnoses and both fixes. Neither applied.

**Why it matters to this plan anyway.** Decision 8 makes the eight benchmark prompts v1's
final acceptance gate, and decision 18 counts the tests a replacement would have to keep
green. Both assume a green baseline to measure against. There is not one today.

## Found in passing

`tests/test_toolbox_disk.py::test_a_symlink_is_not_followed_and_not_counted_twice` fails on
a clean checkout of `376ec4c` -- `assert 31524 < 20000`, "the link's target was counted
again". 2052 pass, 13 skip. Unrelated to anything in this record; the only working-tree
changes are the two untracked files it consists of. Either a real symlink-counting bug in
`disk.py` or a filesystem-dependent test.
