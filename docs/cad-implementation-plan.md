# CAD agent — implementation plan (2026-09-10)

Splits `cad-agent-build-plan.md` into chunks, one subagent each. Every design question is
already settled in the build plan or in `cad-plan-decisions-2026-09-10.md` (cited as `#n`);
nothing here re-opens one. What this file adds is the **seams** — which files each chunk
owns, the **frozen interfaces** across the seams so chunks do not have to talk to each
other, and an **acceptance gate** per chunk that a reviewer can run.

## How to read a chunk

Each chunk states: what it delivers, the files it **owns** (nobody else edits them), the
interfaces it must honour verbatim, and its gate. A chunk is done when its gate passes and
not before. Gates are written to fail an implementation that runs without erroring but is
wrong — the failure mode the whole plan exists for.

## Preconditions

1. **`main` is red.** `tests/test_toolbox_disk.py::test_a_symlink_is_not_followed_and_not_counted_twice`
   fails on a clean tree (2,052 pass, 1 fail, 13 skipped, ~52 s). Filed at `#20`. Fix it or
   record it as known-red before any chunk starts, so no chunk's gate is judged against a
   suite that was already failing.
2. **The toolbox is data to the loop, and a flat package to itself.** Nothing under
   `openreynolds/` imports `openreynolds.toolbox` and nothing may start: `cli.py:42` copies
   the directory to `/work/.toolbox` on the instance. So `cad/check.py` **cannot**
   `import preflight` — it consumes toolbox scripts' `--json` over the backend, which is
   also `test_negative_obligation.py`'s rule about the transport.

   *Between toolbox scripts the opposite holds.* They import each other by the established
   idiom -- `cfmesh.py:107`, `locate.py:65`, `ladder.py:75`, `geometry_view.py:33`,
   `first_look.py:57`, `study_run.py:70`, `progress_report.py:67`:

   ```python
   sys.path.insert(0, str(Path(__file__).resolve().parent))
   import preflight  # noqa: E402  (sibling script, not a package)
   ```

   **No new script copies code from an old one.** C2, C3, C4 and C5 import `preflight` for
   `Finding`, `worst_status`, `summarise`, `read_triangles`, `surface_topology`,
   `scale_diagnosis` and the threshold constants, and import `surfaces` (C2a) for the
   triangle machinery. A second, subtly different edge counter is the mess this rule
   exists to prevent.
3. **Geometry never executes in the loop's process**, even under `LocalBackend` (§2). No
   chunk may `import gmsh`, `import build123d` or `import OCP` from anything under
   `openreynolds/` outside `toolbox/`. Toolbox scripts may, because they run on the
   instance.
4. **Every gate runs on the dev machine, and C1's answers still come from the image.**
   The dev machine carries OpenFOAM v2512 with cfMesh and `surfaceCheck`, gmsh 4.15.2
   (matching the image's pin) and build123d — so no chunk defers its gate to C10.

   *But OCCT differs:* `cadquery-ocp` here is **7.9.3**, the image is **7.8.1**. `#14`
   reserves the unit-declaration re-run precisely because it "is exactly the class of thing
   that moves between OCC versions", so certifying 7.8.1 behaviour from a 7.9.3 run repeats
   that error one version later. **C1's probes are image-of-record**: local runs are
   development, `docs/cad-probes.md` records the image's answers, and where the two differ
   it records both with the versions beside them.

## Decisions taken during planning, recorded because the plan deferred them

**`#2` is settled: a long-lived kernel process on the instance.** §2 named three
architectures — kernel in the loop's process, a long-lived kernel process on the instance,
or a fresh script per call — and deferred the choice. The desk works in a **persistent
IPython kernel on the instance**, one cell per step, reached through the backend.

*The guardrail holds, and holds better.* §2's rule is that geometry executes through the
backend and never by importing a CAD kernel into the loop's process — "on one machine those
two are indistinguishable, which is exactly how the deleted stack acquired an in-process
kernel and the X/GL layer that went with it". A kernel on the far side of the backend
satisfies that by construction rather than by discipline.

*What it makes literal.* §3's "cells concatenated **are** a script" stops being an analogy
over bash blocks. `CellLog.script()` is the concatenation, replay is running it in a fresh
kernel, and the append invariant — "the concatenation still runs" — is directly checkable
rather than simulated.

*Three things that fall out.* The picture arrives through IPython's display protocol instead
of the current scrape-the-command-for-`.png` heuristic. A kernel restart becomes recoverable
by replaying the log, which is also §3's answer to chassis STEPs accumulating in memory. And
OpenFOAM utilities stay reachable inside a cell (`!blockMesh`, `subprocess.run`), so there
is one channel rather than two.

**No copy-modify-run templates.** The agent writes and runs its own cells; it does not copy
a `.py`, edit it and execute it. `templates/duct2d.py` and `templates/body_in_box.py` are
that pattern, and the desk that pointed at them is the one being replaced.

*What replaces them:* **a folder per recipe with a `README.md` at its root**, carrying the
gotchas the template scripts were carrying — the knowledge that existed because prose was
not enough ("the working 2D recipe, not a page describing one"). Data files that must exist
on disk for a utility to read (OpenFOAM dictionaries) live in the folder beside the README.
Copying a dictionary is not the pattern being removed: `snappyHexMesh` reads a file, and
there is no version of that which is not a file. Copying executable logic is.

*Left open, deliberately:* whether `duct2d.py` and `body_in_box.py` are deleted or left for
the main agent's bash. They belong to the desk being replaced; the CAD brief does not point
at them. Decide it in C9, where the rest of `mesher/`'s disposal lives.

## Waves

**C2a and C8a land first.** C2a is small and C2, C3 and C5 import it. C8a is the kernel
channel C8 drives; both are frozen as I8 and I9, so the chunks that consume them develop
against the signature and merge after.

Otherwise **C1 and C3 through C9 are all startable at once**, alongside C2a and C8a: the
interfaces below are frozen so no chunk waits on another's code, only on a signature. Soft
dependencies are noted in place — C2, C3 and C5 merge after C2a; C8 merges after C8a; C5
wants C1's fixture; C9 deletes `mesher/` after C6 and C8 have ported its tests. **C10 runs
last**, against everything.

Twelve chunks: C1, C2a, C2, C3, C4, C5, C6, C7, C8a, C8, C9, C10.

## File ownership

| Path | Owner |
|---|---|
| `tests/data/cad/**`, `scripts/cad_probes.py`, `docs/cad-probes.md` | C1 |
| `openreynolds/toolbox/surfaces.py`, `tests/test_toolbox_surfaces.py` | C2a |
| `openreynolds/toolbox/cad_audit.py`, `tests/test_toolbox_cad_audit.py` | C2 |
| `openreynolds/toolbox/domain_probe.py`, `tests/test_toolbox_domain_probe.py` | C3 |
| `toolbox/templates/snappy/**`, `templates/cfmesh/**`, `toolbox/patch_entries.py`, one new `preflight.py` finding, `tests/test_toolbox_snappy_path.py` | C4 |
| `toolbox/templates/prep/README.md`, `cad_convert.export_patches()`, `tests/test_toolbox_export_patches.py` | C5 |
| `openreynolds/cad/check.py`, `tests/test_cad_check.py`, `openreynolds/toolbox/mesh_look.py` (the `--region` flag **only**) | C6 |
| `openreynolds/cad/brief.py`, `openreynolds/toolbox/b123d_api.py`, `tests/test_cad_brief.py` | C7 |
| `openreynolds/backend/kernel.py` + the `Backend` protocol addition, `tests/test_kernel_channel.py` | C8a |
| `openreynolds/cad/agent.py`, `openreynolds/cad/cells.py`, `tests/test_cad_agent.py` | C8 |
| `openreynolds/cad/__init__.py`, `tools.py`, `cli.py`, `prompt.py`, `config.py`, `mirror.py`, `casebundle.py`, deletion of `openreynolds/mesher/`, `tests/test_cad_wiring.py` | C9 |
| `tests/data/prompts/**`, `scripts/cad_accept.py`, `docs/cad-acceptance.md` | C10 |

**Shared, append-only:** `openreynolds/toolbox/README.md`. C2, C3, C4, C5 and C7 each append
one row for their script to the end of the table. `preflight.py` is edited by **C4 only**,
and by exactly one addition — the `CHANGE_ME` finding. `cad_convert.py` is edited by **C5
only**, additively: it has four importers (`preflight`, `first_look`, `geometry_view`,
`study_run`) and its existing library surface must not move under them. `tests/test_toolbox.py` fails if a script
is missing from the index, and `test_the_index_offers_rather_than_instructs` fails on
imperative wording — write the row as description, not instruction. Resolve any conflict by
keeping both rows.

---

# Frozen interfaces

These are contracts, not suggestions. A chunk that changes one has broken every chunk that
was written against it in parallel.

## I1 — The Finding record

`toolbox/preflight.py:93`. **Toolbox scripts import it** -- they do not restate it.
`cad/check.py` alone re-hydrates findings from JSON into an identical `NamedTuple`, because
it runs in the loop's process and cannot reach the toolbox (Precondition 2). Note
`as_dict()`'s key renaming: `meaning` serialises as `means`.

```python
class Finding(NamedTuple):
    check: str        # a short stable name: "closure", "normals", "coverage", ...
    status: str       # "fail" | "warn" | "ok" | "skipped"
    measured: str     # evidence, quotable on its own, with numbers
    meaning: str = "" # interpretation; allowed to be wrong
    repair: str = ""  # suggestion; allowed to be ignored
```

Thresholds are **imported, not reinvented** (`#11`) -- `preflight.NON_ORTHO_WARN/FAIL`
(70/85), `SKEWNESS_WARN/FAIL` (4/10), `ASPECT_WARN` (1000), `MILLIMETRE_SUSPICION` (100.0),
`SCALE_FACTOR` (100.0), `TOPOLOGY_TRIANGLE_LIMIT` (400,000). Only `cad/check.py`, which
cannot import, carries its own copy, and its gate pins them against preflight's source.

## I2 — Every new toolbox script's `--json` envelope

```json
{
  "script": "cad_audit",
  "ok": true,
  "findings": [{"check": "...", "status": "...", "measured": "...",
                "means": "...", "repair": "..."}],
  "measured": { }
}
```

`ok` is `worst_status(findings) != "fail"`. `measured` is the script's own numbers, free
shape, for a human and for the desk to quote. Exit 0 whatever is found — except where a
script refuses its inputs, which is exit 2 with the refusal on stderr, the discipline
`cad_convert.py` already uses ("refused: ...", naming what to pass).

## I3 — Patch set on disk

The unit every export-side chunk agrees on.

```
<case>/constant/triSurface/<patch>.stl     one file per patch, ASCII or binary
<case>/constant/triSurface/patches.json    the manifest
```

```json
{"unit_metres": 1.0,
 "source": "chassis.step",
 "patches": [{"name": "inlet", "file": "inlet.stl", "triangles": 812,
              "area_m2": 0.0031, "role": "inlet|outlet|wall|symmetry|interface"}],
 "location_in_mesh": [0.05, 0.01, 0.02]}
```

Patch names are chosen where the surface is created, never from a bounding box — the rule
the current brief already carries. `location_in_mesh` is written by C5 and **validated** by
C3; C4 refuses a manifest whose point has not been validated.

## I4 — `cad/check.py`'s public surface

```python
@dataclass
class Check:
    ok: bool = False
    unreachable: bool = False          # semantics unchanged from mesher/check.py
    findings: list[Finding] = ...      # the new register (#11)
    missing: list[str] = ...           # derived: [f.measured for f in findings if fail]
    regions: list[str] = ...           # [""] for a single-region case
    cells: int; points: int; faces: int
    bounds: list[float]; patches: list[dict]; checkmesh: str
    metrics: dict[str, float]; two_d: bool; build: list[str]
    render: str; render_abs: str; error: str; raw: dict
    def as_refusal(self) -> str: ...
    def lines(self) -> list[str]: ...

def verify(backend, case_dir: str, case_rel: str, request: str = "",
           script: str = "") -> Check: ...

def fingerprint(backend, case_dir: str) -> dict:
    """What the geometry in this directory *is*, as numbers: per-patch triangle count,
    area, volume of the closed union, and bounds.

    Used only to compare an accepted run against its own replay. It does not cross the
    C6/C8 seam -- "the accepted geometry" is what is on disk now, so nothing has to be
    recorded at accept time and `CellLog` stays a log."""

def mesh_regions(backend, case_dir: str) -> list[str]:
    """The regions that have a polyMesh: names for `constant/<name>/polyMesh`, `[""]` for a
    single-region `constant/polyMesh`, `[]` for nothing meshed yet.

    Shared with C8's nudge deliberately. Two copies of "is there a mesh here" is how the
    finish check and the nudge come to disagree about whether a CHT case has been meshed --
    `#17`'s bug, which is already in the repo twice."""
def read(payload: dict, case_rel: str = "", case_dir: str = "",
         request: str = "") -> Check: ...
def look_command(case_dir: str) -> str: ...
```

`missing`, `as_refusal()`, `lines()`, `unreachable` and the field names above keep their
current meaning so `cli.py` and `tools.py` need no reshaping. `script` is the concatenated
cell log, for the replay criterion.

## I5 — `cad/cells.py`

```python
@dataclass
class Cell:
    source: str          # the Python the desk wrote
    reasoning: str = ""  # its own words, carried with the action (§3)
    accepted: bool = False

class CellLog:
    def propose(self, source: str, reasoning: str = "") -> Cell: ...
    def accept(self, cell: Cell) -> str:
        """Append it, or return why it cannot be appended -- "" when accepted.

        The refusal string is handed straight to the desk, and it must say that the cell
        already ran: the kernel holds state the log does not, which is correct and
        dangerous if silent. Same shape as `parse_action`'s `(cmd, complaint)`.
        """
    def script(self) -> str: ...      # the concatenation; THE deliverable (#3)
    def cells(self) -> list[Cell]: ...
```

`accept()` runs the **static check**, not a replay: walk the cell's AST, collect free names,
compare against names bound by accepted cells plus imports and builtins. Replay is C6's, at
finish. Re-running the concatenation on every accept is O(n²) against a script that costs
seconds to minutes a run.

**The harness writes the log, and names it `build.py`** — already in
`casebundle.DEFINITION_NAMES`, so the desk never chooses a filename capture would drop. See
C9, where this removes a risk `#18a` had only mitigated.

## I6 — `cad/brief.py`

```python
CAD_DONE = "CAD_DONE"
def system_prompt(step_timeout_s: int) -> str: ...
def task_message(request: str, case_dir: str, case_rel: str,
                 said: list[str] | None = None, geometry: str = "") -> str: ...
def remark_message(text: str) -> str: ...
```

## I7 — The tool

Renamed `mesh` → `cad` (`#18a`). Properties: `request` (unchanged), `case` (unchanged),
`geometry` (new, `#12`) — an absolute path under the workspace root to a
`.step`/`.stp`/`.iges`/`.igs` file, checked for existence and readability **before the run
starts**. Tool names stay exactly eight and sorted; `cad` sorts **second**, after `bash`.

## I8 — `toolbox/surfaces.py`, the shared triangle machinery

The one module C2, C3 and C5 all reach for. It holds what `preflight` does not already have
and what two chunks would otherwise write twice.

```python
class Index:
    """A spatial index over an (n, 3, 3) triangle array."""
    def __init__(self, triangles: np.ndarray, leaf: int = 16) -> None: ...
    def candidates_near(self, point, radius: float) -> np.ndarray: ...   # triangle ids
    def candidates_along(self, origin, direction) -> np.ndarray: ...     # triangle ids
    def candidates_overlapping(self, tri_id: int) -> np.ndarray: ...     # triangle ids

def weld(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(corner indices, unique vertices), on preflight's rounding rule."""

def tri_tri_intersect(a: np.ndarray, b: np.ndarray) -> bool: ...
def ray_hits(index: Index, origin, direction) -> int:
    """Crossings, with the degenerate-hit re-cast handled inside."""
def signed_distance(index: Index, points: np.ndarray) -> np.ndarray:
    """Negative inside. The sign convention every caller depends on."""
```

**The contract that matters:** `candidates_*` may over-return and must never under-return.
Every false negative is a missed intersection, a missed crossing or a wrong sign, and each
of those is silent. C2a's gate is written around that one property.

`weld` exists so C2 and C3 round coordinates the way `preflight.surface_topology()` already
does — "without the weld every edge looks open" — rather than inventing a second rule.

## I9 — The kernel channel

The backend gains a cell channel beside `exec`. Nothing above the protocol learns the
transport exists (`test_negative_obligation.py`).

```python
@dataclass
class CellResult:
    ok: bool                    # False when the cell raised
    stdout: str
    stderr: str
    error: str = ""             # exception type and message, "" when ok
    traceback: str = ""
    images: list[bytes] = ...   # PNG display data, in emission order
    seconds: float = 0.0
    still_running: bool = False # outran the window and was NOT killed
    truncated: bool = False
    log_path: str = ""          # where the whole of a clipped output lives

class Backend:                  # additions only
    def kernel_start(self, cwd: str) -> str: ...        # session id
    def kernel_run(self, code: str, timeout_s: int) -> CellResult: ...
    def kernel_poll(self) -> CellResult: ...            # busy/idle + output since last look
    def kernel_interrupt(self) -> None: ...             # an agent action, not a reflex
    def kernel_restart(self) -> None: ...               # state gone; the log replays it
```

**The window expires; the cell is not killed.** `timeout_s` is harness config — one number
beside `STEP_TIMEOUT_S`, **not agent-configurable**: the step budget keeps the conversation
moving rather than capping compute, and a knob to raise it only lets one stuck cell eat the
run-level budget. On expiry `kernel_run` returns `still_running=True` with the elapsed time
and whatever streamed so far, and the desk's next step either polls or interrupts
deliberately. This is what the backend already does for `exec` — `ExecResult.job_id`, "This
outran the 240 s window and is now running detached as job X. Poll its log with `tail`; do
not start it again." `MAX_SECONDS` still bites on a genuine runaway.

*Why not kill.* With bash a timeout lost nothing, because state lived on disk. In a kernel
it would lose every binding, which is why "the kernel survives" is load-bearing rather than
tidy. Not killing at all is strictly better.

**And the kernel changes what backgrounding means.** `nohup … &` was free under bash because
everything communicated through files; moving work to `subprocess.Popen` inside a kernel
**loses the live binding** — a boolean computed in a subprocess does not leave its result in
the session. §3 already mandates the fix for its own reasons: an expensive stage
"checkpoint[s] its B-rep to a STEP file on the volume and [has] the rest of the script load
that". So a cell that might outrun the window checkpoints, which is also what makes it
replayable. C7's brief says this.

**Images are returned, not discovered.** The current desk scrapes `.png` filenames out of
the command text, stats each, and suppresses re-sends. Display data removes the guesswork:
what the cell drew is what comes back.

**A long-running mesher still goes to the background** — `subprocess.Popen` or `!nohup …&`
inside a cell, polled from a later cell — because a mesher communicates through files and
loses nothing by it.


---

# C1 — Fixtures and the five owed measurements

Everything in §7's "Prerequisites and owed measurements" except the eight prompts (authored
during planning and committed at `tests/data/prompts/`; the runs are C10's) and
the author question (`#0`, not an engineering task). This chunk exists because three of the
four other prerequisites are cheap and every downstream chunk is written on top of an
assumption they have not confirmed.

### Deliverables

- **A real multi-solid STEP fixture** at `tests/data/cad/`, sourced from a real CAD system,
  not written by OpenCASCADE (`#14`: "The probe's file was written by OCC itself, the
  friendly case"). Committed with provenance and licence in `tests/data/cad/PROVENANCE.md`.
  A heatsink or chassis assembly: several solids, at least one fillet of known radius, at
  least one curved surface, and — if findable — one degenerate face.

  **Best effort, not blocking.** Try hard: the constraint is a licence that permits
  redistribution, which rules most of GrabCAD out and makes the openly-licensed hardware
  projects and vendor CAD downloads the places to look. If nothing suitable is found in
  reasonable time, **fall back to an OCC-authored multi-solid fixture and ship the chunk** —
  recording in `PROVENANCE.md` that the real-CAD case is unmet. That is `#14`'s position
  already: supporting multi-solid "does not answer" whether output from a real CAD system
  imports cleanly, it "moves the discovery from before implementation to during it. That is
  an acceptable trade and a deliberate one." The fallback makes the trade explicit rather
  than silent.
- **A synthetic companion set** built by a committed script so the analytic answers are
  exact: a box-with-a-duct (known fluid volume), three solids where two overlap by a known
  volume, a hollow box with a known minimum wall thickness, a fillet of known radius on an
  edge of known length.
- `scripts/cad_probes.py` — runs every probe below and compares against the recorded answer,
  exit non-zero on a change. `docs/cad-probes.md` — the recorded answers with the exact
  commands and the image's `gmsh`/`OCC` versions.

### The probes

1. **Defeaturing is reachable** (`#13`) — `BRepAlgoAPI_Defeaturing` via OCP, or whatever
   build123d surfaces. §2's whole argument for retaining B-rep to export rests on this.
2. **`Mesh.StlLinearDeflection`, twice** (`#6`) — does it govern OCC tessellation on our
   path or only STL import, and how does it compose with `MeshSizeFromCurvature`.
3. **Unit declaration on the pinned image** (`#14`) — the behaviour `cad_convert.py`'s two
   refusals describe was measured on gmsh 4.12.1 / OCC 7.6.3; the image is 4.15.2 / 7.8.1.
4. **`healShapes` on a real file** — what it in fact repairs on the fixture: face count,
   solid count, volume and the free-edge count before and after.

### Gate

- **The fixture is multi-solid**: `importShapes` returns **≥ 3 solids**, asserted, not
  eyeballed.
- **And it is the hard case, or it is on the record that it is not.** Preferred assertion,
  when a real-CAD file was found: the fixture carries at least one face OCCT's checker flags
  **before** `healShapes` and not after — proving there was something real to repair, which
  `FILE_NAME`'s originating-system field cannot (it is routinely blank, generic, or the name
  of a translator rather than the authoring system). On the fallback fixture that assertion
  is expected to find nothing; the gate then requires `PROVENANCE.md` to say so in as many
  words, and C5's thin-wall and degenerate-face outcomes to be reported as untested-on-real-
  input rather than as passes.
- **Probe 1 is answered by doing it, not by importing it.** Remove the fillet of radius `r`
  on an edge of length `L` **from the synthetic fixture, whose edge is a right angle by
  construction**; the resulting solid's volume increases by `(1 - π/4)·r²·L` to within 1%,
  and its count of cylindrical faces drops by exactly the number removed. The formula holds
  for a 90° convex edge only, which is why the probe is pinned to the fixture whose angle is
  ours to choose. Defeaturing the *real* fixture's fillets is run as well and reported as an
  observation — face counts before and after, and whether the operation succeeded at all —
  with no analytic assertion on it. An implementation that only asserts the symbol imports
  fails this gate.
- **Probe 2 produces a 3 × 2 table**, not a yes/no: max chord deviation measured off the
  B-rep at three `StlLinearDeflection` values crossed with `MeshSizeFromCurvature` at 2, at
  fixed `clmax`, on the fixture's curved surface — plus a stated verdict on whether
  deviation responds at all. C5 may not use the knob unless this table says it moves.
- **Probe 3 reproduces both refusals verbatim** on the pinned image: the whole of
  `tests/test_toolbox_cad_convert.py` passes there with `HAS_GMSH` true, and the two refusal
  messages are byte-identical to the ones on `main`. A changed message is a finding, not a
  pass.
- `scripts/cad_probes.py` re-run on the image exits 0; a deliberately corrupted recorded
  answer makes it exit non-zero and name which probe moved.
- The synthetic set's analytic answers are asserted in the fixture builder itself (volume,
  overlap volume, min wall thickness, fillet volume), so a downstream chunk quoting them is
  quoting a checked number.

---

# C2a — `toolbox/surfaces.py`: the shared triangle machinery

Small, lands first, and exists so that C2, C3 and C5 do not each write a spatial index.

### Why it is its own chunk rather than a section of C2

C2 needs an index plus triangle–triangle intersection. C3 needs an index plus ray casting
and signed distance. C5 needs the index to validate its own export before writing it. The
overlap is the index and the welding rule, and left to three chunks in parallel it becomes
three indexes with three sets of tolerances. Making it one owned file with one owner is the
whole point; making it a *chunk* is what stops two agents editing it at once.

It is deliberately **not** merged into `preflight.py`. That file is 2,970 lines and already
carries the house register; adding a BVH to it would put mesh-quality thresholds and
computational geometry in the same module and drag `test_toolbox_preflight.py` into every
chunk's blast radius. `surfaces.py` is a sibling, imported the way `preflight` already is.

### The work

I8's signature, and nothing beyond it. An index over an `(n, 3, 3)` triangle array with
three candidate queries; `weld` on `preflight.surface_topology()`'s existing rounding rule;
`tri_tri_intersect`; `ray_hits` with the degenerate-hit re-cast handled *inside* so no
caller reinvents it; `signed_distance` with negative-inside fixed as the convention.

No findings, no CLI, no I/O. It is a library, and the only toolbox script that is.

### The gate

- **No false negatives, ever — the one property everything else rests on.** For ≥ 50
  randomised triangle sets, each `candidates_*` query's result is asserted a **superset** of
  the true answer computed by exhaustive scan. Over-returning is allowed and measured;
  under-returning fails. Every downstream silent failure — a missed intersection, a missed
  crossing, a wrong sign — enters here.
- **The primitives agree with naive implementations exactly**, under 2,000 triangles:
  `tri_tri_intersect` against a direct SAT test, `ray_hits` against testing every triangle,
  `signed_distance` against a full point-to-triangle scan (to 1e-9).
- **The degenerate ray is handled inside, and provably.** Rays constructed to pass exactly
  through a shared edge and exactly through a vertex return the correct crossing count, and
  the test asserts the re-cast actually happened (a counter or a recorded direction), so a
  pass cannot be luck. C3 depends on this and must not re-solve it.
- **`weld` matches preflight.** On the same surface, `weld`'s vertex count equals the one
  `preflight.surface_topology()` derives internally. Two rounding rules is the failure this
  function exists to prevent.
- **The index earns its place.** On a 200,000-triangle surface the indexed query is at least
  an order of magnitude faster than exhaustive scan, and both return the same answer. An
  index that is correct because it returns everything is not an index.
- **Sign convention is pinned in a test, not only a docstring:** a point inside a closed
  sphere has negative distance. Every caller depends on it and none of them re-check it.

---

# C2 — `toolbox/cad_audit.py`: the exported surface, measured

The half of §5's "Added for CAD" that is about the triangles: closure of the **union**,
manifoldness, consistent outward normals, exhaustive and disjoint patch coverage,
self-intersection.

**It writes no geometry of its own.** `preflight.surface_topology()` already supplies open
edges, non-manifold edges, same-direction edges and degenerate triangles after welding
(`#11`), and `preflight.read_triangles()` reads binary and ASCII STL — both are imported by
the sibling idiom (Precondition 2), not reimplemented. The triangle machinery preflight does
not have comes from `surfaces` (C2a). What this script owns is the *audit logic*: composing
the union, partitioning by patch, and turning numbers into findings.

Genuinely new here, per `#11`'s list of what is missing: **self-intersection**,
**exhaustive-and-disjoint coverage** across the patch set, and **`surfaceCheck` invocation**.
`locationInMesh`, the third item on that list, is C3's.

### Deliverables

`python3 cad_audit.py <case>/constant/triSurface [--json] [--surface-check]`, reading the I3
manifest, emitting the I2 envelope. Findings at least: `closure`, `manifold`, `normals`,
`coverage`, `self_intersection`, `degenerate`, `scale` (reuse `preflight.scale_diagnosis`'s
ladder), and `surface_check` when `surfaceCheck` is on `PATH`.

**The distinction the whole script turns on** (§4): individual patch files are open
surfaces; it is their *union* that must be closed with no free edges and matching shared
edges. A per-file closure check is the wrong check and passes nothing real.

### Gate

- **Self-intersection agrees with brute force, and at scale.** On ≥ 20 randomised cases
  under 2,000 triangles (interpenetrating primitives at random rotations and offsets) the
  reported intersecting-triangle set equals a brute-force O(n²) test's, exactly. Plus one
  large case: a crossing **planted at a known location in a 200,000-triangle surface** must
  be found. The small cases never exercise the spatial index; the large one is the only
  place a broken index shows up, and a missed crossing is silent.
- **Known-answer defect injection**, each verdict naming the right number:
  - delete one quad from a tessellated cube → `closure` fails naming **4** open edges;
  - flip one triangle → `normals` fails naming **3** same-direction edges;
  - assign one face's triangles to two patches → `coverage` fails as *double-assigned*,
    naming the face;
  - drop one face from every patch → `coverage` fails as *unassigned*, naming the count.
    (§4: unassigned faces "land silently in a default patch and take whatever boundary
    condition it carries — a wrong answer rather than an error".)
- **Union-versus-parts, the load-bearing case.** A correct six-patch cube where **every
  individual file has open edges > 0** must produce `closure: ok`. An implementation that
  checks files individually fails here and passes everything else.
- **Invariance.** All finding statuses are identical under a rigid rotation, a uniform
  scale, a permutation of triangle order, and a permutation of vertex order within
  triangles, on the same surface. Any index-order dependence dies here.
- **Degradation, not silence.** Above `TOPOLOGY_TRIANGLE_LIMIT` the expensive findings come
  back `skipped` with the reason in `measured` — never `ok`. Asserted on a 400k-triangle
  surface, which must also complete in under 60 s.
- **Manifest against disk** (moved here from C4, where it was a generator refusal wearing
  a check's job). A patch named in `patches.json` with no STL on disk, and an STL on disk
  with no manifest entry, each produce a **fail** naming the file. It now fires whether or
  not the agent used the template.
- **Nothing is reimplemented.** A test asserts `cad_audit` imports `preflight` and
  `surfaces` and defines no edge-counting, welding or triangle-intersection routine of its
  own; and its topology numbers equal `preflight.surface_topology()`'s on the same surface.
- Runs without gmsh: the whole script is numpy over triangles, so its gate runs on a dev
  machine. Only `--surface-check` is image-gated.

---

# C3 — `toolbox/domain_probe.py`: is the point inside, and how wide is it there

Two additions from §5 with no implementation anywhere in the repo.

**`locationInMesh` validation.** §5a promotes it from nice-to-have to essential once snappy
is first-class: it is a snappy concept, "nothing validates it today", and the point-in-solid
ray cast "exists only as a design note". §4 calls a wrong `locationInMesh` "a classic silent
failure, trivially checkable" — the mesh comes out, `checkMesh` passes, and it is the volume
*outside* the part.

**Local width**, the one definition `#11` keeps from §5 "because nothing in the repo has
it": an SDF computation on the tessellation needing no face naming; the same field taken on
the solid gives minimum wall thickness. Width *along* the flow needs a centreline and is
explicitly left alone. It matters because C5's fluid boolean runs on thin-walled fin
geometry — OCCT's known weak case — and "how thin is the thinnest thing here" is the number
that predicts it.

### The work

`python3 domain_probe.py <triSurface-dir> [--point x y z] [--width-samples N] [--suggest]
[--json]`, emitting the I2 envelope. Findings: `location_in_mesh` (inside, with clearance
measured), `min_width`, `min_wall_thickness`. `--suggest` proposes a point at the maximum of
the interior distance field, so C5 has something to write into the manifest and C4 has
something to refuse the absence of.

**It writes no geometry.** `surfaces.Index`, `ray_hits` and `signed_distance` come from C2a
— including the degenerate-ray re-cast, which C2a proves and C3 must **not** re-solve.
`read_triangles` comes from `preflight`. C3 owns the union assembly, the margin policy, the
width definition and the suggestion search.

### Two definitions pinned here, because a test is a bad place to learn them

**Local width is the diameter of the largest inscribed sphere *containing* the point** — a
maximisation over spheres, not a lookup. `2·|sdf(p)|` is twice the distance to the nearest
wall, which equals the width only on the medial axis: near a wall it reports ≈ 0 for a slab
that is 10 mm thick. **`2·|sdf(p)|` is the wrong answer** and is named here so nobody ships
it.

**Classification within `1e-6 × diagonal` of the surface is reported as on-surface**, with
the measured distance, rather than resolved either way. Inside that band the question is
floating point rather than logic, and saying so is the house rule — "say what you did not
check" — applied to a number instead of a property.

### Scope boundary

Tessellation only: never reads B-rep, never meshes, never writes a dict. C4 consumes its
verdict through the manifest; C6 consumes its findings over the backend.

### Gate

- **Analytic agreement, exact, on a shape that punishes a lazy ray cast.** 10,000 uniformly
  sampled points against a sphere shell, a torus and an **L-shaped duct** (non-convex, so a
  fixed-direction ray crosses the arm twice), classified identically to the analytic
  predicate for every point outside the near-surface band. Zero disagreements, not "within
  tolerance".
- **The band is reported, not guessed.** Points inside `1e-6 × diagonal` come back as
  on-surface with their measured distance, never as a confident verdict.
- **Clearance is a number.** A point 0.5 mm inside a 1 mm gap reports clearance within 1% of
  analytic, and its status follows the margin rule rather than the binary.
- **Width reproduces a field, not a constant.** On a cone whose width varies linearly, the
  measured widths at 20 stations correlate with analytic at **r > 0.99**; slab thickness `t`
  and annulus gap `g` each within 2%. **The slab is sampled off the medial axis** — at 25%
  of its thickness from one wall — which is exactly where `2·|sdf|` gives the wrong answer
  and the pinned definition does not. This item decides whether the implementation
  understood the definition.
- **Wall thickness** on C1's hollow box recovers the known minimum within 2%.
- **`--suggest` survives its own validation**: the proposed point comes back `ok` with
  clearance ≥ the largest inscribed radius minus tolerance, on all three shapes.
- **An unvalidated point is a finding** (moved here from C4). A manifest whose
  `location_in_mesh` is absent, or present and outside the domain, produces a **fail** — the
  check that stops snappy meshing the outside of the part and exiting 0.
- **Nothing is reimplemented.** Asserts `domain_probe` imports `surfaces` and `preflight`
  and defines no ray cast, spatial index or distance routine of its own.

---

# C4 — the snappy path: template files, a patch enumerator, and a CHANGE_ME rule

### What this replaces, and why

An earlier draft of this chunk was `snappy_gen.py`, a generator with `--dx-surface`, a
defaulted-knob table and `--set key=value`. **That was a facade over `snappyHexMeshDict`** —
the same mistake `#13` rejects for build123d ("a real library it saw in training, with no
facade in front of it"), moved to the meshing half. `snappyHexMeshDict` is a format the
model has seen thousands of times. The tell was the reductio: if snappy's knobs are
enumerated in a house CLI, so are `simpleFoam`'s, and nothing in this repo does that.

**Deviation from `#5a`, recorded rather than reinterpreted.** `#5a` says "The missing
`snappyHexMeshDict` writer is no longer avoidable". It is right that the gap must be filled
and wrong that a *writer* fills it, on the plan's own anti-facade argument. What fills it is
below.

The toolbox's own division settles the shape: **scripts measure, templates build.** Every
script in there measures or dresses something that exists — `mesh_look`, `preflight`,
`locate`, `layer_report`, `cells_estimate`, `cfmesh`, `first_look`, and `case_gen`, which
reads patch names and types off the mesh. The one thing that authored a dictionary from
parameters was deleted, "and with it the habit it encouraged of adding one more shape every
time a new case turned up."

### Deliverables

**1. A recipe folder with a README at its root.**

```
templates/snappy/README.md              the recipe and the gotchas
templates/snappy/snappyHexMeshDict      starting dictionaries, CHANGE_ME where
templates/snappy/surfaceFeaturesDict    nothing can be guessed, and the
templates/snappy/blockMeshDict          PATCH-ENTRIES markers
templates/cfmesh/README.md
templates/cfmesh/meshDict
```

The README carries what the template scripts used to carry in their docstrings — the
"known gotchas, baked in rather than left to be rediscovered by failing". For this path:
that `castellatedMesh`, `snap` and `addLayers` are three stages and failing late still exits
0; that layer coverage is read off the mesh, never off snappy's own summary (the `cfmesh.py`
finding — 88.6% eroded to 66.7%, printed 66.7%, exited 0); that normal orientation is how
snappy decides inside from outside, so an inverted patch turns a solid into a void; that a
real chassis outruns a step window and goes to the background.

**No `.py` to copy, edit and run.** The agent reads the README and writes its own cells.
The dictionaries are data — `snappyHexMesh` reads a file and there is no version of that
which is not a file — so copying one is not the pattern being removed.

**2. `CHANGE_ME` where nothing can be guessed.** The shipped dictionaries carry the literal
token:

```
locationInMesh (CHANGE_ME CHANGE_ME CHANGE_ME);   // domain_probe.py --suggest
```

Everything with a defensible default carries it as a literal in plain sight with a comment
beside it — `nCellsBetweenLevels 3;`, `maxNonOrtho 65;  // preflight warns at 70, fails at
85`. Stated, not imported: a template that imports the house thresholds is a facade again.
`CHANGE_ME` lives in the artifact, is greppable, and survives being copied — where a CLI
refusal only fires if the script is the thing that runs.

**3. `toolbox/patch_entries.py`, the one thing that is genuinely a script.** Enumerating N
patches into three dictionary sections that must agree by name is mechanical bookkeeping
over what is on disk, and expecting it by hand at twenty patches is expecting the failure §4
names — "unassigned faces land silently in a default patch and take whatever boundary
condition it carries". It reads `constant/triSurface/` and I3's manifest and fills marked
regions:

```
// PATCH-ENTRIES-BEGIN geometry
// PATCH-ENTRIES-END
```

`--insert <dict>` rewrites them in place, **idempotent and re-runnable** — re-export the
patches, run it again, the dict follows. `--print` emits the fragments. It measures what is
on disk and decides nothing.

**4. The refusals move to where checks live.** A patch in the manifest with no STL, an STL
with no manifest entry, and an unvalidated `locationInMesh` are findings, not generator
behaviour: manifest-versus-disk to **C2's `cad_audit`**, the point to **C3's
`domain_probe`**, and a dictionary still containing `CHANGE_ME` to a **`preflight.py`
finding** — that script's exact charter, "the cheap questions asked before an expensive
run". They fire whether or not the README was read.

### Gate

Runs on the dev machine: OpenFOAM v2512 with cfMesh and `surfaceCheck` is installed here, so
C4 merges on these passing locally rather than deferring to C10.

- **The patch names survive.** Copy `templates/snappy/`'s dictionaries into a case holding
  a committed three-patch duct STL set, run `patch_entries.py --insert`, replace the two
  `CHANGE_ME` values, and run the mesher: it completes, `checkMesh` is clean, and
  `constant/polyMesh/boundary` carries **exactly** `inlet`, `outlet`, `walls`, each with
  more than zero faces, with no `defaultFaces` and no `patch0`.
- **Inside, not outside.** Meshed cell count within 10% of `fluid_volume / dx³`. An inverted
  patch normal or a wrong `locationInMesh` meshes the outside of the duct and exits 0; this
  is what catches it.
- **Refinement is honoured in the mesh, not in the dict.** Doubling the surface refinement
  level in the dictionary halves the measured near-wall cell size, measured off `polyMesh`.
  Reading the number back out of the dictionary is not this gate.
- **Layer coverage is measured, not claimed.** With layers on, achieved coverage comes from
  the mesh via `layer_report.py`. Under 50% is a `warn` with the number in it — never a
  silent success, which is the exact `cfmesh.py` failure: 88.6% eroded to 66.7%, printed
  66.7%, exited 0.
- **`CHANGE_ME` is loud.** The shipped templates contain the token (asserted, so no default
  is ever quietly baked in), and a dict that still contains it produces a `preflight`
  **fail** naming the file and the line — before any mesher runs.
- **The enumerator is exhaustive, disjoint and idempotent.** On a **twenty**-patch set —
  which is where hand-writing fails and three patches prove nothing — every STL on disk
  appears exactly once in each of `geometry`, `features` and `refinementSurfaces`, and no
  name appears that has no file. Running it twice produces byte-identical output; adding one
  patch and re-running adds exactly one entry per section and leaves the rest unchanged.
- **The README is sufficient on its own.** Every gotcha it names is asserted present, and
  the dictionaries it ships mesh the committed duct set with only the `CHANGE_ME` values
  replaced — the property that separates a recipe from a description of one.
- **The README offers rather than instructs**, checked with the same word list
  `test_the_index_offers_rather_than_instructs` applies to the toolbox index. The free-will
  contract does not stop at the index.
- **Nothing is reimplemented** — `patch_entries.py` imports `preflight` for the manifest and
  triangle reading, and the templates restate no house threshold as an import.

---

# C5 — prep: a recipe folder, and one tessellation function

### What this replaces, and why

An earlier draft was `cad_prep.py`, a script with nine subcommands — `inspect`, `heal`,
`defeature`, `select`, `boolean`, `interfere`, `fluid`, `tag`, `export`. **That was a facade
over the CAD kernel**, which is `sketch.py` with a different vocabulary, and `#13` rules it
out: the model writes build123d directly because it is "a real library it saw in training,
with no facade in front of it".

`cad_convert.py`'s own docstring refuses the first subcommand outright, and the argument
generalises to the rest:

> "**There is no statistics report here on purpose.** The module is importable
> (`import gmsh`), so the B-rep answers questions directly — entity lists,
> `getBoundingBox`, `getMass` for lengths and areas, `getCurvature`, `getEntities` — and a
> fixed set of numbers chosen in advance would only tell you which numbers somebody else
> thought mattered. The case that matters is the one with a feature nobody thought to
> measure."

Under **scripts measure, templates build**: `inspect` and `interfere` measure, and the
docstring above says even those are the agent asking gmsh. The other seven build. So the
chunk is a recipe folder plus the one piece of mechanical work that is not a house verb.

### Deliverables

**1. `templates/prep/README.md`** — the recipe the agent reads before writing its cells:
import STEP/IGES, repair with `healShapes` (`fixDegenerated`, `fixSmallEdges`,
`fixSmallFaces`, `sewFaces`), defeature by identifying cylindrical and toroidal faces,
select among solids, boolean, extract the fluid domain by subtraction with capping, tag
faces with build123d selectors, export. Snippets, and the gotchas beside them.

The gotchas this folder owes, each paid for in the plan: **B-rep is retained to export** and
tessellation happens only at per-patch STL write (§2), because fillet removal and face
tagging are both face-level operations that on a mesh would need re-segmentation into
primitives — an open research problem. **Multi-solid from the outset** (`#14`):
`importShapes` returns a list regardless, so a single solid is the n=1 case. **Checkpoint
the expensive stage** (§3): import-and-repair writes its B-rep to a STEP file and later
cells load it — "a cache the script writes and reads, not a second source of truth", the
move `scratch.py` already makes against network-filesystem cost. **Selectors are re-derived
on each run**, never face indices. **`clmax` is primary and mandatory** (`#6`).

**No `.py` to copy and run.** The agent writes build123d in cells, with C7's introspected
exact-signature API reference as its support.

**2. `cad_convert.export_patches()`** — the one tessellation implementation.

`cad_convert.py` is two things fused. As a **library** — `CAD_SUFFIXES`, `declared_unit`,
`render_tessellation` — it has four importers (`preflight`, `first_look`, `geometry_view`,
`study_run`) and must survive. As a **CLI** it converts one STEP to one merged STL, which
per-patch export supersedes.

So rather than a second script duplicating tessellation, cad_convert **absorbs per-patch**:
a function taking face groups to files, writing the per-patch STLs and `patches.json` of I3,
carrying the two refusals it already owns and the sagitta, facet-count and longest-edge
report it already computes. The existing CLI becomes the n=1 caller of the same function.
One implementation, one refusal set. The agent calls it from a cell — a library call like
`import numpy`, not a house verb, because it decides nothing about the design.

### What it does not do

**It does not change an import's design** (§1). Prep operations that remove or cut geometry
are expected; redesigning is not — "a distinction of intent, not of operation type, so it
cannot be enforced by the type system". It lives in C7's brief.

**No reverse-engineering an import into a parametric spec** — a research problem.

### The risk this carries, stated because `#14` insists

Supporting multi-solid does not retire the untested thing: whether output from a *real CAD
system* imports cleanly at all — degenerate faces, what `healShapes` in fact repairs, how
assembly structure arrives, whether units are declared per component. `#14`: "it moves the
discovery from before implementation to during it. That is an acceptable trade and a
deliberate one." If C1's fixture search came up empty, this risk is larger, and the gate
says so.

### Gate

The gate is on `export_patches()` and on the README's recipe run as cells — not on an API,
because there is no longer an API to test.

- **Exhaustive and disjoint at the source.** Every B-rep face of the exported domain lands
  in exactly one patch file, asserted over face **count**, before tessellation. C2 checks
  the same property downstream on triangles; both must hold, and a disagreement between them
  is a bug in one of the two.
- **Tag stability, the load-bearing property.** Run the README's tagging recipe three times
  — twice from scratch, once on the model after a rigid transform. The face → patch
  assignment is identical as a partition of face centroids mapped through the transform.
  This is what "selectors re-derived on each run" means, and it is what an implementation
  leaning on OCCT face ordering fails.
- **Round-trip conservation.** import → `healShapes` → `export_step` → re-import: solid
  count, total volume, total surface area and every per-solid bounding box to 1e-6 relative.
  A repair that quietly drops a solid dies here.
- **The fluid boolean is arithmetic.** On C1's box-with-a-duct, fluid volume equals box
  minus solid within 0.1%, and each capping face is planar with the analytic area within
  0.1%.
- **Defeaturing measured against analysis.** On C1's synthetic right-angle fillet, volume
  grows by `(1 - π/4)·r²·L` within 1% and the cylindrical-face count drops by exactly the
  number removed. C1's probe proves the operation exists; this proves the recipe uses it.
- **The two refusals survive the move.** `export_patches()` with no `clmax`, and a source
  declaring no unit with none supplied, each refuse with the reason and exit 2 — the desk
  reports up and the main agent asks the user (`#12`), never guesses. Asserted on the
  message text, which must be unchanged from cad_convert's.
- **The CLI is the same code.** Converting a single-solid STEP through the CLI and through
  `export_patches()` with one group produces byte-identical STL and identical reported
  numbers. Two tessellation paths that agree today drift tomorrow.
- **The checkpoint is a cache.** Delete the checkpoint STEP and re-run the recipe from
  empty: the exported patch set is identical to floating-point tolerance and every C2
  finding on it is unchanged.
- **Thin walls, honestly.** Run the fluid boolean on thin-walled fin geometry — OCCT's known
  weak case (§2) — and record the outcome either way. A failure is information the Manifold
  fallback was held in reserve for, not a reason to weaken the gate.
- **The README offers rather than instructs**, on the same word list as the toolbox index.

---

# C6 — `cad/check.py`: the finish check

The desk's finish line, rewritten for CAD-and-mesh as one job, reporting in the Finding
register (`#4`, `#11`, `#17`). **One finish check, no phase ladder** — §5 lists six places
the phase ordering breaks, two of them inside the plan itself.

### What it checks

*Floor, inherited from the current desk:* mesh present and non-empty, `checkMesh` passes,
patches carrying names somebody chose, a rebuild script present, a picture drawn, scale
matching the request's stated dimensions (`SCALE_SLACK`), unreachable reported as
unreachable rather than as a verdict.

**Deviation from the inherited floor, recorded rather than reinterpreted: the two-patch
count goes.** `mesher/check.py` fails a mesh with fewer than two named patches — "a flow
case needs at least an inlet, an outlet and walls". True of through-flow, false of
buoyancy-driven flow: a sealed domain whose convection is driven by volumetric sources on
cell zones, every boundary an adiabatic wall, legitimately has **one** patch. `checkMesh`
passes, the mesh is right, and the desk cannot finish — `#17`'s sentence in a second place,
"not absent support; it is a gate rejecting a valid result".

The count was a proxy anyway; the criterion's own docstring says what it is for — "patches
with names somebody chose, because `patch0`/`patch1` out of `gmshToFoam` is what a mesh
looks like when nobody said which end was the inlet". The property is **named, not
counted**.

- **fail** on zero patches with faces, or any patch carrying a mesher's default name
  (`_UNNAMED` already matches `patch\d+`, `region\d+`, `surface\d+`, `volume\d+`,
  `defaultFaces`). A single patch named `walls` passes.
- **warn**, not fail, on fewer than two named patches, stating the assumption: one boundary
  patch is consistent with a closed domain driven from inside it — volumetric sources on
  cell zones, or a body force — where every boundary carries the same condition; it is not
  consistent with buoyancy driven by wall temperature, which needs a hot patch and a cold
  patch, nor with through-flow, which needs an inlet and an outlet. Repair: split it with
  `topoSet` + `createPatch`, or name the surfaces separately where they are built. `ok` is
  `worst_status != "fail"`, so a warn does not block — "say what you did not check" rather
  than a guess dressed as a gate.

**And cellZones are reported**, which nothing in the repo currently does. Where a mesh's
zones are the mechanism — a heat source and sink driving convection — a verdict that cannot
see them is describing half the mesh. The check reads `constant/polyMesh/cellZones` over the
backend (`mesh_look.py`'s payload stays frozen) and reports names with cell counts in
`measured`. A measurement, not an assertion.

*Added for CAD (§5):* closure of the exported union, manifoldness, consistent outward
normals, exhaustive and disjoint patch coverage, self-intersection, a validated
`locationInMesh` — all by consuming C2's and C3's I2 envelopes over the backend — and **the
concatenated script re-running clean from empty** (`#3`).

**What this is worth over bare `checkMesh`, stated because it is a fair question.** The
mesh-quality half *is* `checkMesh`, wrapped. Everything C6 adds is something OpenFOAM's own
checker cannot see: that `locationInMesh` was inside the fluid (mesh the outside of the part
and `checkMesh` passes a perfectly valid mesh of the wrong volume); that the *input* surface
was closed with consistent normals and disjoint patches (a leak gives the wrong volume and
still checks clean); that the mesh is the size the request named (74 mm built in millimetres
is "a 74 m duct, with `checkMesh` and the picture both perfectly happy"); that patches carry
names somebody chose; that the script replays; that a picture exists; and that unreachable is
distinguished from failure. C6 is a **coordinator**: it runs `checkMesh` correctly — including
`-region` per region, which nothing in the repo does today (`grep -rn "\-region "` returns 0)
— assembles C2's and C3's findings, and adds the criteria that are about the request and the
artifacts rather than the mesh. **The novel geometric checks live in C2 and C3, not here.**

**So `checkMesh` is the sole authority on mesh quality.** The finish check reports its
verdict and does **not** re-decide pass/fail against `preflight`'s `NON_ORTHO_WARN/FAIL`,
`SKEWNESS_WARN/FAIL` or `ASPECT_WARN`. Those constants phrase a finding — the `measured`
value and its `meaning` — and never overrule the verdict. Two authorities on one number is
a desk being told contradictory things about a mesh OpenFOAM has already judged.

*Region-aware* (`#17`): discover `constant/*/polyMesh`, fall back to `constant/polyMesh`,
run `checkMesh -region <name>` per region, require all to pass, apply the named-patch
criterion per region. Today `check.py` refuses a multi-region mesh however correctly it was
built and reports "nothing has been meshed yet" about a mesh that exists.

*Deliberately not:* a cheap coarse mesh as a proxy for validity. "A coarse mesh can ignore
small features entirely, so it would pass on exactly the cases where defeaturing was
supposed to have resolved something."

*The rebuild-script criterion is checked against `casebundle.DEFINITION_NAMES`* (`#18a`), so
a script name capture would drop fails the run loudly instead of losing the artifact
quietly. C9 owns that set; import it.

### The one exception to "do not touch `mesh_look.py`"

Its `--json` keys are a three-way contract with `case_gen.py` and the hosted UI's Mesh
panel, which lives in a separate repo — **the payload shape does not change**. This chunk
adds a `--region <name>` flag that changes only *which* `polyMesh` is read, producing a
payload with identical keys. `#17`'s "not adopted: region-aware instruments" rules out
reporting per region *in one payload*; one payload per region is compatible. The gate pins
this.

### Gate

- **A real multi-region case passes.** Build one (`splitMeshRegions` on a tutorial, or a
  synthetic two-region mesh). The current `mesher/check.py` reports "nothing has been meshed
  yet" on it — assert that, then assert the new check passes it. Then break one region's
  mesh and assert the check fails naming **that region**.
- **Replay runs the script as a script.** `verify(..., script=...)` shells out
  `python3 <script>` through `exec`, **not** through C8a's kernel: the artifact's promise is
  that it is "runnable top to bottom", and running it in the kernel that already holds the
  bindings would test the convenience instead of the claim. This also keeps C6 independent
  of C8a.
- **Script replay, three cases, and the third is the point:**
  1. a clean cell log → pass;
  2. a cell referencing a binding defined only in a rejected cell → replay raises
     `NameError` → fail, naming the cell;
  3. **a log that replays without error but produces a different shape** — e.g. a cell that
     mutated a file in place, so a run from empty yields a different volume → fail on the
     **`fingerprint()` comparison** (per-patch triangle count, area, union volume, bounds),
     not on the exit code. An implementation that only checks the replay's exit status
     passes 1 and 2 and fails here.
- **No finding is prose.** Every finding the check emits has a non-empty `measured` and,
  when `status == "fail"`, a non-empty `repair`. This is the stated upgrade over today's
  check, which "emits prose with no status, no measured value and no repair".
- **`checkMesh` is not second-guessed.** A mesh `checkMesh` calls OK is not failed by the
  finish check on a quality metric, and a mesh it fails is not passed. Constructed both
  ways: a mesh with non-orthogonality between preflight's warn and fail thresholds that
  `checkMesh` accepts must reach `ok`, with the number and its meaning in `measured`. Two
  authorities on one number is the bug this forbids.
- **`mesh_look.py`'s payload keys are unchanged**, with and without `--region`: set equality
  against a key list pinned in the test, and `tests/test_toolbox_mesh_look.py` still green.
- **`unreachable` survives on both paths.** The twice-asked retry and the distinct
  `unreachable` verdict hold for single- and multi-region cases alike — a workspace that
  will not answer is never a fail (`83fb3a9`).
- **The closed-cavity case passes.** A sealed mesh with one patch named `walls` and two
  cellZones (`heater`, `cooler`) reaches `ok`. Assert the *current* `mesher/check.py`
  refuses it first — the before/after is the evidence the gate fixes something real, the
  same way the multi-region item works.
- **Dropping the count does not drop the naming rule.** A single patch named `defaultFaces`
  or `patch0` still fails, and zero patches with faces fails.
- **The one-patch warn is present, reasoned and non-blocking.** `ok` is true with a `warn`
  in `findings`; the warn names both mechanisms one patch is inconsistent with — wall-
  temperature-driven buoyancy, which needs a hot and a cold patch, and through-flow, which
  needs an inlet and an outlet — and carries a repair.
- **cellZones appear in `measured`** with names and cell counts, and a mesh with none says
  so rather than omitting the line.
- **The CAD findings are consumed, not recomputed.** C2's and C3's envelopes arrive over the
  backend; a `fail` in either is a `fail` here, carrying their `measured` and `repair`
  through unchanged. A test asserts `check.py` computes no geometry of its own.
- **The rebuild script is one capture will take** (`#18a`) — validated against
  `casebundle.DEFINITION_NAMES` itself, so a name nobody listed fails the run loudly instead
  of losing the artifact quietly.
- `tests/test_mesher_check.py` is ported to `tests/test_cad_check.py` with every criterion
  it pinned still pinned, `SCALE_SLACK` included — **except** the two-patch count, whose
  replacement is the three items above.

---

# C7 — `cad/brief.py`: what the desk is told

The brief for the whole job, inheriting §2's list. **Not** in the frozen system prompt —
`tests/test_prompt.py` holds that free of mandated workflow, and the read-only-import rule
lives here (`#4`).

### What must be in it

*§2's paid-for inheritance, every item:* any PNG a command writes comes back attached; the
finish token is a request to be checked, never a result; a budget-exhausted run is still
checked, "because an unexamined mesh is the failure this desk exists to end"; unreachable is
not failure; the user's verbatim words outrank the caller's paraphrase, and a mid-run remark
reaches the desk while it works; patch names come from the surfaces as they are created,
never from a bounding box; metres, always; coarse first; say what you did not check.

*New:* build123d in algebra mode, written directly, no facade (`#13`); §3's cell-authoring
requirements as the **correctness condition** they are — parameters as named constants, no
dependence on anything outside an accepted cell, no cell whose effect requires having been
run once already, reasoning carried with the action; the read-only rule for imports; snappy
as the default and gmsh-OCC as the quick shape check (`#5a`); the cheap-checks-before-
expensive-operations advice that replaces the phase ladder (`#4`).

- **A cell that might outrun the step window checkpoints to disk.** This is the one habit
  the kernel requires that bash did not: backgrounding work into a subprocess loses the live
  binding, so an expensive stage writes its B-rep to a STEP file and later cells load it —
  which is also §3's replay requirement, arrived at from the other direction. The window
  expiring does not kill the cell; it is reported still running, and can be polled or
  interrupted.

*Instruments by name and by what they answer* (`#4`) — `geometry_view.py` was used **zero
times across every persona run** while `read_file` on a PNG was used constantly. Name
`mesh_look.py`, `cad_convert.py`, `cad_audit.py`, `domain_probe.py`, `patch_entries.py`,
`cfmesh.py`, `layer_report.py`, `preflight.py`, each with the question it answers — and the
two recipe folders, `templates/snappy/README.md` and `templates/prep/README.md`, which are
instruments in the same sense: unnamed, they are unread.

### Also delivered: `toolbox/b123d_api.py` and the reference it generates

The **build123d capability index** (§3, `#13`) — what the library offers and what each thing
is for. It replaces the facade's benefits, and with copy-modify-run templates gone and no
`cad_prep` facade, it plus the recipe READMEs are the whole of the desk's build123d support.
`#13` files it as cheap and worth scheduling early; its cited result is a vendor blog with
n=1, so it is worth doing and not worth quoting as evidence.

**Three tiers, because the sizes force it.** Measured on the installed library:

| | entries | tokens |
|---|---|---|
| `dir(build123d)`, raw | 473 | ~7.4k |
| with every public method | 9,998 | ~179k |
| top-level with OCCT/stdlib/builder noise stripped | 311 | ~3.7k |
| + all public methods on the 13 core classes | +659 | ~9.1k |

*Tier 1, in the brief:* **one line** naming the file and its question — "what build123d
offers, grouped by what it is for; `help()` and `inspect.signature()` in your kernel give
the exact call". Nothing more. The brief is ~1.5k tokens; an index in it would be five times
the brief, paid at every step of a thirty-step run, which is §0's context complaint with
different filler.

*Tier 2, on disk at `/work/.toolbox/b123d_api.md`:* the curated index, **15-20 KB**. Read on
demand. `!grep` is its search — at this size a search script would wrap a tool the agent
already has, and `#19` records a phantom `search.py` already sitting in the toolbox index.

*Tier 3, the kernel:* signatures and docstrings live, against the installed library, which
cannot go stale. Named in the brief because a model used to bash will not reach for it by
default. This is what `#2`'s resolution buys, and it is why tier 2's job is **discovery**
— knowing a callable exists in order to introspect it — rather than reference.

**The cut is relevance, not top-level-versus-method.** The operations are top-level
functions (`extrude`, `fillet`, `chamfer`, `offset`, `revolve`, `loft`, `sweep`, `split`,
`mirror`, `scale`, `section`, `thicken`, `draft`, `project`), but two things that matter are
invisible to a top-level dump: **the boolean operators are dunders** — `+`, `-`, `&`, which
*are* algebra mode and are what §3's `y = x + sphere` is written in — and **the selector
idiom is methods** on `ShapeList`, which is what tagging is written in and therefore what
C5's tag-stability gate depends on.

- **In:** operations; primitives; geometry helpers (`Axis`, `Plane`, `Location`, `Vector`,
  `Rotation`, `Align`); import/export; the operators spelled out; and the ~30 selector and
  query methods — `faces`, `edges`, `vertices`, `solids`, `filter_by`, `filter_by_position`,
  `sort_by`, `sort_by_distance`, `group_by`, `first`, `last`, `center`, `bounding_box`,
  `moved`, `located`.
- **Out:** raw OCCT bindings (`BRep*`, `Bnd_*`, `TopoDS*`, `gp_*`), stdlib and typing
  leakage (`cos`, `sqrt`, `dataclass`, `Any`, `Callable`), and **builder-mode machinery**
  (`BuildPart`, `BuildSketch`, `BuildLine`, `Mode`) — `#13` chose algebra mode, and offering
  builder mode invites the ambient-state failure class the choice was made to avoid.

**So its provenance claim changes.** It is not "generated, never hand-written": it is
**curated selection, introspected content, verified live** — a reading list whose entries
are checked to exist, rather than a dump. The gate enforces the verification half.

### Gate

- **Every instrument is named with its question.** For each script above, the brief contains
  its filename *and* a sentence saying what it answers. Asserted per script — this is the
  measured failure the requirement comes from.
- **The inheritance list is complete.** One assertion per item in §2's list, each on a
  distinct phrase. A brief missing "say what you did not check" or "coarse first" fails.
- **The read-only rule is in the brief and not in the prompt.** `test_prompt.py` stays
  green, and the phrase is absent from `SYSTEM_PROMPT`.
- **Every entry resolves.** Regenerate the index and assert each entry names a live
  attribute of the installed build123d. An entry naming something that does not exist is
  the failure mode now that the file's job is discovery rather than reference.
- **The operators are present and explicit** — `+`, `-`, `&`, each with what it does.
  Asserted, because no introspection pass produces a dunder and §3's case for algebra mode
  is written on them.
- **The selector group is present and non-empty**, naming at least `filter_by`, `sort_by`
  and `group_by`. C5's tag-stability gate is written in these, so their absence would leave
  the one correctness-critical idiom undocumented.
- **Builder mode and OCCT bindings are absent**, asserted by pattern (`BuildPart`,
  `BuildSketch`, `BuildLine`, `Mode`; `BRep*`, `Bnd_*`, `TopoDS*`, `gp_*`). The index must
  not offer the surface `#13` rejected.
- **Grouped, not flat, and under 20 KB**, with the measured size recorded. A flat
  alphabetical dump is where the twenty things that matter drown.
- **Methods beyond the curated set are deliberately absent**, and the brief names `help()`
  and `inspect.signature()` in the kernel as how to get them. Asserted on both, so the
  omission reads as a decision rather than an oversight.
- **Tier 1 is one line, and the brief does not carry the index.** The brief's own token
  count is measured and recorded, and a test asserts the index is not inlined into it — the
  cost that would otherwise be paid at every step of a thirty-step run.
- **The finish token is `CAD_DONE`** and the brief presents it as a request that is checked,
  in the same words the current brief uses ("it is not a formality and it does not take your
  word for anything").

---

# C8a — the kernel channel

`#2` is settled above: a long-lived IPython kernel on the instance, one cell a step. This
chunk is the transport, and nothing else.

### The work

I9's four methods on the `Backend` protocol, implemented for `LocalBackend` first and for
`HostedBackend` behind the same signature. A kernel per desk run, started in the case
directory, torn down with the run.

**The harness never executes in the desk's kernel.** Every harness-side probe — the finish
check, the nudge, anything the loop wants to know about the workspace — goes through `exec`,
never `kernel_run`. Three reasons: a harness cell mutates the session the desk is reasoning
about; it would land in the cell log, or force filtering to keep it out; and the kernel may
be **busy** with a still-running cell, where a probe would block or fail. C6's finish check
already works this way for `mesh_look.py`, so this is consistency rather than a new rule.

**The rule this chunk lives under:** `tests/test_negative_obligation.py` forbids anything
above the protocol from knowing the transport exists. `CellResult` carries no session
handle, no socket, no notion of where the kernel is — the same discipline `exec` already
keeps. §2's guardrail is the reason: on one machine an in-process kernel and a remote one
are indistinguishable, "which is exactly how the deleted stack acquired an in-process kernel
and the X/GL layer that went with it".

**Output is bounded the way `exec`'s already is.** A cell that prints a mesh log is clipped
in the middle — "the news is at both ends" — with the whole of it left at `log_path`.
Display data is capped at `images.MAX_ATTACH_BYTES` per image and downscaled by the existing
`images.downscale`, not by new code.

### Gate

- **State persists across cells, and dies on restart.** `x = 1` in one cell is readable in
  the next; after `kernel_restart()` it raises `NameError`. Both halves matter: the first is
  the feature, the second is what makes the cell log the source of truth rather than the
  kernel.
- **A raising cell is a result, not an exception.** A cell that raises returns `ok=False`
  with the exception type, message and traceback in the result — the desk is handed the
  failure as work, exactly as a non-zero exit code is today. No exception escapes into the
  middle of a tool call.
- **Images come back without being named.** A cell that draws with matplotlib and never
  writes a file returns PNG bytes in `images`. This is the whole reason the channel exists
  in preference to bash, and it is the one behaviour the scrape-the-command heuristic cannot
  have.
- **A slow cell is not killed, and a hung one is recoverable.** A cell that outruns the
  window returns `still_running=True` with its elapsed time and streamed output, and is
  **still running** — asserted by polling and seeing it finish with its binding in place.
  Separately, `while True: pass` followed by an explicit `kernel_interrupt()` returns the
  kernel to a usable state, and the next cell runs in the same session with earlier bindings
  intact. A channel recoverable only by restart loses the run's state to one bad cell; a
  channel that kills on expiry loses a computation that was merely slow.
- **The timeout is not in the model's hands.** A test asserts no cell content can change the
  window — it comes from config, the way `STEP_TIMEOUT_S` does.
- **Background work outlives the cell.** A `subprocess.Popen` started in one cell is still
  running when a later cell polls it, and its log is readable — the pattern a real mesher
  needs, and the one the brief's `nohup … &` rule already assumes.
- **The transport stays invisible.** `test_negative_obligation.py` passes unchanged, and a
  test asserts `CellResult` exposes no field naming a socket, port, session id or file
  descriptor.
- **`LocalBackend` and `HostedBackend` agree.** The same cell sequence produces the same
  `CellResult` fields — `ok`, `error`, `stdout`, image count — on both. Divergence here is
  the class of bug that only appears in production.

---

# C8 — `cad/agent.py` and `cad/cells.py`: the loop and the cell log

The loop, ported from `mesher/agent.py` — the three things it owns stay: the picture
arrives, the finish is verified, the budget bites. Two things change: it drives **cells
through C8a's kernel channel** rather than bash blocks, and it keeps the accepted-cell log,
which **is** the script (`#3`).

*What the kernel changes about the port.* `parse_action` takes one fenced `python` block
rather than one `bash`; the two complaints (no block, several blocks) survive verbatim,
because both were about the model's discipline rather than the language. `_picture()` and
its `_sent` mtime bookkeeping are **deleted** — display data arrives with the result. The
eviction rule that keeps the last `KEEP_IMAGES` pictures and drops older images while
keeping their words survives unchanged; it was never about where images came from.

**The append invariant is not "the cell succeeded"** (§3). A cell can run correctly in a
live session and fail in sequence — it referenced a binding from a cell that was never
accepted, or it is not idempotent under replay. The invariant is **the concatenation still
runs**. That is what `accept()` must enforce.

**Enforcing it cannot mean replaying on every accept.** That is O(n²) against a script whose
every run costs seconds to minutes; twenty cells would spend the whole wall-clock budget
re-running themselves. Two layers instead:

- *On accept, a static check.* "No dependence on anything outside an accepted cell" is
  statically decidable — walk the cell's AST, collect free names, compare against the names
  bound by accepted cells plus imports and builtins. Cheap, immediate, and exact on the
  common case.
- *At finish, the full replay*, which is C6's job and gated there.

Each layer catches what the other cannot: the static pass catches a reference to a rejected
cell at the moment it happens; replay catches non-idempotence, which no static analysis
sees.

**A rejected cell has already run, and the rejection has to say so.** If a cell executes
cleanly but fails the static check, the kernel now holds state the log does not. That
divergence is correct — the log is truth, the kernel is convenience — and dangerous if
silent, because the desk would keep building on a binding that will not exist at replay. So
the rejection says: this ran, its result is in your session, it is **not** in the script
because it uses `x`, which no accepted cell binds; re-emit with `x` defined, or accept the
cell that defines it.

### Deliverables

`Mesher` → `CadDesk` (same constructor shape: `cfg, backend, store, home, on_step,
interject`), `MeshResult` → `CadResult` keeping every field `cli.py` and `tools.py` read
(`ok`, `case_rel`, `case_dir`, `summary`, `check`, `png`, `steps`, `seconds`, `tokens`,
`error`, `remarks`, `stopped`). `Step` keeps its field **names** (`cmd`, `exit_code`,
`seconds`, `output`, `image`) because `cli.py` renders per-step progress from that shape —
`cmd` now holds the cell's source and `exit_code` is 0/1 from `CellResult.ok`, so the
renderer needs no change. Plus `cells.py` per I5.

`run()` gains `geometry: str = ""`, passed to `task_message`.

### Gate

- **The append invariant holds under randomised drive.** With a scripted fake provider, run
  ≥ 50 randomised accept/reject sequences; after every one, `CellLog.script()` executes
  clean **in a fresh kernel**. Every sequence, not sampled — and with **trivial cells**
  (`x = 1`), because the invariant is bookkeeping rather than CAD. A gate that ran real
  geometry here would be too slow to run at all, which is how a gate becomes decorative.
- **The cell that runs live and fails in sequence is rejected, and the rejection says it
  already ran.** Construct it: a cell using a binding from a proposed-but-rejected cell.
  Assert it is absent from `script()`, that the refusal names the binding, **and** that it
  tells the desk the cell executed and its result is live in the session. The second half is
  what stops the desk compounding the error.
- **The static check is sound on what it claims.** Names bound by accepted cells, imports
  and builtins are all in scope; a free name no accepted cell binds is caught. Asserted both
  ways — no false positive on a legitimately-scoped cell, no false negative on the reference
  case.
- **The checkpoint is a cache, not state.** Drive a run whose early cell writes a STEP
  checkpoint and whose later cells load it. Delete the checkpoint, replay `script()` from
  empty, and assert the same geometry comes out. A log that only works because the
  checkpoint survived fails here — which is the §3 requirement stated as a test.
- **The inherited behaviours are still pinned**, ported from `tests/test_mesher.py` rather
  than rewritten: one fenced block per message and the two complaints, finish-token
  semantics (`_is_finish` matches the whole command, not a mention in a comment), image
  eviction keeping the words, **budget exhaustion still runs the check**, mid-run remarks
  landing in the thread and beating a simultaneous finish, and the empty-turn recovery.
  Dropped with the bash channel: the PNG-scrape and not-sent-twice tests, which pinned a
  heuristic that no longer exists — a cell's images arrive with its result.
- **The nudge asks the filesystem, not the transcript.** `_has_mesh`'s command-grep worked
  when every action was a shell command; in a kernel, prep and meshing both look like
  Python, and `snappyHexMesh` may appear inside a `Popen` string or not at all. It becomes a
  filesystem check through **`check.mesh_regions()`**, and the objection the grep was chosen
  over — "another round trip on the very path this is trying to make cheaper" — is spent,
  because the loop already talks to the workspace every step.
  - **It runs through `exec`, never in the desk's kernel.** The nudge is harness-driven and
    automatic; a cell it injected would mutate the session the desk reasons about, land in
    the cell log, and fail outright while the kernel is busy with a still-running cell.
    Asserted: no `kernel_run` call originates in the loop's own bookkeeping.
  - **It is region-aware.** `constant/polyMesh` **or** `constant/*/polyMesh` — a CHT run
    that has meshed two regions and no singular mesh must **not** be nudged. This is `#17`'s
    failure in a third place, and the shared `mesh_regions()` is what stops a fourth.
  - Twelve cells of build123d prep with nothing meshed gets nudged; meshing on step three
    does not. The failure it exists for is on the record: 942 s, eight commands, nothing on
    disk, "deriving tangent geometry in closed form" — which a kernel full of live bindings
    makes easier, not harder.
- **A kernel death mid-run is survivable.** Restart the kernel between two steps and assert
  the desk recovers by replaying the accepted log rather than losing the run — §3's answer
  to a live process not surviving a dormant period, exercised.
- **`geometry` reaches the desk.** A run given a geometry path has it in the first message,
  verbatim.

---

# C9 — Wiring, and the blast radius

Everything `#18` and `#18a` priced. Small edits in many owned files; the gate is that
nothing silently stops travelling.

### Deliverables

- **`mirror.KEEP_SUFFIXES` gains `.step`, `.stp`, `.iges`, `.igs`, `.stl`** (`#18a`,
  settled without needing a choice). `casebundle.build()` reads from the mirror, so what the
  mirror drops cannot be bundled.
- **`casebundle.classify()` reorders**: the geometry-suffix test runs *before* the
  `constant/` membership test, so per-patch STLs land in the `geometry` tier where they are
  budgeted, not in `definition` — tier 0, which travels even when oversized and which
  `tests/test_casebundle.py` assumes stays around 4 MB a study.
- **`casebundle.DEFINITION_NAMES` grows** to admit the names a CAD desk would choose, and
  C6's check validates the rebuild script's name against that same set — so a name nobody
  listed fails the run loudly instead of losing the artifact quietly.
- **The tool is renamed `mesh` → `cad`** with the `geometry` property (I7). Follow-on:
  `tests/test_tools.py`'s sorted-eight literal, `tools.py`'s module docstring ("The eighth,
  `mesh`, delegates to the mesh desk"), `cli.py`'s import and construction, and
  `toolbox/README.md`'s `case_gen.py` row ("that is the `mesh` tool's job").
- **Env aliases.** `OPENREYNOLDS_MESHER_MODEL/_EFFORT/_MAX_STEPS/_MAX_SECONDS` and
  `OPENREYNOLDS_MESH_TOOL` are env-or-JSON only, absent from `_CONFIG_KEYS`, with no alias
  mechanism except the hand-rolled one for `anthropic_api_key`. Renaming without aliases
  silently breaks every existing environment. `OPENREYNOLDS_MESH_TOOL` also carries `5b588c3`'s
  unresolved experiment and must keep working for C10's A/B.
- **The prompt bullet is rewritten inside the existing cap.** 5,947 characters against
  6,000; the bullet is 297, so ~350 to work in. State the fact: what it takes (a shape in
  words, or a CAD file on the volume), what it returns, what stays with the caller. If the
  cap binds, raise it deliberately with a written reason rather than compressing facts into
  ambiguity — `design.md` permits changing the test with the code.
- **`openreynolds/mesher/` is deleted**, after C6 and C8 have ported `test_mesher_check.py`
  and `test_mesher.py`, **and after C10 has measured its baseline against it** — the old desk
  is the only source of the number C10 compares to, because its recorded pass rate is not in
  the tree either. "Gone, not extended."
- **The image gains `ipykernel`.** The kernel channel is the `#2` decision, and neither
  `ipykernel` nor `jupyter_client` is in `pyproject.toml` or on the image, which has no
  network — so this is a workspace-image change, not only a repo change. Taken deliberately
  over a hand-rolled `python3 -i` protocol: display data is the reason the channel exists
  (C8a's image gate), and hand-rolling its capture is where the bugs would be.
  `tests/test_toolbox.py` asserts every toolbox script imports only what the image has, so
  the manifest in `ENVIRONMENT.md` is updated with it.
- **The two copy-modify-run templates are disposed of.** `templates/duct2d.py` and
  `templates/body_in_box.py` are the pattern the kernel decision removes, and the desk whose
  brief points at them is the one being deleted. Either delete them with `mesher/`, or keep
  them for the main agent's bash and say so in the toolbox index — but not left ambiguous.
  Note `agent.py`'s step-12 nudge names `templates/` by path, so whichever way this goes,
  C8's nudge text follows it.
- **`README.md:250` is corrected** (`#16`). It still carries a `geometry/` row describing
  the stack `b6ac498` deleted three days ago — a segment that does not exist, documented in
  detail. Replace it with the new desk's row.

### Gate

- **Capture end to end, with byte totals.** A case holding `build.py`, twenty per-patch
  STLs under `constant/triSurface/` **totalling over 2 MB**, and a `.step`: after a mirror
  pass and `casebundle.build()`, the STLs are in the **`geometry`** tier, the `.step`
  arrived through the mirror and is bundled, `build.py` travels, and the `definition` tier
  stays within the size assumption `tests/test_casebundle.py` makes. Assert tier byte
  totals, not membership — the size rule is the half a membership test misses.
  *Measure that assumption rather than quoting it:* the ~4 MB figure comes from the
  decisions doc, and the gate pins whatever `test_casebundle.py` actually assumes today.
- **The oversized-STL path specifically.** A single 5 MB STL under `constant/triSurface/`
  survives the mirror. This exercises `reason_by_size`, not `KEEP_SUFFIXES`, and a fix to
  only one of the two passes every other item in this gate.
- **The rebuild script is the one capture takes.** `cad/check.py` reads
  `casebundle.DEFINITION_NAMES` itself rather than a second list, asserted by importing
  both; and the harness-written log lands under a name in that set, so the check passes by
  construction rather than by the desk having chosen its filename well.
- **Aliases resolve with precedence.** Old names still work; new names win when both are
  set; `Config.save()` behaviour is unchanged for keys that were never in `_CONFIG_KEYS`;
  `tests/test_wiring.py` (every config field must be read somewhere) stays green.
- **The prompt survives its own tests.** `len(SYSTEM_PROMPT) < 6000`, every imperative
  pattern absent, every pinned fact (`v2512`, `pyvista`, `24 hours`, `sandbox_expired`,
  `latestTime`, `/work`) present, nothing volatile interpolated — and the new bullet states
  the CAD-file fact.
- **The `geometry` refusal costs nothing.** A missing, unreadable or wrong-suffix path is
  refused before the run starts: assert **zero provider calls** were made and the refusal
  names the path and what is wrong with it. "Refused early when unreadable" (`#12`) means
  not discovered on step nine.
- **Tool names are exactly eight and sorted**, `cad` **second** after `bash`, and the whole
  suite green with `openreynolds/mesher/` gone — no import of it anywhere, asserted by a
  grep test.
- **The image manifest matches the image.** `ENVIRONMENT.md` names `ipykernel`, and a test
  asserts the kernel channel's imports are all in the manifest — the discipline
  `tests/test_toolbox.py` already applies to toolbox scripts.

---

# C10 — Acceptance

Runs last. §5: "**The eight benchmark prompts the current desk was accepted on are v1's
final gate** (`#8`), run against the replacement."

**The prerequisite was not satisfied, and is now.** The original eight were judged
unrecoverable on 2026-09-11 — they existed only in a chat history, which is §7's complaint:
"This is the third time evidence cited in a commit message lives outside the tree, after the
deleted stack's `DESIGN.md` and `qa-runs/`. A gate whose test set exists only in a chat
history is not a gate."

**Eight replacements are authored and committed** at `tests/data/prompts/`, written before
any chunk is built against them so the gate is not shaped by what turned out to be easy:

| | prompt | the failure it exists to catch |
|---|---|---|
| T1 | 2D U-bend | a "2D" case meshed several cells deep; patches named by bounding box |
| T2 | body in a flow box | layer coverage claimed rather than measured; scale |
| T3 | Tesla valve | the failure that ended the previous stack — "circles connected by a line" |
| T4 | cold plate with fins | thin-wall booleans, OCCT's weak case; minimum width unmeasured |
| T5 | STEP assembly, fluid domain | ingest, repair, multi-solid, tagging, per-patch export |
| T6 | STEP with no declared unit | guessing a unit — a factor of 1000 on every length |
| T7 | sealed cavity driven by cell zones | a correct one-patch mesh refused by a two-patch rule |
| T8 | conjugate two-region case | a correct multi-region mesh called "nothing has been meshed" |

Four exercise authoring, two exercise STEP prep, and **two exercise cases the desk being
replaced cannot return at all** — T7 and T8 are the C6 deviation and `#17`, made executable.
Each file carries the properties the request names, which the finish check cannot know and
the desk must measure, and each names *a pass that is really a failure* — the residual the
checks cannot see, which is where the reviewer agent's closing condition looks.

**The comparison this costs.** A pass rate on this set is **not like-for-like** with the old
desk's. Both facts travel with the number wherever it is reported; substituting a new set
quietly and reporting the rate as continuous would be §7's failure wearing a better
disguise.

### Deliverables

- `tests/data/prompts/` — **done**: eight authored prompts and an index recording their
  provenance. What remains is the runs.
- `scripts/cad_accept.py` — runs all eight against the desk on the image, collecting for
  each: the script, the per-patch STLs, the mesh, the render, the findings JSON, wall clock
  and token spend.
- The `OPENREYNOLDS_MESH_TOOL=0` A/B, which settles the experiment `5b588c3` opened and left
  open — whether "a slow non-deterministic natural-language sub-agent is a worse interface
  than the bash the caller already has".
- `docs/cad-acceptance.md` — the results, both arms, with the old desk's recorded numbers
  beside them.
- **The reviewer's closing condition** (§5): a one-off manual pass over authoring output the
  checks accepted. Outputs that pass everything and are visibly wrong define the reviewer
  agent's job and give its false-negative rate; if there are none, it is not needed.

**No golden-file regression suite**, deliberately: CAD preparation is not deterministic — a
Tesla valve has many valid representations — so golden-file testing is impossible. The
desk's own finish check is the continuous invariant grader, and it costs nothing because it
is the gate rather than a second suite.

### Gate

- **All eight run, and each is judged on its own named properties**, not only on the finish
  check — "a property you did not measure is a property you did not build". A run that
  passes the finish check and misses a named property is a fail with the property named.
- **T6 passes by not finishing.** The desk reports up with the unit reason and returns its
  `stopped` state; a run that guesses millimetres fails this prompt even though its mesh is
  right, and a run that blocks waiting for a human fails it too.
- **T7 and T8 pass, and are shown to be new.** Both are run against the desk being replaced
  first, which must refuse them — one on the two-patch rule, one on the missing singular
  `constant/polyMesh`. The before/after is what makes them evidence rather than assertion.
- **The aggregate is reported with its caveat.** Pass rate against the old desk's recorded
  rate, with the statement that the sets differ and the comparison is indicative. Any
  individual regression is named even where the aggregate holds.
- Each passing run leaves the full artifact set on disk, and **re-running its committed
  script from empty reproduces a mesh that passes the finish check again** — the §3
  invariant, exercised on real output rather than on a fixture.
- The A/B is run and reported: desk versus bash-with-the-toolbox on the same eight prompts,
  wall clock and token spend for both. §5 records that on the hardest prompt measured, bash
  did better; the answer is whatever it is, and it is written down either way.
- The manual review pass is done and its count recorded — **including zero**, which is a
  result ("if there are none, it is not needed") and not a skipped step.
- Where v1 is exposed is restated against what was observed: authoring is the part that
  cannot be checked into correctness (the J-Geo ≈ 0.35 against J-Sem ≈ 0.8 gap across all
  eleven models in P3D-Bench, with no automated check behind it). Any prompt that passed
  every check with a visibly wrong shape is named.

---

# Notes for whoever runs this

**Do not let a chunk widen.** Each gate is written so that passing it is evidence the thing
works, not evidence it ran. If a gate looks unreachable, that is a finding about the design
worth reporting — not a reason to weaken the gate quietly.

**Three things the plan deliberately does not build**, so no chunk should drift into them:
the reviewer agent (§5, deferred, C10 only measures whether it is needed); the ask-the-user
half of the iteration loop (`#9`, `#10` — the desk reports up and asks no one); and
solver-aided placement (§3, filed as an experiment, and `#0`'s reading is unconfirmed with
its author).

**Where it runs is still deferred** (`#2`). Everything above is developed against
`LocalBackend`. The guardrail that keeps the deferral live: geometry executes through the
backend, never by importing a CAD kernel into the loop's process. On one machine those two
are indistinguishable, "which is exactly how the deleted stack acquired an in-process kernel
and the X/GL layer that went with it".
