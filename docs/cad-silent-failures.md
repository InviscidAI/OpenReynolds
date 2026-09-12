# The silent failures, and what is watching for each

Some defects never present as failures. Mesh the outside of the part and `checkMesh`
passes a perfectly valid mesh of the wrong volume. A leak in the exported surface gives
the wrong fluid volume and still checks clean. Neither can be found by waiting for a
failure, so neither can wait for the build-up loop to surface it.

So detection is separated from delivery. **The supervisor reads every instrument below**,
out of band, on the case as it stands on disk, and records what each measured. It then
vets the mesh itself, which is the part that carries the judgement.
The agent is not told, the checks are not in its brief, and nothing is mounted in its
workspace -- the contamination grep of §3 is what proves the last part. The
implementations are `openreynolds/buildup/probes.py`, which imports `cad_audit.py`,
`domain_probe.py` and `surfaces.py` **in the supervisor's process**; the workspace does
not contain them.

**A check is activated for the agent only once a real instance of its failure has been
seen** -- once some run has produced a silently-wrong result of that kind, and the
supervisor has said so with the case in front of it. On that trigger and not before, the
criterion joins the finish check and the measuring script becomes reachable. A row that
never earns an activation stays dormant forever, and that is the point: the agent pays
for a check only after the failure it catches has been observed to happen.

## The probes measure; they do not judge

They used to return `fired` / `pass` / `skipped`, and a firing was a failed exit even on a
run that ended `done`. The first baseline sweep (`core-20260912-083752-e3a3`) is why that
is gone.

Three probes fired across eight cases and the supervisor overturned all three. T2's seed
point sits 0.312 m outside the sphere because it is an external-flow case, and the desk's
own `Total volume = 0.127967` against a 0.8 x 0.4 x 0.4 box minus the sphere confirms the
mesh was right. T4's 132 flipped edges on 88 triangles are one surface present twice in
`constant/triSurface` while `meshDict` named a single file that `surfaceCheck` certified
as `Number of zones (connected area with consistent normal) : 1` -- and the delivered mesh
came from blockMesh and never touched the STL.

Meanwhile the two runs that really were wrong produced no probe signal at all. T6 guessed
a length unit and shipped a mesh `checkMesh` passed; T3 burned 28 cells to a mesh that
never got a verdict. Both had every probe return no reading, because five of the eight
cases never write a `triSurface` for a probe to read -- they are blockMesh, gmsh, or an
analytic O-grid.

**Every measurement was correct and every verdict was wrong.** A probe can count free
edges, flipped edges, crossing pairs and the clearance of a point. It cannot know whether
the surface it counted is the one that was meshed, or whether `outside` is where the seed
point belonged, because that is the case's intent -- and the supervisor is the thing that
reads the case. So the probes report numbers and `n/a`, the supervisor vets the mesh
directly on **every** run rather than only when something fires, and no exit code turns on
a reading.

The sharpest evidence that a fixed check cannot carry intent is `scale`. It is written for
"millimetres read as metres" and measures the surface against *the extent the case states*.
T6 is a STEP that declares no unit -- the factor-of-a-thousand case -- so it states no
extent, and the probe written for that failure is structurally unable to fire on it.

Nothing here says the numbers are worthless: the supervisor used all of them while
reasoning about T2 and T4. It says the conclusion was never the instrument's to draw.

Written by hand, from the supervisor's own output. `python3 scripts/cad_supervise.py
registry` prints the states as the code holds them, and
`tests/test_buildup_probes.py` pins the two together, so a row here that has drifted from
the code is a failing test rather than a stale document.

| `id` | `catches` | `detect` | `state` | `triggered` | `activated` |
|---|---|---|---|---|---|
| `location_in_mesh` | The meshing point sits outside the fluid, so the mesh is of the volume around the part. Snappy succeeds, `checkMesh` passes, and the mesh is of the complement. | Ray cast and parity count from the point **the case actually used** -- read off `system/snappyHexMeshDict`, not off what the desk said -- against the exported surface after the run. | dormant | | |
| `union_closure` | The exported patch set has free edges, so the meshed volume is not the one intended. Nothing errors, because an open surface is a legal STL. | Open edges on the welded **union**, never per file: individual patch files are open surfaces by construction and a per-file check passes nothing real. | dormant | | |
| `normals` | Inconsistent outward orientation turns a solid into a void for snappy, which meshes the complement without complaint. | Edges walked twice in the same direction on the welded union. | dormant | | |
| `coverage` | A face assigned to two patches, or to none. An unassigned face lands silently in a default patch and takes whatever boundary condition it carries. | `cad_audit`'s coverage finding. It needs the patch manifest to know which face was meant to be whose, so a case without one records `skipped` rather than a pass. | dormant | | |
| `scale` | The case is a factor of a thousand from the dimensions the request stated -- millimetres read as metres -- and every number downstream is self-consistent. | The union's largest extent against the extent the case states, fired outside a factor of a hundred. Measured against the request, which means the request must state its dimensions. | dormant | | |
| `self_intersection` | The exported surface crosses itself, so the volume it bounds is not well defined and the mesher resolves it silently either way. | Triangle-triangle intersection between non-neighbours on the union, skipped above 200,000 triangles and saying so. | dormant | | |

## The three states a probe reports

`pass` and `fired` are the two that mean something about the case. **`skipped` is neither**
-- it is the probe saying it could not measure, and it is recorded separately precisely so
that a case nobody could probe is not counted as a case that came back clean. A registry
full of `skipped` is not a corpus that is going well.

## Where an activated check lives

Open, deliberately. Folding a probe into the finish check makes it a gate; leaving it as a
reported finding makes it advice. The first activation decides it on the evidence of that
case, and the decision is recorded in the row's `activated` column -- with the run id, so
the reasoning can be found again.
