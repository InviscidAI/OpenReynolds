# The silent failures, and what is watching for each

Some defects never present as failures. Mesh the outside of the part and `checkMesh`
passes a perfectly valid mesh of the wrong volume. A leak in the exported surface gives
the wrong fluid volume and still checks clean. Neither can be found by waiting for a
failure, so neither can wait for the build-up loop to surface it.

So detection is separated from delivery. **The supervisor probes for every row below**,
out of band, on the case as it stands on disk, and grades the result into the run record.
The agent is not told, the checks are not in its brief, and nothing is mounted in its
workspace -- the contamination grep of §3 is what proves the last part. The
implementations are `openreynolds/buildup/probes.py`, which imports `cad_audit.py`,
`domain_probe.py` and `surfaces.py` **in the supervisor's process**; the workspace does
not contain them.

**A check is activated for the agent only once its probe has actually fired** -- once some
run has produced a silently-wrong result of that kind. On that trigger and not before, the
criterion joins the finish check and the measuring script becomes reachable. A probe that
never fires across the whole corpus stays dormant forever, and that is the point: the
agent pays for a check only after the failure it catches has been observed to happen.

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
