# cfmesh — a hex-dominant mesh whose layers survive

Data, not a script: one dictionary and this page. Nothing here is a procedure — it is
the recipe written out so it can be read and changed, and a case that wants something
else is a case that wants something else. Ignore them where they do not fit.

| file | goes to | what it is |
|---|---|---|
| `meshDict` | `<case>/system/` | one `CHANGE_ME` surface file, and every other knob a literal with a comment beside it |

## The recipe, as commands

```bash
cp .toolbox/templates/cfmesh/meshDict <case>/system/
cat <case>/constant/triSurface/{inlet,outlet,walls}.stl > <case>/constant/triSurface/all.stl
# replace surfaceFile with that path
python3 .toolbox/preflight.py <case> --checks change_me
cartesianMesh && checkMesh
python3 .toolbox/layer_report.py <case> --patch walls --ref <case-with-no-layers>
```

## Gotchas, baked in rather than rediscovered by failing

**cfMesh reads one surface file, not a directory of them.** An I3 patch set is one STL
per patch; cfMesh wants them concatenated. An ASCII STL is a sequence of `solid <name>`
records, so `cat` is a legal merge and each solid becomes a patch carrying its own name.
A binary export has to be converted first — `surfaceMeshConvert` or `cfmesh.py`.

**`renameBoundary/defaultName` renames every patch that is not listed under
`newPatchNames`.** With `defaultName walls` and an empty `newPatchNames`, a three-patch
duct meshes into a single patch called `walls` carrying every face, `checkMesh` passes,
and the inlet is gone. This is the same failure snappy has with unassigned faces,
arriving through a different door: the names have to be listed to survive.

**A key that matches no patch is a warning and an exit code of 0.** `localRefinement`,
`patchBoundaryLayers` and `newPatchNames` are matched as regexes against the patch names
in the surface file; a key that matches nothing produces `Cannot find any patch names
matching ...` in the log and a mesh with no refinement and no layers where they were
asked for. `cfmesh.py --check --log log.cartesianMesh` is the thing that reads for it.

**cfMesh meshes the inside of a closed surface, and has no `locationInMesh`.** Closure
is what decides inside, so an external flow needs the body enclosed in a box and the
union handed over as one file. `cfmesh.py --box` does that and says what it did.

**Layer coverage is measured off the mesh, whatever the mesher said.** This is where
cfMesh earns the trip: snappy inserts prisms and deletes the ones a quality metric
rejects — on one hull 88.6% coverage at the first growth iteration, 66.7% by the
fiftieth, printed 66.7%, exited 0 — where `cartesianMesh` extrudes a sheet and optimises
it afterwards and covered 100.0% of the same wall. Both numbers came out of
`layer_report.py`, which reads first-cell heights off `polyMesh`.

**A real chassis outruns a step window and belongs in the background.** `cartesianMesh`
is multi-threaded and its `ExecutionTime` is the sum over threads, so a log that says
112 s may have taken four; either way a foreground call killed at the step timeout
leaves a `constant/polyMesh` that reads like a mesh.

```bash
nohup cartesianMesh > log.cartesianMesh 2>&1 &
```

## When snappy is the better path

Snappy takes per-patch surfaces directly, has refinement regions, and has a feature-edge
stage driven by `surfaceFeatureExtract`. `../snappy/` holds that recipe.
