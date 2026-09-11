# snappy — a hex-dominant mesh around an exported patch set

What is in this folder is data: three OpenFOAM dictionaries and this page. There is no
script here to copy, edit and run. The recipe is written out so it can be read, argued
with, and turned into whatever the case at hand actually needs — a case that wants
something else is a case that wants something else, and these are a starting point, not
a procedure. Ignore them where they do not fit.

## The files

| file | goes to | what it is |
|---|---|---|
| `blockMeshDict` | `<case>/system/` | the background box snappy carves out of. Two `CHANGE_ME` corners and a literal `dx` |
| `surfaceFeatureExtractDict` | `<case>/system/` | one block per STL, filled by `patch_entries.py`. The utility here is `surfaceFeatureExtract`; `surfaceFeatures` / `surfaceFeaturesDict` is not in v2512 |
| `snappyHexMeshDict` | `<case>/system/` | three marked patch lists, one `CHANGE_ME` point, and every other knob as a literal with a comment beside it |

The inputs they expect are the patch set on disk — `constant/triSurface/<patch>.stl`,
one file per named patch, plus `patches.json` — and nothing else.

## The recipe, as commands

```bash
cp .toolbox/templates/snappy/*Dict  <case>/system/
python3 .toolbox/patch_entries.py <case> \
        --insert system/snappyHexMeshDict \
        --insert system/surfaceFeatureExtractDict
# replace the two things nothing can guess: the background box corners in
# blockMeshDict, and locationInMesh in snappyHexMeshDict
python3 .toolbox/preflight.py <case> --checks change_me   # names any that are left
blockMesh && surfaceFeatureExtract && snappyHexMesh -overwrite && checkMesh
python3 .toolbox/mesh_look.py <case>
```

`snappyHexMesh` also reads `system/fvSchemes` and `system/fvSolution` even though it
solves nothing, and stops with a fatal error if either is missing. A `fvSchemes` with
the six empty scheme blocks and a `fvSolution` with an empty `solvers {}` is enough to
get past it.

## What nothing can guess, and what is guessed on purpose

Two values in these dictionaries are `CHANGE_ME`, and it is a literal token in the file
rather than a check inside a script, so it survives the dictionary being copied
somewhere else:

- **the background box** — `boxMin` / `boxMax` in `blockMeshDict`. For an internal flow
  it is the surface bounds grown by a cell or two; for an external one it is a
  wind-tunnel decision. `surfaceCheck <stl>` prints the bounds it has to contain.
- **`locationInMesh`** — the point that says which side of the surface is fluid.
  `domain_probe.py --suggest` proposes one and validates it.

Everything else is a literal with a comment beside it: `nCellsBetweenLevels 3;`,
`maxNonOrtho 65;` beside the note that preflight warns at 70 and fails at 85,
`minThickness 0.1;` beside the note that this is the knob that silently shrinks a layer
coverage number. They are stated rather than imported, so editing one is editing a
number and not guessing what a default was hiding.

## Gotchas, baked in rather than rediscovered by failing

**`castellatedMesh`, `snap` and `addLayers` are three stages, and failing late still
exits 0.** Each reads the mesh the one before it wrote. A snap that did not converge
and a layer stage that inserted layers and then deleted them again both leave a mesh
behind and both return zero to the shell. The stages are individually switchable in the
dictionary for exactly this reason: castellate and snap to a clean `checkMesh` before
`addLayers` is worth turning on at all.

**Layer coverage is read off the mesh, never off snappy's own summary.** On one hull
snappy reached 88.6% coverage on its first growth iteration, eroded it to 66.7% by its
fiftieth, printed 66.7% and exited 0 — and the mesh it left had two thirds of a layer on
it. `layer_report.py` measures the first-cell height at each patch face off `polyMesh`
and compares it against the same wall meshed with no layer request:

```bash
python3 .toolbox/layer_report.py <case> --patch walls --ref <case-with-no-layers>
```

The number that came out of the mesh is the number. What snappy said about itself is a
log line.

**Normal orientation is how snappy decides inside from outside, so an inverted patch
turns a solid into a void.** The exported union has to be closed with consistent
outward normals, one coordinate frame, no per-file transforms; one patch wound the
other way and snappy meshes the wrong side of that surface and exits 0. The measurement
that catches it is arithmetic on the mesh: the cell count of an internal-flow mesh is
about `fluid_volume / dx^3`, and a mesh of the outside is nothing like it.
`cad_audit.py` checks closure and winding on the surfaces themselves.

**Every surface has to appear in `geometry`, in `features` and in
`refinementSurfaces`, by the same name.** A surface left out of `refinementSurfaces` is
still meshed; its faces land in whatever patch snappy defaults to, they take that
patch's boundary condition, and the run finishes. That is a wrong answer rather than an
error, and it is why `patch_entries.py` writes all three lists from one reading of the
directory instead of leaving twenty patches to be typed three times. `boundary` after
a good run carries the patch names that were exported and no `defaultFaces`.

**A real chassis outruns a step window and belongs in the background.** The duct in
`tests/data/snappy/duct/` meshes in about a second; a vehicle at a useful cell size is
minutes to hours, and a foreground call that is killed at the step timeout leaves a
half-written `constant/polyMesh` that reads like a mesh. Redirect it, background it, and
come back to the log:

```bash
nohup snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 &
tail -5 log.snappyHexMesh          # later, in another step
```

`cells_estimate.py` predicts the count from the dictionary beforehand, which is the
cheap way to find out that a level was one too high.

**Refinement level is a mesh fact, not a dictionary fact.** `level (1 1)` asks for
cells half the background size at that surface; whether it arrived is measurable —
`layer_report.face_spacing(mesh, patch)` is the near-wall size straight off `polyMesh`.

## When cfMesh is the better path

Snappy's layer stage inserts prisms and deletes the ones a quality metric rejects.
cfMesh's `cartesianMesh` extrudes a sheet and optimises it afterwards, and on the hull
above it covered 100.0% of the same wall. `../cfmesh/` holds that recipe.
