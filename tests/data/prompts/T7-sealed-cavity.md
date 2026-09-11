# T7 — sealed cavity driven by cell zones

**Capability:** authoring. **Runtime expectation:** short; the shape is simple and the point
is elsewhere.

## Request

> A sealed rectangular cavity of air, 100 mm x 60 mm x 40 mm, for a natural convection solve.
> There is no inlet and no outlet — every boundary is an adiabatic wall. Inside it, mark two
> cell zones I can put a heat source and a heat sink on: a 20 mm cube near the floor at one
> end called `heater`, and a matching one near the ceiling at the other end called `cooler`.

## Properties the desk must measure and print

- cavity dimensions, 100 x 60 x 40 mm
- both cell zones present by name, with their cell counts and their centroids
- zone size, 20 mm cube each, and their positions relative to floor and ceiling
- the boundary is a single named patch covering the whole cavity

## What this catches

**A correct mesh refused by a rule that assumed through-flow.** The desk being replaced fails
any mesh with fewer than two named patches — "a flow case needs at least an inlet, an outlet
and walls" — which is true of through-flow and false of buoyancy-driven flow. This cavity
legitimately has one patch, `checkMesh` passes, the mesh is right, and the old desk cannot
finish. It is `#17`'s sentence in a second place: not absent support, but a gate rejecting a
valid result.

**Cell zones invisible in the verdict.** Here the zones *are* the mechanism. A finish check
that reports patches and cell counts and never mentions `heater` or `cooler` is describing
half the mesh.

**The warn that must not become a fail.** One patch earns a stated assumption, not a
refusal: consistent with a domain driven from inside, not consistent with wall-temperature
buoyancy (which needs a hot patch and a cold patch) or with through-flow. The run finishes
with the warning attached.

## A pass that is really a failure

The zones exist with the right names and the wrong cells in them — a `topoSet` box that
caught the floor cells as well as the cube. The cell count and centroid are what show it.
