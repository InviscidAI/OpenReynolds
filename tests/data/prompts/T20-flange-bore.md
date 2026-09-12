# T20 — the simplest possible case, and six patches to name

**Capability:** authoring, patch naming where the geometry is trivial.
**Fixture:** none — authored from the request.

## Request

> Mesh the water in a bolted pipe flange. The flange is 180 mm outside diameter and 20 mm
> thick, with a 60 mm diameter central bore, and it is bolted to an identical flange with a
> 2 mm gasket between them, so the bore runs 42 mm from face to face. Six 14 mm bolt holes
> sit on a 140 mm pitch circle; they are dry and not part of the flow. The bore is filled
> with water entering one end and leaving the other. Mesh it. Name the inlet, the outlet,
> the two flange bore walls and the gasket's inner face separately, and export one STL per
> patch to `constant/triSurface` before meshing.

## Properties the desk must measure and print

- the bore diameter, measured on each of the two meshed flange bore patches separately
- the face-to-face length, measured on the mesh, and the gasket face's axial position within it
- the water volume, against the analytic cylinder
- the bolt holes' absence from the flow domain, established on the mesh rather than asserted
- the number of connected mesh regions `checkMesh` reports, and the number expected
- every exported patch, exhaustive and disjoint over the domain's faces

## What this catches

**A control case, and the corpus has none.** Every other case here is hard somewhere, which
means a failure is ambiguous between the desk and the difficulty. This geometry is a cylinder
42 mm long and 60 mm across. If the desk substitutes arithmetic for a measured bore diameter
*here*, the pattern the last two sweeps found is not about hard geometry at all. If it
measures properly here and not elsewhere, that localises the failure to difficulty. **Either
outcome is informative, which is what a control is for.**

**Five patches over a shape with four natural faces.** The two flange bore walls and the
gasket inner face are three coaxial cylindrical bands that a selector keying on surface type
and axis cannot tell apart — they differ only in axial position. T1's recorded failure was
"patches named by bounding box"; this is the minimal geometry that reproduces the temptation.

**A feature that must be shown absent.** The bolt holes do not touch the flow, so the honest
answer establishes on the mesh that nothing of them appears in it. Proving an absence is a
different act from measuring a presence and nothing in the corpus asks for one.

## A pass that is really a failure

The desk exports one `walls.stl` covering all three bands, or names them by bounding box so
the gasket face and a flange band swap, and everything else is perfect. The volume is right,
the length is right, `checkMesh` is clean, and a boundary condition applied to the gasket face
lands on a flange instead. On a cylinder this is invisible; on a real case it is a wrong
answer with no symptom.

## Provenance

Adapted from the circular flange benchmark that is P2 in
[Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD) — "circular flange
+ central bore + 6 bolt holes + double fillet", 10 features, scored 10/10 by both MAC and the
`cad` skill and their second-cheapest prompt. It is here precisely because it is the easiest
thing either suite contains.
