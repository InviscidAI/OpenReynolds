# T1 — 2D U-bend

**Capability:** authoring. **Runtime expectation:** a few minutes.

## Request

> A plane (2D) U-bend duct in water. The passage is 10 mm wide throughout. Both legs run
> horizontally, 120 mm long, and are joined by a 180 degree bend whose centreline radius is
> 15 mm. The inlet is the open end of the lower leg and the outlet is the open end of the
> upper leg. Mesh it for an incompressible solve.

## Properties the desk must measure and print

- passage width, 10 mm, measured on the mesh at the inlet and at the crown of the bend
- centreline bend radius, 15 mm, measured on the mesh
- the mesh is exactly one cell thick, and both side patches are typed `empty`

## What this catches

**The 2D trap.** "OpenFOAM has no 2D mesh, only a 3D mesh one cell thick with the two faces
normal to the thickness typed `empty`. The common way a 2D case is actually a slow, thin 3D
one is a mesh built several cells deep, or a side patch left `patch`/`wall` where `empty`
was meant." A mesh three cells deep passes `checkMesh` and solves, slowly and wrongly.

**Patch naming by position.** Inlet and outlet are both at the same end of the bounding box
in x and differ only in y. A desk that classifies patches by where a face sits in the
bounding box labels one of them wrong and says nothing about it — the failure the brief's
"name the patches where you create the surfaces" rule exists for, and the reason a U-bend is
the shape that exposes it.

## A pass that is really a failure

The two side patches carry names somebody chose but are typed `patch`. `checkMesh` is happy,
the picture looks right, and the solve is a three-dimensional one nobody asked for.
