# T8 — conjugate two-region case

**Capability:** authoring, multi-region. **Runtime expectation:** medium.

## Request

> A 30 mm square duct 200 mm long with a solid copper block set into its floor halfway along
> — 40 mm long, 30 mm wide, 10 mm thick — for a conjugate heat transfer solve. I need the
> fluid and the solid as separate mesh regions with a conformal interface between them, so I
> can run `chtMultiRegionFoam` on it. Inlet and outlet at the two ends of the duct.

## Properties the desk must measure and print

- duct cross-section 30 x 30 mm and length 200 mm
- block dimensions 40 x 30 x 10 mm, and its position along the duct
- both regions present, named, with their cell counts
- the interface between them is conformal — matching faces, no gap and no overlap
- `checkMesh` passes **for every region**

## What this catches

**A correct multi-region mesh the old desk cannot return.** `mesher/check.py` fails on a
missing `constant/polyMesh`, so it "cannot return a multi-region mesh however correctly it
built one", and it reports "nothing has been meshed yet" about a mesh that exists — "a wrong
diagnosis, not merely an unhelpful one". `checkMesh -region` is invoked nowhere in the
package today.

**A CHT case is reachable but unassisted** (`#17`). The desk has bash, root, every utility
including `splitMeshRegions`, and a populated `$FOAM_TUTORIALS`; it gets no leverage from the
toolbox while doing this, because `case_gen.py`, `preflight.py`, `mesh_look.py`,
`first_look.py`, `layer_report.py` and `locate.py` all hardcode the singular mesh path. This
prompt measures how expensive that is in practice, which is worth knowing before deciding
whether to fix it.

**Conformality, which `checkMesh` per region cannot see.** Two regions can each be perfectly
valid and not share a face. Only a check across the interface finds it.

## A pass that is really a failure

Two regions, both clean, and the interface patches do not match face for face — the solve
runs and the heat flux across the interface is wrong by an amount nobody can see in a
picture.
