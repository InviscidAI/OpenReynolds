# T13 — a helical leak path 0.2 mm wide

**Capability:** authoring, helical sweep, a clearance that is the whole flow path.
**Fixture:** none — authored from the request.

## Request

> A screw-top jar has leaked and we want to see the path. The neck is 40 mm outside
> diameter with a single-start trapezoidal external thread, 2 mm pitch, 1 mm thread depth,
> over 12 mm of neck height. The lid's internal thread matches it but is cut 0.2 mm
> shallower on every flank, so when the lid is screwed fully down a 0.2 mm gap follows the
> thread helix from the inside of the jar to the outside air. Mesh the air in that gap —
> the helical channel only, not the jar's contents. Name the neck flank, the lid flank,
> the inner end of the helix and the outer end separately, and export one STL per patch to
> `constant/triSurface` before meshing.

## Properties the desk must measure and print

- the flank gap, measured between the two meshed flank patches, at three positions a third of a turn apart
- the number of turns the channel makes, and its developed centreline length
- the number of connected mesh regions `checkMesh` reports, and the number expected
- the channel's cross-sectional area at the inner end and at the outer end, measured on the mesh
- every exported patch, exhaustive and disjoint over the domain's faces

## What this catches

**A gap that is the product, not a defect.** Every other clearance case in this corpus asks
whether a narrow gap survived meshing. Here the gap *is* the domain: if it closes anywhere
along the helix the region count goes to two, and if it is opened too far the leak rate is
wrong by the cube of the error. There is no bulk volume to hide in.

**A length at a station, three times, on a surface that is not planar or axisymmetric.** The
baseline and the first bench sweep both found the desk substituting arithmetic on its own
parameters for a width. A flank gap on a helix cannot be recovered that way without
reconstructing the thread analytically, which is itself the tell.

**A helical sweep, which nothing in the corpus has.** T10's blades twist; nothing sweeps a
profile along a helix, and it is the operation both source suites treat as a headline
capability.

## A pass that is really a failure

The thread is modelled as a stack of annular grooves rather than a helix — visually and even
volumetrically close, and it meshes cleanly — but the channel is then a set of disconnected
rings, not one path from inside to outside. The region count catches it, and the developed
centreline length is what distinguishes a real helix from a good imitation.

## Provenance

Adapted from the threaded jar and screw-on lid (06) and hex bolt and nut (03) benchmarks in
[Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM), whose stated point is "two **mating**
threaded parts" and "**real ISO threads**". Both score the threads as present and printable.
The 0.2 mm flank clearance and the leak path are authored here, because a thread that prints
and a thread that seals are different claims about the same geometry.
