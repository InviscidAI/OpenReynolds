# prep — a customer STEP to a tagged fluid domain, exported patch by patch

What is in this folder is this page. There is no script here to copy, edit and run: the
recipe is build123d written in cells, and `b123d_api.md` beside the toolbox scripts is
the exact-signature reference for it. Written out rather than packaged so it can be
read, argued with, and cut down to what the part in front of you actually needs — a
geometry that wants something else is a geometry that wants something else. Ignore them
where they do not fit.

The pieces, and each is a cell or two:

| | |
|---|---|
| import and repair | `gmsh.model.occ.importShapes`, then `healShapes` with its sewing turned off — and the solid count taken across the call |
| checkpoint | the repaired B-rep written to a STEP file, so the cells after it start from a file rather than from a session |
| defeature | cylindrical and toroidal faces identified by radius, handed to OCCT's `BRepAlgoAPI_Defeaturing` |
| select and boolean | `.solids()` is a list whatever came in; pick, fuse, cut, and the overlap measured rather than assumed |
| the fluid domain | a flow volume minus the part, the open ends capped, and the capping faces are the ones that become inlet and outlet |
| tag | `ShapeList` selectors re-derived on this run's geometry, never a face index kept from the last one |
| export | `cad_convert.export_patches()`, one STL per patch plus `patches.json` |

**The B-rep survives all the way to export, and tessellation happens once, at the end.**
That is not a preference. Fillet removal identifies a cylindrical face, deletes it and
reintersects its neighbours; face tagging asks a face what surface it lies on and where
its centre is. Both are face-level questions, and on a triangle mesh both first require
re-segmenting the mesh back into primitives — an open research problem, not a library
call. Tessellate early and the rest of this page stops being available.

**Multi-solid from the start.** `importShapes` hands back a list however many solids are
in the file, and `.solids()` on the build123d side does the same. A single solid is the
n = 1 case of that, not a different path — so nothing here special-cases it, and a second
solid arriving in a customer file six weeks from now changes no code.

---

## Units — three places they bite, all three measured

Everything downstream of here is in **metres**: `clmax`, the STLs, the case. The CAD
files are not, and the conversions are not where anyone expects them.

**`Geometry.OCCTargetUnit = "M"` converts a file that declares a unit and is a no-op on
one that does not.** It converts *from* a declaration, so it cannot carry an override.
A file with no declaration needs an explicit scale; `cad_convert.py` refuses rather than
guess which, and that refusal is the one place in this pipeline where the answer has to
come from the person who owns the part.

**It also outlives the session that set it.** The gmsh option is reset by
`gmsh.finalize()`; the OpenCASCADE static it writes through is not, and that static is
read by the STEP **writer** as well as the reader. Measured: a session that sets it to
`M` and imports leaves every later STEP write in the process multiplied by a thousand
and still declaring millimetres — across two `finalize()` calls. In a kernel that lives
for a whole session that is a standing hazard rather than a curiosity.
`cad_convert.export_patches()` puts the static back when it is done; a cell of your own
that sets the option should check the number in the file it writes afterwards.

**`import_step` does not read the file's unit at all.** It hands back the numbers as
written, so a file declaring millimetres arrives as millimetres whatever it says. The
conversion is yours:

```python
from build123d import *
part = import_step("prep/healed.step").scale(1e-3)   # mm in the file -> metres here
```

And in the other direction, `export_step(shape, path, unit=Unit.M)` means *my numbers
are metres*: it writes a file declaring millimetres whose coordinates are a thousand
times your numbers, which is a **truthful** millimetre file. `Unit.MM` means the numbers
are already millimetres. Either is fine; what is not fine is a metre-scale model written
under `Unit.MM`, which declares millimetres over numbers that are metres and is wrong by
a thousand in a file that says nothing about it. The real fixture in
`tests/data/cad/real/` is exactly that mistake made by a commercial CAD system: it
declares metres and is 54 of them across.

---

## Import and repair, and what repair does when there is nothing to repair

`healShapes` is the repair primitive, and at its defaults it is a damage primitive on a
multi-solid assembly. Measured on a real three-solid STEP: **3 solids became 0**, free
edges went 0 → 354, and **the volume moved by five parts in ten million** — which is the
trap, because a volume check reads that as success. `sewFaces=True` is the single
argument that does it, it is on by default, and a sweep of every other flag and four
tolerances found all of them innocent. Sewing an assembly merges faces across bodies
that were never one shell, the shells stop closing, and `makeSolids` cannot put back
what sewing took apart.

```python
import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setString("Geometry.OCCTargetUnit", "M")     # the file declares a unit
gmsh.model.add("prep")
gmsh.model.occ.importShapes("customer.step")
gmsh.model.occ.synchronize()
before = len(gmsh.model.getEntities(3))

gmsh.model.occ.healShapes(
    fixDegenerated=True, fixSmallEdges=True, fixSmallFaces=True,
    sewFaces=False,          # the one that loses the solids. Measured, not assumed.
    makeSolids=True,
)
gmsh.model.occ.synchronize()
after = len(gmsh.model.getEntities(3))
print(f"solids {before} -> {after}, surfaces {len(gmsh.model.getEntities(2))}")

gmsh.write("prep/healed.step")        # the checkpoint; see below
gmsh.finalize()
```

`before != after` is the result, not an exception: repair that changes the solid count
has changed the assembly, and the honest move is to say so and go on without it rather
than to tune the tolerance until the number comes back. Volume alone will not tell you —
it did not in the measurement above. **Solid count and free-edge count are what catch
it**, and `cad_audit.py` measures closure on the exported surface at the other end.

The other honest limit: the one real multi-solid file in this tree has **nothing wrong
with it** — zero faces flagged by OCCT's checker, zero free edges, three closed solids.
So what repair does to a *broken* file is untested here, and a repair that appears to do
nothing may be telling you the file was already fine.

## The checkpoint is a cache, not a second source of truth

Import and repair is the expensive stage and the one nothing later changes, so it writes
its B-rep to `prep/healed.step` and the cells after it load that file. Deleting the
checkpoint and running the cells again gets the same geometry; it is the run that is
cheaper, not the answer that is different. Nothing is edited in the checkpoint, nothing
is read from it that is not also derivable from the source, and no decision is recorded
there — a file that is only ever written by one cell and read by the next.

```python
part = import_step("prep/healed.step").scale(1e-3)
print(len(part.solids()), "solids", part.volume, "m^3")
```

## Defeature, at the scale the tolerances expect

OCCT's defeaturing is not surfaced by build123d, so it comes from OCP directly. It takes
faces, not edges: a fillet is a cylindrical face of small radius, a corner blend is a
torus of small minor radius, and identifying them by radius is the whole selection step.

```python
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Defeaturing
from OCP.TopoDS import TopoDS

small = []
for face in part.faces():
    kind = str(BRepAdaptor_Surface(face.wrapped).GetType()).rsplit("_", 1)[-1]
    if kind == "Cylinder" and BRepAdaptor_Surface(face.wrapped).Cylinder().Radius() <= 6e-4:
        small.append(face.wrapped)
    elif kind == "Torus" and BRepAdaptor_Surface(face.wrapped).Torus().MinorRadius() <= 1e-3:
        small.append(face.wrapped)

op = BRepAlgoAPI_Defeaturing()
op.SetShape(part.wrapped)
for face in small:
    op.AddFaceToRemove(face)
op.Build()
defeatured = Solid(op.Shape()) if op.IsDone() else part
```

On a right-angle edge the arithmetic is closed-form and worth checking against: removing
a fillet of radius `r` from an edge of length `L` puts back `(1 - pi/4) * r**2 * L`, and
a measured recovery within a per cent of that is defeaturing having reconstructed the
corner rather than approximated it away. It is true for 90° and false for anything else.

**The scale trap, found by falling into it.** OCCT's defeaturing tolerances are
absolute. Run on a model at millimetre magnitudes the same call had not returned after
fifteen minutes; run at metre magnitudes — the model's true size — the whole operation
took seconds. Convert to metres, then defeature.

Offering nine faces and getting five removed is the expected outcome, not a failure:
defeaturing takes what it can and leaves the rest. Count the cylindrical faces before and
after and say which number you got.

## Select among solids, and boolean

```python
solids = part.solids()                               # a list at n = 1 as well
body   = max(solids, key=lambda s: s.volume)
rest   = [s for s in solids if s is not body]

overlap = body & rest[0]                             # interference, measured
print("overlap", overlap.volume, "m^3")              # 0.0 means they only touch
fused = body + rest[0]
assert abs(fused.volume - (body.volume + rest[0].volume - overlap.volume)) < 1e-12
```

A fuse of solids that overlap is not the sum of their volumes, and a fuse of solids that
do not touch is a compound rather than one solid. Both are visible in `.volume` and in
`len(result.solids())` for the price of one line each.

## The fluid domain: subtract, then cap

The fluid is what is left of a flow volume once the part is taken out of it. For an
internal flow the flow volume is a body that plugs the passage and runs past both open
ends; the subtraction cuts it to the passage, and the faces where it was cut off flat are
the **capping faces** that become inlet and outlet.

```python
duct  = Cylinder(radius=0.02, height=0.20, rotation=(0, 90, 0)).locate(Location((0.1, 0.05, 0.05)))
fluid = duct - part                                  # the passage, capped by the ends
print(fluid.volume, "m^3")
```

Two checks cost nothing and catch the boolean that silently did nothing:

- `fluid.volume` against `flow_volume.volume - part.volume` where the part is wholly
  inside the flow volume — arithmetic, to a fraction of a per cent.
- each capping face `is_planar` with the area you can compute by hand. A cap that came
  out curved is a cap taken off the wrong surface.

**Thin walls are where this operation is known to be weak.** A large subtraction against
thin-walled fin geometry is OCCT's documented soft spot, and the failure is not always an
exception — it can be a result with a face missing or a wall gone. `domain_probe.py`
measures minimum wall thickness on the material between two shells, which is the number
that predicts it, and `cad_audit.py` will find an open union afterwards. Report what
happened either way: a boolean that failed on a thin fin is information, and a
tessellated fallback exists for exactly that case.

## Tag: selectors re-derived on this run

Patch names are chosen where the surface is created. A face is named for what it is —
the cap at the upstream end, the wall of the passage, the plane of symmetry — and the
selector that finds it is written against geometry, never against a face index carried
over from a previous run. Face order is OpenCASCADE's: it is a property of the file and
of every operation that has touched it since, so an export keyed on index is correct
until the model changes and quietly wrong afterwards.

```python
faces  = fluid.faces()
inlet  = faces.filter_by_position(Axis.X, -1e-6, 1e-6)
outlet = faces.filter_by_position(Axis.X, 0.2 - 1e-6, 0.2 + 1e-6)
walls  = [f for f in faces if f not in list(inlet) + list(outlet)]
```

`filter_by(GeomType.PLANE)`, `filter_by(Axis.Z)`, `sort_by(Axis.X)`, `group_by()` and
`filter_by_position()` are the rest of that vocabulary; `b123d_api.md` has them with
their signatures. Two runs of the same selectors on the same geometry give the same
partition, and a rigid transform of the model moves every face centre through the
transform and changes nothing about who is in which patch. That property is what makes
the export reproducible, and it is the one worth a check of your own.

## Export

```python
import sys
sys.path.insert(0, "/work/.toolbox")
import cad_convert

export_step(fluid, "prep/fluid.step", unit=Unit.M)

key = lambda fs: [tuple(f.center(CenterOf.MASS)) for f in fs]
report = cad_convert.export_patches(
    "prep/fluid.step",
    {"inlet":  {"role": "inlet",  "faces": key(inlet)},
     "outlet": {"role": "outlet", "faces": key(outlet)},
     "walls":  {"role": "wall",   "faces": key(walls)}},
    "case/constant/triSurface",
    clmax=0.002,                                  # metres, and there is no default
    location_in_mesh=[0.1, 0.05, 0.05],
)
print(cad_convert.render(report))
```

Faces are named by their **centre of mass in metres** — `face.center(CenterOf.MASS)`,
not `face.center()`, which is the bounding-box centre and lands off a cylinder by its
sagitta. Every B-rep face is asserted into exactly one patch **before** anything is
tessellated: a face nobody named is a refusal, because an unassigned face is not dropped
— it lands in whatever patch the mesher defaults to, takes that patch's boundary
condition, and the run finishes with a wrong answer rather than an error. One group may
say `"faces": None` and take whatever is left, which is also how the one-STL command line
is the n = 1 case of the same function.

**`clmax` is the decision this export exists to make explicit, and it has no default.**
It follows the mesh, not the geometry: about half the finest surface cell size at the
wall. Facet finer than the mesh and you pay for triangles the mesher cannot resolve;
facet coarser and the faceting *becomes* the geometry — the mesh reproduces the flat
spots faithfully and they reach the pressure field looking like physics. The report says
what came out: the tightest curve found, the sagitta that implies, the facet count, the
longest edge produced, and the gmsh calls in order.

`location_in_mesh` goes into the manifest and is the point that says which side of the
surface is fluid. `domain_probe.py --suggest` proposes one and measures its clearance;
`cad_audit.py` then measures the exported set as one surface, which is the only level at
which closure means anything.
