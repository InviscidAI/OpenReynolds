# What is on this instance

Facts about the sandbox you are in, so nothing here has to be rediscovered by trying
it. Like everything in the toolbox this is a description, not an instruction.

## The network is sealed

There is no outbound network. `pip install` and `apt-get install` cannot reach an index
and will fail — fast, on purpose, with a pointer back here — so reaching for a package is
a dead end, not a slow success. Everything below is already installed. If something you
want is genuinely not here, it is a constraint to work within, not to install around.

## Python (`python3`)

One interpreter, and `python3` in a shell is it — the one the renderers use. Installed:

- **numpy**, **matplotlib**, **pandas**
- **pyvista** with **vtk-osmesa** — headless rendering through OSMesa
  (`PYOPENGL_PLATFORM=osmesa` is already set). `pyvista.OpenFOAMReader` reads a `<case>.foam`;
  when it chokes on a `0/` field that is a `$variable`, a `flowVelocity`/`Tinf` symbol or an
  `#includeEtc`, `render.py` falls back to `foamToVTK` — reach for `render.py` before writing
  pyvista by hand.
- **imageio** with the ffmpeg plugin, and a system **ffmpeg**/**ffprobe** — so frames can be
  turned into a `.gif`/`.mp4`/`.webp` *on this instance*, not only on your machine. `encode.py`
  does exactly that from a `*_frames/` directory, so a finished animation need not be hand-written.

- **gmsh** (the Python module, `import gmsh`) — the OpenCASCADE geometry kernel and the mesher
  in one: `gmsh.model.occ` builds boxes, cylinders, spheres, cones, booleans, fillets, imports
  STEP, and meshes the result body-fitted; `gmshToFoam` turns the `.msh` into a polyMesh, and
  types every patch `patch`, so walls are retyped afterwards with `foamDictionary` or
  `createPatch`. The same kernel builds a *planar* face from 2D primitives, meshes it in quads
  and extrudes it one cell thick into hexahedra with the two z faces typed `empty`, which is
  how a plane case is meshed here. Name a physical group per surface as you build it and the
  patch names come out of the geometry rather than out of where a face happens to sit.
- **build123d** — parametric CAD in readable Python on the same OpenCASCADE kernel (via OCP):
  `Cylinder(5, 40) - Cylinder(4, 40)` is a penne, `export_step(part, "body.step")` writes the
  B-rep gmsh imports with `merge`/`importShapes` and meshes. This is the editable layer: a
  solid as a short script, and STEP as the file any CAD program opens.
- `mesh_look.py` in the toolbox draws any meshed case with one colour per patch and prints
  its cells, bounds, patch areas and normals and checkMesh's verdict — the fastest way to
  see whether a mesh is the shape that was meant and whether the inlet is the end you think.

Not installed: **scipy**, **PyMuPDF/`fitz`**. For PDFs use the poppler tools below, not `fitz`.

## Binaries on PATH

- **OpenFOAM 2512** (ESI): the classic solver and utility names — `blockMesh`,
  `snappyHexMesh`, `simpleFoam`, `pimpleFoam`, `interFoam`, `foamToVTK`, `checkMesh`,
  `decomposePar`, `reconstructPar`, `postProcess`, … cfMesh's executables are here too.
- **HiSA** — implicit density-based shock-capturing solver, on PATH with no exports; see
  `hisa_env.py` and its ONERA M6 example case.
- **gmsh** — geometry and meshing.
- **poppler-utils**: **`pdftoppm`** (PDF page → image, `-r` for DPI) and **`pdftotext`**.
  This is the PDF path; there is no `fitz`.
- **mpirun** (OpenMPI) — parallel solves work as root out of the box
  (`OMPI_ALLOW_RUN_AS_ROOT` and `PMIX_MCA_gds=hash` are set for you; no `--allow-run-as-root`
  needed, and PMIx will not fail to enumerate interfaces under the sealed network).

## Where compute runs

The image runs as **root**. Long work belongs in a **job** (`job_start`), which is detached
and survives; a synchronous `bash` command that runs past ~150 s is cut off at the edge, so
mesh, solve and `reconstructPar` on a real case belong in a job, not a foreground exec.

See `README.md` for the scripts that already do the common jobs (rendering, animation,
geometry views, case generation, digests) — most of what gets written by hand is in there.
