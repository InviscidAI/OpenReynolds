# What is in here

Small scripts and some notes, refreshed from the distribution at the start of every
session. They are offered, not imposed: use them, edit them, replace them, or ignore
them. Nothing checks whether you did.

Each takes `--help`.

| | |
|---|---|
| `geometry_view.py` | Surfaces (`.stl`, `.obj`, `.ply`, `.vtk`) drawn from four fixed views with every facet edge and the bounding-box ticks on, plus open edges, non-manifold edges, connected bodies, extents and face-area range. No verdicts: whether 40 open edges matter depends on where they are. |
| `mesh_digest.py` | `checkMesh` output as a metric table — the numbers and the patch table, no thresholds applied. |
| `cells_estimate.py` | Cell count predicted from the STL and the snappy dictionaries, before the build rather than after it. |
| `log_digest.py` | Solver log to a residual plot, a last-iteration table, and the continuity and bounding summary. |
| `render.py` | Fixed pyvista scenes for a case: mesh cuts and field slices, as PNGs, with cameras that do not move between runs so two of them are a visual diff. |

Anything that writes a PNG is worth knowing about twice over, because `read_file` on an
image path hands the picture back to you rather than its bytes — so a render is
something you can look at, not only something you can produce.

`notes/openfoam-field-notes.md` is OpenFOAM practice written as field notes rather than
procedure. `notes/bundle-layout.md` is one suggested `/work` layout, labelled a
suggestion. `notes/openfoam-agent-architecture.md` is a longer design document, optional
reading.
