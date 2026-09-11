# CAD acceptance — the eight prompts, run (C10)

v1's final gate (`#8`), with the `OPENREYNOLDS_MESH_TOOL=0` A/B beside it.

## Where this was run, which changes what it proves

**These runs measure the CAD desk on the dev machine, not on the shipped image.** The plan
frames `#8` as the eight prompts "run against the desk on the image". That is not what is
below. The harness (`scripts/cad_accept.py`) still has its hosted arm, unchanged and
working; the runs reported here were taken with `--local`, against `LocalBackend` over the
OpenFOAM v2512 / cfMesh / gmsh / build123d / OCP 7.9.3.1 installation on this machine. Every
number below is a measurement of the desk's behaviour. None of them is a measurement of the
image.

The reason is the first item in *still owed*, and it is a blocker rather than a preference.

## Still owed, plainly

### 1. The image ships no Jupyter kernel — blocking, and outside this tree

The pinned workspace image carries IPython and traitlets but **not** `jupyter_client`,
`ipykernel`, `pyzmq`, `tornado`, `jupyter_core`, `comm` or `debugpy`. C8a's kernel channel
drives every cell through `jupyter_client.manager.KernelManager`, so on the image the desk
answers

    no kernel on this workspace: No module named 'jupyter_client'

and **all eight prompts fail before a model is ever called**. The image is also
network-sealed: `pip install` there answers "needs the network, which is sealed in this
sandbox", so the dependency cannot be satisfied at run time by the thing that needs it. The
only way the hosted arm ran at all was carrying wheels in from the dev machine and putting
them on `sys.path` behind site-packages via a `.pth` file — reproducible, recorded in
`ensure_kernel_deps`, and **not a fix**.

C9 declared `ipykernel` and `jupyter_client` in `pyproject.toml` and `ENVIRONMENT.md`, so the
repository now states what the image owes. The image is outside the repository. **Until it is
rebuilt with a kernel, nothing on it can run a cell**, and running locally sidesteps that; it
does not resolve it.

### 2. `scripts/cad_probes.py` has not been re-run on the image

C1's six answers are *image-of-record* claims measured on this machine's **OCP 7.9.3.1.1**
against the image's **7.8.1**. A local re-run cannot satisfy that gate — it would only
re-measure what C1 already measured, on the same OCP, and report agreement with itself. The
harness refuses `probes --local` for exactly that reason rather than producing a green
number that means nothing. **This gate is outstanding.**

### 3. Acceptance-on-image is outstanding

A local pass does not certify the image. Item 1 has to be closed first; then
`python3 scripts/cad_accept.py desk` (no `--local`) is the run that would.

## The caveat that travels with every number here

**The eight prompts are authored replacements, not the originals.** The set the current
desk was accepted on is not in the repository and neither are its T-numbered runs; they
existed only in a chat history and were judged unrecoverable on 2026-09-11.
`tests/data/prompts/README.md` records that in full. Two consequences, and they attach to
every figure below wherever it is quoted:

1. **A pass rate on this set is not like-for-like with the old desk's.** The comparison is
   indicative, not continuous.
2. **The old desk's pass rate is not recoverable at all.** §7's complaint is precisely that
   the evidence is not in the tree, and it is still not: searching the repository and its
   history finds the *durations* of the old runs quoted in prose — successes in 1.9 to 5.2
   minutes, one 10.6-minute aerofoil as the longest success, one 942-second failure that
   produced nothing — and **no pass count and no denominator anywhere**. So the aggregate
   below stands alone, with that stated, rather than beside an invented baseline. Inventing
   one would be the failure §7 names, wearing a better disguise.
