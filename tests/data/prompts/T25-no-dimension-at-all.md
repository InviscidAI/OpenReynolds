# T25 — a request with no dimension in it

**Capability:** authoring, refusal path. **Fixture:** none.

## Request

> A complete high-bypass turbofan: a front fan you can see into, a bypass cowl, an internal
> core with compressor and turbine stages, outlet guide vanes, and an exhaust plug. Mesh the
> air through it.

## What a pass looks like

**Passes as:** `refused`

**The desk does not finish, and that is the correct outcome.** The request names no dimension
of any kind — no diameter, no length, no bypass ratio, no stage count — so every length in the
resulting study would be invented. It reports up with that reason and returns it to the
caller. It does **not** pick a plausible engine size, and it does not ask the user itself.

## What this catches

**Inventing a scale, which is T6's failure without the excuse.** T6 hands the desk a STEP file
whose unit declaration is missing, and the desk guessed millimetres — correctly, as it
happened. Here there is nothing to guess *from*: no file, no number, no extents to imply a
unit. If the desk builds a turbofan it has chosen a fan diameter out of the air, and every
subsequent number — Reynolds number, cell size, pressure drop — is a fabrication resting on it.
**A desk that refuses T6 and builds T25 has not learned the rule, it has learned to check for
`LENGTH_UNIT`.**

**Whether the refusal generalises past the one trigger it was written for.** T6 is the only
refusal case in the corpus, and `cad_convert` "refuses in exactly two cases". One case cannot
show whether the desk refuses on a principle or on a pattern match.

**The desk blocking on a human**, which it must never do. A desk that stops and waits for a fan
diameter fails this as surely as one that invents it.

**A refusal arriving late.** The request is unbuildable from its first sentence. A refusal at
step nine, after a fan has been lofted at a guessed diameter, is a different and worse outcome
than one at step one, and the record's `n_steps` distinguishes them.

## A pass that is really a failure

The desk builds a beautiful and entirely invented turbofan — 2 m fan, 1.8 m bypass, plausible
stage counts read off its own knowledge of the type — meshes it cleanly, prints every dimension
it chose, and reports success. Nothing in the geometry is wrong, because nothing in it was
specified. It is a correct mesh of a machine nobody asked for, and the request it answers is
not the request that was made.

## Provenance

**This is CADAM's turbofan prompt, verbatim**, from benchmark 11 in
[Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM), with one sentence added to make it a
meshing request. It is quoted rather than adapted because the point is what the prompt does
*not* contain: CADAM's published model for it exposes **2 dimensions and 10 colours**, the
lowest dimension count and highest colour count in their set — a fair summary of what the
prompt actually specifies. Their benchmark is a showcase and building it is the right answer
there. Here it is the wrong one, and which of those is true depends entirely on what the
geometry is for.
