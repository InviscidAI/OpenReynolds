# The acceptance prompts

v1's final gate (`#8`). Each is run against the CAD desk on the image; the set passes when
the pass rate is at least the recorded rate of the desk being replaced, with every
individual regression named even where the aggregate holds.

**Twelve, not eight.** T1–T8 are the original set. T9–T12 were added on 2026-09-12 from
the two open-source benchmark suites named in §"Where T9–T12 come from" below, each one
carrying a gap that the first sweep's §6 named and the original eight could not reach.

## Provenance, stated plainly

**These are authored, not recovered.** `docs/cad-agent-build-plan.md` §7 records that the
original eight "are not in the repository, and neither are the T-numbered acceptance runs" —
they existed only in a chat history and were judged unrecoverable on 2026-09-11. These eight
were written on that date to cover the same ground.

The consequence has to travel with the number: **a pass rate measured on this set is not
like-for-like with the old desk's**. Where the two are compared, both facts are stated, and
the comparison is treated as indicative rather than as a regression test. §7's complaint was
that "a gate whose test set exists only in a chat history is not a gate"; substituting a new
set quietly and reporting the number as continuous would be the same failure wearing a
better disguise.

## What the set covers

| | prompt | the failure it exists to catch |
|---|---|---|
| T1 | 2D U-bend | a "2D" case meshed several cells deep; patches named by bounding box |
| T2 | body in a flow box | blockage and domain extent; layer coverage claimed rather than measured |
| T3 | Tesla valve | the failure that ended the previous stack — "circles connected by a line" |
| T4 | cold plate with fins | thin-wall booleans, OCCT's known weak case; minimum width unmeasured |
| T5 | STEP assembly, fluid domain | ingest, repair, multi-solid selection, tagging, per-patch export |
| T6 | STEP with no declared unit | guessing a unit — a factor of 1000 on every length in the study |
| T7 | sealed cavity driven by cell zones | a correct one-patch mesh refused by a two-patch rule |
| T8 | conjugate two-region case | a correct multi-region mesh reported as "nothing has been meshed yet" |
| T9 | flooded bearing, 1 mm clearance | a gap narrower than the cell, bridged silently, with the region count still right |
| T10 | one turbine blade passage | a trailing edge thinner than the cell, rounded to fit, every printed number still agreeing |
| T11 | honeycomb flow straightener | a boolean over ~100 cutters that loses a few, invisibly to `checkMesh` |
| T12 | centrifugal impeller passage | a root fillet that fails to apply and is not reported — MAC's one published failure |

Of the original eight, four exercise authoring, two exercise STEP prep, and two exercise
cases the desk being replaced could not return at all. T9–T12 exercise authoring against
features at the cell scale, and are the corpus's first cases that export a surface without
ingesting a STEP.

## Where T9–T12 come from

Adapted, with the same honesty the eight are owed, from two open-source text-to-CAD
benchmark suites:

- [Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM) — 13 OpenSCAD benchmarks, GPL-3.0.
  T10 from its axial turbine blisk (12), T11 from its honeycomb bracket (04), T12 from its
  centrifugal impeller (08).
- [Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD) — 10 build123d
  benchmarks P1–P10 with a 141-feature pass/fail record, MIT, sharing its prompt set with
  [earthtojake/text-to-cad](https://github.com/earthtojake/text-to-cad). T9 from its
  print-in-place articulable gyroscope, T12 also from its P8.

**What was taken is the geometry, not the case.** Both suites score a *solid model for
printing*: pass means the feature is present, the body is watertight, the part comes off
the plate. This corpus scores a *meshable fluid domain*: pass means `checkMesh` is clean
and every named property was measured and printed. Those are different questions of the
same shape, so a prompt cannot cross over — the vase, the bolt and the V8 have no fluid
domain and nothing for this desk to be right or wrong about. What crosses over is the
*topology*: a twisted blade, a hundred-cell lattice, a 1 mm clearance, a fillet OCCT
refuses. Each of the four re-poses one of those as an internal-flow case, states the
dimensions it inherited, and adds the fluid domain, the patches and the property list
here.

**One of the four is a deliberate overlap.** T12 is MAC's P8, whose item 13 — the blade
root fillets — is the single failure in its published 140/141. It is in the corpus so that
one point of the sweep is comparable to a number someone else measured, rather than only
to our own baseline.

No code, prompt text, or licensed material from either repository is vendored into this
repository; CADAM's GPL-3.0 makes that a decision rather than a convenience, and it was
not taken.

## How each is judged

The desk's own finish check is the floor and is not restated per prompt. What each file adds
is the **properties the request names**, which the finish check cannot know and which the
desk is required to measure and print — "a property you did not measure is a property you
did not build". A run that passes the finish check and misses a named property is a fail
with the property named.

Wall clock and token spend are recorded for every run, passing or not. The desk being
replaced finished its successes in 1.9 to 5.2 minutes, with one 10.6-minute aerofoil as the
longest success on record and one 942-second failure that produced nothing at all.
