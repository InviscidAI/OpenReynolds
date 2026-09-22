# The acceptance prompts

v1's final gate (`#8`). Each is run against the CAD desk on the image; the set passes when
the pass rate is at least the recorded rate of the desk being replaced, with every
individual regression named even where the aggregate holds.

**Twenty-six, not eight.** T1–T8 are the original set. T9–T26 were added on 2026-09-12 from
the two open-source benchmark suites named in §"Where T9–T26 come from" below.

The corpus grew by breadth on purpose. The alternative on the table was repeating three cases
five times each to tighten an estimate on them; fifteen distinct geometries at one run each
tests the same behavioural claims without confounding them with any one case, and the unit of
this corpus is the corpus.

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
| T13 | helical thread leak path | a 0.2 mm gap that *is* the domain; a width on a non-planar surface |
| T14 | tapered NACA wing, external | a profile the desk generated and can therefore restate |
| T15 | flooded bevel gear mesh | two solids in near contact, where OCCT may weld them |
| T16 | turbofan bypass annulus | a non-monotonic area schedule, right at both ends and wrong between |
| T17 | V8 intake manifold | eight near-identical port faces to name, and eight tubes to union |
| T18 | finned cylinder, external | twelve instances of the same narrow gap |
| T19 | shaft seal with keyway | 0.05 mm clearance beside a keyway that hides its loss |
| T20 | pipe flange bore | **the control** — a cylinder, three coaxial bands to tell apart |
| T21 | enclosure with standoffs | a blind hole `checkMesh` cannot distinguish from a through hole |
| T22 | vented brake disc | 36 passages and a count that must come off the mesh |
| T23 | caged ball | a closed solid touching nothing, and where `locationInMesh` lands |
| T24 | swirl chamber | a property that is a direction, not a length |
| T25 | turbofan, no dimensions given | inventing a scale — T6's failure with no file to blame |
| T26 | spiral stairwell, tangent treads | a spec that does not close, resolved silently by a tolerance |

Of the original eight, four exercise authoring, two exercise STEP prep, and two exercise
cases the desk being replaced could not return at all.

T9–T26 exercise authoring against features at the cell scale. **Every one of them requires a
per-patch STL export in the request itself**, which T9–T12 did not — they named patches and
left the export implied, and all four returned `n/a` on every probe as a result. Two of them
(T25, T26) have no mesh as their correct outcome, which doubles the corpus's refusal-path
coverage from one case to three counting T6.

## Where T9–T26 come from

Adapted, with the same honesty the eight are owed, from two open-source text-to-CAD
benchmark suites:

- [Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM) — 13 OpenSCAD benchmarks, GPL-3.0.
  T10 ← blisk (12), T11 ← honeycomb bracket (04), T12 ← impeller (08), T13 ← threaded jar
  (06) and hex bolt (03), T14 ← NACA wing (05), T15 ← bevel gears (07) and planetary stage
  (09), T16 ← turbofan (11), T17 ← V8 (13), T18 ← radial engine (10), **T25 ← turbofan (11),
  quoted verbatim**.
- [Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD) — 10 build123d
  benchmarks P1–P10 with a 141-feature pass/fail record, plus a 10-piece show gallery; MIT,
  sharing its prompt set with
  [earthtojake/text-to-cad](https://github.com/earthtojake/text-to-cad). T9 ← articulable
  gyroscope, T12 ← P8, T18 ← P7, T19 ← P4, T20 ← P2, T21 ← P5, T22 ← S10, T23 ← ball-in-cage,
  T24 ← S9, T26 ← P9.

**Three cases exist for the external comparison.** T12 is MAC's one published failure (P8
item 13, the blade root fillets). T26 is its one published **defensive correction** (P9 item 8,
the tangent tread it fixed unprompted and the `cad` skill did not). T20 is P2, which both
pipelines scored full marks, so it serves as the control. Those are the only points where this
corpus can be set beside a number somebody else measured — and even there the scores are not
comparable, only the behaviour.

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

## What a case may grade, and what it may not

Settled on 2026-09-17, after a re-vet threw out two findings of its own.

A case has three kinds of prose and only one of them is an acceptance test.

**`## Request` is what the desk is judged against.** Everything it is allowed to want goes
in here, including an opinion about *how* -- a requester who says "build the block and cut
the channel out of it, the 3 mm webs are where I expect trouble" is an ordinary customer
with a view, and a desk that ignores it has ignored the request. If a case wants a
particular route exercised, **the request has to ask for it in words.**

**`## Properties, measured on the delivered mesh` is the checklist**, and each bullet has
to be answerable from the request plus the delivered mesh. A property naming a quantity the
request never mentions is either an engineering property of any mesh -- cells across a gap,
one cell thick, layer coverage -- or it is the case wanting something it did not ask for.

**`## What this catches` and `## A pass that is really a failure` are neither.** They are
provenance: why the case was written, which section of the plan it answers, what the author
was afraid of. They cite plan sections and name kernel weaknesses. **They are not criteria
and must not be graded against.**

That distinction was not written down, and the cost of leaving it implicit is on record.
The `core+cad_export` vet failed T4 for building the water volume as a union rather than
by subtraction -- against a request whose own words are "Mesh the water side only", on the
authority of a `What this catches` note citing OCCT's thin-wall boolean. It failed T10 for
a sharp trailing edge, against a request that stated no trailing-edge thickness at all.
Both findings were withdrawn. Three of the six failures in that sweep were the harness's
or the corpus's, and **none of the three was the desk's**.

T4 and T10 have since had their requests rewritten to ask, in the requester's own voice,
for the thing the notes were wishing for. That is the fix: not a stricter vet, a clearer
request.
