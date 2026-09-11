# The eight acceptance prompts

v1's final gate (`#8`). Each is run against the CAD desk on the image; the set passes when
the pass rate is at least the recorded rate of the desk being replaced, with every
individual regression named even where the aggregate holds.

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

Four exercise authoring, two exercise STEP prep, and two exercise cases the desk being
replaced could not return at all.

## How each is judged

The desk's own finish check is the floor and is not restated per prompt. What each file adds
is the **properties the request names**, which the finish check cannot know and which the
desk is required to measure and print — "a property you did not measure is a property you
did not build". A run that passes the finish check and misses a named property is a fail
with the property named.

Wall clock and token spend are recorded for every run, passing or not. The desk being
replaced finished its successes in 1.9 to 5.2 minutes, with one 10.6-minute aerofoil as the
longest success on record and one 942-second failure that produced nothing at all.
