# This sweep is void. It is kept, not deleted, and it is not a baseline.

**The API account ran out of credits partway through it.** 21 of the 26 cases never made a
single model call: `provider in 2.1s, 0 cells, $0.00`. Two more died mid-build — T10 at 18
cells and $1.49, T11 at 12 — and exactly one case, T1, ran to completion before the
outage. T8 and T9 had restarted on reloaded credits when the sweep was stopped.

| | |
|---|---|
| label | `core+declare_gate` |
| core sha | `3f5500e` |
| cases | 26 attempted, **1 completed**, 2 partial, 21 never started |
| spend | ~$3.22, nearly all of it T1, T10 and T11 |
| status | **void** |

## Why it is not salvageable, and not partially usable either

The rule for a dirty environment applies here and applies harder: *"one contaminated run
can be discarded, but an environment that was dirty at run 6 was dirty at run 1."* This is
not a noisy sweep, it is a sweep that mostly did not happen. Reading T1 as a result would
be reading a single case as a result, which is the thing `/cad-sweep` says not to do; and
resuming would produce a mongrel where some cases ran before the outage and some after,
on a corpus whose whole value is that every case saw the same desk on the same day.

**The addition is untested by this.** Nothing here says whether the declare gate works,
because the only case that completed is T1 — a `blockMesh` U-bend that exports no
surface, so every advisory probe returns `n/a` and there was nothing for the gate to say.

## One thing it did surface, which is a real defect

`stopped: provider` records **no reason**. Every one of those 21 records carries
`error: ""`, so the record cannot say whether the provider refused for credits, for rate
limits, for a bad key or for an outage. The runner log has the duration and nothing else.
A sweep that dies on the provider should be able to say why without somebody remembering
an email. Worth an addition of its own, and cheap.

## What happens next

Re-run from scratch against the same baseline, `core+bench26-20260912-133719-4bbd`. This
directory stays so the chain can show a gap rather than imply continuity across one.
