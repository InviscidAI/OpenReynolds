# Void, and it bought something: the addition it measured was broken.

This sweep was cut short when the session that launched it exited — 10 of 26 cases, 8 of
them complete, $8.54, **no provider failure this time**. It is void for the ordinary
reason that 8 of 26 is not a corpus result. It is kept for the other one.

## What the eight completed runs showed

The declare tool was used on every one of them, and the gates ran:

| state | count |
|---|---|
| `n/a` | 26 |
| `clean` | 10 |
| `warned` | 9 |
| `xpass` | 3 |
| `xfail` / `waived` | **0** |

Four cases drew real warnings — T10 (63 free edges), T16 (1,096), T12 (2,177), T17
(4,257), each with crossing triangles alongside — and the gate's numbers matched the
supervisor's independent reading exactly, which is the cross-check working.

**Not one of those warnings was ever delivered to the desk.**

```python
check, advisory = self._declare(declare, case_rel, request)
if check.ok:
    result.ok = True
    break          # the advisory is computed here and dropped
```

The loop broke on a passing `checkMesh` before handing the advisory back, and a passing
`checkMesh` is the path all eight took. So the addition reproduced, one layer up, exactly
the failure it was built to close: `union_closure` measures the truth, the record keeps
it, and nothing tells the desk. T26's 259 free edges and T17's 4,257 are the same story.

The three `xpass` entries are the tell that nothing was delivered. Every waiver in the
sweep was a blind pre-emptive prediction, because no desk ever saw a warning to react to.

## And the xpass reasons correct a claim I made for the design

All three predicted a warning that did not come, and all three reasoned correctly about
the geometry while being wrong about the check:

> *"Each of the four patch STLs is individually an open surface by construction; only the
> union…"* — T13, on `union_closure`

That is true of the geometry and irrelevant to the probe, which welds the union precisely
so that individually-open patches are fine — `probes.py` says so in as many words:
*"open edges on the welded union, never per file: individual patch files are open surfaces
by construction and a per-file check passes nothing real."* T14 made the same prediction
and the same mistake, and its `location_in_mesh` xpass is the same shape.

So **`xpass` measures whether the desk can predict the check, not whether it understands
its geometry** — which is what it was claimed to measure. The desk understood its
geometry in all three cases. It had not been told how the check works, and nothing in the
tool description or the brief tells it.

## What happens next

Fixed in the commit that follows this file: a clean `checkMesh` with a warning the desk
has not been shown holds the finish for exactly one turn, so the desk can fix it, waive
it with a reason, or declare again unchanged — which is itself the answer, and is what a
second declare in the record means. Once per check, so it cannot loop; `_told` is what
remembers.

Three tests pin it, the first being the one that should have been written before this
sweep ran: `test_a_passing_checkmesh_does_not_swallow_the_advisory`.

Re-run from scratch. The 8 runs here measured a version of the addition that is now gone.
