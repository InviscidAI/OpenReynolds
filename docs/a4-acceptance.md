# A4 acceptance — the elbow, free-form

**Study `20260823-213712-babc`** · instance `974f4406` · `claude-opus-5`, effort `high`
· 212 messages across two sessions · 2026-08-23/24

What Plan 2 §13 asked for:

> "simulate airflow through an L-shaped junction" → the agent clarifies as *it* sees
> fit, authors geometry and case, meshes, solves, sanity-checks however it chooses,
> delivers numbers + renders + a write-up. **Success includes the negative check: zero
> forced stops, zero harness-imposed ordering occurred; it asked the user only when it
> actually wanted to**

## Verdict

**The negative check passed, twice.** Across 212 messages and two sessions, exactly two
harness-authored messages entered the conversation — one job-status line and one resume
blurb, both factual, both recorded as `event` rather than `user`. No forced ordering, no
tool call refused on policy grounds, no injected checklist, no grading of output, no
forced stop. Every `is_error` tool result was a fact about the world, never a refusal.

**The CFD was delivered, with the limitations stated.** The study ended without a
converged steady state — the flow settles into a limit cycle — and the agent said so
before it was asked.

## The numbers

Loss coefficient between the monitor planes (2 D_h upstream, 15 D_h downstream), time-
averaged over the second half of each run. Uncertainty is the standard error of twenty
block means, which accounts for the sample-to-sample correlation an ordinary standard
error would ignore.

| mesh | cells | iterations | window | K_raw | spread |
|---|---|---|---|---|---|
| coarse | 100,048 | 20,000 | 10k–20k | **1.622 ± 0.009** | sd 0.164, range 1.23–2.21 |
| medium | 358,400 | 20,000 | 10k–20k | **1.570 ± 0.008** | sd 0.137, range 1.18–2.02 |
| fine | 1,209,600 | 3,100 | 1.5k–3.1k | 1.553 ± 0.020 | sd 0.110, range 1.35–1.92 |

Averaging is what made the grid comparison legible. On the 6,000-iteration run the
mesh-to-mesh difference was *smaller* than the oscillation amplitude, so the ladder said
nothing; over 20,000 iterations the differences (0.052, then 0.017) are several times the
standard error, and the trend is monotone and settling.

Other measured quantities, medium mesh:

- **Mass conservation** — 0.12052 kg/s in, 0.12052 out, against a 0.12050 target.
- **Dimensional check** — `pTotal` verified numerically to be ρ(p + ½|U|²) in pascals, max
  deviation 6×10⁻⁵ Pa. This is the kinematic-vs-dynamic pressure trap the field notes
  flag; the agent pinned it down rather than assuming it.
- **Inlet friction factor** — 0.0188 against Blasius 0.0197, confirming the recycling
  mapped inlet delivers a properly developed profile.
- **Reattachment** — L/D ≈ 0.60 off the inner wall.
- **y⁺ on walls** — min 1.8, max 60, mean 25.7.

### Against the literature

The agent named its reference band *before* it had a result: 1.1–1.3 for a sharp 90°
mitred elbow (Idelchik diagram 6-7; Crane TP-410 ≈ 1.14; ASHRAE ≈ 1.2).

K_raw includes the duct friction between the two monitor planes. At the 6,000-iteration
snapshot the agent separated the two by extrapolating the fully-developed total-pressure
gradients back to the corner, giving K_raw 1.803 → K_exc 1.403. Applying that same
proportion to the time-averaged K_raw of 1.570 suggests a friction-corrected **K of
roughly 1.2**, inside the handbook band.

That last figure is an estimate, not a computed result: the friction correction was
fitted at one snapshot and has not been recomputed against the averaged field. It is
offered as an indication of where the number sits, not as the deliverable.

## What the agent chose, unprompted

Nothing below was suggested to it.

- Read "L-shaped junction" as a **sharp miter** rather than a swept radius, and said why.
- Wrote a **17 KB Python generator** to emit the blockMeshDicts programmatically instead
  of hand-authoring three of them, which is what made a 1.5× refinement ladder cheap.
- Chose a **`mappedPatch` recycling inlet** over a guessed profile, reasoning that the
  developed profile is the fixed point of the recycling map — then verified it against
  Blasius afterwards.
- Escalated **5 → 40 → 60 iterations** as cheap probes before committing to a production
  solve, and hit and fixed the MPI-as-root failure itself along the way.
- On resume, noticed the old runner would **wipe prior results** and wrote a new one that
  resumes from `latestTime` instead.
- **Killed its own sweep** when it realised 6,000 iterations could not separate mesh
  effect from oscillation, and relaunched at 20,000.

## What it verified, and what it did not

Its own words, unprompted:

> **The run does not converge to a steady state — it settles into a limit cycle.**
> Residuals flatten at ~6·10⁻³ (U) and ~5·10⁻² (p) and stay there … The loss coefficient
> swings between about 1.5 and 2.3 with no trend over 5500 iterations.

> a snapshot value of K is meaningless here

> I cannot tell from a steady solver whether that is physical shedding off the sharp
> corner (very plausible at Re_Dh = 67 000) or a numerical limit cycle.

It localised the oscillation rather than just noting it: σ(p₀) was identical to three
decimals at all five inlet-leg stations and decayed downstream (5.8 Pa at 2D past the
corner, 0.55 at 20D), so the inlet leg moves as a rigid pressure column and the
fluctuation is generated in the bend.

**Known limitations of the numbers above:**

1. **Not converged.** These are time averages over a limit cycle, not a steady solution.
2. **Physical or numerical is unresolved.** Settling that needs URANS — perturb and fit a
   growth rate, per the field notes — which was not run.
3. **The outlet leg is short.** At 20 D_h, p₀ is still decaying at ~1.5× the
   fully-developed rate at 17D, so the friction correction is biased and K_exc reads
   somewhat high.
4. **The fine mesh is under-sampled.** 3,100 iterations against 20,000 for the others, so
   its average carries a wider band and the Richardson extrapolation it would support is
   not claimed here.
5. **y⁺ moves with refinement** (mean 25 medium, 40 coarse), so part of the mesh trend is
   wall-treatment sensitivity rather than pure discretisation error — the agent named this
   itself.

## What the run cost, and what it bought

Two sessions, ~16 min and ~46 min of agent time, plus remote solve time. It was
interrupted mid-solve when the Modal workspace was disabled at iteration 444 of 6,000; the
volume survived, and the resumed session found its own checkpoint and restarted from
`latestTime` with no help.

Seven harness defects surfaced that no unit test had caught:

| # | Defect | Why the tests missed it |
|---|---|---|
| 1 | `fetch` flattened paths in the fake, preserved them for real | The fake diverged from the service |
| 2 | System prompt claimed `foamToC` exists; it is not in the image | Never verified against the container |
| 3 | `mpirun` fails as root without two env vars | Only reachable by running a parallel solve |
| 4 | A redirected log destroyed every non-ASCII character | Only visible with stdout redirected |
| 5 | Resume blurb reported a stale job status | Needed a job in the `unknown` state |
| 6 | A timed-out command returned a bare `exit_code: -1` | Needed a command to actually time out |
| 7 | A bodyless 200 raised `JSONDecodeError` through error handling | Needed a slow request through a proxy |

All seven are fixed, each with a regression test.

## Reproducing

```bash
openreynolds --study 20260823-213712-babc
```

Artifacts fetched to `studies/20260823-213712-babc/files/`: `medium_p0profile.png`,
`miter_medium/renders/miter_medium_corner.png`, `miter_medium/renders/miter_medium_Umag.png`.
Time-average statistics in `studies/20260823-213712-babc/timeaverage.json`. The case,
its generator, and all three meshes remain on the instance volume under `/work/elbow/`.
