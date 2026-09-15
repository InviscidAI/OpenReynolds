# core-kimi-k3-moonshot-20260915-060615-0057

**A screen.** The same ten cases as `core-gpt-5.6-sol`, run to answer one question: is
kimi-k3's 2/26 a fact about the model, or about how Aster serves it? The model is held and
everything about the serving changes -- Moonshot instead of Aster, the `anthropic` adapter
instead of `openai`, on a core carrying the brief fix.

## 1. The sweep

| | |
|---|---|
| model | `kimi-k3` at `medium`, Moonshot |
| adapter | `anthropic` -- the same one Opus 5 scores 7/10 on |
| core sha | `3c147ee` |
| passed | **2 / 10** (T1, T7) |
| stopped on **time** | **8** |
| stopped on steps | 0 |
| spend | $5.26 (81.8% cache hit) |
| contamination | **0 / 10** |

## 2. The four arms, same ten cases

| arm | pass | time | steps | $ | tok/s | secs | cells |
|---|---|---|---|---|---|---|---|
| Opus 5, `anthropic` | 7/10 | 0 | 2 | 9.54 | 51.6 | 4143 | 200 |
| kimi-k3 Aster, `openai` | 2/10 | 6 | 2 | 3.33 | 14.8 | 8002 | 207 |
| **kimi-k3 Moonshot, `anthropic`** | **2/10** | **8** | 0 | 5.26 | 25.5 | 8370 | **167** |
| gpt-5.6-sol, `openai-responses` | 8/10 | 0 | 2 | 5.12 | 39.3 | 3508 | 193 |

**The adapter line is retired.** kimi-k3 scores 2/10 on Chat Completions at Aster *and* on
the Messages API at Moonshot -- the adapter Opus 5 passes 7 of these same ten on. Two
providers, two adapters, 1.7x the throughput, the fixed brief. Nothing about our wiring
explains it, and `openai_api.py` no longer needs to be suspected.

## 3. But the runs are not the same runs

The pass count is identical and the failures are not. Every case was re-checked with
`checkMesh` directly, on the two modes the Aster sweep established:

| case | Aster delivered | Moonshot delivered |
|---|---|---|
| T22 | 1 of 36 passages, 2.9% of CAD volume | **36 regions, 0.000398715 against CAD 4.0028e-4 -- 0.4%** |
| T16 | -- | **0.27639417 against analytic 0.275573 -- 0.3%** |
| T1 | checkMesh wrote no readable verdict | **2.8712e-06, matching gpt-5.6-sol's 2.87115e-06** |
| T11 | `patch0..patch7`, three empty | 3 patches, 0.00086962 against CAD 8.5627e-4 |
| T18 | `inlet`/`outlet`/`walls` at 0 faces | 6 named patches, all populated |
| T14 | unrefined background box | 5 named patches |

**Five of the eight failures carry `why: ""`** -- the finish check found nothing missing.
They built a valid, correctly shaped mesh with proper named patches and the wall clock
killed them between 907 and 1040 s. Only T3, T5 and T24 never meshed.

At Aster the meshes were *wrong*. Here they are *late*. Same score, different failure.

## 4. Why -- and it is not throughput

Moonshot is 1.7x faster (25.5 vs 14.8 tok/s) and used **the fewest cells of any arm**
(167 against Opus's 200), and still spent the most wall clock. So the difference is not
steps and it is not tokens per second. It is how much the model thought:

| arm | turns | thinking chars/turn | thinking total | output tok/turn |
|---|---|---|---|---|
| kimi-k3 **Aster** | 221 | **774** | 171,018 | 467 |
| kimi-k3 **Moonshot** | 178 | **2,326** | 413,996 | 1,018 |
| Opus 5 | 219 | 526 | 115,154 | 796 |
| gpt-5.6-sol | 208 | 383 | 79,641 | 580 |

**The same model reasoned three times as much per turn at Moonshot.** The cause was
measured on the live endpoints before either run: Aster accepts `reasoning_effort` and
ignores it -- low/medium/high returned 114/73/78 reasoning tokens, no monotonic effect --
while Moonshot accepts `thinking={"type": "adaptive"}` with `output_config.effort` and
honours it (`lean` stayed False).

So the Aster arm never measured kimi-k3. It measured kimi-k3 with its reasoning throttled
by an endpoint that takes the knob and does nothing with it, and an under-reasoned desk
makes exactly the errors that sweep found: the background box kept as the answer, one
passage of thirty-six, the patch that was named and left empty.

**Caveat on attribution.** The two arms differ in provider, adapter *and* brief, so this
comparison alone cannot assign the whole gap to reasoning. The 3x thinking difference is
measured directly; its cause is measured independently on the two endpoints; neither rests
on the sweep.

## 5. What this leaves open

kimi-k3 here is **time-bound, not incapable**. Reasoning properly is what costs the clock:
2,326 chars of thinking a turn is why 167 cells took 8,370 s, against a ~900 s budget
calibrated for a model that thinks a quarter as much.

The test is a rerun of the five cases that built the right mesh and ran out of clock --
T11, T14, T16, T18, T22 -- at a doubled budget, with T1 and T7 as controls that already
pass. `--seconds` could not have delivered that until `43bc706`: the supervisor's deadline
was a constant and would have killed a 1800 s run at 1500 s and filed it `wedged`.
