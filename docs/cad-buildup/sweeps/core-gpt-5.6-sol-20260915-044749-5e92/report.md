# core-gpt-5.6-sol-20260915-044749-5e92

**A screen, not a corpus sweep.** Ten of the twenty-six cases, run to answer one question:
does the desk work at all for a strong model that is not Claude? It is not a measurement
of gpt-5.6-sol against Opus 5, and the section below says why it cannot be.

## 1. The sweep

| | |
|---|---|
| label | `core-gpt-5.6-sol` |
| model | `gpt-5.6-sol` at `medium`, OpenAI direct |
| adapter | `openai-responses` (`/v1/responses`) -- **new code, written the same day** |
| core sha | `4d9bb56` |
| cases | T1, T3, T5, T7, T11, T14, T16, T18, T22, T24 -- one run each |
| ended `done` | 8 / 10 |
| passed | **8 / 10** |
| `checkMesh` ok | 10 / 10 |
| stopped on **time** | **0** |
| total spend | $5.12 |
| total run seconds | 3,508 (125-585 s per case, against a ~900 s budget) |

## 2. Contamination

**0 / 10.** The numbers below measure what they claim.

## 3. What this screen can and cannot support

**Can:** the shared layer is exonerated. The loop, `base.py`'s block helpers, the brief and
the corpus all work with a non-Anthropic model, well enough to pass 8 of 10 cases that
Opus 5 passed 7 of. "The desk only works with Claude" is dead.

**Cannot:** this is `responses_api.py`. The Kimi K3 sweep that scored 2/26 ran
`openai_api.py` -- different code. A good result here says nothing about that adapter, and
the two share only `base.py`. This screen eliminates; it does not identify.

**Also cannot:** compare models. This ran on core `4d9bb56`, which carries the brief fix;
the Opus and K3 arms ran older cores whose brief still described the deleted fence
protocol. 8 against 7 is not a model result, and no sign test is computable here.

## 4. The mesh vet -- the part that decides whether 8/10 is real

`checkmesh_ok` was 10/10, and in the K3 sweep fourteen runs carried a green `checkMesh` on
a mesh that was wrong. So every case was re-checked directly, on the two failure modes
that sweep established: a bare background box, and named patches landing empty.

**Neither appears.** Every case has its named patches, all with non-zero faces. No
`defaultFaces`-only mesh, no `patch0..patch7` anonymisation, no empty patch:

| case | Kimi K3 delivered | this run delivered |
|---|---|---|
| T22 | 1 of 36 passages, 2.9% of the CAD volume | 36 regions, `Total volume = 0.000400087` against CAD `4.0028e-4` |
| T24 | bare box, one `defaultFaces` patch | 12 named patches, all populated |
| T14 | unrefined box, one `defaultFaces` patch | 5 named: inlet, outlet, rootWall, farField, wing |
| T18 | right volume, `inlet`/`outlet`/`walls` at **0 faces** | 6 named, all populated |
| T11 | `patch0..patch7`, three empty | 4 named, `0.00084227` against CAD `8.5627e-4` |
| T7 | -- | one `allWalls` patch, `0.00024` = 0.1 x 0.06 x 0.04 exactly |

The volumes that can be checked against a number the case or the desk computed agree:
T22 exactly, T11 within 2%, T7 exactly, T18 to `0.0540717`.

**One reading not yet run down:** T5 (`steps`, did not pass) reports `Total volume =
41887.1` with `Min volume = 2.925`. Those magnitudes are millimetre-scale, not metre-scale,
which would mean the geometry was never scaled -- the failure `undeclared_step_unit_...`
names. It did not pass and so does not touch the 8/10, but it is not a clean run either.

## 5. Probes

Unchanged from the corpus in the way that matters: `coverage` and `scale` read `n/a` on
all ten, as they did on all twenty-six before, because no case supplies a patch manifest or
an `extent_m`. `union_closure`, `normals` and `self_intersection` read on most cases. No
probe caught anything; the mesh vet above is again the only instrument that looked at what
was delivered.

## 6. Properties

Not graded in this screen. The question here was whether the desk runs, and grading 119
named properties across ten cases is the corpus sweep's job, not a screen's. The
`killed_run_record_loses_the_cases_named_properties` fix (`47012e6`) is in this core, so a
future sweep on it will have the properties to grade.

## 7. What it leaves open

1. `openai_api.py` is still untested with a strong model, and the only OpenAI models that
   can exercise it are gpt-5.2 and below -- every model above refuses function tools beside
   a reasoning effort on Chat Completions. A paired gpt-5.2 run on both adapters is the
   available test; a third-party OpenAI-compatible gateway serving a frontier model is the
   better one.
2. Zero `time` stops here against twelve in the K3 corpus, on a provider measured at 3.5x
   K3-on-Aster's throughput. Consistent with throughput mattering; not a demonstration.
3. Whether K3 itself can do this work is the open question this screen does not touch.
