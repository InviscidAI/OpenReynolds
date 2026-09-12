# Every case, and the external benchmark it came from

*2026-09-12. Corpus at `af131f1`. Sources read at their `main`; neither pipeline was run,
so every external number here is theirs, quoted from their own report.*

Provenance has to travel with the number. This is the full map: what each case was taken
from, what the source scored on it, and what was changed. The short version lives in
`tests/data/prompts/README.md`; this is the long one, and it is the file to correct if a
mapping here is wrong.

## The two sources, and the third behind one of them

| | [Adam-CAD/CADAM](https://github.com/Adam-CAD/CADAM) | [Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD) |
|---|---|---|
| license | GPL-3.0 | MIT |
| kernel | OpenSCAD → polyhedra | build123d / OCCT → STEP |
| benchmark set | 13 models, `.scad` + render | 10 scored prompts **P1–P10**, 10-model show gallery **S1–S10**, 2 print-in-place demos |
| what it publishes | dimension and colour counts; no pass/fail | 141 features, binary pass/fail, tokens and cost per prompt |
| scored against | nothing — a showcase | its own `cad skill` baseline, same prompts, same model |

**P1–P10's prompts are not MAC's own.** They come from
[earthtojake/text-to-cad](https://github.com/earthtojake/text-to-cad) ("CAD Skills"), whose
`cad` skill is the baseline MAC measures itself against — and whose `cadpy` package MAC
vendors. So a P-case carries **two** external results, MAC's and the skill's, on the same
prompt at the same model (`qwen3.7-max`), with agent architecture the only variable. Where
the two disagree, that is the most informative thing either suite offers.

## T1–T8 — no external source

Authored 2026-09-11 to replace the original eight, which existed only in a chat history and
were judged unrecoverable. `tests/data/prompts/README.md` records that provenance and its
consequence. **Nothing in T1–T8 comes from either repo**, and a pass rate over T1–T8 is not
comparable to anything published elsewhere.

| | case | source |
|---|---|---|
| T1 | 2D U-bend | authored |
| T2 | body in a flow box | authored |
| T3 | Tesla valve | authored |
| T4 | cold plate with fins | authored |
| T5 | STEP assembly, fluid domain | authored; fixture is C1's real multi-solid STEP |
| T6 | STEP with no declared unit | authored; fixture is C1's box-with-duct, `LENGTH_UNIT` emptied |
| T7 | sealed cavity, cell zones | authored |
| T8 | conjugate two-region | authored |

## T9–T26 — the map

The fluid domain, patches and properties are authored here in every case except T25, which
is quoted.

| | case | ← source | what the source scored | what changed |
|---|---|---|---|---|
| **T9** | flooded bearing, 1 mm clearance | MAC **articulable gyroscope** (print-in-place demo) | no pass/fail published; criterion is the ring spinning free off the plate | dimensions theirs (30/23/22/15 mm, 0.4–1 mm clearance). Spinner height changed to 8 mm **after the first sweep** — at their flush height the regions cannot connect and the brief was unpassable |
| **T10** | one turbine blade passage | CADAM **12, axial turbine blisk** — 14 dims · 1 colour | showcase; scored on blades present and twisted | blade dims theirs (28 blades, 22 mm chord, 6% thick, 30° stagger, 35° twist). Passage, periodicity, stations authored. **Their `offset(r=0.6)` trailing-edge thickening is the failure this case catches** |
| **T11** | honeycomb flow straightener | CADAM **04, honeycomb bracket** — 13 dims · 1 colour; MAC **S1, honeycomb organizer** | showcase both sides | lattice geometry theirs; duct, through-flow and open-cell count authored. Neither source asks whether every cell is open — for a lightening cutout a fused wall is cosmetic |
| **T12** | centrifugal impeller passage | CADAM **08, impeller** — 10 dims · 1 colour; MAC **P8** — 15 features | **MAC 14/15 — its only failure in 141, item 13 "blade root fillets"**, ¥1.20. **`cad` skill 15/15**, ¥32.75 | 7 blades (CADAM's) not MAC's 12, so the fillet is the marginal feature and not the passage. **The two external results disagree here** |
| **T13** | helical thread leak path | CADAM **06, threaded jar & lid** — 9 dims · 2 colours; **03, hex bolt & nut** — 3 dims · 2 colours | showcase; "two **mating** threaded parts", "**real ISO threads**" | thread form theirs; the 0.2 mm flank clearance and the leak path authored. A thread that prints and a thread that seals are different claims |
| **T14** | tapered NACA wing, external | CADAM **05, NACA 2412 wing** — 9 dims · 1 colour | showcase; "true airfoil from the NACA equations, tapered loft" | 120→80 mm chord over 200 mm span theirs. Spar tubes and lightening holes dropped — internal structure, no wetted surface. Flow domain and three stations authored |
| **T15** | flooded bevel gear mesh | CADAM **07, bevel gear drive** — 9 dims · 3 colours; **09, planetary stage** — 10 dims · 4 colours, which is MAC **P10** — 21 features | **MAC 21/21** ¥0.73, **skill 21/21** ¥11.43 (P10, their largest) | gear pair at 90° theirs; the 0.3 mm backlash, flooded cavity and flank-gap measurement authored. Cavity resized to 160×140×120 mm after the 90 mm wheel did not fit the first draft |
| **T16** | turbofan bypass annulus | CADAM **11, turbofan** — **2 dims · 10 colours** | showcase | reduced to the bypass annulus, the only part with a wetted path. Radii and the mid-duct bulge authored — the source specifies two dimensions in total |
| **T17** | V8 intake manifold | CADAM **13, V8 engine** — 22 dims · 8 colours, their largest | showcase; "an intake manifold in the valley" | reduced to the manifold's internal air path. All dimensions authored |
| **T18** | finned cylinder, external | CADAM **10, radial aircraft engine** — 15 dims · 6 colours; MAC **P7, radial-engine cylinder** — 17 features incl. "12 fins" | **MAC 17/17** ¥0.53, **skill 17/17** ¥17.42 | one cylinder, not nine. Fin stack theirs; cooling duct and the twelve measured gaps authored |
| **T19** | shaft seal with keyway | MAC **P4, stepped shaft + keyway + chamfer** — 11 features | **MAC 11/11** ¥0.57, **skill 11/11** ¥6.53 | shaft and keyway theirs; housing bore, the 0.05 mm clearance and the leak path authored. Full marks as a solid is why it is interesting as a fluid domain |
| **T20** | pipe flange bore — **the control** | MAC **P2, circular flange + bore + 6 bolt holes + double fillet** — 10 features | **MAC 10/10** ¥0.34, **skill 10/10** ¥8.07 — their second-cheapest prompt | flange theirs; bolted pair, gasket and bore flow authored. Here *because* it is the easiest thing either suite contains |
| **T21** | enclosure with standoffs | MAC **P5, open-top enclosure + 4 standoffs + blind holes + outer fillets** — 12 features | **MAC 12/12** ¥0.36. **`cad` skill 11/12 — failed item 11**, fillet scope, ¥2.88 | standoffs and blind holes theirs; cooling flow, vent and measured depths authored. Outer fillets dropped — outside the wetted surface. **The two external results disagree here** |
| **T22** | vented brake disc | MAC **S10, brake disc** (show gallery) | no pass/fail published | 36 vanes and the through-flow authored. Deliberately a radial array *without* blade curvature, so T12's failure can be attributed to the curvature or not |
| **T23** | caged ball | MAC **ball-in-cage** (print-in-place demo; cousin of **S5**) | no pass/fail; stated as the harder of their two articulable cases | **dimensions used exactly as given** — 40 mm cube, 16 mm hollow, 15 mm ball, 1 mm clearance, six 12 mm holes. Their criterion is the ball rattling; here it is which volume gets meshed |
| **T24** | swirl chamber | MAC **S9, plasma reactor** (show gallery) | no pass/fail published | only the class of object is theirs. All dimensions and the 15° tangential entry authored |
| **T25** | turbofan, no dimensions given | CADAM **11, turbofan** — **prompt quoted verbatim** | showcase — its published model exposes **2 dims · 10 colours**, the lowest dimension count and highest colour count in their set | one sentence added to make it a meshing request. Nothing else. **Passes as `refused`** — building it is the right answer in a showcase and the wrong one here |
| **T26** | spiral stairwell, tangent treads | MAC **P9, miniature spiral staircase** — 16 features | **MAC 17/16 — a published *defensive correction*** on item 8: it found the tread–column tangency would fracture and fixed it unprompted, ¥1.45. **`cad` skill 16/16** but shipped the disconnected version; their report shows both renders | scaled to a full-size stairwell, flow domain authored. **The tangency is reproduced deliberately** — 100 mm column radius + 1100 mm tread = the 1200 mm wall, exactly. **The two external results disagree here** |

## The three cases where the external pipelines disagree with each other

The only points in the corpus where two independent systems, same prompt and same model,
produced different outcomes — which makes them worth more than the cases both passed.

| case | prompt | MAC | `cad` skill |
|---|---|---|---|
| **T12** | P8, impeller root fillets | **failed** (item 13, fillet code conflict) | passed 15/15 |
| **T21** | P5, enclosure fillet scope | passed 12/12 | **failed** (item 11) |
| **T26** | P9, staircase tangent tread | **corrected it unprompted** | passed 16/16 with the defect |

**And every failure either system recorded, across all 141 features and both runs, was a
fillet.** The skill's three are scope errors (P5 item 11, P6 items 16 and 18 — over- and
under-applying); MAC's one is a code conflict (P8 item 13). Four failures out of 282
feature-judgements, all on the same operation: the one whose input is an edge set on geometry
that prior booleans have already fragmented.

## What was left on the table

Not every benchmark has a fluid domain, and a prompt with no wetted surface has nothing for
this desk to be right or wrong about.

**CADAM, unused:** 01 twisted hex vase, 02 knurled control knob.

**MAC, unused:** P1 block with through-holes, P3 L-bracket, P6 aerospace clevis bracket (the
prompt the `cad` skill did worst on, 16/18); gallery S2 gyroscope ornament, S3 lighthouse,
S4 smartphone stand, S6 articulable gyroscope (its dimensions are in T9), S7 multi-link
chain, S8 Geneva mechanism.

S7 and S8 are the two worth revisiting: a multi-link chain and a Geneva mechanism are both
multi-body assemblies with clearances, which is the class T9 and T23 each cover from one
angle.

## What none of this licenses

**No code, prompt text or licensed material from either repository is vendored here.** CADAM
is GPL-3.0 against this repo's MIT, which makes that a decision rather than a convenience.
T25 quotes one prompt, as quotation, and says so in its own file.

**The scores in the tables above are not comparable to this corpus's.** Their criterion is
that a feature is present on a printable solid; this corpus's is a clean `checkMesh` plus
every named property measured *and printed*. A 99.3% and a 9/12 are answers to different
questions, and the only thing that compares across the two is behaviour — did the pipeline
attempt what its brief required, and did it say what happened.
