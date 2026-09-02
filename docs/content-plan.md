# The content plan

A distribution strategy for Inviscid AI, built from what could be verified about how simulation
content actually travels in 2026.

**The scoreboard is followers.** The point of this is a warm account at launch — an audience that
already exists when there is something to announce. Views, likes and upvotes are instrumental;
follower delta is the number that decides whether any of this worked.

**The bet is breakouts, on the platform where breakouts convert.** Optimise for the ceiling and
accept the variance — median performance in this genre is in the hundreds of views and one post
carries an account. But breakout reach and follower growth pull apart hard, and the split is by
platform:

| Account | Reach | Followers | Conversion |
|---|---|---|---|
| @cfd_guy (TikTok) | ~13.1M views | 22.1K | 0.17% |
| @theairflowguy (TikTok) | 3.2M views | 4,586 | 0.14% |
| @atmosmelancholy (TikTok) | ~4M views | 1,617 | 0.04% |
| **AeroJAX (Instagram)** | ~186K likes | **~62K** | ~10× better |

AeroJAX's top posts *are* breakouts. They happened on Instagram and they converted. TikTok breakouts
convert to almost nothing — the flip side of follower count not being a ranking input there.

**X is not the answer either, despite appearances.** Measured today: @fyfluiddynamics, the
fluid-dynamics brand, has **11,795** followers on 19 posts. The official APS Division of Fluid
Dynamics has **931**. The author of the WebGL fluid simulation that scored 868 points on Hacker News
has **7,323**. And @primerlearning — whose entire channel is simulation videos — has **8,187 on X
against 1.94M on YouTube**, a 237× gap for the same creator and content. The realistic ceiling is
@gabrielpeyre at 101,765, after 6,116 posts of daily computational-maths visuals.

The mechanism explains it. X's timeline is ~50% in-network by design and out-of-network content is
discounted (`OonWeightFactor 0.75`), so **followers are a superb asset once held and there is no
firehose to acquire them with** — a cold-start trap, exactly inverse to TikTok. X also runs the worst
engagement rate of the four major platforms (0.10%) and needs ~70 posts/month to sustain it against
Instagram's 20.

The startup-audience premise is also weaker than it feels: Stack Overflow's 2025 survey (49,000+
responses) ranks X **10th at 17.1%** among developer community platforms — behind GitHub 66.9%,
YouTube 60.5%, Reddit 53.7%, Discord 38.9%, LinkedIn 37.2% and Hacker News 19.6%. Founders and VCs
are loud on X; developers are elsewhere.

So: **chase the ceiling on Instagram, and treat TikTok, X and Reddit as free distribution whose
numbers do not count toward the scoreboard.** Absurdity is a lever worth using; mass venues beat
specialist ones; and every experiment below is judged on follower delta, not reach.

Every number below carries a source. Claims that did not survive checking are listed at the end so
they do not creep back.

---

## The format

Five criteria. The first four come from the posts that won; the fifth is a gate.

1. **Binary A vs B** — a comparison with a winner, not a beautiful render.
2. **Legible in one second**, even when the subject is obscure. Zero HVAC knowledge required to
   understand "which duct is better."
3. **Contested or non-obvious**, ideally with folk wisdom to confirm or destroy.
4. **The viewer has a stake** — they have done this, or will.
5. **The answer is a fluid-dynamics answer** — carried by moving fluid, not by radiation,
   conduction, chemistry, or an equation.

Criterion 5 reads as too obvious to write down. It is here because a subject was added to this
document carrying 1,259 Reddit points and no fluid in it. Appetite research measures whether people
will argue; it says nothing about whether the answer moves. A subject failing 5 is worse than
nothing — it advertises a case the tool cannot address.

**Stake is the load-bearing variable.** AeroJAX's top posts are single-sided vs cross ventilation
(~186K likes) and T-branch vs Y-branch junctions (~186K) — the two dullest subjects imaginable.
Their llama post is absurd *and* an A/B — numpy vs jax — and did ~7K. That is above their average,
so absurdity does real work; what it cannot rescue is that numpy vs jax is not a question. It says
*we run on jax, so we are fast*. The answer was decided before the simulation ran.

FluidX3D shows the same on Reddit: NASA X-59 at 117 billion cells scored **484**; its head-to-head
"Siemens: 12 hours on 8×A100 — FluidX3D: 147 seconds" scored **27**. **A comparison you obviously
win is not a comparison** — a direct warning about our own benchmark posts, the easiest ones to
reach for.

**Assertions get memed; verdicts get argued.** Belzona posts serious product guides and its comments
are pure jokes. AeroJAX posts comparisons and its comments are objections. A guide offers nothing to
disagree with, so the only available response is a joke. The A/B structure manufactures the
objections — which matters, because the objection is the next video.

---

## The objection loop

Real comments on AeroJAX's top posts: *"this is ignoring the air that will go under the truck"*,
*"what about if flow is going opposite direction?"*, *"cross-ventilation moves hot air through the
whole room while single-sided keeps the cold parts untouched."*

Each is a simulation request with the physics already attached. Answer it, and the answer generates
its own next objection. Better than "Day X of the top comment": no cold start, no hostile
agenda-setting, and it compounds.

Dassault Systèmes has run the mechanic for years — commenters request absurd simulations, engineers
run them. The [fork video](https://www.tiktok.com/@dassaultsystemes/video/7481653954972093718) opens
"after more than a year of requests." **Their bottleneck was a year. Ours is hours.** That gap is the
product demo.

Copy their measured best practice: **credit the submitter by name.** Their all-time top post (12.2M
views) is a chicken-in-an-oven animation someone else made, captioned "Thanks to Tayfun PEKTAŞ." For
an open-source project that doubles as contributor recruitment.

**No fake mistakes.** Every simulation genuinely is simplified — state the simplification and let
people find it. Planting a known error produces an identical comment section short-term and is a lie
this audience can detect, especially with case files public. Getting caught would poison every
previous post.

---

## Subjects

Backed by live Reddit scores, verified 26 Aug 2026, subject to drift. The criterion is not what
renders well but what people already fight about.

**There is no separate industrial track.** Ceiling fans, window fans, portable AC, vents, attic fans,
range hoods and ducts *are* HVAC — a large industry whose professionals argue about these exact
questions. r/HVAC carries nine threads on closing vents in unused rooms; a "your ductwork design is
probably bad" PSA scored 497/236, written by and for contractors. Most companies choose between
content that reaches people and content that reaches buyers; here they coincide.

### First six

| # | Question | Evidence |
|---|---|---|
| 1 | **A fan blows a narrow jet but sucks from everywhere.** Where should it go? | [8,671 / 321](https://reddit.com/r/explainlikeimfive/comments/nqdhft/) |
| 2 | **Window fan: blow in or out?** | Two mass LPTs give *opposite* advice in their titles — [3,906](https://reddit.com/r/LifeProTips/comments/1l2jo9n/) vs [2,765](https://reddit.com/r/LifeProTips/comments/8uzu9b/) |
| 3 | **Ceiling fan reversal in winter — does it do anything?** | [ELI5 attacking the folklore's own logic, 6,576 / 485](https://reddit.com/r/explainlikeimfive/comments/uo34qc/); a [53,647-point TIFU](https://reddit.com/r/tifu/comments/cvj28f/) |
| 4 | **Portable AC: single vs dual hose.** | [12,738 / 550](https://reddit.com/r/YouShouldKnow/comments/keqtps/); people 3D-print fixes |
| 5 | **Golf ball dimples — why aren't planes dimpled?** | [5,815 / 685](https://reddit.com/r/askscience/comments/ze0j1f/), recurring a decade |
| 6 | **Shower curtain: why does it attack you?** | [16,135 / 1,102](https://reddit.com/r/askscience/comments/6cmj7m/); 12+ ELI5 threads |

1 and 2 are the same physics and pair naturally. 5 pairs with aero socks.

### Bench

Campfire smoke following you ([12,621 / 1,136](https://reddit.com/r/explainlikeimfive/comments/rvn1ac/)) · windows down vs A/C on the highway · closing vents in unused rooms · air fryer vs convection oven ([47,340 / 4,519](https://reddit.com/r/unpopularopinion/comments/qvk5sz/)) · wood stove backdraft · PC case pressure · attic fan short-circuit · cycling draft distance · aero socks · motorcycle windscreen (taller is worse) · wake turbulence · [F1 dirty air](https://reddit.com/r/F1Technical/comments/1rpun3x/) · wind chill on objects · fridge door.

**Recognisable object beats canonical test case.** Everything above 1M views in the Instagram sample
is flow over a helicopter, airliner, warship, sports car, cow or exhaust manifold. Everything in the
triple digits is a cylinder, NACA 0012 or a centrifugal pump. Same physics, different recognisability.

**Scale is not our hook.** FluidX3D's big-number posts work because it is a bespoke GPU lattice-
Boltzmann code doing 117-billion-cell runs. OpenFOAM will not get there. The hook is relatability or
absurdity — both cheaper than compute.

### Avoid

AeroJAX has done single-sided vs cross ventilation, T vs Y junctions, and tailgate up vs down. Their
content could not be enumerated (Instagram blocks anonymous fetch), so adjacency is flagged, not
proven. Duct geometry is the closest neighbour — use flex vs rigid or sharp elbow vs sweep. Heat-wave
windows overlaps their ventilation post; differentiate by making it buoyancy-driven.

### Nothing from the industry research fits

The startup-buyer research (fission, launch, defence, batteries, fusion) produced no usable subjects,
and the failure mode is worth recording.

**Subjects that are genuinely CFD have no appetite** — reentry aerothermodynamics, blunt bodies,
nozzle flow, reactor thermal-hydraulics, EV battery fires all returned nothing above 150 points where
such questions live. **The one subject with enormous appetite is not CFD** — "how do computers in
space dissipate heat" did 1,259/381 on r/explainlikeimfive, and heat rejection in vacuum is
radiation, not flow. Both failures came from grading on shareability without asking whether the
answer moves.

Reach those buyers by naming the companies and calling them — Antares, Valar Atomics, Oklo,
TerraPower, Kairos, Relativity, Ursa Major, Inversion, Anduril. A week of work; no quantity of Reels
shortens it.

### Not supported by evidence

Looked for real argument and found none: sunroof buffeting, vortex generators on cars, radiator and
humidifier placement, chimney height, arrow fletching, drone prop wash. Run-or-walk-in-the-rain
recurs but scores 4–23 points — well below its reputation.

**Audience facts worth not forgetting:** r/HVAC has never had a single CFD post, ever, across 204,578
subscribers — professional HVAC people are not a CFD-literate audience. r/datacenter is 21,057 subs
and r/MEPEngineering 14,638: dense, not reach.

---

## Platforms

**Instagram is the scoreboard platform. Everything else is free distribution.**

| | Instagram | X | TikTok | Reddit | LinkedIn |
|---|---|---|---|---|---|
| **Role** | **the account** | sharing | cross-post | stars + karma | credibility |
| **Counts toward followers?** | **yes** | some | effectively no | no | no |
| **Artifact** | 20–40s vertical A/B | 10–20s + still | same cut | GIF/video | carousel + text |
| **Optimise for** | **follows**, watch time, sends | copy-link shares | shares, bookmarks | upvotes → stars | text retrieval |

Effort follows that first row. Instagram gets the tuning, the retention analysis and the experiment
budget; the others get the same asset reposted at near-zero marginal cost.

YouTube is dropped: long-form narrated explainers are the only format that builds an audience there
and they are not automatable. The evidence is still instructive — ProjectPhysX (silent renders) runs
**221 views per subscriber** with one 11-second cow at 62% of lifetime views; AirShaper (narrated)
runs **21.6**, ten times better conversion. Silent renders buy views, not audiences. The graveyard is
real: @openlb 1,260 subs after 18 years, @DualSPHysics 3,630 across 271 videos, and SimScale — a
funded platform claiming 900,000 engineers — whose last five uploads got 95, 63, 91, 219 and 135
views.

### Instagram — primary

Ranking predictors, from Meta's own
[Reels system card](https://transparency.meta.com/features/explaining-ranking/ig-reels-chaining/):
probability of watching under three seconds (using **skips within two seconds**), watching over 95%,
resharing, sharing off-platform, commenting, following, using the audio. Mosseri's stated top three
are watch time, likes and sends, sends weighted slightly more for non-followers. The "sends are 3–5×
a like" figure has no primary source.

**Never post a muted Reel** — Instagram makes "reels that are muted" less visible. **No trending
audio**: business accounts are licensing-barred from the main catalog, and within-account music-vs-
original across 35 accounts came out 7–7. Use Meta Sound Collection.

**Captions do not move reach, and long irrelevant ones now hurt.** As of mid-2026 the
[recommendations policy](https://help.instagram.com/313829416281232) names the engagement-farming
tactic directly — "long captions unrelated to the underlying content and coordinated comment
networks intended to artificially drive engagement" — making that content ineligible for
recommendations. The clause was added between 21 May and 26 Aug 2026. A 2026 dataset agrees
independently: 500+ character captions reached 141 median vs ~380.

**Debunked by Instagram directly:** the first-30-minutes rush ("we can confirm this isn't true!"),
hashtags as a reach lever, and "link in bio kills reach."

**Tool and tutorial framing is fatal** — every caption in the sample leading with a software name,
webinar number or course pitch came in under ~1,000 views. The machine appears in replies and case
files, never in the hook.

### X

Weights, verified from [`param.rs`](https://raw.githubusercontent.com/xai-org/x-algorithm/main/home-mixer/params/param.rs)
(open-sourced [13 Aug 2026](https://techcrunch.com/2026/08/13/x-open-sources-its-ranking-algorithm-letting-users-see-if-theyve-been-shadowbanned/)):

| Signal | Weight | | Signal | Weight |
|---|---|---|---|---|
| Share via copy-link | **20.0** | | Video open | 0.05 |
| Reply / Quote / DM share | 5.0 | | **Profile click** | **0.0** |
| Follow author | 4.0 | | **Dwell** | **0.0** |
| Repost | 1.0 | | Report | −234.0 |
| Like | 0.5 | | Mute / Not interested | −58.8 / −43.2 |

**Copy-link share is the top signal by 4×** — forty times a like. It rewards *"someone pastes this to
one colleague"*, and a flow visualisation is close to the most DM-able artifact that exists. Replies
and quotes beat reposts 5:1, vindicating the objection loop. Dwell and profile clicks are zero.

Weights multiply **predicted probabilities**, not counts, so "a report costs 468 likes" is a
misreading. Video weights are trivial but that is the weight for *opening* a video, not a penalty —
video reaches through the same door by raising P(reply) and P(copy-link). Post one still frame
alongside: a screenshot is more paste-able than a clip.

The "Premium gives 2×/4× reach" claim is from the **2023** repo and is not in current code.

### TikTok

Platform risk is gone — divestiture closed 22 Jan 2026. But **follower conversion is 0.04–0.17%
measured**, so on a follower scoreboard TikTok is close to worthless as a growth channel. Budget on
the assumption that a 1M-view video yields low thousands of followers at best, and never let TikTok
numbers stand in for progress.

It stays in the plan because the asset already exists and posting it costs minutes. Treat any
follower it produces as a bonus, and route attention off-platform where possible.

Format from measured data: **6–25 seconds** (the two biggest are 6s), one relatable or absurd object,
and a **human-scale anchor** in the caption — "Blue whale for scale" (5.9M), "don't try this on the
road" (3.2M). Shares and bookmarks track reach; comments do not — the 5.9M turbine video has 119
comments, the 4M chicken has 14.3K shares.

**No trending audio**: TikTok's own docs describe a similarity check that de-duplicates feed slots
for videos sharing a sound. Use the Commercial Music Library. Never `#cfd` — it belongs to Cheyenne
Frontier Days, a rodeo. Use the commercial-disclosure toggle once promoting.

### Reddit

**Mass subs are content; specialist subs are launch.** r/interestingasfuck (16.7M),
r/oddlysatisfying, r/nextfuckinglevel take clips with no product mention; that audience cannot detect
a physics error and does not remember either way. The specialist subs — r/CFD (46K), r/FluidMechanics
— are small, have memory, hold the people who would adopt, and carry the strongest 2026 anti-AI
reflex. Do not seed there and do not use them as QA; **the case file is the QA artifact**.

Two posts one day apart in July 2026: "CFD simulation through the vents" scored **21,667** in
r/interestingasfuck; "CFD simulations of airflow through vents" scored **255** in r/FluidMechanics.
An ~85× multiplier for one hop. (The multiplier is measured; the intent behind it is not.)

Repost economics are fat-tailed: one render scored 11,277 / 520 / 335 / 98 across four subs, plus
five r/Damnthatsinteresting attempts at 335, 211, 60, 56, 21. **Conversion prior: ~0.6–1.0 GitHub
stars per upvote** (LeafWiki: 168 upvotes → ~136 stars in 48h, "bigger than every press mention
combined"; PocketPaw ~131 → ~104).

**The 2026 immune response is the constraint.** r/programming has banned LLM content outright;
r/selfhosted auto-removes posts pending AI disclosure; r/Python bans wrapper showcases. Lead with the
artifact, never the agent. Answer "what software?" plainly — OpenFOAM, here is the case file — and
stop.

Start mass-sub posting now: it builds account age and karma, which are hard gates, and costs nothing.

### LinkedIn

Retrieval is **text-only** — verified from LinkedIn's own paper,
[arXiv:2510.14223](https://arxiv.org/abs/2510.14223): embeddings for members and content "using only
textual input." A vorticity render is a null input; it earns dwell only after the words win the
match. Reshares run 0.29× reach, so there is no viral mechanism. Video is the *worst* native format
(0.86×, down 36% YoY). Company pages are ~2% of the feed.

Expect a few hundred reactions and treat that as the ceiling. The playbook inverts usual advice:
carousels beat everything (1.39×), stills beat video, write *more* technically because the embedding
reads your words, 20+ short paragraphs beat ≤5 (1.13× vs 0.70×), reply to every commenter (up to
2.4×), post 2–4×/week, skip polls. **Engagement pods are actively enforced against** — two named VPs
on record in early 2026.

### Not worth it

**Threads** — 500M MAU but 28.4M outbound referrals/month against 115M DAU and ~4-minute sessions.
Zero-cost cross-post, never a channel. **Discord** — retention, never discovery: 1,000 members and 8
weeks before Discovery eligibility, and there is no OpenFOAM, FEniCS or ParaView server at all.

**The sleeper:** [cfd-online.com](https://www.cfd-online.com/Forums/), 190,141 members, is quiet —
the busiest OpenFOAM sub-forum has gone from ~8 threads/day in 2019 to ~0.4/day. Quiet means a
substantive post is actually seen.

### Hacker News — the launch surface

Not an audience-building channel, but the highest-leverage single moment available, and worth naming
now so it is planned rather than improvised.

**Fluid simulation is a proven HN front-page topic four times over**: a business card running a fluid
sim 1,135 points, WebGL Fluid Simulation 868 (front-paged three separate times as a straight repost),
Fluid Paint 605, a fluid-simulation pendant 501. The format has a floor there that most content does
not, and render quality carries as much weight as subject.

The one open-source launch retrospective with real source-level analytics — Plausible Analytics'
first four months — reports **Hacker News 43.6K visitors against Twitter's 10K**, and describes the
difference as *"Hacker News gives you thousands of visitors within several hours; Twitter is
something that can give you a few visitors every day consistently."* Their Product Hunt launch day
delivered 1,000 visitors and 15 trials.

Note the shape: HN is a spike, not a curve, and it rewards the artifact rather than the agent — the
same constraint the Reddit anti-AI gates impose.

---

## Production

One simulation produces one **master asset**; each platform gets a different composition.

**Master:** PNG sequence, 60fps, 2160×3840 **vertical native** (90.8% of 119 measured top performers
are exactly 9:16; zero square, zero landscape), dark background `#0B0E14`. A and B rendered with
**identical camera, identical colour scale, identical time range**, plus **one shared colorbar across
the full width** — that bar is the visible proof of a shared scale, and two panels on different
scales is the easiest way to make a comparison that convinces and means nothing.

**Colormap: rainbow/jet, deliberately.** Perceptually bad, and the field knows it. Also what every
winning post uses, and reach is measurably decoupled from correctness. Jet in the video; the case file
lets anyone re-render. Aesthetics is a distribution decision, physics lives in git.

**Structure: state the question, withhold the answer.** Across 48 measured top comparison videos,
**exactly one** declared a winner in its caption — and that was a creator advertising her own product.
Seven were framed as open questions. The convention is to pose the contrast and let the comments
resolve it, which is the objection loop firing on the post itself.

**Layout: split composite, panels joined directly, no outer box.** Instagram demotes "reels that are
muted or **contain borders**, reels that are **majority text**, or reels that have already been
posted" — that clause targets letterboxed repurposing, a box around the *outside*, not an internal
seam. A simulation A/B has no alternative: you cannot put "fan blowing in" and "fan blowing out" in
one room, because two fans in one domain is a different simulation. Avoid chrome — bevels, shadows,
gaps — that reads as two videos in boxes. Orientation follows geometry: stack top/bottom for a wide
room section; side-by-side for anything with a vertical axis. Never left/right on a wide subject.

**On-screen text is the highest-leverage element** — across 5,354 videos, Visual+Text median 25,492
views vs Visual-only 973. On Instagram, **adding a spoken hook actively hurts** (34,680 vs ~13,000).
No voiceover. Labels are words, not identifiers: "fan blowing out", never `case_02_outflow`. Short,
unboxed, heavy outline, clear of the bottom 40% and TikTok's icon rail.

**Duration: 20–40s, moved by retention data.** Meta normalises retention against **same-length
peers**, so length is not directly penalised — a 60s video competes against other 60s videos.
Measured top technical accounts cluster at 60–75s; the short-seamless-loop convention appears only in
legacy 2021–22 hits.

**Branding: the Inviscid AI mark**, bottom-left, every frame, identical across platforms. Instagram
permits your own logo and limits reach on third-party watermarks, so never export anything carrying a
TikTok or CapCut stamp. Reposting is the normal fate of anything that works — the cow's biggest
numbers landed on a car account with 7.9M followers — and the mark is what makes theft traceable.

---

## Case files

Publish the full case for every post: a repo, one directory per video, mesh and dicts and the command
that produced it.

Four things at once. It answers "Colorful Fluid Dynamics" permanently — the derision is real, and the
8M-follower account carrying the cow reel had a false physics claim in its caption. It makes every
objection checkable, which is what makes the loop honest. It gives people something to clone, so an
objection converts directly into an activation. And it clears the anti-AI gates on Reddit, where the
artifact is the only acceptable pitch.

AeroJAX cannot copy this — his solver is a black box. It is the one mechanic that is structurally
ours.

Be clear-eyed that **reach is decoupled from correctness**: a 3.2M-view rotor-downwash reel's top
comment called it "AI Slop" for putting the downwash at the fuselage. Rigor is the conversion and
trust play, not the reach play.

---

## Experiments

**Read this first.** Performance here is fat-tailed and subject-dominated, so **variance between
subjects swamps variance between format choices.** A 20s ceiling-fan clip against a 40s golf-ball clip
tells you nothing about duration. Every test below is within-subject or accepts a long horizon.

The useful instrument is **Instagram Trial Reels**, which send a Reel straight to non-followers,
skipping connected ranking — a cold-audience bench. Caveat: duplicate uploads get downranked, so
variants must be materially different edits.

| | Question | Design | Decision rule |
|---|---|---|---|
**Every decision rule below reads follower delta first.** Instagram reports follows attributed to a
given reel; that is the primary number. Watch time and comments are diagnostics for *why* a variant
won, not the verdict.

| | Question | Design | Decision rule |
|---|---|---|---|
| **E1** | Duration | Same sim, ~15s / ~30s / ~60s as Trial Reels, 5 subjects | Follows per 1K views. Tiebreak on watch time as a *fraction of length*; within noise → shortest, it's cheapest |
| **E2** | Which audio track | Hold constant for 20 clips, then vary | Null after 20 paired posts → stop, pick on brand grounds |
| **E3** | Text vs labels-only | Same sim, two variants, 5 subjects | Follows per 1K views. Tie → drop the question card, relieves the "majority text" risk |
| **E4** | State the question or imply it | Same test as E3 | Follows per 1K views, then **substantive comments per view**, judged by hand |
| **E5** | Does the stake ranking predict? | Write a predicted rank before each of the first 10; compare after 14 days | Correlates → delegate subject selection. Doesn't → curation is a lottery, shift to volume |
| **E6** | Cost per clip, yield rate | 5 subjects end to end | Cadence follows from cost × yield, once both are real |
| **E7** | Does the objection loop outperform? | Alternate fresh subject / objection answer, 6 pairs | Follows per 1K views. Reach is not the test |
| **E8** | Which mass subreddits carry | Each clip to 3–4 subs over weeks, never simultaneously | Stars and referral traffic, by **median** not best case. Followers are not the goal here |

**E5 is the priority and costs one written prediction per clip.** The subject thesis was built by
reading winners backwards and has never predicted anything forward. A negative result is useful: it
says stop curating and start producing.

**E7 matters more than its position suggests.** People follow on the belief that an account will keep
answering questions they have — a promise of continuity, which a one-off spike is not. If the
objection loop wins on follows while losing on reach, that is not a demotion; that is the scoreboard
working, and the loop should move to the centre of the plan rather than the edge.

---

## Still open

**Cadence** — decided by E6, not by argument. Daily is mechanically possible; agents remove the human
labour constraint that caps everyone else. The real limits are instance-hours per clip and yield rate.
AeroJAX posts constantly because his solver is 2D real-time on a laptop CPU; ours is not. The loop also
has a natural clock — comments must accumulate before there is an objection worth answering. Build ~10
clips of buffer before committing publicly.

**The pilot** — five subjects end to end. Closes E1–E5 and E6 simultaneously. This is the only
remaining item that is a build rather than a decision.

---

## Do not re-import these

Widely circulated, did not survive checking:

- **"Sends are 3–5× a like"** on Instagram — no primary source, ever.
- **"55% of Reels views come from non-followers"** — not an Instagram statistic.
- **X Premium gives 2×/4× reach** — from the 2023 repo, not current code.
- **"Links cut LinkedIn reach 60%"** — a marketing blog laundered through Forbes, contradicted by
  named LinkedIn staff.
- **"A LinkedIn save is worth 5× a like"** — real vendor data, but correlational; "save" is not in
  LinkedIn's published label set.
- **360Brew as "the LinkedIn algorithm"** — the paper was withdrawn by arXiv administrators.
- **"AI labels cut Instagram reach up to 80%"** — no Meta statement that the label demotes anything.
- **"85% of video is watched without sound"** — 2016 Facebook, three self-reporting publishers,
  internal spread 50–80%.
- **AFFiNE's "2,000 stars from Reddit in month one"** — partly disconfirmed; their launch post
  scored 92.
- **AeroJAX is a viral account** — AeroJAX is Arno Meijer's differentiable JAX/Equinox CFD framework.
  The Instagram account is real and does well (~62K followers, top posts ~186K likes), but the
  project and the account are different things.

- **X is where startups and developers are** — they are 10th at 17.1% in Stack Overflow's 2025
  survey. Founder and VC visibility is not audience presence.
- **Bluesky as the technical migration** — mobile MAU 10.4M and falling 27% YoY, roughly half its
  Q4 2024 peak; the largest institutional fluids account there has 175 followers.

**Ideas that did not survive an earlier draft of this document:** the three-beat
question/run/verdict structure · seamless short loops as a ranking lever · trending audio · "shorter
is safer" · scale as the hook · a separate industrial content track · specialist-then-mass Reddit
seeding · data-centre neighbourhood heat · X as the primary follower channel (its in-network
weighting makes followers valuable *and* hard to acquire — good retention, no cold-start firehose).

Two claims were retracted by research agents and then verified **true** by direct measurement — the X
weight table, and @theairflowguy's 3.2M-view drafting video credited to r/CFD (`followerCount: 4586`,
`heartCount: 225300`). Retractions were checked, not relayed. Day-count data for the "Day X of the top
comment" format does not exist in any reachable source; anyone quoting a number is guessing.
