# What running it found

Every defect here survived a green test suite. None was found by unit tests, and most
were invisible until something above them was fixed first — which is the argument for
driving the whole product from the outside rather than testing its parts.

They are listed in the order they became findable.

## `stop --force` never killed anything

`pkill` takes exactly one pattern. `pkill -9 -x mpirun -x simpleFoam` exits 2 with
*"only one pattern can be provided"* and kills nothing — and `2>/dev/null; true` turned
that into a silent success. So `stop --force` printed a clean report while eight cores
kept running, which is worse than not having the flag: you read "done" and walk away.

I had previously reported this as verified. It wasn't: the run I checked had no
surviving solvers, so the force path never executed. Verifying the happy path is not
verifying the feature.

**Fix:** one `pkill` per name, exit codes read rather than discarded (0 matched,
1 already gone, anything else reported). Proved against two real named processes on the
live instance: running before, gone after.

## A ladder script restarts the solver you just killed

Visible only once the first bug was fixed. A study is usually driven by a script working
through a mesh ladder, so killing the solver it is currently running frees it to start
the next one. The check three seconds later finds a brand-new `simpleFoam` and reports
failure — while everything it killed did in fact die.

**Fix:** keep looking, up to four passes, and say how many it took.

## The harness could not tell a stall from a finished turn

Both came back as "here is the reply", so a run that outlived its cap kept talking to
something that had stopped listening.

**Fix:** a turn that ends by timing out, by the agent exiting, or by producing nothing
is labelled as such. A long wait prints a heartbeat, so a seven-minute wait and a dead
run no longer look identical from outside.

## The harness talked over the agent

It treated 12 s of quiet as "turn over". A command taking longer than that cut the reply
mid-sentence — literally `…render it in 3D so you` — and fired the next message while
the agent was still working. Two personas spent their last turns exchanging remarks with
a progress line.

**Fix:** wait for the prompt the product already prints, not for silence. Silence stays
as the fallback for watch mode, where there is no prompt. Seeing the prompt required
reading bytes rather than lines: it has no trailing newline, since nothing follows it
until somebody answers, so a line reader cannot see it until other output arrives — and
none ever will.

## A pasted paragraph became four separate turns

Visible only once turn detection was correct. Two bugs had been cancelling into a
plausible-looking transcript.

Every line of stdin was its own turn, so a message with a paragraph break in it became
two user turns: the agent answered the first half, and the second half arrived mid-turn
as an interruption. Four messages went in; the session recorded six turns; every reply
after that landed against the wrong question.

The transcript read as an agent dodging a direct question and then answering with
silence. The session log showed the opposite — a 2,600-character answer, and then
*"Bet my job on it."* **The user never saw either.**

This is not a test artifact. Anyone pasting a paragraph into the plain terminal hits it,
and what they see is an agent that answers your first sentence and ignores the rest.

**Fix:** lines that arrive together are one message. Nobody types a second line inside
200 ms; a pasted paragraph arrives all at once. EOF arriving mid-paste is put back,
because it ends the session, not the message.

## Watch mode showed no prompt

Three of four personas timed out at ten minutes with the same shape: the agent launches
a mesh ladder, says it will be back in twenty minutes, and the session shows nothing but
`watching 1 job(s)` repeated. You *can* type — watch mode polls for it — but nothing on
screen says so.

**Fix:** watch mode prompts. It is the same state the idle prompt describes: the session
is waiting on you. A one-shot run does not prompt, because inviting input where there is
no way in is a lie.

## A dead session kept accepting input

The TUI worker could end — normally or by raising — and the input box stayed live:
everything typed into it accepted, echoed, and discarded.

**Fix:** say the session ended and disable the box. Writing the test for it surfaced two
more: reporting the end crashed on a torn-down screen, and hopping to an event loop
mid-shutdown blocked for seconds.

## Every study opened into everybody else's files

The volume outlives every session, and a new study opened straight into it, among
every other study's cases. That is not a clean slate by any reading of the words.

One run inherited a velocity from a case an earlier session had abandoned and carried
it several turns before noticing. A later one opened by finding somebody else's
simulations, rebuilding one from scratch and re-running it to check they were real
before it would use them -- which is exactly right, and is also several minutes spent
on a problem that should not have existed.

**Fix:** a study gets `/work/<study-id>`, made at session start, and it is the default
working directory for `bash` and `job_start`. Starting a new project starts somewhere
empty. Nothing to clear, no flag to remember. The volume still persists; studies made
before this keep the whole workspace, because moving their files out from under them
would be worse.

## Three things that were defined and never called

Each looked done from the outside -- named, documented, visible in `--help` -- and did
nothing.

- **`--fresh`** moved earlier work aside so personas could not contaminate each other.
  It was never invoked. Four runs shared one workspace because of it. Found by reading
  an agent's own words in a transcript: *"the old simulations in this workspace were
  done by someone else."*
- **`--turns`** was clipped by `min()` against the persona's own default, so asking for
  14 silently gave 8 -- and the run then reported "ran out of turns" as though the
  conversation had run its course.
- **`args.turns or persona.turns`** treated an explicit `--turns 0` as "unset".
- **The line telling the user where their files are.** The edit silently failed, the
  shell chain hid it, and it was reported as done. The test written for it would have
  passed against a no-op, because its own fixture routed the output to `/dev/null`.

The lesson is the same each time: a flag that is not exercised by a test is a comment.

So `tests/test_wiring.py` checks the structure rather than the instances. Every click
option arrives as a function parameter, and a parameter nothing reads is an option that
does nothing while `--help` still advertises it. Every argparse destination has to be
read. Every `View` method has to be called, every `Config` setting read, every
`ToolContext` field used, every `Backend` method exercised by something above the
protocol.

It found a fifth on its first run: `files --depth` was declared, documented, given a
`show_default`, and threaded nowhere. Fixed rather than exempted.

`ToolContext.view` is the case that justifies the whole file. It was added, wired into
the tools so job state would reach a panel, and then not passed in by the session. The
panel never updated for a release, and every test stayed green, because they built the
context themselves and never went through the session that was failing to.

## Smaller ones, same origin

- **A stalled model connection hung forever.** No timeout meant a dead socket was
  indistinguishable from a model thinking hard. 300 s explicit: failing is recoverable,
  silence is not.
- **A five-minute command looked like a hang.** One line when it starts and nothing
  after. Slow calls now report elapsed time every 10 s — which also stops anything
  watching the terminal from concluding the turn ended.
- **cp1252 stdout killed the test harness** on the first sigma in the first reply — the
  exact failure the product already defended against, in a script that didn't. The
  defence moved to one module so the two cannot be fixed separately again.
- **Git Bash rewrites `/work`** into `C:/Program Files/Git/work` before the process sees
  the argument, so a correct command came back as a 404 naming a path nobody typed.
- **One persona's crash cost four runs.** The suite died on persona one, turn one, and
  the other three never started.

## What the geometry renders taught

`toolbox/geometry_view.py` worked on the first live run, and looking at what it produced
found two defects a test never would have:

- The enclosing box, drawn solid, hid the sphere and the lid inside it — which is the
  normal case in CFD, where the outer domain hides the interesting geometry.
- Every axis tick read `0.0` on a 6 cm case, pyvista's default `%.1f`. That is precisely
  the mm-versus-metres question the picture exists to answer, and it was unanswerable
  from it.

A third came from looking again after fixing those: a sphere rendered as a black blob,
because its facets were smaller than the lines outlining them. Edge visibility is now
decided per part from how many pixels a facet actually gets.

## Things that were true and nothing kept true

A different shape of the same problem. Not code that does nothing — prose that says
something the code stopped doing.

- **The design document** still described one shared `/work` after studies got
  directories of their own. It is the document someone reads first, and it will be
  believed.
- **`openreynolds studies`** existed from the first commit and the README never
  mentioned it. A command nobody is told about is a command nobody uses, which is the
  same outcome as not having written it.
- **A test count in the README** was already wrong by 112 the moment it was written. A
  stale count in a document about verification is worse than none: it is a claim
  nothing keeps true, sitting next to claims that are.

All three are now checked. The command list is read out of the click decorators, so it
cannot fall behind them.

## What only shows up with a model in the loop

- **Capture fails quietly on purpose** — it must never delay or break a study — so
  nothing ever announced that it had stopped. `doctor` opens a study and says. It was
  working, but establishing that meant reading eight `session.json` files by hand, which
  is archaeology rather than a check.
- **Nothing said how long anything took.** A command that ran four minutes and one that
  ran two seconds came back as the same line. A job going two hours read like one
  started a minute ago.
- **The loop had no visible joints.** The activity pane coloured tool calls but never
  marked where one round of think-then-act ended, so a turn of three rounds looked like
  one of thirty. A live run then showed seven rounds inside a single turn at roughly
  twenty seconds each — which is also why anyone watching reaches for the keyboard.

## Two messages sat unread for twenty-five minutes

A transient `pimpleFoam` run on six ranks, t = 1.19 s of 6 s. The model ended its turn
with a question — "anything you want added while it runs?" — and the user answered it,
twice: *I want a gif, how far along is the sim?* and, a while later, *yo?* Neither was
answered. The screenshot showed a finished turn, two echoed messages, and nothing else.

Neither message had reached the model. The transcript ended at the model's question;
the mirror's local copy was still receiving `processor1/0.35/U_0.gz` when the screenshot
was taken. The turn-end sync ran on the session thread, the session thread is the one
that reads what was typed, and with `everything=True` a six-rank transient case gives a
sync an unbounded amount to do — hundreds of small per-processor field files in batches,
each batch a round trip, each failure retried a file at a time. The messages waited in
the queue behind all of it.

Two things were wrong and they compounded. The block itself: fixed by asking the
background thread to sync (`LiveMirror.catch_up`) instead of doing it in line. And the
silence: the screen had no way to show that anything was happening, so a busy session
and a dead one looked identical. The one grey stage line had been overwritten by the
last event and then cleared at the turn's end. That is what the bar is for: a running
solve stays on it with a percentage while the model thinks or the mirror copies, and
the thinking has a clock on it. A session that is doing something should look like it.

## "It keeps doing its own work and doesn't respond"

The next screenshot after the mirror fix: the agent 134 seconds into a single `bash`
step (step 34, one call), a solve running, and the user's "how long will this simulation
take?" sitting under a dim "(sent - it reads this at its next step)" with no answer for
minutes. The user's words: "it still doesn't stop and respond, it keeps doing its own
work, and it's especially problematic when the task takes a lot of time."

This one is not a bug to fix in place — it is structural, and worth writing down as a
limit rather than a defect. The agent runs on one thread. A `bash` call holds that
thread for up to 300 seconds and cannot be interrupted; a thinking phase holds it for
minutes. The interject mechanism delivers a typed message when the *next* tool result is
assembled — which, mid-`bash`, is 134 seconds away. Nothing on that thread can answer
sooner, and the free-will contract forbids the harness from forcing the model to answer
at all.

So the fix is a *second* thread with a *second*, cheap agent: the front desk (`desk.py`).
It is read-only, it never touches the main agent's thread or tools, and it answers the
user from the transcript and the live job facts within seconds — labelled `desk` so it
is never mistaken for the agent. The main agent still gets the message unchanged and
answers in its own time. Two things this is careful about: the desk speaks in the third
person and never promises on the agent's behalf (it is a narrator, not an impersonator),
and it only speaks when the agent is actually busy (an idle session answers for itself).
The same desk writes the plain-language "now" line the user also asked for — "what am I
doing right now?" — which the mechanical phase labels on the bar were not.

The companion complaint in the same message: the bar was always on. A progress strip
that only pulses "busy" is noise. It now appears only when real compute is running — a
solve, a mesh, a decomposition, a sync — and takes no room otherwise.

## The session stopped answering because it kept re-uploading its own pictures

A four-scenario container study, hours in, rendering a large visual set. The user
typed "are you still working? / hello? / yo?" and got nothing. The activity pane told
the story if you knew to read it: "the thread is intact — say something to continue"
printed over and over, which is what prints when a *model request fails*. Every turn was
failing, so nothing could answer.

The cause was in the thread. A `read_file` on a render returns the image to the model as
an image block, and that block stays in the conversation and is re-sent on every
subsequent turn. Over a long render-and-look session the thread had accumulated
**twenty-one images, five megabytes of them, one a single 1.9 MB PNG** — and all of it
was re-uploaded, as ~7 MB of base64, in every request, on top of a 298k-token context.
The requests got large enough that the model API began refusing them, and once it was
refusing them the session was simply stuck: each new message started a turn that failed
the same way.

Nothing was actually broken with the work — every job had finished and all forty-six
renders were already on the user's disk. It was pure context bloat. Three fixes: the
thread now **sheds the pixels of images the model has already looked at** (keeps the
one-line description and the path, drops the base64; the two most recent stay whole), so
requests stay a size the API will accept. After two failures in a row the harness stops
repeating "the thread is intact" and **says plainly** that the API is failing, that the
work is safe, and to resume in a fresh thread with `--study`. And a message typed at the
prompt now reaches the front desk even when the main turn is failing — the reason someone
asks "are you still working?" is that the agent went quiet, which is exactly a failing
turn.

The lesson is the same one the delivery fix taught from the other side: an image is
cheap to look at once and ruinous to carry forever. Looking is a moment; the thread is
forever, and the two must not be the same thing.

## A test case that was the problem

The controller persona ran out of time twice, and nothing was broken. Its goal asked how
air divides between two branches of a symmetric wye. A symmetric wye fed symmetrically
splits 50/50 by symmetry; the real question is what breaks the symmetry — the fan
upstream, the resistance downstream — and none of that is in the geometry.

The agent said exactly that on its second turn and was right, then spent twenty-eight
minutes on a research question wearing the clothes of a simple one.

Worth recording because the failure looked identical to a product failure from the
outside: a run that does not finish. The persona exists to test whether the agent can
keep you informed while it works, not whether it can close an open problem. Goals now
name a geometry, a flow rate, and one number to produce, and a test holds them to it.

## What the runs say about the toolbox

Across every persona run: `geometry_view.py` was used zero times, and `read_file` on a
PNG was used constantly — the agent renders its own figures and then looks at them,
including one it named `WHAT_TO_LOOK_FOR.png` and read back twice.

That is the design working rather than failing. The capability — being able to see what
you drew — gets used every session. The pre-built script does not, because the agent
would rather write the render its own case needs. It also says which half was worth
building: the image plumbing is load-bearing, the script is a convenience.

## A field was added to the index and every index on disk went stale invisibly

`search.py` rebuilds when the stamp's schema version does not match the code's, which is
the mechanism that stops a stale index answering authoritatively about something else.
A `notes` field was added to the earned row so `failure` queries had words to match on,
and the schema version was left where it was.

So the index already on disk passed the staleness check, was loaded, and answered
`failure "diverged"` with nothing at all — against a study whose phase table says
`diverged at t=0.31` in as many words. The rows had no `notes`, and nothing anywhere
said so. A guard defeated by forgetting to move the number it reads.

**Fix:** the version bumped, and a test that pins the exact field set of both rows
against it, so the next change to a row has to move the number rather than remember to.

## The ranking put a coincidence of filing above a match on a field

The rule is that an exact match on an indexed field outranks the same word appearing
somewhere in a path — `incompressible/simpleFoam/pitzDaily` contains `simpleFoam` and so
does the `application` entry of a case that runs it, and only one of those is an answer.

Running the query the design document itself uses, `internal incompressible steady`, put
at the top a case that had matched `incompressible` *in its directory name*, above cases
that matched it as a field. The regime class is offered to the matcher whole and split on
its hyphen, so a class that is a single word — `steady`, which 21 cases carry because
they have no properties file to read compressibility from — yielded the same pair twice
and scored six points for one concept.

**Fix:** the pairs de-duplicated before scoring. Two tests, one from each side: a
one-word class must not outscore a two-field match, and a path coincidence must never
add up to a field.

## A concentration report collapsed on the one family it was written for

The tutorial tree is not a neutral sample: 89 of its 557 cases are one solver from one
family. So a report of "what do cases set this key to" says how many *families* hold each
value as well as how many cases, because five cases from one directory of siblings and
five unrelated ones are different claims that the counts cannot tell apart.

Families were grouped by the two directory levels immediately above each case. Run
against `div(-phi,Ua)` — a key only the adjoint tutorials set, every one of them under
`incompressible/adjointOptimisationFoam` — it reported **47 families**, the largest named
`1_Inlet_2_Outlet/levelSet`. The tree is not uniformly deep, that family nests further
down than the others, and the proxy fell apart on the single case it existed to catch,
in the direction that hides the problem.

**Fix:** families measured down from the tree root, which the caller already knows,
rather than up from the case. The same query now reports 2 families, the largest holding
68 of the 69. It also showed what the honest version is worth: for `div(phi,U)`, the
fourth most common value across the corpus — 37 cases, 12% — turns out to be held by one
family alone.

## An adoption was recorded against a query that never offered it

The retrieval log keeps the whole ranked list rather than the winner, because the failure
worth catching is a vendor hit and an earned hit disagreeing and the earned one being
taken. `took <path>` records what was actually used, and it attached to the most recent
retrieval.

Three queries were run and the case adopted came from the first. The record hung it on
the third, so the log said a `failure` query had produced a tutorial it never returned.
"Most recent" is a guess, in the one file whose whole purpose is not guessing.

**Fix:** it attaches to the most recent retrieval whose hits actually contained that
path, which is a fact about what was on the screen. An adoption no query offered is kept
unattached and surfaced separately — the agent used a case the corpus never handed it is
the more interesting record, not a broken one.

## Smaller ones, same origin

- **The reader could not read the key the design document asks about.** Its own example
  query is `div(phi,U)`, and the dictionary reader keyed on identifiers. `fvSchemes` is
  keyed by the term being discretised, so `div(phi,U)` — the most common non-trivial key
  in the tree, in 299 cases — and every quoted regex key in `fvSolution` read as nothing.
  Found by checking the documented example against the real tree rather than by testing
  the reader, which passed.
- **Checking `blockMeshDict` before `snappyHexMeshDict` mislabels the snappy tier.**
  62 of the 64 snappy cases also carry a `blockMeshDict`, because snappy cuts its mesh
  out of a background block that blockMesh builds. In the order the design document's
  table gives, a query for a snappy precedent finds 2 cases instead of 64. Found by
  counting the corpus before writing the schema, which is the cheap version of finding
  it by using it.
- **Paths compared as strings are two cases when one is pasted from a shell.** A hit
  recorded as `...\simpleFoam\motorBike` and an adoption typed as
  `.../simpleFoam/motorBike` did not match, so the log reported that the corpus had never
  offered the case that was used. Compared as paths, with a `normpath` to settle a stray
  `..`, they are the one case they are.
- **A test fixture that shared a directory built a case the corpus cannot contain.** The
  single-case harvest helper reused one tree across calls, so a second call's properties
  file landed beside the first's and produced a case carrying both
  `thermophysicalProperties` and `transportProperties`. The compressibility test failed
  for a reason that was entirely the fixture's.

## An unset `$FOAM_TUTORIALS` indexed the working directory and called it a success

`Path(os.environ.get("FOAM_TUTORIALS", ""))` is not an empty path. `Path("")`
normalises to `Path(".")`, and a `Path` defines no `__bool__`, so it is truthy and
`is_dir()` is True. The guard `if not args.tutorials or not args.tutorials.is_dir()`
could therefore never fire, and the message it would have printed was unreachable code.

With the variable unset, `corpus.py build` walked the current directory, found whatever
`system/controlDict` happened to be under it, and printed `indexed 1 / built against
unknown unknown` with exit 0. `search.py` was worse: it has no guard, and `ensure()`
builds on the first query, so a single query wrote a stamped index of the wrong tree and
answered from it. An index of the wrong corpus that reports success is worse than a
refusal, because nothing downstream can tell.

**Fix:** the environment is read as a string and only becomes a `Path` when it is
non-empty, so the guard sees `None`. The guard belongs on *building*, not on asking — an
index already on disk and not stale is answerable with no tree in sight, which is every
query after the first.

## A stale index could not be detected because the stamp did not say what it was built from

Found while verifying the previous one. Having produced an index of the working
directory, correcting `$FOAM_TUTORIALS` and querying again returned the bogus row with
no rebuild and no warning: the stamp matched on schema and on version, and the tree was
the one thing it recorded nothing about.

`$WM_PROJECT_VERSION` being unset made it permanent. `staleness` returns None on an
unknown live version — deliberately, so it does not rebuild on every query forever — and
that short-circuit sat *above* every remaining check. So an agent that fixed its
environment mid-session kept being answered about the wrong corpus for the rest of it.

**Fix:** the stamp records the tutorial and work trees, and the tree is compared *before*
the version short-circuit. An older stamp records no tree, which is something unknown
rather than a mismatch, so the schema version was bumped to make those rebuild once
instead of being special-cased forever.

## Deleting one index file left every query answering "nothing matched" for good

The module docstring said the index is rebuilt when it is missing. It was not:
`staleness` read `corpus.stamp.json` and nothing else, and the loader passes over a file
it cannot open, one tier at a time. So with a valid stamp in place and
`tutorials.index.jsonl` removed — a hand clean of the corpus directory, or a partial
copy — every query printed `nothing matched` and exited 0, indefinitely.

**Fix:** both index files have to exist, checked alongside the stamp, and the docstring
now describes what the code does.

## The one command that did not print a tier merged the two tiers

`keyword` answers "what does the corpus set this to". It took a tier filter and the
command line passed its default of `None`, so a shipped tutorial and this instance's own
finished study were counted into one table — and the footer called all of them "tutorial
families". Verified with one of each disagreeing about `div(phi,U)`: the output was a
50/50 split with nothing on screen to say that half of it was the system's own prior
output.

The test suite already had a test whose docstring said mixing them "is how a house style
becomes a fact". The capability was there; the default was wrong. Every other command
prints the tier of every hit.

**Fix:** one distribution per tier, each labelled, and the footer says "tutorial
families" only when they are tutorials. Reported separately rather than restricted to
the vendor tier, because "what have my own studies used" is a real question — just a
different one, and the answer has to say which it is answering.

## A file named `physicalProperties` was taken as proof of incompressibility

`compressible_of` read `transportProperties` **or** `physicalProperties` as
incompressible. On the Foundation fork that second name replaced *both*
`transportProperties` and `thermophysicalProperties`, so its presence says nothing
either way — and the review raised it as an unverified fork risk.

It turned out to be verifiably wrong on the tree being indexed. The single
`physicalProperties` in the whole of v2512 belongs to
`electromagnetics/electrostaticFoam/chargedWire` and contains `epsilon0` and `k`: a
vacuum permittivity and a mobility. There is no fluid transport in it at all, and it was
being indexed as an incompressible case on the strength of the filename.

**Fix:** the file is opened. A `thermoType` in it is a thermophysical model, a `nu` or a
`viscosityModel` is the incompressible form, and anything else is null. The charged-wire
case now reads null, which is what had actually been read about it.

## A query reported that it understood words the tier it searched had never seen

`failure` searches the earned tier only, and the unmatched-word notice was computed over
the rows of both. So `failure "pitzDaily"` — a word that appears solely in a vendor path
— printed a bare "nothing matched" and wrote `unmatched: []` into the retrieval log,
telling the reader and the record that the query had been understood and simply had no
answer.

**Fix:** the notice takes the same tier filter the search does.

## The provenance log lost an eighth of its records when two processes wrote at once

`retrievals.jsonl` was appended with `open(path, "a")` and one `write`, on the reasoning
the manifest already states: concurrent appends interleave whole lines but never halves
of one, which is why these files are JSONL rather than a JSON array that has to be read,
edited and rewritten.

The reasoning was inherited rather than measured, and it is wrong. Six processes
appending 150 records each produced **763 lines instead of 900, 23 of them cut mid-JSON**
-- records lost outright, not merely interleaved. Buffered append mode is a seek-to-end
followed by a write with nothing joining them, so two writers agree on an offset and one
overwrites the other.

Rewriting it as a single `os.write` to an `O_APPEND` descriptor did not fix it either:
796 of 900. That is one syscall, which POSIX makes atomic for a regular file, but the
Windows CRT implements append as the same seek-then-write pair.

**Fix:** a lock file -- created `O_EXCL`, released after the write, no platform branch
and nothing imported. 900 of 900, none torn. Past the retry budget the write proceeds
unlocked on purpose: a record written into a possible race beats a record dropped for
certain, and a process that died holding its lock must not silence the log for good.

A provenance log that quietly loses an eighth of its records is worth less than no log,
for the same reason a stale index is worth less than no index.

Measured on Windows. On Linux a small record is one buffered `write()` and the loss mode
probably does not arise, but "probably" is not what a record of what happened should
rest on, and the suite runs on developer machines too.

**`study_state.record` has the same defect and is not fixed here.** The same test
against the study manifest: 805 of 900 artifacts survived, and `artifacts()` duly
reported 805. It is the file the claim came from, five toolbox scripts write to it, and
changing it is a wider change than the corpus work -- so it is written down rather than
quietly altered.
