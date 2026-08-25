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
