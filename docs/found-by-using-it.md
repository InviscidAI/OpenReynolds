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

## Smaller ones, same origin

- **A stalled model connection hung forever.** No timeout meant a dead socket was
  indistinguishable from a model thinking hard. 300 s explicit: failing is recoverable,
  silence is not.
- **A five-minute command looked like a hang.** One line when it starts and nothing
  after. Slow calls now report elapsed time every 10 s — which also stops anything
  watching the terminal from concluding the turn ended.
- **Workspace contamination.** The volume outlives every session, so a new study opens
  in a directory full of other studies' work. A live run picked up a velocity from an
  abandoned case and carried it for several turns. The session brief now says what is
  there and whose it is.
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
