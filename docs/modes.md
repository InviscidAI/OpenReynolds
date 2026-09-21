# Modes: full auto, ask before compute, structured

A session runs in one of three modes. The mode decides one thing: how much the agent
does before it asks you. It never changes what the agent is told about CFD, and it
never touches `bash`, `write_file`, `read_file`, `fetch`, `job_check` or `job_kill`.

| Mode | Label | What it gates |
| --- | --- | --- |
| `auto` | Full auto | Nothing. The agent decides everything and nothing waits for you. The default. |
| `partial` | Ask before compute | Every `job_start` and every `mesh` call is put to you before it runs. |
| `structured` | Structured | The study goes in stages: you approve a plan, then each stage at a checkpoint. `job_start` and `mesh` are held until a plan has been approved. |

The source of truth is `openreynolds/modes.py`. The loop, the briefing, `/help modes`
and these docs all read their wording from there or describe it.

## Why there are modes at all

OpenReynolds' design (`docs/design.md` section 1) forbids the harness from requiring
approvals or enforcing an order of work, because a harness that tells the model how to
do CFD caps the work at whatever its author knew. That rule is unchanged in full auto.

What it never meant was that *you* cannot ask to be consulted. A solve on forty cores
is your money, and a study you intend to defend is your plan. So the other two modes
exist for the person who wants them, and the harness gates exactly what that person
chose to have gated and nothing else. The judgement inside each step stays the
agent's.

## Full auto (`auto`)

The briefing is the same bytes it was before modes existed, the tool list has no
`checkpoint`, and no tool call is ever held. This is how every earlier version of
OpenReynolds ran.

## Ask before compute (`partial`)

`job_start` and `mesh` are the two calls that spend compute: a detached solve, and the
mesh desk (a second agent building a mesh on the workspace). Each one is put to you
before it runs, with what would run:

- a job shows its name, its command and its working directory;
- a mesh shows the shape as the agent described it and the case it goes into.

You answer with one of three things:

| Answer | What happens |
| --- | --- |
| `/yes`, or `y`, `yes`, `ok`, `go`, `approve` typed on its own | The call runs. |
| `/no [reason]`, or `n`, `no`, or any other words | The call does not run. The agent gets an error result saying you declined, with your words verbatim when you gave any. |
| `/all`, or `/yes all`, `a`, `all` | The call runs and the session switches to full auto for the rest of the session. The agent is told so. |

Anything you type that is not a yes, a no or an "all" is steering: it declines the call
and goes back to the agent as your reason. `/status`, `/files`, `/renders`, `/open`,
`/help`, `/mode`, `/model` and `/effort` are answered while the question stays open.
`/exit` declines the call and then leaves.

Only what you type after the question appears answers it. Something you typed while the
agent was still working reaches it as an ordinary message alongside the tool results,
and the question waits for a fresh answer.

The briefing carries one sentence saying you chose this, so the agent is not surprised
by a declined call.

The mesh desk runs its own `bash` loop inside a `mesh` call. The question is asked once,
on the outer `mesh` call: approving it approves the desk's work.

## Structured (`structured`)

The agent gets a ninth tool, `checkpoint`, with three fields: `stage`, `summary` and
`next`. Calling it puts the summary and what comes next in front of you and waits:

- **approve** (`/yes`): the result tells the agent you approved and to carry on with
  what it said comes next;
- **anything else**: the result carries your words as the changes you asked for, and
  nothing is approved;
- **approve all** (`/all`): approves, and switches the session to full auto.

`job_start` and `mesh` are **held** until at least one checkpoint has been approved
since the session entered structured mode. A held call does not run; the agent gets an
error result saying it was held, why, and that nothing ran. After that first approval,
the plan, they run without further questions. Further checkpoints are the agent
reporting back at the end of a stage: the briefing says you want one after each stage,
but nothing in the harness forces a checkpoint, so when to call one is the agent's call.

The stages are the guided pipeline's own phases, from `toolbox/study_state.py`:

    geometry, preview, mesh, checkMesh, probe, solve, reconstruct, render, animate, report

The briefing says, in one sentence, that you chose structured mode, names those stages,
and says you want to agree a plan at a checkpoint and see one after each stage. What a
stage involves, and whether a study needs all of them, is still the agent's call.

## Choosing a mode

At the start:

```bash
openreynolds --mode partial          # or: auto, structured
OPENREYNOLDS_MODE=structured openreynolds
```

The aliases are accepted wherever a mode name is: `full`, `full-auto`, `fullauto` and
`free` mean `auto`; `ask`, `approve`, `approvals` and `partial-auto` mean `partial`;
`plan`, `staged`, `stages` and `guided` mean `structured`.

The order of precedence is `--mode`, then `OPENREYNOLDS_MODE`, then `mode` in the config
file, then `auto`. A value that is not a mode is ignored: a bad `OPENREYNOLDS_MODE` is
warned about and the config file's `mode` applies (or full auto), and a bad `mode` in
the config file falls back to full auto without a warning. On `--study <id>` an ignored
value names no mode, so the study carries on in its stored mode.

**Resuming.** The mode is stored in the study's `session.json`. `--study <id>` carries
on in the stored mode unless `--mode` or `OPENREYNOLDS_MODE` names one.

**Non-interactive runs.** `-p` has nobody at the terminal to answer a question, so `-p`
with `partial` or `structured` (given, or stored on the study being resumed) is refused
as a usage error with exit code `2` before an instance is acquired.

## Switching mid-session

```
/mode                 which mode the session is in, and the three to choose from
/mode partial         switch
```

A switch applies from the next tool call, including the next call in a turn that asked
for several. Switching into structured mode starts it
afresh, so a plan has to be approved in it before compute is spent again. The agent is
told about the switch in the harness's voice, as a fact.

Switching into or out of structured mode adds or removes `checkpoint` from the tool
list, which rewrites the prompt cache once. That cost is paid only when you switch.

If you switch while a question is already open, that question still waits for your
answer.

## In the hosted app

*New study* has a mode choice (Full auto, Ask before compute, Structured), remembered
for next time. Under the composer a mode chooser sends `/mode <name>` for the running
session, and the top bar shows the current mode. A question arrives as a card in the
conversation with *Approve*, *Decline* (*Ask for changes* at a checkpoint) and
*Approve all*; each button sends the same line you could have typed.

## For a program

With `--output-format stream-json` a question is an `approval` event and its answer an
`approval_done` event; a mode change is a `mode` event. Answer by sending
`{"type": "user", "text": "/yes"}` (or `/no <reason>`, or `/all`) on stdin. See the
README's *Driving it from a program*.
