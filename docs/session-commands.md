# In a session: commands, help and completion

Anything you type in a session reaches the agent at its next step, so you can steer a
run without stopping it. A line that starts with one of the verbs below is a command
instead. A line that starts with `/` but is not one of them (a path, say) is sent as an
ordinary message rather than refused.

Every verb lives in one registry, `COMMANDS` in `openreynolds/commands.py`. The parser,
`/help`, the terminal's completion and the hosted app's suggestion list are all read
from it, so a command cannot exist in one and be missing from another.

## The commands

| Command | Also | What it does |
| --- | --- | --- |
| `/btw <something>` | `/bytheway`, `/aside` | Say something without asking the agent to stop. Your words reach it at its next step, marked as an aside. On its own, `/btw` is `/status`. |
| `/status` | `/what` | What is happening right now: jobs, thread size, tokens and cache, last sync. Answered by the harness; the agent is not told and no model call is made. |
| `/files [path]` | `/ls` | List the study's directory on the workspace, or the path you give. |
| `/renders` | `/pics`, `/images` | Show the pictures folder, newest first. |
| `/open` | | Open the study's local folder in your file browser. Terminal only. |
| `/mode [name]` | | Show the mode, or switch: `auto`, `partial`, `structured`. See [modes.md](modes.md). |
| `/model [model]` | | Show the provider, model, effort and mode, or switch model. See [switching-models.md](switching-models.md). |
| `/effort [level]` | | Show or set how hard the model thinks: `low`, `medium`, `high`. |
| `/yes` | `/approve`, `/y` | Approve the open question. |
| `/no [reason]` | `/deny`, `/n` | Decline it; the reason goes back to the agent in your words. |
| `/all` | `/yes all` | Approve it and switch the session to full auto. |
| `/help [topic]` | `/?` | The overview, or a topic. |
| `/exit` | `/quit` | Leave. Jobs keep running on the workspace; `openreynolds --study <id>` picks the study up again. |

`/status`, `/files`, `/renders`, `/open`, `/help`, `/mode`, `/model` and `/effort` are
answered by the harness and never become a turn. `/yes`, `/no` and `/all` with no
question open answer "nothing is waiting for an answer".

`/exit` typed while the agent is mid-turn (the hosted app's End button sends the same
word; the interface's `ctrl+c` and a closed stdin count the same) ends the turn at its
next safe point rather than when the model happens to stop: a held `job_check` or
`mesh_wait` returns at once, a command already running finishes and its result is
recorded, nothing further is started, the model is not asked again, and the transcript
carries one line from the harness saying the session was ended by the person. Then the
ordinary close-down runs. A mesh desk still building is told to stop at its next
command.

## /help

`/help` on its own lists every command with its arguments and a summary, the three
modes, how completion works where you are, and the topics:

| Topic | What it covers |
| --- | --- |
| `/help commands` | Every command with its aliases and its longer description. |
| `/help modes` | The three modes, what each gates, how to answer, how to switch. |
| `/help model` | `/model` and `/effort`: the spec forms, when a switch takes effect, and what it costs. |
| `/help tools` | A plain-language line for each of the agent's tools, `checkpoint` included. |
| `/help keys` | The keys: completion, and in the terminal interface the pane shortcuts. |

An unknown topic answers with the list of topics and then the overview. In the hosted
app, `/help` gives the web version: `/open` is left out and the keys are the composer's.

## Completion in the terminal interface

Type `/` in the prompt and a list of matching commands opens directly above it, each
with its summary. The best match also appears as grey text in the prompt.

| Key | What it does |
| --- | --- |
| Up, Down | Move the highlight through the list (it wraps). The grey text follows. |
| Tab | Take the highlighted suggestion, or the grey text when the list is closed. While the text starts with `/`, Tab never moves focus to another pane. |
| Right | Take the grey text. |
| Enter | Send the line. On a half-typed verb (`/stat`) or an argument cut short of a choice (`/mode par`), it takes the highlighted suggestion instead. |
| Esc | Close the list for the current text. Typing opens it again. |
| Click | Take a suggestion. |

A verb that takes an argument completes with a space after it, so taking it moves
straight on to that argument's choices:

- `/mode ` lists the three modes with their labels;
- `/model ` lists the current model and the models this provider is known to have;
- `/effort ` lists `low`, `medium`, `high`;
- `/help ` lists the topics;
- `/yes ` offers `all`.

Aliases are suggested only when typed in full.

While a question is open, the prompt's placeholder becomes the answer hint and `/yes`,
`/no` and `/all` come first in the list.

The other keys in the interface: `ctrl+t` shows or hides the agent's thinking, `ctrl+f`
the workspace files, `ctrl+g` the renders, `ctrl+r` refreshes the file list, `ctrl+l`
clears the activity pane, and `ctrl+c` quits with jobs left running.

`--plain` has no completion; the commands work the same when typed in full.

## Completion in the hosted app

Typing `/` in the message box opens the same list above it, fetched from
`GET /api/commands` rather than written into the page. Up and Down move, Tab picks,
Enter picks unless the line is already a whole command (then it sends), Esc closes, and
a click picks. Under the box are a *help* button and mode, model and effort choosers
for the running session; each sends the line you could have typed.

## For a program

`--output-format stream-json` takes commands the same way as a terminal: send the line
as `{"type": "user", "text": "/status"}`. What the harness answers comes back as a
`status` event.
