# Security

## Reporting a problem

Write to **security@inviscidai.com**. Say what you found, how to reproduce it, and which
version or commit you were on. Please do not open a public issue for anything exploitable
until there has been a chance to fix it; ordinary bugs belong in the
[issue tracker](https://github.com/InviscidAI/OpenReynolds/issues). You will hear back
within a few working days.

## Supported versions

OpenReynolds is pre-1.0. Fixes go to `main` and to the latest release only.

| Version | Supported |
|---|---|
| latest release on `main` | yes |
| anything older | no -- upgrade |

## What this program does by design

Most of what follows is not a vulnerability to be fixed; it is how the agent works, and
the honest thing is to say so plainly so that nobody is surprised by it. The design is
in [`docs/design.md`](docs/design.md).

**The agent runs an unsandboxed shell in its workspace, with no approval gate.**
The `bash` tool executes whatever the model decides to run. There is no allow-list, no
confirmation step, and no rewriting of commands -- the harness may not block or gate a
tool call on policy grounds, and that is deliberate. The workspace is the sandbox, and
where it is depends on how you run OpenReynolds:

- **A Reynolds workspace (the default):** commands run as root inside the study's own
  container in the cloud, and never on your machine.
- **`--local`:** commands run in a Docker container on your machine, as your user, with
  no network. They reach the workspace directory (`~/.openreynolds/work`, mounted as
  `/work`) and nothing else on your machine.
- **`OPENREYNOLDS_LOCAL=native`** (and `OPENREYNOLDS_LOCAL=1`, its older spelling): commands
  run directly on your machine, as you, with **no sandbox at all**. Use it only where
  you would let the model type into your own terminal.

Whatever the workspace, what comes back to your machine from a study is written under
`./studies/<id>/`.

**The modes are about compute, not about safety.** In the default mode, full auto,
nothing is ever held. If you choose *ask before compute* (`--mode partial`), every
`job_start` call is put to you before it runs, and in *structured* mode it is held
until you approve a plan. `bash`, `write_file`, `read_file`, `fetch`, `job_check` and
`job_kill` run without asking in every mode, so a mode is a way to decide when compute
is spent, not a sandbox or an allow-list for commands.

**`/work` is readable by the model.** On a Reynolds workspace each study has its own,
holding what that study and you put there (`openreynolds push`, uploads). A local
workspace is one directory shared by every local study, so whatever an earlier study
left there is visible to every later one. A file in the workspace can contain text
that looks like instructions; the model reads files as part of its work and nothing
stands between the two. Keep secrets out of `/work`.

**Transcripts are uploaded unless capture is off.** By default every message, every
tool call and its (capped) output, every fetched artifact and the end-of-session
results go to the workspace service you configured, on a background thread, so the
study is kept somewhere other than one laptop. `--no-capture` turns it off for a
session, `OPENREYNOLDS_CAPTURE=0` for an environment, and `openreynolds doctor` says
which it is. The local mirror under `./studies/` is written regardless.

**A trace of each study is shared while this machine is signed in, unless sharing is
off.** Separately from capture, and whether the workspace is the service's or this
machine's, a signed-in OpenReynolds sends a trace of each study for improving Reynolds:
usage (tools and whether they worked, the model, tokens, timings, the workspace kind
and solver, the operating system and versions) and study content (messages, the agent's reasoning, tool inputs and outputs, cut
at 20,000 characters each). It says so in one line at sign-in, and a study's trace is sent
after the study ends. Credentials it knows (the model key, the service key, environment
variables named like secrets) and the shapes of common keys are removed before a trace
is written to its queue (`openreynolds/redact.py`); that does not remove all personal or
engineering information. `openreynolds sharing off` or `OPENREYNOLDS_SHARING=off` stops
it and discards the queue, and nothing is sent without a service key. The web app's
Account page deletes every trace shared. What a trace is, and how long it is kept:
the privacy policy at https://tryreynolds.com/privacy.

**`preferences.md` is sent verbatim to the model provider.** So is everything you type,
everything the model reads with `read_file`, and every tool output it sees. The
standing note beside `config.json` is relayed word for word into the briefing of every
session, which is its whole purpose; it is the wrong place for a credential or anything
you would not send to the provider directly.

**Credentials** live in `config.json` under your user configuration directory, written
readable only by you on POSIX systems, and never in the repository or the study mirror.
`doctor` prints keys redacted. The model API key is yours (bring your own key); the
workspace service never sees it, and the model provider never sees the service key.

**Out of scope**, then: the model running a command you would not have chosen, the
model reading a file it found on the volume, and text in a file steering what the
model does next. Those are the free-will contract working as written. In scope is
anything where the *harness* does more than carry out the model's decisions and report
facts: a path that escapes `./studies/<id>/` on your machine, a key printed in full,
an upload that happens with capture off, a trace sent with sharing off or with no
service key, a known credential left in a trace, a command that runs somewhere other
than the instance.
