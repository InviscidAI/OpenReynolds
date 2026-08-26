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

**The agent runs an unsandboxed shell on your hosted instance, with no approval gate.**
The `bash` tool executes whatever the model decides to run, as root, inside the
instance's own container. There is no allow-list, no confirmation step, and no
rewriting of commands -- the harness may not block or gate a tool call on policy
grounds, and that is deliberate. The instance is the sandbox: a per-user container
with a persistent volume, and nothing else. The agent's tools reach the instance only;
they never execute anything on your machine. What comes back to your machine is
written under `./studies/<id>/` and nowhere else.

**`/work` on the instance persists across studies and is readable by the model.** The
volume outlives every session. Whatever a previous study left there, whatever you
carried up with `openreynolds push`, and whatever another study is writing at the same
time is visible to the model in every later session on that instance, and it may read
it. A file in the workspace can contain text that looks like instructions; the model
reads files as part of its work and nothing stands between the two. Keep secrets out
of `/work`, and do not share an instance with anyone whose files you would not show
the model.

**Transcripts are uploaded unless capture is off.** By default every message, every
tool call and its (capped) output, every fetched artifact and the end-of-session
results go to the workspace service you configured, on a background thread, so the
study is kept somewhere other than one laptop. `--no-capture` turns it off for a
session, `OPENREYNOLDS_CAPTURE=0` for an environment, and `openreynolds doctor` says
which it is. The local mirror under `./studies/` is written regardless.

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
an upload that happens with capture off, a command that runs somewhere other than the
instance.
