# Contributing

## Setting up

```bash
git clone https://github.com/InviscidAI/OpenReynolds
cd OpenReynolds
pip install -e ".[dev]"
python -m pytest tests -q
git config core.hooksPath .githooks
```

The last line installs the pre-commit hook, which runs the whole suite before a commit
is allowed. It takes about half a minute. It exists because twice in one session a
commit went out red on the strength of the tests just written rather than the whole
suite, and discipline had already failed twice.

The suite runs against fakes -- no service account, no model key, no network. The
scripts under `scripts/` drive a live service and need credentials; `openreynolds doctor`
says whether yours work.

## The tests that will bite

A good part of the suite checks the *shape* of the code rather than its behaviour.
These are the ones a reasonable change trips over, and what each one wants:

- **Everything declared is read** (`tests/test_wiring.py`). Every click option, every
  `Config` field, every `ToolContext` field, every `View` method and every `Backend`
  method has to be read somewhere in the package. Four separate times something was
  defined, documented, listed in `--help` and never wired up; the test turns that into
  a failure. Adding a field means wiring it end to end in the same change.
- **The README mentions every subcommand** (`tests/test_wiring.py`). A command nobody
  is told about is a command nobody uses. A new `@main.command("x")` needs the string
  `openreynolds x` in the README.
- **Documents carry no hand-edited test counts** (`tests/test_wiring.py`). A number
  that has to be edited by hand goes stale.
- **No imperative language in the system prompt or the briefing**
  (`tests/test_prompt.py`, `tests/test_briefing.py`). "Always", "you should", "make
  sure", "first,", "workflow" and their relatives fail the build. The prompt states
  facts; decisions are the model's. Know-how belongs in the toolbox and the field
  notes, where it is offered rather than imposed. The system prompt is also
  byte-frozen (it is the cache prefix) and capped in length, so it changes rarely and
  deliberately.
- **HTTP stays below the backend protocol** (`tests/test_negative_obligation.py`).
  `openreynolds/backend/hosted.py` is the only module that knows a service exists.
  Nothing above `backend/base.py` may import a transport or mention an endpoint.

The contract these tests enforce is in [`docs/design.md`](docs/design.md): the harness
may cap output, sync the toolbox, poll jobs and wake the model with facts, and capture
the transcript. It may not order the model's actions, gate or rewrite a tool call,
require approvals, inject checklists, or grade the output.

## Style

- Docstrings explain *why*, often with the incident that taught it.
  [`docs/found-by-using-it.md`](docs/found-by-using-it.md) is the casebook; a fix that
  came from using the thing usually earns an entry there.
- Commit messages are a sentence about what is now true, in the style of `git log`.
- Files are UTF-8. Git normalises line endings to LF (`.gitattributes`); on Windows the
  working copies are typically CRLF, and an editor that rewrites a whole file's endings
  makes a diff nobody can review -- keep each file the way you found it.
- Tests are named for the behaviour they protect (`test_a_failed_turn_does_not_...`),
  and a test that exists because something went wrong says so in its docstring.

## Sending a change

Open a pull request against `main` with the suite green. Say what is now true that was
not before, and if the change came from a session that went wrong, what happened. A
change to the prompt, the briefing or the tool schemas is a change to what the model
sees and gets a closer look than the rest.
