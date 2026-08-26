# openreynolds (npm launcher)

Agentic OpenFOAM CFD: describe a flow problem, get an evidence-backed answer.

This npm package is a **launcher**, not the program. The real thing is the
[`openreynolds` Python package](https://pypi.org/project/openreynolds/); everything
about what it does, what it needs (an Anthropic key, a compute backend) and how to use
it is in the
[project README](https://github.com/InviscidAI/OpenReynolds#readme).
`npm install -g openreynolds` exists so people who live in Node get the same
one-command install as everyone else.

## Install

```sh
npm install -g openreynolds
openreynolds "flow over a cylinder at Re=100"

# or without installing
npx openreynolds --help
```

Prefer the Python-native route if you already have `uv` or `pipx`:

```sh
uvx openreynolds            # run without installing
pipx install openreynolds
```

## What the launcher does

On every run, in order, saying what it is doing before it does it:

1. **Finds `uv`** (`OPENREYNOLDS_UV`, then `PATH`, then the usual install directories).
   If `uv` is missing it offers the official installer
   (`curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux,
   `irm https://astral.sh/uv/install.ps1 | iex` on Windows). It asks first on a
   terminal; pass `--yes` or set `OPENREYNOLDS_YES=1` to skip the question. In a
   non-interactive shell without either it prints the instructions and exits with
   code 2. Nothing is ever installed silently.
2. **Installs the Python package once**, with `uv tool install "openreynolds>=0.1,<0.2"`
   (the range is pinned in the launcher; `uv` fetches a Python 3.10+ if you have
   none). Skipped when `uv tool list` already shows it.
3. **Execs the installed `openreynolds`** with all of your arguments, inheriting
   stdin/stdout/stderr and the environment. Exit codes are passed through; Ctrl-C
   goes to the Python process, which does its own cleanup.

Node is only involved for those few milliseconds of bookkeeping.

## Launcher-only commands and flags

Everything not listed here is forwarded to the Python CLI untouched. Launcher flags
must come before any forwarded arguments.

| Command / flag | Meaning |
| --- | --- |
| `openreynolds upgrade` | Upgrade the Python package within the launcher's pinned range (`uv tool install --upgrade`). Update the launcher itself with `npm update -g openreynolds`. |
| `openreynolds --launcher-version` | Print the npm launcher version and the Python spec it pins. |
| `--yes` | Run the uv installer without asking. |
| `--from-path <dir>` | Developers: install a local checkout in editable mode (`uv tool install --editable <dir>`) instead of the PyPI release. |

## Environment variables

| Variable | Effect |
| --- | --- |
| `OPENREYNOLDS_UV` | Absolute path to the `uv` executable to use. |
| `OPENREYNOLDS_YES=1` | Same as `--yes`. |
| `OPENREYNOLDS_PYTHON_SPEC` | Override the pinned requirement, e.g. `openreynolds==0.1.3` or `openreynolds>=0.2.0a1`. |
| `OPENREYNOLDS_ASCII=1` | Plain ASCII output (no box-drawing glyphs). |
| `NO_COLOR` / `FORCE_COLOR` | Disable / force ANSI color in the launcher's own output. |

Anything else (`ANTHROPIC_API_KEY`, backend configuration, ...) belongs to the Python
program and is passed through unchanged.

## Uninstall

```sh
npm uninstall -g openreynolds
uv tool uninstall openreynolds
```

## Note on the `reynolds` npm package

`reynolds` on npm (v0.1.x) is the earlier, different product: a bundled terminal app
with its own hosted-compute setup. It is not this launcher and does not install
`openreynolds`.

## License

MIT. Source: https://github.com/InviscidAI/OpenReynolds/tree/main/launcher
