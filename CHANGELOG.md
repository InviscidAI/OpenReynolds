# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A live mirror: the study's files are synced to `./studies/<id>/files/` in the
  background for the whole session, and a render the model just looked at arrives at
  once. The interface's files pane reads from it.
- A progress bar for real compute -- solver time against `endTime`, residuals, Courant
  number, snappyHexMesh phase -- shown only while a solve, mesh, decomposition or sync
  is running.
- A front desk: a second, cheap agent that answers the user within seconds while the
  main agent is mid-turn, and writes the plain-language "now" line. It is read-only
  and cannot steer the agent. `OPENREYNOLDS_DESK=0` turns it off.
- Render delivery: every image the instance writes lands in a flat
  `./studies/<id>/renders/` folder, frame directories are assembled into gifs locally,
  and the interface has a renders tab. `openreynolds renders` lists them.
- `openreynolds video` encodes a frame set on the user's machine (ffmpeg, or imageio as
  a fallback); the instance never carries an encoder.
- `openreynolds push` carries a local file or directory up to the study's workspace.
- `job_check` takes `wait_s`, holding the answer until the job ends or the user types.
- `python -m openreynolds` as an entry point beside the console script.
- `cli.session(..., interface=)`: a seam through which another interface (a web page,
  a test harness) can drive a session using the same `View` the terminal uses.
- `-p` runs exit `1` when the model API would not complete a turn and `2` when
  `--max-wait` ran out with a job still running, so a script can tell the two apart
  from a run that finished.
- `OPENREYNOLDS_CAPTURE=0` switches transcript upload off from the environment; the
  README says up front that transcripts go to the workspace service by default.
- Packaging metadata: project URLs, authorship, keywords, classifiers, a PEP 639
  licence expression, the version read from the package, `video` and `toolbox`
  extras, and an sdist that leaves study mirrors and engineering scaffolding behind.
  CI runs on Windows and Python 3.11 as well.
- `SECURITY.md`, `CONTRIBUTING.md` and this changelog.

### Changed

- Turn-end syncs no longer block the session thread; the mirror is poked instead, so a
  message typed during a long solve is read at once.
- After two model-API failures in a row the session says plainly what is happening and
  how to resume, instead of repeating that the thread is intact.
- The thread sheds the pixels of images the model has already looked at, keeping the
  path and description, so long render-and-look sessions stay a size the API accepts.
- Field notes: steady-versus-transient, the 2D recipe, `job_check wait_s` in place of
  `sleep`, rendering next to the data, and what a steady solve that will not converge
  usually means.
- The design document moved to `docs/design.md`.

### Fixed

- `openreynolds doctor` opened a real study on the platform every run to check that
  capture worked. It now makes a read-only call and changes nothing.
- A duplicate copy of the architecture notes at the repository root is gone; the one
  under the toolbox notes is the one that ships.

## [0.1.0] - 2026-08-24

The first cut: the tool-use loop over a hosted OpenFOAM workspace, seven tools, the
terminal interface, watch mode with factual wakes, the local study mirror, capture,
stopping that verifies, the toolbox and the field notes.

[Unreleased]: https://github.com/InviscidAI/OpenReynolds/compare/0c50b8a...HEAD
[0.1.0]: https://github.com/InviscidAI/OpenReynolds/commits/0c50b8a
