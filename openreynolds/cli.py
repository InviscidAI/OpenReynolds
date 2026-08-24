"""Terminal entry point."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
import click
from rich.console import Console

from . import __version__
from .backend import hosted
from .backend.base import Backend, BackendError, WORKSPACE_ROOT
from .browse import Browser
from .capture import Capture
from . import commands, images
from .config import Config, config_path
from .loop import Loop
from .stopping import running_solvers, stop_everything
from .store import Store, list_studies, new_study_id
from .terminal import tolerant_stdout
from .tools import ToolContext
from .view import ConsoleView, View
from .watch import NOTHING, LineReader, NullReader, situation, watch

TOOLBOX_SOURCE = Path(__file__).parent / "toolbox"
TOOLBOX_DEST = f"{WORKSPACE_ROOT}/.toolbox"
RESULTS_FILE = "results.json"
"""Picked up from the study's own directory if it happens to be there."""

tolerant_stdout()
console = Console()


@click.group(invoke_without_command=True)
@click.version_option(__version__, "-V", "--version")
@click.option("-p", "--prompt", "one_shot", help="Run non-interactively and exit.")
@click.option("--study", "study_id", help="Resume a local study by id.")
@click.option("--instance", "instance_id", help="Use a specific workspace instance.")
@click.option("--model", help="Override the model for this session.")
@click.option("--no-capture", is_flag=True, help="Do not send anything to the platform.")
@click.option("--plain", is_flag=True, help="Plain streaming terminal instead of the interface.")
@click.option(
    "--max-wait",
    type=float,
    default=0.0,
    help="With -p, stop waiting on jobs after this many minutes (0 = no limit).",
)
@click.pass_context
def main(
    ctx: click.Context,
    one_shot: str | None,
    study_id: str | None,
    instance_id: str | None,
    model: str | None,
    no_capture: bool,
    plain: bool,
    max_wait: float,
) -> None:
    """A CFD agent with a real OpenFOAM workspace."""
    if ctx.invoked_subcommand is not None:
        return

    cfg = Config.load()
    if model:
        cfg.model = model
    if no_capture:
        cfg.capture = False

    missing = cfg.missing()
    if missing:
        console.print(f"[red]Missing configuration:[/] {', '.join(missing)}")
        console.print(f"Set them in the environment, or run [bold]openreynolds config[/].")
        raise SystemExit(1)

    session(
        cfg,
        study_id=study_id,
        instance_id=instance_id,
        one_shot=one_shot,
        plain=plain,
        max_wait=max_wait,
    )


@main.command("studies")
def studies_cmd() -> None:
    """List local studies."""
    cfg = Config.load()
    sessions = list_studies(cfg.studies_dir)
    if not sessions:
        console.print(f"No studies under {cfg.studies_dir}")
        return
    for item in sessions:
        live = sum(1 for job in item.jobs.values() if job.status == "running")
        suffix = f" - {live} job(s) running" if live else ""
        title = item.title or "(untitled)"
        console.print(f"[bold]{item.study_id}[/]  {title}  instance={item.instance_id}{suffix}")


@main.command("config")
@click.option(
    "--from-env",
    is_flag=True,
    help="Take values from the environment instead of prompting.",
)
@click.option(
    "--key-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Read the Anthropic key from this file (it is not echoed).",
)
def config_cmd(from_env: bool, key_file: Path | None) -> None:
    """Set credentials and defaults."""
    cfg = Config.load()

    if from_env or key_file:
        if key_file:
            cfg.anthropic_api_key = key_file.read_text(encoding="utf-8").strip()
        path = cfg.save()
        console.print(f"[green]Saved[/] {path}")
        for name, value in (
            ("service url", cfg.foamd_url),
            ("service key", _redact(cfg.foamd_api_key)),
            ("anthropic key", _redact(cfg.anthropic_api_key)),
            ("model", cfg.model),
        ):
            console.print(f"  {name:14} {value or '[red]not set[/]'}")
        return

    if not _can_prompt():
        console.print(NO_TERMINAL_HELP)
        raise SystemExit(1)

    console.print(f"Writing to [bold]{config_path()}[/]\n")
    try:
        cfg.foamd_url = (
            click.prompt("Service URL", default=cfg.foamd_url or "").strip().rstrip("/")
        )
        cfg.foamd_api_key = click.prompt(
            "Service API key", default=cfg.foamd_api_key or "", hide_input=True
        ).strip()
        cfg.anthropic_api_key = click.prompt(
            "Anthropic API key", default=cfg.anthropic_api_key or "", hide_input=True
        ).strip()
        cfg.model = click.prompt("Model", default=cfg.model).strip()
    except (click.Abort, EOFError):
        # Some shells report a terminal and then deliver EOF, so isatty() alone cannot
        # tell whether prompting will work. Explain rather than dying on "Aborted!".
        console.print(f"\n{NO_TERMINAL_HELP}")
        raise SystemExit(1) from None

    path = cfg.save()
    console.print(f"\n[green]Saved[/] {path}")


@main.command("stop")
@click.option("--study", "study_id", default=None, help="Which study. Defaults to the newest.")
@click.option("--force", is_flag=True, help="Also kill solver processes that outlived their job.")
def stop_cmd(study_id: str | None, force: bool) -> None:
    """Stop this study's jobs, and confirm the work actually stopped."""
    cfg = Config.load()
    studies = list_studies(cfg.studies_dir)
    if not studies:
        console.print(f"No studies under {cfg.studies_dir}")
        raise SystemExit(1)
    chosen = study_id or studies[0].study_id
    store = Store(cfg.studies_dir, chosen)
    if not store.session.instance_id:
        console.print(f"[red]{chosen} has no instance recorded.[/]")
        raise SystemExit(1)

    console.print(f"stopping [bold]{chosen}[/] on {store.session.instance_id[:8]}\n")
    try:
        backend, _client, _iid = hosted.acquire(
            cfg.foamd_url, cfg.foamd_api_key, store.session.instance_id
        )
    except BackendError as exc:
        console.print(f"[red]Could not reach the workspace service:[/] {exc}")
        raise SystemExit(1) from exc

    try:
        report = stop_everything(backend, store, force=force)
    finally:
        backend.close()

    for line in report.lines():
        style = "green" if report.clean else "yellow"
        console.print(f"  [{style}]{line}[/]")
    if not report.clean and not force:
        console.print("\n[dim]openreynolds stop --force also kills the leftover processes[/]")
    raise SystemExit(0 if report.clean else 1)


@main.command("files")
@click.argument("path", default="")
@click.option("--study", "study_id", default=None, help="Which study. Defaults to the newest.")
@click.option("--depth", default=4, show_default=True, help="How far down to list.")
@click.option("--cat", "cat_path", default=None, help="Print one file instead of listing.")
@click.option("--pull", "pull_path", default=None, help="Copy something out to this machine.")
@click.option("--open", "open_dir", is_flag=True, help="Open the study folder here.")
def files_cmd(
    path: str,
    study_id: str | None,
    depth: int,
    cat_path: str | None,
    pull_path: str | None,
    open_dir: bool,
) -> None:
    """Look at the workspace: what is in it, and what has been copied out.

    Read-only, and nothing here starts a session or costs a token. The model is not
    involved and is not told.
    """
    cfg = Config.load()
    studies = list_studies(cfg.studies_dir)
    if not studies:
        console.print(f"No studies under {cfg.studies_dir}")
        raise SystemExit(1)
    chosen = study_id or studies[0].study_id
    store = Store(cfg.studies_dir, chosen)

    if open_dir:
        _open_folder(store.dir, ConsoleView(console))
        return

    path = workspace_path(path)
    cat_path = workspace_path(cat_path) if cat_path else None
    pull_path = workspace_path(pull_path) if pull_path else None

    # No instance recorded is not a reason to refuse to show anything: the workspace
    # is a volume that outlives instances, and someone asking to see their files does
    # not want a lecture about bookkeeping. Reuse whatever is already up.
    try:
        backend, _client, instance = hosted.acquire(
            cfg.foamd_url, cfg.foamd_api_key, store.session.instance_id or None
        )
    except BackendError as exc:
        console.print(f"[red]Could not reach the workspace service:[/] {exc}")
        raise SystemExit(1) from exc

    browser = Browser(backend, store, home=store.session.home)
    try:
        if cat_path:
            text, _is_text = browser.read(cat_path)
            console.print(text, highlight=False, markup=False)
            return
        if pull_path:
            written = browser.pull(pull_path)
            for local in written:
                console.print(f"[green]{local}[/]")
            if not written:
                console.print("[yellow]nothing was copied[/]")
            return

        console.print(f"[bold]{chosen}[/] on {instance[:8]}\n")
        view = ConsoleView(console)
        view.workspace(browser)
        view.show_files(path, depth)

        local = browser.local()
        console.print(f"\n[bold]already on this machine[/] ({store.dir})")
        for local_path in local[-20:] if local else []:
            console.print(f"  {local_path}")
        if not local:
            console.print("  [dim]nothing pulled out yet - openreynolds files --pull <path>[/]")
    except BackendError as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(1) from exc
    finally:
        backend.close()


@main.command("doctor")
def doctor_cmd() -> None:
    """Check configuration, connectivity and credentials."""
    cfg = Config.load()
    console.print(f"config file: [bold]{config_path()}[/]"
                  f"{'' if config_path().exists() else '  (absent; using the environment)'}\n")

    failures = 0
    for label, ok, detail in run_checks(cfg):
        mark = "[green]ok  [/]" if ok else "[red]FAIL[/]"
        console.print(f"  {mark}  {label}")
        if detail:
            console.print(f"        [dim]{detail}[/]")
        failures += 0 if ok else 1

    if failures:
        console.print(f"\n[red]{failures} check(s) failed.[/] Run [bold]openreynolds config[/].")
        raise SystemExit(1)
    console.print("\n[green]Ready.[/]")


def run_checks(cfg: Config) -> list[tuple[str, bool, str]]:
    """Each check reports a fact. Nothing here changes anything."""
    return [
        _check_settings(cfg),
        _check_service(cfg),
        _check_model(cfg),
        _check_capture(cfg),
        _check_toolbox(),
        _check_terminal(),
    ]


NO_TERMINAL_HELP = (
    "[yellow]config could not prompt here[/] - this channel has no usable terminal.\n"
    "Set the values without prompts instead:\n\n"
    "  openreynolds config --key-file <path to a file holding the key>\n"
    "  ANTHROPIC_API_KEY=... openreynolds config --from-env\n\n"
    "or run [bold]openreynolds config[/] from an ordinary terminal window."
)


def _can_prompt() -> bool:
    """Whether there is a terminal to prompt in.

    `sys.stdin` is None when stdin is closed, which is not rare in an automation
    channel, so this cannot simply call isatty().
    """
    stream = getattr(sys, "stdin", None)
    try:
        return bool(stream is not None and stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _redact(secret: str) -> str:
    return f"{secret[:12]}..." if len(secret) > 12 else "set"


def _check_settings(cfg: Config) -> tuple[str, bool, str]:
    missing = cfg.missing()
    if missing:
        return f"settings: missing {', '.join(missing)}", False, ""
    return (
        "settings",
        True,
        f"service {cfg.foamd_url}, key {_redact(cfg.foamd_api_key)}, model {cfg.model}",
    )


def _check_service(cfg: Config) -> tuple[str, bool, str]:
    if not (cfg.foamd_url and cfg.foamd_api_key):
        return "workspace service: not configured", False, ""
    client = hosted.FoamdClient(cfg.foamd_url, cfg.foamd_api_key)
    try:
        instances = [i for i in client.list_instances() if i.get("status") != "deleted"]
    except BackendError as exc:
        return "workspace service", False, str(exc)
    finally:
        client.close()

    if not instances:
        return "workspace service", True, "reachable; no instance yet (one is made on demand)"
    described = ", ".join(f"{i.get('id', '?')[:8]} {i.get('status', '?')}" for i in instances[:3])
    return "workspace service", True, f"reachable; {len(instances)} instance(s): {described}"


def _check_model(cfg: Config) -> tuple[str, bool, str]:
    """Validates the key, the base URL and the model id in one free call."""
    if not (cfg.anthropic_api_key or cfg.llm_base_url):
        return "model API: no key", False, ""
    try:
        client = anthropic.Anthropic(
            api_key=cfg.anthropic_api_key or None, base_url=cfg.llm_base_url or None
        )
        counted = client.messages.count_tokens(
            model=cfg.model, messages=[{"role": "user", "content": "ping"}]
        )
    except anthropic.APIStatusError as exc:
        return "model API", False, f"{exc.status_code}: {getattr(exc, 'message', exc)}"
    except anthropic.APIError as exc:
        return "model API", False, str(exc)
    return "model API", True, f"{cfg.model} reachable ({counted.input_tokens} tokens for a ping)"


def _check_capture(cfg: Config) -> tuple[str, bool, str]:
    """Whether transcripts are reaching the platform.

    Capture is on by default and fails quietly on purpose -- it must never delay or
    break a study. The cost of that is there being no moment at which anyone finds
    out it stopped working, so this is the moment.
    """
    if not cfg.capture:
        return "capture", True, "off for this configuration; nothing is uploaded"
    if not (cfg.foamd_url and cfg.foamd_api_key):
        return "capture: not configured", False, ""

    client = hosted.FoamdClient(cfg.foamd_url, cfg.foamd_api_key)
    try:
        study_id = client.create_study("openreynolds doctor", None)
    except BackendError as exc:
        return "capture", False, f"a study could not be opened: {exc}"
    except Exception as exc:  # the platform is optional; naming the failure is not
        return "capture", False, f"{type(exc).__name__}: {exc}"
    finally:
        client.close()
    return "capture", True, f"transcripts are being kept (opened study {study_id[:8]})"


def _check_toolbox() -> tuple[str, bool, str]:
    if not TOOLBOX_SOURCE.is_dir():
        return "toolbox: not found in the installed package", False, ""
    scripts = sorted(p.name for p in TOOLBOX_SOURCE.glob("*.py"))
    notes = sorted(p.name for p in (TOOLBOX_SOURCE / "notes").glob("*.md"))
    return "toolbox", True, f"{len(scripts)} scripts, {len(notes)} notes -> {TOOLBOX_DEST}"


def _check_terminal() -> tuple[str, bool, str]:
    kind = images.protocol()
    detail = f"{kind} graphics: renders show inline" if kind else "renders print their path"
    return "terminal", True, detail


# -- the session ---------------------------------------------------------------


def session(
    cfg: Config,
    *,
    study_id: str | None,
    instance_id: str | None,
    one_shot: str | None,
    plain: bool = False,
    max_wait: float = 0.0,
) -> None:
    resuming = study_id is not None
    store = Store(cfg.studies_dir, study_id or new_study_id())

    try:
        backend, client, resolved_instance = hosted.acquire(
            cfg.foamd_url,
            cfg.foamd_api_key,
            instance_id or store.session.instance_id or None,
        )
    except BackendError as exc:
        console.print(f"[red]Could not reach the workspace service:[/] {exc}")
        raise SystemExit(1) from exc

    store.session.instance_id = resolved_instance
    store.session.model = cfg.model
    store.session.home = _home_for(store, backend, resuming)
    if not store.session.title and one_shot:
        store.session.title = one_shot[:80]
    store.save()

    capture = None
    if cfg.capture:
        if resuming and store.session.remote_study_id:
            capture = Capture(client, store.session.remote_study_id, warn=_warn)
        else:
            capture = Capture.start(
                client, store.session.title or store.session.study_id, resolved_instance, warn=_warn
            )
            if capture:
                store.session.remote_study_id = capture.study_id
                store.save()

    _sync_toolbox(backend)

    ctx = ToolContext(
        backend=backend,
        store=store,
        max_output=cfg.max_tool_output,
        home=store.session.home,
        on_fetch=_fetch_hook(capture),
    )
    browser = Browser(backend, store, home=store.session.home)

    def drive(view: View, reader: Any) -> None:
        """One session, against whichever interface is running it."""
        # The tools report job state through the view, so a panel showing what is
        # running is current the moment it changes rather than only while polling.
        ctx.view = view
        view.workspace(browser)
        loop = Loop(cfg, ctx, store, view, capture=capture)
        loop.interject = lambda: _typed_while_working(loop, view, browser, store, reader)
        view.header(store.session.study_id, resolved_instance, cfg.model, store.dir)
        loop.brief(
            _situation_brief(
                store, backend, resuming, interactive=not one_shot, browser=browser
            )
        )
        try:
            if one_shot:
                _run_one_shot(loop, backend, store, one_shot, view, reader, max_wait)
            else:
                _run_interactive(loop, backend, store, view, browser, reader)
        except KeyboardInterrupt:
            view.info("interrupted - jobs keep running on the instance")

    force_exit = False
    try:
        if one_shot or plain or not _tui_available():
            drive(ConsoleView(console), LineReader() if not one_shot else NullReader())
        else:
            force_exit = bool(_run_tui(drive))
    finally:
        _pickup_results(backend, capture, store.session.home or WORKSPACE_ROOT)
        if capture:
            capture.close()
        _report_on_exit(backend, store)
        backend.close()
        if force_exit:
            # The session thread is still inside a network call it cannot be pulled out
            # of. Everything worth keeping is written; waiting for it would leave the
            # user unable to close a program they asked to close.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


WORKSPACE_LISTED = 40


def _home_for(store: Store, backend: Backend, resuming: bool) -> str:
    """This study's own directory, made if it is not there.

    A new study used to open straight into the shared volume, among every other
    study's cases. That is not a clean slate by any reading of the words, and it cost
    real work: one run inherited a velocity from a case an earlier session had
    abandoned, another opened by having to verify somebody else's results before it
    could start on its own.

    So a study gets a directory named after it. The volume still persists -- that is
    the whole point of it -- but starting a new study now starts somewhere empty.
    Studies made before this have no home recorded and keep the whole workspace,
    because moving their files out from under them would be worse.
    """
    if store.session.home:
        home = store.session.home
    elif resuming:
        home = WORKSPACE_ROOT
    else:
        home = f"{WORKSPACE_ROOT}/{store.session.study_id}"

    if home != WORKSPACE_ROOT:
        try:
            backend.exec(f"mkdir -p {home}", timeout_s=60)
        except BackendError as exc:
            console.print(f"[yellow]could not make {home} ({exc}); using {WORKSPACE_ROOT}[/]")
            return WORKSPACE_ROOT
    return home


def _workspace_note(browser: Browser, home: str, resuming: bool) -> str:
    """What is in this study's directory, and what else is on the volume.

    The listing is of the study's own directory, because that is the one whose
    contents are its business. The rest of the volume gets a single line: it exists,
    it belongs to other studies, and nothing in it was written for this request.
    """
    try:
        entries = [
            entry for entry in browser.tree(home, depth=1) if not entry.name.startswith(".")
        ]
    except BackendError:
        return ""

    if home == WORKSPACE_ROOT:
        neighbours = ""
    else:
        # Saying whose it is without saying what they are leaves a real question open,
        # and a live run spent turns on it: it found several near-identical studies
        # made minutes apart by a user who did not remember commissioning them, and
        # worked through whether that meant an intruder. The answer is dull and the
        # harness has always known it -- they are this same tool's other sessions.
        others = _other_studies(browser, home)
        neighbours = (
            f"\n{WORKSPACE_ROOT} also holds {others} directory of this tool's own "
            "earlier sessions on this instance, each named after its study id. They "
            "are readable, and none of them was written for this request."
            if others == "one other"
            else f"\n{WORKSPACE_ROOT} also holds {others} directories of this tool's "
            "own earlier sessions on this instance, each named after its study id. "
            "They are readable, and none of them was written for this request."
        )

    if not entries:
        return f"Your directory is {home}. It is empty.{neighbours}"

    listed = "\n".join(
        f"  {entry.name}{'/' if entry.is_dir else ''}"
        f"   last written {_when(entry.mtime)}"
        for entry in entries[:WORKSPACE_LISTED]
    )
    if len(entries) > WORKSPACE_LISTED:
        listed += f"\n  ... and {len(entries) - WORKSPACE_LISTED} more"

    if resuming:
        head = f"Your directory is {home}, as this study left it:"
    else:
        head = (
            f"Your directory is {home}. It already contains the following, which was "
            "left by earlier sessions and not written for this request:"
        )
    return f"{head}\n{listed}{neighbours}"


def _other_studies(browser: Browser, home: str) -> str:
    """How many other session directories share the volume."""
    try:
        siblings = [
            entry
            for entry in browser.tree(WORKSPACE_ROOT, depth=1)
            if entry.is_dir and not entry.name.startswith(".") and entry.path != home
        ]
    except BackendError:
        return "some"
    if not siblings:
        return "no"
    return "one other" if len(siblings) == 1 else f"{len(siblings)} other"


def _when(mtime: float) -> str:
    if not mtime:
        return "unknown"
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(mtime)) + "Z"


def _situation_brief(
    store: Store,
    backend: Backend,
    resuming: bool,
    interactive: bool,
    browser: Browser | None = None,
) -> str:
    """Facts about this session, assembled by the harness.

    Whether anyone is at the terminal is one of them. It is the difference between a
    question that gets answered and a turn that ends on a question nobody will ever
    see, and the model has no other way to know which kind of session this is.
    """
    lines = []
    if resuming:
        lines.append(situation(store, backend))
    else:
        lines.append(f"study {store.session.study_id} on instance {store.session.instance_id}.")
    if browser is not None:
        note = _workspace_note(browser, store.session.home or WORKSPACE_ROOT, resuming)
        if note:
            lines.append(note)
    if interactive:
        lines.append(
            "A person is at the terminal for this session and can answer you. Anything "
            "they type while you are working reaches you at your next turn."
        )
    else:
        lines.append(
            "This is a non-interactive run. Nobody is at the terminal and no answer "
            "can arrive, so a question asked here will not be seen."
        )
    return "\n".join(lines)


def _report_on_exit(backend: Backend, store: Store) -> None:
    """Say what is still running, because it is still being paid for.

    Jobs outliving the session is the design, not an accident -- but leaving without
    being told is how an idle laptop keeps eight cores busy overnight.
    """
    study = store.session.study_id
    home = store.session.home or WORKSPACE_ROOT
    live = store.live_jobs()
    if not live:
        # Knowing a study has a directory of its own is no use without being told
        # which one it is.
        console.print(f"\n[dim]this study's files are in {home} on the instance[/]")
        console.print(f"[dim]resume with:  openreynolds --study {study}[/]")
        console.print(f"[dim]look at them: openreynolds files --study {study}[/]")
        return
    names = ", ".join(job.name or job.job_id[:8] for job in live)
    console.print(f"\n[yellow]{len(live)} job(s) still running on the instance:[/] {names}")
    console.print("[dim]they keep going, and keep costing, until they finish[/]")
    console.print(f"[dim]  resume: openreynolds --study {study}[/]")
    console.print(f"[dim]  stop:   openreynolds stop --study {study}[/]")


def _tui_available() -> bool:
    """The interface needs a real terminal and the library that draws it."""
    if not _can_prompt():
        return False
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


def _run_tui(drive: Any) -> None:
    """Hand the session to the interface, falling back if it cannot start.

    An interface failure should cost the look of the thing, not the session.
    """
    try:
        from .tui import OpenReynoldsApp, TuiReader, TuiView
    except Exception as exc:  # pragma: no cover - import-time only
        console.print(f"[yellow]interface unavailable ({exc}); using the plain terminal[/]")
        drive(ConsoleView(console), LineReader())
        return

    app = OpenReynoldsApp(lambda running: drive(TuiView(running), TuiReader(running)))
    app.run()
    return app.quitting


QUIT = object()
"""Returned by the command handler when the user asked to leave."""


def _apply(
    command: commands.Command, loop: Loop, view: View, browser: Browser, store: Store
) -> Any:
    """Act on one typed line. Returns what goes to the model, or None, or QUIT."""
    if command.kind == commands.EXIT:
        return QUIT
    if command.kind in (commands.SAY, commands.ASIDE):
        if not command.text:
            return None
        loop.say(command.text)
        return command.text
    _local(command, view, browser, store, loop)
    return None


def _local(
    command: commands.Command,
    view: View,
    browser: Browser,
    store: Store,
    loop: Loop | None = None,
) -> None:
    """Commands answered here, out of what the harness already knows.

    None of these reach the model. That is the whole point of them: a question that
    costs a turn and derails the work is a question people stop asking, and then they
    have no idea what is going on.
    """
    if command.kind == commands.STATUS:
        view.status(
            commands.status_lines(
                store,
                tokens=getattr(loop, "context_tokens", 0) or 0,
                local_files=len(browser.local()),
            )
        )
    elif command.kind == commands.FILES:
        view.show_files(command.text)
    elif command.kind == commands.OPEN:
        _open_folder(store.dir, view)
    elif command.kind == commands.HELP:
        view.status(commands.HELP_TEXT.splitlines())


def _typed_while_working(
    loop: Loop, view: View, browser: Browser, store: Store, reader: Any
) -> str | None:
    """Drain what was typed mid-turn, and hand back only what was meant for the model.

    `/status` and `/files` are answered here and now, without a turn. Everything else
    rides along with the next tool result, so it lands at the model's next turn rather
    than sitting unread until the whole turn is over.
    """
    for_model: list[str] = []
    while True:
        typed = reader.poll()
        if typed is NOTHING:
            break
        if typed is None:
            # EOF belongs to whoever waits at the prompt; swallowing it here would
            # leave the session unendable.
            reader.putback(None)
            break
        command = commands.parse(typed)
        if command.kind == commands.EXIT:
            reader.putback(typed)
            break
        if command.kind in (commands.SAY, commands.ASIDE) and command.text:
            for_model.append(command.text)
        else:
            _local(command, view, browser, store, loop)
    return "\n".join(for_model) or None


def workspace_path(value: str) -> str:
    """Undo the translation a POSIX-emulating shell applies to a workspace path.

    Git Bash on Windows rewrites a leading `/work` into `C:/Program Files/Git/work`
    before this process ever sees the argument, so a perfectly correct command comes
    back as a 404 naming a path the user never typed. Every path taken here is a
    workspace path, which makes recovering it unambiguous.
    """
    if not value or value.startswith(WORKSPACE_ROOT):
        return value
    text = value.replace("\\", "/")
    if not re.match(r"^[A-Za-z]:/", text):
        return value
    index = text.find(WORKSPACE_ROOT + "/")
    if index > 0:
        return text[index:]
    return WORKSPACE_ROOT if text.endswith(WORKSPACE_ROOT) else value


def _open_folder(path: Path, view: View) -> None:
    """Show a directory in whatever the platform calls a file browser."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606 - the platform's own opener
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(path)])
    except (OSError, AttributeError) as exc:
        view.warn(f"could not open {path}: {exc}")
        return
    view.info(f"opened {path}")


def _run_turn(loop: Loop, view: View) -> bool:
    """Run a turn, surviving a model-API failure. Returns whether it completed.

    A long study will meet a rate limit or a dropped connection eventually, and
    losing the whole session to one is a poor trade when the thread is still intact
    and the jobs are still running on the instance.
    """
    try:
        loop.run()
        return True
    except anthropic.APIStatusError as exc:
        detail = getattr(exc, "message", None) or str(exc)
        console.print(f"\n[red]The model API returned {exc.status_code}:[/] {detail}")
    except anthropic.APIError as exc:
        console.print(f"\n[red]Could not reach the model API:[/] {exc}")

    loop.settle()
    view.info("the thread is intact - say something to continue, or /exit")
    return False


def _run_interactive(
    loop: Loop, backend: Backend, store: Store, view: View, browser: Browser, reader: Any
) -> None:
    while True:
        if store.live_jobs():
            wake = watch(backend, store, view, reader)
            if wake.kind == "eof":
                return
            if wake.kind == "job":
                loop.inform(wake.text)
            elif wake.kind == "user":
                # A long solve is exactly when someone wants to leave, or to ask what
                # is happening without setting the whole thing off again.
                spoken = _apply(commands.parse(wake.text), loop, view, browser, store)
                if spoken is QUIT:
                    return
                if spoken is None:
                    continue
            else:
                continue
        else:
            view.prompt()
            line = reader.get()
            if line is None:
                return
            spoken = _apply(commands.parse(line), loop, view, browser, store)
            if spoken is QUIT:
                return
            if spoken is None:
                continue

        if _run_turn(loop, view) and loop.needs_refresh:
            loop.refresh(situation(store, backend))


def _run_one_shot(
    loop: Loop,
    backend: Backend,
    store: Store,
    prompt: str,
    view: View,
    reader: Any,
    max_wait_minutes: float = 0.0,
) -> None:
    """Run until the model is done and no jobs remain.

    There is nobody here to answer a question, so if the model ends its turn wanting
    one, this waits on the job instead -- possibly for hours. `--max-wait` bounds that.
    Stopping only ends the waiting: the job carries on out on the instance and the
    study resumes.
    """
    loop.say(prompt)
    if not _run_turn(loop, view):
        return

    deadline = time.monotonic() + max_wait_minutes * 60 if max_wait_minutes else None
    while store.live_jobs():
        wake = watch(backend, store, view, reader, deadline=deadline)
        if wake.kind == "timeout":
            view.info(
                f"stopped waiting after {max_wait_minutes:g} min; the job is still "
                f"running - resume with --study {store.session.study_id}"
            )
            return
        if wake.kind != "job":
            break
        loop.inform(wake.text)
        if not _run_turn(loop, view):
            return
        if loop.needs_refresh:
            loop.refresh(situation(store, backend))


def _sync_toolbox(backend: Backend) -> None:
    """Push the toolbox into the workspace. It is offered, never imposed."""
    if not TOOLBOX_SOURCE.is_dir():
        return
    try:
        backend.put_tree(TOOLBOX_SOURCE, TOOLBOX_DEST)
    except BackendError as exc:
        console.print(f"[yellow]toolbox sync skipped:[/] {exc}")


def _fetch_hook(capture: Capture | None):
    """Register fetched files as artifacts, and draw them if the terminal can."""

    def handle(paths: list[Path]) -> None:
        for path in paths:
            if capture is not None:
                capture.artifact(path)
            images.show(path)

    return handle


def _pickup_results(backend: Backend, capture: Capture | None, home: str) -> None:
    """If a results file happens to be there, capture it. Nothing requires one."""
    if capture is None:
        return
    try:
        raw = backend.get_file(f"{home.rstrip('/')}/{RESULTS_FILE}")
    except BackendError:
        return
    try:
        capture.result(json.loads(raw.decode("utf-8")))
    except (ValueError, UnicodeDecodeError):
        return


def _warn(message: str) -> None:
    console.print(f"[yellow]{message}[/]")


if __name__ == "__main__":
    main()
