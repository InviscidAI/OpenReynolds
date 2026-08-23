"""Terminal entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic
import click
from rich.console import Console

from . import __version__
from .backend import hosted
from .backend.base import Backend, BackendError, WORKSPACE_ROOT
from .capture import Capture
from . import images
from .config import Config, config_path
from .loop import Loop
from .store import Store, list_studies, new_study_id
from .tools import ToolContext
from .watch import LineReader, NullReader, situation, watch

TOOLBOX_SOURCE = Path(__file__).parent / "toolbox"
TOOLBOX_DEST = f"{WORKSPACE_ROOT}/.toolbox"
RESULTS_PICKUP = f"{WORKSPACE_ROOT}/results.json"

def _tolerant_stdout() -> None:
    """Keep an undecodable character from killing the session.

    A CFD conversation is full of Greek letters, superscripts and arrows, and stdout
    is not always UTF-8 - a redirect to a file on Windows lands on cp1252, where a
    single mu raises mid-stream.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


_tolerant_stdout()
console = Console()


@click.group(invoke_without_command=True)
@click.version_option(__version__, "-V", "--version")
@click.option("-p", "--prompt", "one_shot", help="Run non-interactively and exit.")
@click.option("--study", "study_id", help="Resume a local study by id.")
@click.option("--instance", "instance_id", help="Use a specific workspace instance.")
@click.option("--model", help="Override the model for this session.")
@click.option("--no-capture", is_flag=True, help="Do not send anything to the platform.")
@click.pass_context
def main(
    ctx: click.Context,
    one_shot: str | None,
    study_id: str | None,
    instance_id: str | None,
    model: str | None,
    no_capture: bool,
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

    session(cfg, study_id=study_id, instance_id=instance_id, one_shot=one_shot)


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
        # Prompting needs a terminal. Say what to do instead of aborting cryptically.
        console.print(
            "[yellow]config needs a terminal to prompt in.[/] Either run it from a "
            "normal terminal window, or set the values without prompts:\n\n"
            "  openreynolds config --key-file <path to a file holding the key>\n"
            "  ANTHROPIC_API_KEY=... openreynolds config --from-env\n"
        )
        raise SystemExit(1)

    console.print(f"Writing to [bold]{config_path()}[/]\n")
    cfg.foamd_url = click.prompt("Service URL", default=cfg.foamd_url or "").strip().rstrip("/")
    cfg.foamd_api_key = click.prompt(
        "Service API key", default=cfg.foamd_api_key or "", hide_input=True
    ).strip()
    cfg.anthropic_api_key = click.prompt(
        "Anthropic API key", default=cfg.anthropic_api_key or "", hide_input=True
    ).strip()
    cfg.model = click.prompt("Model", default=cfg.model).strip()
    path = cfg.save()
    console.print(f"\n[green]Saved[/] {path}")


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
        _check_toolbox(),
        _check_terminal(),
    ]


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
        on_fetch=_fetch_hook(capture),
    )
    loop = Loop(cfg, ctx, store, console, capture=capture)

    console.print(
        f"[bold]study[/] {store.session.study_id}   "
        f"[bold]instance[/] {resolved_instance}   [bold]model[/] {cfg.model}"
    )
    console.print(f"[dim]fetched files land in {store.dir}[/]\n")

    if resuming:
        loop.brief(situation(store, backend))

    try:
        if one_shot:
            _run_one_shot(loop, backend, store, one_shot)
        else:
            _run_interactive(loop, backend, store)
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted - jobs keep running on the instance[/]")
    finally:
        _pickup_results(backend, capture)
        if capture:
            capture.close()
        backend.close()
        console.print(f"\n[dim]resume with: openreynolds --study {store.session.study_id}[/]")


EXIT_WORDS = ("/exit", "/quit")


def _run_turn(loop: Loop) -> bool:
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
    console.print("[dim]the thread is intact - say something to continue, or /exit[/]")
    return False


def _run_interactive(loop: Loop, backend: Backend, store: Store) -> None:
    reader = LineReader()
    while True:
        if store.live_jobs():
            wake = watch(backend, store, console, reader)
            if wake.kind == "eof":
                return
            if wake.kind == "job":
                loop.inform(wake.text)
            elif wake.kind == "user":
                # Honoured here too: a long solve is exactly when someone wants out,
                # and jobs keep running on the instance either way.
                if wake.text.strip() in EXIT_WORDS:
                    return
                loop.say(wake.text)
            else:
                continue
        else:
            console.print("\n[bold green]>[/] ", end="")
            line = reader.get()
            if line is None:
                return
            text = line.strip()
            if not text:
                continue
            if text in EXIT_WORDS:
                return
            loop.say(text)

        if _run_turn(loop) and loop.needs_refresh:
            loop.refresh(situation(store, backend))


def _run_one_shot(loop: Loop, backend: Backend, store: Store, prompt: str) -> None:
    """Run until the model is done and no jobs remain."""
    reader = NullReader()
    loop.say(prompt)
    if not _run_turn(loop):
        return
    while store.live_jobs():
        wake = watch(backend, store, console, reader)
        if wake.kind != "job":
            break
        loop.inform(wake.text)
        if not _run_turn(loop):
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


def _pickup_results(backend: Backend, capture: Capture | None) -> None:
    """If a results file happens to be there, capture it. Nothing requires one."""
    if capture is None:
        return
    try:
        raw = backend.get_file(RESULTS_PICKUP)
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
