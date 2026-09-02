"""Terminal entry point."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from . import __version__
from .backend import hosted
from .backend.local import LocalBackend
from .backend.base import Backend, BackendError, WORKSPACE_ROOT
from .browse import Browser
from .capture import Capture
from . import commands, images
from .config import Config, config_path
from .delivery import Gallery
from .llm import PRESETS, ProviderError, make_provider, preset_for
from .loop import Loop
from . import video as video_mod
from .desk import Concierge
from .mirror import LiveMirror, local_for, sync as mirror_sync
from .progress import Tracker
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
    "--keep-alive",
    is_flag=True,
    help="Leave the instance and its jobs running after this session ends.",
)
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
    keep_alive: bool,
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
        console.print(
            "[bold]openreynolds login[/] gets a service key; [bold]openreynolds config[/] "
            "sets the model key. Or set them in the environment."
        )
        raise SystemExit(1)

    outcome = session(
        cfg,
        study_id=study_id,
        instance_id=instance_id,
        one_shot=one_shot,
        plain=plain,
        keep_alive=keep_alive,
        max_wait=max_wait,
    )
    code = ONE_SHOT_EXIT_CODES.get(outcome or "ok", 0)
    if code:
        raise SystemExit(code)


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
    help="Read the model API key from this file (it is not echoed).",
)
@click.option(
    "--provider",
    "provider",
    type=click.Choice(sorted(PRESETS), case_sensitive=False),
    help="Whose model: a preset that fills in the endpoint and a model id.",
)
def config_cmd(from_env: bool, key_file: Path | None, provider: str | None) -> None:
    """Set credentials and defaults."""
    cfg = Config.load()
    if provider:
        _apply_preset(cfg, provider)

    if from_env or key_file or (provider and not _can_prompt()):
        if key_file:
            cfg.llm_api_key = key_file.read_text(encoding="utf-8").strip()
        path = cfg.save()
        console.print(f"[green]Saved[/] {path}")
        _print_settings(cfg)
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
        if not provider:
            console.print("Model providers: " + ", ".join(sorted(PRESETS)))
            chosen = click.prompt(
                "Provider", default=cfg.provider, type=click.Choice(sorted(PRESETS), case_sensitive=False)
            )
            if chosen.lower() != cfg.provider:
                _apply_preset(cfg, chosen)
        chosen_preset = preset_for(cfg.provider)
        if key_file:
            cfg.llm_api_key = key_file.read_text(encoding="utf-8").strip()
        elif chosen_preset is not None and not chosen_preset.needs_key:
            console.print(f"  ({chosen_preset.note})")
        else:
            cfg.llm_api_key = click.prompt(
                "Model API key", default=cfg.llm_api_key or "", hide_input=True
            ).strip()
        cfg.model = click.prompt("Model", default=cfg.model).strip()
    except (click.Abort, EOFError):
        # Some shells report a terminal and then deliver EOF, so isatty() alone cannot
        # tell whether prompting will work. Explain rather than dying on "Aborted!".
        console.print(f"\n{NO_TERMINAL_HELP}")
        raise SystemExit(1) from None

    path = cfg.save()
    console.print(f"\n[green]Saved[/] {path}")
    _print_settings(cfg)


LOGIN_POLL_CAP_S = 15.0
"""The service says how often to ask; this is how often is too often to be polite."""


@main.command("login")
@click.option("--service", default=None, help="The service to sign in to (default: the configured one).")
@click.option("--name", default=None, help="What to call this machine's key (default: its hostname).")
@click.option("--email", default=None, help="Sign in as this address without being asked.")
@click.option("--password-stdin", is_flag=True, help="Read the password from standard input.")
@click.option("--browser", is_flag=True, help="Approve a short code in the browser instead of typing a password.")
@click.option("--no-browser", is_flag=True, help="With --browser: print the address instead of opening it.")
def login_cmd(
    service: str | None, name: str | None, email: str | None, password_stdin: bool,
    browser: bool, no_browser: bool,
) -> None:
    """Sign in with your email and password; this machine gets its own service key."""
    cfg = Config.load()
    url = (service or cfg.foamd_url).rstrip("/")
    label = name or socket.gethostname() or "openreynolds"
    if browser:
        _login_browser(cfg, url, label, no_browser)
        return

    if password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
        if not email:
            console.print("[red]--password-stdin needs --email.[/]")
            raise SystemExit(2)
    elif not _can_prompt():
        console.print("Nothing to type into here. Use [bold]--email[/] with [bold]--password-stdin[/], or [bold]--browser[/].")
        raise SystemExit(1)
    else:
        email = (email or click.prompt("Email")).strip()
        password = click.prompt("Password", hide_input=True)

    try:
        auth = hosted.auth_config(url)
    except BackendError as exc:
        console.print(f"[red]Could not reach {url}:[/] {exc.message}")
        raise SystemExit(1) from None

    session: dict[str, Any] | None
    try:
        session = hosted.password_session(auth["supabase_url"], auth["publishable_key"], email, password)
    except BackendError as exc:
        if "captcha" in exc.message.lower():
            # The identity provider wants proof of a person, which a terminal cannot
            # give. The browser can: hand over to the code flow without making the
            # person type anything again.
            console.print("Sign-in needs a quick check in the browser (it proves you are a person).")
            _login_browser(cfg, url, label, no_browser)
            return
        if exc.code != "invalid_credentials":
            console.print(f"[red]Sign-in failed:[/] {exc.message}")
            raise SystemExit(1) from None
        session = _offer_account(auth, url, email, password)
        if session is None:
            raise SystemExit(1)

    jwt = str(session.get("access_token") or "")
    try:
        hosted.accept_terms(url, jwt)
        token = hosted.mint_key(url, jwt, label)
    except BackendError as exc:
        console.print(f"[red]Signed in, but the service would not issue a key:[/] {exc.message}")
        raise SystemExit(1) from None

    cfg.foamd_url = url
    cfg.foamd_api_key = str(token["key"])
    path = cfg.save()
    console.print(f"\n[green]Signed in as {email}.[/] This machine's key ([bold]{token.get('name') or label}[/]) is saved to {path}")
    _say_what_is_next(cfg)


def _offer_account(auth: dict[str, Any], url: str, email: str, password: str) -> dict[str, Any] | None:
    """Wrong password, or no account yet -- the service cannot tell which, so ask."""
    console.print("Wrong password, or no account with that address yet.")
    if not _can_prompt() or not click.confirm(f"Create an account for {email} with this password?", default=False):
        return None
    console.print(f"The terms are at {url}/terms and the privacy note at {url}/privacy.")
    if not click.confirm("Accept them?", default=False):
        return None
    try:
        session = hosted.sign_up(auth["supabase_url"], auth["publishable_key"], email, password)
    except BackendError as exc:
        console.print(f"[red]Could not create the account:[/] {exc.message}")
        return None
    if session is None:
        console.print(
            f"[green]Account created.[/] Confirm the address from the email sent to {email}, "
            "then run [bold]openreynolds login[/] again."
        )
        raise SystemExit(0)
    return session


def _say_what_is_next(cfg: Config) -> None:
    still = cfg.model_key_missing()
    if still:
        console.print(
            f"The model key is still needed: set [bold]{still}[/] or run "
            "[bold]openreynolds config[/]. Then [bold]openreynolds doctor[/]."
        )
    else:
        console.print("Next: [bold]openreynolds doctor[/], then [bold]openreynolds[/].")


LOGIN_POLL_CAP_S = 15.0
"""The service says how often to ask; this is how often is too often to be polite."""


def _login_browser(cfg: Config, url: str, label: str, no_browser: bool) -> None:
    """The code flow: for a terminal with no keyboard of its own (a remote box, a CI
    runner) that still has a person with a browser somewhere."""
    try:
        offer = hosted.device_code(url, label)
    except BackendError as exc:
        console.print(f"[red]Could not start signing in at {url}:[/] {exc.message}")
        raise SystemExit(1) from None

    code = str(offer.get("user_code", ""))
    where = str(offer.get("verification_url", url))
    console.print(f"\nYour code: [bold]{code}[/]")
    console.print(f"Approve it at [bold]{where}[/]")
    opened = False
    if not no_browser:
        try:
            opened = bool(webbrowser.open(where))
        except Exception:  # noqa: BLE001 - a browser that will not open is not an error here
            opened = False
    console.print("Waiting for the approval..." if opened else "Open that address, then come back here.")

    # The service's numbers are taken literally: a zero lifetime is an expired code,
    # not a request for the default.
    interval = min(max(float(offer.get("interval", 5)), 0.5), LOGIN_POLL_CAP_S)
    deadline = time.monotonic() + float(offer.get("expires_in", 600))
    device = str(offer.get("device_code", ""))
    token: dict[str, Any] | None = None
    while token is None:
        if time.monotonic() > deadline:
            console.print("[red]The code expired before it was approved.[/] Run login again.")
            raise SystemExit(1)
        time.sleep(interval)
        try:
            token = hosted.device_token(url, device)
        except BackendError as exc:
            console.print(f"[red]Signing in failed:[/] {exc.message}")
            raise SystemExit(1) from None

    cfg.foamd_url = str(token.get("base_url") or url).rstrip("/")
    cfg.foamd_api_key = str(token["api_key"])
    path = cfg.save()
    console.print(f"\n[green]Signed in.[/] Key [bold]{token.get('name') or label}[/] saved to {path}")
    _say_what_is_next(cfg)


def _apply_preset(cfg: Config, name: str) -> None:
    """Point the settings at a vendor: its family, endpoint, model ids and window.

    The key is left alone -- it is the one thing a preset cannot know -- and so is
    anything the person already set explicitly for this same provider."""
    preset = preset_for(name)
    if preset is None:
        raise click.BadParameter(f"unknown provider {name!r}", param_hint="--provider")
    if cfg.provider != preset.name:
        cfg.llm_api_key = ""
    cfg.provider = preset.name
    cfg.llm_base_url = None
    cfg.model = preset.model
    cfg.desk_model = preset.desk_model
    cfg.context_window = preset.context_window


def _print_settings(cfg: Config) -> None:
    for name, value in (
        ("service url", cfg.foamd_url),
        ("service key", _redact(cfg.foamd_api_key)),
        ("provider", cfg.provider),
        ("model key", "via the service key" if cfg.provider == "reynolds" else _redact(cfg.llm_api_key)),
        ("model", cfg.model),
    ):
        console.print(f"  {name:14} {value or '[red]not set[/]'}")


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
        # Scoped to the study being stopped. `--force` is what widens it to the whole
        # instance, which is the only place that sweep still belongs: a person naming a
        # study and asking twice.
        report = stop_everything(backend, store, force=force, home=store.session.home)
        _release(backend)
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
        console.print(f"\n[bold]already on this machine[/] ({store.files_dir})")
        for local_path in local[-20:] if local else []:
            console.print(f"  {local_path}")
        if not local:
            console.print(
                f"  [dim]nothing here yet - openreynolds pull --study {chosen}[/]"
            )
    except BackendError as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(1) from exc
    finally:
        _release(backend)
        backend.close()


@main.command("pull")
@click.argument("path", default="")
@click.option("--study", "study_id", default=None, help="Which study. Defaults to the newest.")
@click.option(
    "--readable-only",
    "selective",
    is_flag=True,
    help="Only images, reports, logs and case dictionaries, instead of everything.",
)
def pull_cmd(path: str, study_id: str | None, selective: bool) -> None:
    """Copy this study's files down to this machine.

    A session already does this at the end of every turn, so this is for asking
    again, for asking about one directory, or for `--all` when the default judgement
    about what is worth keeping is the wrong one for this study.
    """
    cfg = Config.load()
    studies = list_studies(cfg.studies_dir)
    if not studies:
        console.print(f"No studies under {cfg.studies_dir}")
        raise SystemExit(1)
    chosen = study_id or studies[0].study_id
    store = Store(cfg.studies_dir, chosen)

    try:
        backend, _client, instance = hosted.acquire(
            cfg.foamd_url, cfg.foamd_api_key, store.session.instance_id or None
        )
    except BackendError as exc:
        console.print(f"[red]Could not reach the workspace service:[/] {exc}")
        raise SystemExit(1) from exc

    browser = Browser(backend, store, home=store.session.home)
    console.print(f"[bold]{chosen}[/] on {instance[:8]}\n")
    try:
        report = mirror_sync(
            browser, path=workspace_path(path), everything=not selective
        )
    finally:
        _release(backend)
        backend.close()

    for line in report.lines():
        console.print(line, highlight=False, markup=False)
    console.print(f"\n[dim]your copy of this study is in {store.files_dir}[/]")
    # A sync that reached nothing and had something to complain about is a failure,
    # and someone who put this in a script needs to be able to tell.
    raise SystemExit(1 if report.warnings and not report.pulled else 0)


@main.command("push")
@click.argument("local", type=click.Path(exists=True, path_type=Path))
@click.option("--to", "dest", default=None, help="Where on the instance to put it. Defaults under the study's own directory.")
@click.option("--study", "study_id", default=None, help="Which study. Defaults to the newest.")
def push_cmd(local: Path, dest: str | None, study_id: str | None) -> None:
    """Upload local files to this study's workspace on the instance.

    The sync only runs one way -- the instance's work comes home on its own, but
    nothing carries your files up, and the agent's tools reach the instance, not your
    disk. This is the other direction: a local file or directory goes to the instance,
    where the agent can then read it. A directory keeps its name (`push ./case` lands
    at `/work/<id>/case/`); a file lands in the study's own directory unless `--to`
    says otherwise. Purely an upload -- it starts the instance if it is asleep and
    spends no tokens.
    """
    cfg = Config.load()
    studies = list_studies(cfg.studies_dir)
    if not studies:
        console.print(f"No studies under {cfg.studies_dir}")
        raise SystemExit(1)
    chosen = study_id or studies[0].study_id
    store = Store(cfg.studies_dir, chosen)
    home = store.session.home or f"{WORKSPACE_ROOT}/{chosen}"

    # A --to is a workspace path (undo any Git-Bash mangling); a bare name hangs off
    # the study's own directory so "push into inputs" does not have to be spelled out.
    if dest:
        target = workspace_path(dest)
        if not target.startswith(WORKSPACE_ROOT):
            target = f"{home}/{target.lstrip('/')}"
    else:
        target = home

    try:
        backend, _client, instance = hosted.acquire(
            cfg.foamd_url, cfg.foamd_api_key, store.session.instance_id or None
        )
    except BackendError as exc:
        console.print(f"[red]Could not reach the workspace service:[/] {exc}")
        raise SystemExit(1) from exc

    console.print(f"[bold]{chosen}[/] on {instance[:8]}")
    try:
        if local.is_dir():
            remote = target if dest else f"{home}/{local.name}"
            backend.put_tree(local, remote)
            count = sum(1 for _ in local.rglob("*") if _.is_file())
            console.print(f"[green]uploaded[/] {count} file(s) from {local} -> {remote}")
        else:
            remote = target if (dest and _looks_like_file(target)) else f"{target}/{local.name}"
            backend.put_file(remote, local.read_bytes())
            console.print(f"[green]uploaded[/] {local.name} -> {remote}")
    except (BackendError, OSError) as exc:
        console.print(f"[red]upload failed:[/] {exc}")
        raise SystemExit(1) from exc
    finally:
        _release(backend)
        backend.close()


def _looks_like_file(remote: str) -> bool:
    """A --to that names a file (has a suffix on its last segment) rather than a dir."""
    return "." in remote.rsplit("/", 1)[-1]


@main.command("renders")
@click.option("--study", "study_id", default=None, help="Which study. Defaults to the newest.")
@click.option("--open", "open_it", is_flag=True, help="Open the folder in the file browser.")
def renders_cmd(study_id: str | None, open_it: bool) -> None:
    """Show this study's pictures: every render and assembled gif, newest first.

    A session copies them into one flat folder as they are made -- no `fetch`, no
    hunting through the workspace tree. This lists that folder and, on a graphics
    terminal, draws the newest. Purely local: no instance is started.
    """
    cfg = Config.load()
    studies = list_studies(cfg.studies_dir)
    if not studies:
        console.print(f"No studies under {cfg.studies_dir}")
        raise SystemExit(1)
    chosen = study_id or studies[0].study_id
    store = Store(cfg.studies_dir, chosen)
    store.renders_dir.mkdir(parents=True, exist_ok=True)
    view = ConsoleView(console)
    view.show_renders(store.renders_dir)
    if open_it:
        _open_folder(store.renders_dir, view)


@main.command("video")
@click.argument("frames", default="")
@click.option("--study", "study_id", default=None, help="Which study. Defaults to the newest.")
@click.option("--fps", default=None, type=float,
              help="Frames per second. Defaults to what the frames were rendered for "
                   f"(their {video_mod.SIDECAR}), else {video_mod.DEFAULT_FPS:g}.")
@click.option("--out", "out_path", default=None, help="Where to write the video. Defaults beside the frames.")
def video_cmd(frames: str, study_id: str | None, fps: float | None, out_path: str | None) -> None:
    """Assemble mirrored render frames into a video, on this machine.

    Stills render on the instance, next to the data; encoding happens here, next
    to the player, with the ffmpeg (or imageio) this machine already has. Give it
    a directory of frames -- a workspace path is mapped to your local mirror -- or
    let it find the study's biggest frame set. Purely local: no instance is
    started and no token is spent.
    """
    cfg = Config.load()
    studies = list_studies(cfg.studies_dir)
    if not studies:
        console.print(f"No studies under {cfg.studies_dir}")
        raise SystemExit(1)
    chosen = study_id or studies[0].study_id
    store = Store(cfg.studies_dir, chosen)

    if frames:
        mapped = workspace_path(frames)
        if mapped.startswith(WORKSPACE_ROOT):
            directory = local_for(store.files_dir, mapped)
        else:
            directory = Path(frames)
    else:
        found = video_mod.best_frame_dir(store.files_dir)
        if found is None:
            console.print(
                f"[yellow]no directory under {store.files_dir} holds two or more "
                "frames[/] - point me at one: openreynolds video <dir>"
            )
            raise SystemExit(1)
        directory = found

    sequence = video_mod.frames_in(directory)
    # The frames say what they were rendered for; the flags say what was asked for
    # now. What is typed wins, and the sidecar answers when nothing was.
    wanted_name, wanted_fps = video_mod.intent(directory)
    fps = wanted_fps if fps is None else fps
    if out_path:
        target = Path(out_path)
    elif wanted_name:
        target = directory.parent / wanted_name
    else:
        target = directory.parent / f"{directory.name}.mp4"
    try:
        tool = video_mod.assemble(sequence, target, fps=fps)
    except video_mod.VideoError as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(1) from exc
    console.print(f"[green]{target}[/]  ({len(sequence)} frames at {fps:g} fps, via {tool})")


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
        _check_video(),
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
    """Enough of a key to tell which one it is, never the whole thing -- and nothing
    at all for an empty one, so "not set" is what a missing key reads as."""
    if not secret:
        return ""
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
    """Validates the key, the endpoint and the model id together, as cheaply as the
    provider allows."""
    if cfg.model_key_missing():
        return "model API: no key", False, ""
    try:
        # With an image in the probe: the agent reads renders, and a model that
        # cannot is the wrong model, however well it answers text.
        detail = make_provider(cfg).probe(cfg.model, vision=True)
    except ValueError as exc:
        return "model API", False, str(exc)
    except ProviderError as exc:
        if exc.status_code:
            return "model API", False, f"{exc.status_code}: {exc.message}"
        return "model API", False, exc.message
    return "model API", True, f"{detail} via {cfg.provider}"


def _check_capture(cfg: Config) -> tuple[str, bool, str]:
    """Whether transcripts can reach the platform.

    Capture is on by default and fails quietly on purpose -- it must never delay or
    break a study. The cost of that is there being no moment at which anyone finds
    out it stopped working, so this is the moment.

    The probe is a read. The capture plane itself is post-only, so this asks the
    same service, with the same key and the same client, for something it already
    has -- which is what capture needs to be true. An earlier version proved the
    point by opening a real study, and every `doctor` run left one behind named
    after itself: a check that changes the thing it is checking.
    """
    if not cfg.capture:
        return "capture", True, "off for this configuration; nothing is uploaded"
    if not (cfg.foamd_url and cfg.foamd_api_key):
        return "capture: not configured", False, ""

    client = hosted.FoamdClient(cfg.foamd_url, cfg.foamd_api_key)
    try:
        client.list_instances()
    except BackendError as exc:
        return "capture", False, f"the platform is not taking this key: {exc}"
    except Exception as exc:  # the platform is optional; naming the failure is not
        return "capture", False, f"{type(exc).__name__}: {exc}"
    finally:
        client.close()
    return "capture", True, "transcripts will be kept; nothing was opened to find out"


def _check_toolbox() -> tuple[str, bool, str]:
    if not TOOLBOX_SOURCE.is_dir():
        return "toolbox: not found in the installed package", False, ""
    scripts = sorted(p.name for p in TOOLBOX_SOURCE.glob("*.py"))
    notes = sorted(p.name for p in (TOOLBOX_SOURCE / "notes").glob("*.md"))
    return "toolbox", True, f"{len(scripts)} scripts, {len(notes)} notes -> {TOOLBOX_DEST}"


def _check_video() -> tuple[str, bool, str]:
    """Which encoder `openreynolds video` would use on this machine.

    Not having one is not a failure -- videos are optional -- but finding that out
    at the moment the frames are ready is the wrong moment."""
    tool = video_mod.encoder()
    if tool == "ffmpeg":
        import shutil as _shutil

        return "video assembly", True, f"ffmpeg at {_shutil.which('ffmpeg')}"
    if tool == "imageio":
        return "video assembly", True, "imageio will encode (ffmpeg not on PATH)"
    return (
        "video assembly",
        True,
        "no encoder found; openreynolds video needs ffmpeg or `pip install imageio`",
    )


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
    keep_alive: bool = False,
    max_wait: float = 0.0,
    interface: Any = None,
) -> str | None:
    """Run one study to its end.

    `interface`, when given, is a callable that takes `drive` -- the one-session
    function below, `drive(view, reader)` -- and runs it against an interface of its
    own: a web page, a test harness, anything that implements `View` and the reader
    protocol. It is the same seam the terminal interface uses (`_run_tui`), so
    supplying one can change what the user reads and never what the model does.
    Its return value says whether the process has to be force-exited afterwards.

    Returns how a one-shot run ended (see `_run_one_shot`), and None for an
    interactive session, where whatever happened was said on screen to someone.
    """
    outcome: str | None = None
    resuming = study_id is not None
    store = Store(cfg.studies_dir, study_id or new_study_id())
    known_here = (store.dir / "session.json").is_file()
    """Whether this machine already held the study before this run. A resume without
    it is a study opened somewhere else, and it is named by its id rather than
    inheriting the shared workspace root."""

    local = bool(os.environ.get("OPENREYNOLDS_LOCAL"))
    try:
        if local:
            # A workspace on this machine: no container, no bill, and nothing to
            # reach. Meant for working on the harness itself, where the question is
            # what the agent does and the answer comes from running it repeatedly.
            backend = LocalBackend()
            client, resolved_instance = None, "local"
            console.print(f"[dim]local workspace: {backend.workspace_root}[/]")
        else:
            backend, client, resolved_instance = hosted.acquire(
                cfg.foamd_url,
                cfg.foamd_api_key,
                instance_id or store.session.instance_id or None,
            )
    except BackendError as exc:
        console.print(f"[red]Could not reach the workspace service:[/] {exc}")
        raise SystemExit(1) from exc

    if getattr(backend, "was_already_running", False):
        # The account is capped at one instance and `acquire()` joins the one that is
        # already there, which is the right default and was completely silent. Two
        # terminals, or a terminal and the web app, then shared four cores with nothing
        # said on either screen -- one live pair ran at a fifth of the throughput each
        # had alone, and both were billed for it.
        console.print(
            f"[yellow]joining the workspace already running on {resolved_instance[:8]}[/] "
            "- another session may be using it, so they share its cores"
        )
    store.session.instance_id = resolved_instance
    store.session.model = cfg.model
    if resuming:
        # A study this machine has never seen -- opened in the browser, or on another
        # laptop -- knows nothing about itself until the platform is asked.
        _recover_session(store, client, study_id)
    store.session.home = _home_for(store, backend, resuming, known_here)
    if not store.session.title and one_shot:
        store.session.title = one_shot[:80]
    store.save()

    capture = None
    if cfg.capture and client is not None:
        if resuming and store.session.remote_study_id:
            capture = Capture(client, store.session.remote_study_id, warn=_warn)
        else:
            capture = Capture.start(
                client, store.session.title or store.session.study_id, resolved_instance,
                study_id=store.session.study_id, home=store.session.home, warn=_warn,
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
    live_mirror = LiveMirror(browser, interval_s=cfg.mirror_interval_s)
    # Delivery rides on the mirror: every render that arrives is surfaced into one
    # flat folder and a directory of frames is assembled into a gif on this machine,
    # so the pictures reach the user without the agent ever running `fetch`. The same
    # moment sends it to the platform, which is how a study run here can be looked at
    # from the browser -- before this, capture only saw a render if the model had
    # happened to `fetch` one, and delivery exists precisely so that it need not.
    live_mirror.gallery = Gallery(store.files_dir, store.renders_dir, capture=capture)
    # A render the model just looked at should be on the user's machine now, not at
    # the next cycle. poke() is non-blocking, so looking costs the model nothing.
    ctx.on_render = lambda _path: live_mirror.poke()

    def drive(view: View, reader: Any) -> None:
        """One session, against whichever interface is running it."""
        nonlocal outcome
        # The tools report job state through the view, so a panel showing what is
        # running is current the moment it changes rather than only while polling.
        ctx.view = view
        # A held job_check ends early the moment the user speaks; the peek takes
        # nothing, so the message still arrives through the usual channel.
        ctx.on_wait_input = getattr(reader, "pending", None)
        # The mirror runs for the whole session -- through turns, and through the
        # hours a solve spends writing while the model's turn is over -- so the
        # user's copy of the study is never more than one interval behind.
        live_mirror.view = view
        # The bar. It reads job logs on its own thread and is told what this thread
        # is doing, so a solve stays on screen with a percentage while the model
        # thinks, and a thought has a clock on it while the solve runs.
        tracker = Tracker(
            view,
            backend=backend,
            store=store,
            home=store.session.home,
            local_dir=store.fetch_dir(),
        )
        live_mirror.progress = tracker
        # The front desk: a second cheap agent that answers the user while this one
        # is mid-turn. Off without a key (BYOK) or when disabled. One-shot runs have
        # nobody at the keyboard, so it does not run there.
        concierge = None
        if cfg.desk and not cfg.model_key_missing() and getattr(reader, "accepts_input", True):
            concierge = Concierge(cfg, store, view, tracker)
            tracker.concierge = concierge
            concierge.start()
        tracker.start()
        live_mirror.start()
        view.workspace(browser)
        loop = Loop(cfg, ctx, store, view, capture=capture, progress=tracker)
        loop.interject = lambda: _typed_while_working(
            loop, view, browser, store, reader, progress=tracker, concierge=concierge
        )
        view.header(store.session.study_id, resolved_instance, cfg.model, store.dir)
        loop.brief(
            _situation_brief(
                store,
                backend,
                resuming,
                interactive=not one_shot,
                browser=browser,
                preferences=cfg.preferences,
            )
        )
        try:
            if one_shot:
                outcome = _run_one_shot(
                    loop, backend, store, one_shot, view, reader, max_wait,
                    live=live_mirror, progress=tracker,
                )
            else:
                _run_interactive(
                    loop, backend, store, view, browser, reader,
                    live=live_mirror, progress=tracker, concierge=concierge,
                )
        except KeyboardInterrupt:
            view.info(_interrupt_note(keep_alive))
        finally:
            tracker.stop()
            if concierge is not None:
                concierge.stop()

    force_exit = False
    try:
        if interface is not None:
            force_exit = bool(interface(drive))
        elif one_shot or plain or not _tui_available():
            drive(ConsoleView(console), LineReader() if not one_shot else NullReader())
        else:
            force_exit = bool(_run_tui(drive))
    finally:
        live_mirror.stop()
        _pickup_results(backend, capture, store.session.home or WORKSPACE_ROOT)
        # The interface is gone by now, so this reports to the plain console. A
        # one-shot run never had turn ends to sync at, which makes this its only one.
        # It runs before anything is stopped: the work comes home first — and it
        # goes through the mirror's own lock, because stop()'s join is bounded: a
        # cycle that outlived it is still writing these same files, and two syncs
        # interleaving over one path is how a local copy ends up a hybrid of two
        # versions of the file.
        live_mirror.view = None
        _final_sync(live_mirror, ConsoleView(console))
        if capture:
            capture.close()
        _close_down(backend, store, keep_alive=keep_alive)
        backend.close()
        if force_exit:
            # The session thread is still inside a network call it cannot be pulled out
            # of. Everything worth keeping is written; waiting for it would leave the
            # user unable to close a program they asked to close.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    return outcome


ONE_SHOT_EXIT_CODES = {"ok": 0, "failed": 1, "timeout": 2}
"""What a `-p` run's ending means to the shell that started it. Documented in the
README, so a script can tell "the model could not be reached" from "the solve is
still going" without parsing the output."""

WORKSPACE_LISTED = 40


def _recover_session(store: Store, client: Any, study_id: str | None) -> None:
    """Fill in what this machine does not know about a study it is resuming.

    `openreynolds --study <id>` does not require the study to have been run here, and
    that is the point: a study run in the browser should be openable on a laptop. But
    with no local `session.json` there was nothing to say which directory on the volume
    belonged to it, and `_home_for` fell back to the workspace root -- so the study
    opened among every other study's files, the mirror tried to bring the whole volume
    down, and capture opened a second row because it could not find the first. The
    platform knows all three facts; ask it.

    Never raises: an older service without the read route, or no network, leaves the
    session exactly as it was and the caller carries on with local state.
    """
    if not study_id or store.session.home:
        return
    try:
        row = client.get_study(study_id)
    except Exception:  # noqa: BLE001 - an older service, or none reachable
        return
    if not isinstance(row, dict) or not row.get("id"):
        return
    store.session.remote_study_id = str(row["id"])
    if row.get("home"):
        store.session.home = str(row["home"])
    if row.get("instance_id") and not store.session.instance_id:
        store.session.instance_id = str(row["instance_id"])
    if row.get("title") and not store.session.title:
        store.session.title = str(row["title"])


def _home_for(store: Store, backend: Backend, resuming: bool, known_here: bool = True) -> str:
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
    elif resuming and known_here:
        # A study this machine already had, whose session predates homes: it keeps
        # the whole workspace, because moving its files out from under it would be
        # worse than the untidiness.
        home = WORKSPACE_ROOT
    else:
        # A new study, or one resumed on a machine that has never seen it -- opened
        # in the browser, or on another laptop. `known_here` is false there, and the
        # id names the directory. Falling back to the workspace root instead is what
        # put one run among every other run's files, and made the mirror try to bring
        # the whole volume down.
        home = f"{WORKSPACE_ROOT}/{store.session.study_id}"

    if home != WORKSPACE_ROOT:
        try:
            backend.exec(f"mkdir -p {home}", timeout_s=60)
        except BackendError as exc:
            console.print(f"[yellow]could not make {home} ({exc}); using {WORKSPACE_ROOT}[/]")
            return WORKSPACE_ROOT
    return home


def _machine_note(backend: Backend) -> str:
    """What this machine is, measured rather than assumed.

    The tool descriptions carry the same arithmetic, and a live run showed they can
    be passed over entirely: it chose a serial solve on a 34,764-cell mesh without
    the words serial, parallel or decompose appearing anywhere in its reasoning. The
    question never came up. The briefing arrives as a message rather than as a
    description of a tool, which makes it the one place a fact cannot be skipped --
    so a fact this expensive to miss belongs here. It says what the machine is and
    what was measured on it; what to do about that stays where every other decision
    stays.
    """
    try:
        result = backend.exec("nproc", timeout_s=30)
        cores = int((result.output or "").strip().split()[0])
    except Exception:  # noqa: BLE001 - not knowing is not a failed session
        return ""
    if cores <= 1:
        return ""
    return (
        f"This machine has {cores} cores. A solver run as one process uses one of "
        f"them, and the session is billed for all {cores} either way; `decomposePar` "
        f"and `mpirun -np N` spread it over N. What the extra ranks return falls away "
        f"as the cells each one holds get small, so the N worth using depends on the "
        f"mesh."
    )


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
        neighbours = _neighbours(browser, home)

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


def _neighbours(browser: Browser, home: str) -> str:
    """One sentence about the other session directories, or nothing when there are none.

    Saying whose the work is without saying what those sessions are leaves a real
    question open, and a live run spent turns on it: it found several near-identical
    studies made minutes apart by a user who did not remember commissioning them, and
    worked through whether that meant an intruder. The answer is dull and the harness
    has always known it -- they are this same tool's other sessions.
    """
    try:
        siblings = [
            entry
            for entry in browser.tree(WORKSPACE_ROOT, depth=1)
            if entry.is_dir and not entry.name.startswith(".") and entry.path != home
        ]
    except BackendError:
        return ""
    if not siblings:
        return f"\nNothing else of this tool's is on {WORKSPACE_ROOT}; yours is the first."
    count = (
        "one other directory"
        if len(siblings) == 1
        else f"{len(siblings)} other directories"
    )
    return (
        f"\n{WORKSPACE_ROOT} also holds {count} from this tool's own earlier sessions "
        "on this instance, one per study id. They are readable, and none of them was "
        "written for this request."
    )


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
    preferences: str = "",
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
    machine = _machine_note(backend)
    if machine:
        lines.append(machine)
    if preferences:
        # The user's standing note, in the user's voice. The harness relays it
        # verbatim and adds nothing: what to do about it stays the model's call,
        # like anything else the user says.
        lines.append(
            "The user keeps a standing note that they ask to have passed on at the "
            "start of every session. In their own words:"
        )
        lines.append(preferences.strip())
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


def _final_sync(live: LiveMirror, view: View) -> None:
    """The close-down sync, serialized with any cycle that outlived stop().

    Same reporting contract as _mirror: everything that can go wrong becomes a
    line of text, never an exception -- a convenience may not end a session."""
    try:
        report = live.sync_now()
    except Exception as exc:  # noqa: BLE001
        view.warn(f"could not mirror this study's files: {exc}")
        return
    for line in report.brief():
        view.info(line)


def _mirror(browser: Browser, view: View, everything: bool = True) -> None:
    """Copy this study's files down, and say briefly what came.

    Nobody asks for this. That is the point of it: what shipped before was a study
    that ran for half an hour on the instance and left two files on the machine of
    the person who commissioned it, and there is no version of that which is what
    they wanted.

    It is never allowed to be the reason a session ends, so everything that can go
    wrong here becomes a line of text instead of an exception.
    """
    try:
        report = mirror_sync(browser, everything=everything)
    except Exception as exc:  # noqa: BLE001 - a convenience may not end a session
        view.warn(f"could not mirror this study's files: {exc}")
        return
    for line in report.brief():
        view.info(line)


def _release(backend: Backend) -> None:
    """Put a borrowed instance back down.

    Only a session is a reason for a container to be running. Everything else here --
    listing files, pulling them, checking the plumbing -- lazy-starts one as a side
    effect of asking a question, and every one of those routes was leaving it up
    afterwards to idle for fifteen minutes on somebody's bill. If it was already
    running, it belongs to whoever started it and is left alone.
    """
    if getattr(backend, "was_already_running", False):
        return
    shutdown = getattr(backend, "shutdown", None)
    if shutdown is None:
        return
    try:
        shutdown()
    except BackendError as exc:
        console.print(f"[dim]could not stop the instance ({exc}); it will idle out[/]")


def _interrupt_note(keep_alive: bool) -> str:
    """What Ctrl+C actually does, said at the moment it is done.

    The old line promised "jobs keep running on the instance" and then the
    close-down stopped every one of them -- exactly the kind of claim nothing kept
    true. The note now states what the teardown that follows will do."""
    if keep_alive:
        return "interrupted - jobs keep running on the instance (--keep-alive)"
    return (
        "interrupted - stopping jobs and the instance; the volume and your local "
        "copy of the study stay"
    )


def _close_down(backend: Backend, store: Store, keep_alive: bool = False) -> None:
    """End the session: stop the work, then put the container down.

    A container left running is a container being paid for, and the only thing that
    knows the session is over is the session. The earlier design left jobs running on
    purpose -- closing a laptop on a long solve is a real thing to want -- but the
    default it produced was worse: an idle machine burning eight cores until a reaper
    noticed, and a bill arriving before anyone did.

    So the default is now the other way round, and it is safe to be: the study's files
    have already come home, and stopping an instance leaves its volume untouched.
    `--keep-alive` is there for the laptop case, and says what it is choosing.
    """
    study = store.session.study_id
    home = store.session.home or WORKSPACE_ROOT
    shared = bool(getattr(backend, "was_already_running", False))
    """Whether this session joined a workspace somebody else had already started.

    An account is capped at one instance and `acquire()` joins the existing one without
    saying so, so a second terminal -- or the web app, or `openreynolds files` -- lands in
    the same container. Stopping it, or sweeping it, then reaches work this session never
    started. One live run lost a 22-minute solve to exactly that."""
    console.print(f"\n[dim]this study's files are in {store.dir}[/]")
    console.print(f"[dim]on the instance they are at {home}[/]")

    live = store.live_jobs()
    if keep_alive:
        if live:
            names = ", ".join(job.name or job.job_id[:8] for job in live)
            console.print(f"[yellow]{len(live)} job(s) still running:[/] {names}")
            console.print("[dim]they keep going, and keep costing, until they finish[/]")
            console.print(f"[dim]  stop:   openreynolds stop --study {study}[/]")
        console.print(f"[dim]  resume: openreynolds --study {study}[/]")
        return

    if live:
        names = ", ".join(job.name or job.job_id[:8] for job in live)
        console.print(f"[dim]stopping {len(live)} running job(s): {names}[/]")
    # Scoped to this study's own directory: the sweep still catches mpirun ranks that
    # outlived their job's process group, which is why it exists, and can no longer
    # reach a solve another session is running on the same instance.
    report = stop_everything(backend, store, home=home)
    for line in report.lines():
        console.print(f"  [{'green' if report.clean else 'yellow'}]{line}[/]")

    if shared:
        # Somebody else's session had this workspace up before this one joined it, so
        # it is theirs to stop. `_release` has said so for every read-only command
        # since it was written; the session path is the one that never asked.
        console.print(
            "[dim]this workspace was already running when this session joined it, "
            "so it is left up[/]"
        )
    else:
        try:
            shutdown = getattr(backend, "shutdown", None)
            if shutdown is not None:
                shutdown()
                console.print("[dim]instance stopped; the workspace volume is untouched[/]")
        except BackendError as exc:
            console.print(f"[yellow]could not stop the instance ({exc}); it will idle out[/]")

    console.print(f"[dim]  resume: openreynolds --study {study}[/]")


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
        console.print(f"[dim]your copy of them: {store.files_dir}[/]")
        console.print(f"[dim]resume with:  openreynolds --study {study}[/]")
        console.print(f"[dim]look at them: openreynolds files --study {study}[/]")
        console.print(f"[dim]bring more:   openreynolds pull --study {study} --all[/]")
        return
    names = ", ".join(job.name or job.job_id[:8] for job in live)
    console.print(f"\n[yellow]{len(live)} job(s) still running on the instance:[/] {names}")
    console.print("[dim]they keep going, and keep costing, until they finish[/]")
    console.print(f"[dim]  your copy of this study so far: {store.files_dir}[/]")
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
    command: commands.Command,
    loop: Loop,
    view: View,
    browser: Browser,
    store: Store,
    progress: Any = None,
) -> Any:
    """Act on one typed line. Returns what goes to the model, or None, or QUIT."""
    if command.kind == commands.EXIT:
        return QUIT
    if command.kind in (commands.SAY, commands.ASIDE):
        if not command.text:
            return None
        loop.say(command.text)
        return command.text
    _local(command, view, browser, store, loop, progress)
    return None


def _local(
    command: commands.Command,
    view: View,
    browser: Browser,
    store: Store,
    loop: Loop | None = None,
    progress: Any = None,
) -> None:
    """Commands answered here, out of what the harness already knows.

    None of these reach the model. That is the whole point of them: a question that
    costs a turn and derails the work is a question people stop asking, and then they
    have no idea what is going on.
    """
    if command.kind == commands.STATUS:
        # The same picture the bar shows, so /status and the bar cannot disagree.
        stage = progress.snapshot().headline if progress is not None else ""
        view.status(
            commands.status_lines(
                store,
                stage=stage,
                tokens=getattr(loop, "context_tokens", 0) or 0,
                token_totals=getattr(loop, "token_totals", None),
                local_files=len(browser.local()),
                sync_age=browser.cache_age(),
            )
        )
    elif command.kind == commands.FILES:
        view.show_files(command.text)
    elif command.kind == commands.RENDERS:
        store.renders_dir.mkdir(parents=True, exist_ok=True)
        view.show_renders(store.renders_dir)
    elif command.kind == commands.OPEN:
        _open_folder(store.dir, view)
    elif command.kind == commands.HELP:
        view.status(commands.HELP_TEXT.splitlines())


def _typed_while_working(
    loop: Loop,
    view: View,
    browser: Browser,
    store: Store,
    reader: Any,
    progress: Any = None,
    concierge: Any = None,
) -> str | None:
    """Drain what was typed mid-turn, and hand back only what was meant for the model.

    `/status` and `/files` are answered here and now, without a turn. Everything else
    rides along with the next tool result, so it lands at the model's next turn rather
    than sitting unread until the whole turn is over -- and, because "the next turn"
    can be five minutes away when the agent is deep in a solve, the same text is handed
    to the front desk, which answers within seconds without waiting on the main thread.
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
            if concierge is not None:
                concierge.ask(command.text)
        else:
            _local(command, view, browser, store, loop, progress)
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


RETRYABLE_MODEL_STATUSES = frozenset({408, 409, 425, 429})
"""Model-API failures worth waking the model on again: a timeout, a conflict, a
rate limit. Everything else the API answers with a 4xx is about this account or
this request -- the budget, the key, the model id -- and answers the same way in a
minute."""

_REFUSALS = {
    400: "the request itself was rejected",
    401: "the key was not accepted",
    402: "the account cannot pay for this call -- a budget or a usage cap",
    403: "this account is not allowed to make this call",
    404: "there is no such model at that endpoint",
    413: "the request was too large to accept",
}


def _run_turn(loop: Loop, view: View) -> bool:
    """Run a turn, surviving a model-API failure. Returns whether it completed.

    A long study will meet a rate limit or a dropped connection eventually, and
    losing the whole session to one is a poor trade when the thread is still intact
    and the jobs are still running on the instance.

    A refusal is the other case, and it is not survivable by waiting. When the API
    says 402 or 401, every later call says it too: a live session watched a job for
    twenty-six minutes making ninety refused calls, answering nobody, because the
    harness could not tell "try again" from "this will never work". `blocked_reason`
    is that distinction, and it is the caller's cue to stop asking until a person
    says something.
    """
    status: int | None = None
    said = ""
    try:
        loop.run()
        loop.api_failures = 0
        loop.blocked_reason = None
        return True
    except ProviderError as exc:
        loop.api_failures += 1
        status = exc.status_code
        if status:
            said = f"The model API returned {status}: {exc.message}"
        else:
            said = f"Could not reach the model API: {exc.message}"
        console.print()
        console.print(f"[red]{said}[/]")
        # The terminal sees the console; a web page sees the view. A refused call
        # that only reached the console looked, on the page, like an agent that
        # had gone quiet -- for ten minutes, to a person typing "what's going on?".
        view.notice(said)

    loop.settle()
    refused = bool(status) and 400 <= status < 500 and status not in RETRYABLE_MODEL_STATUSES
    study = loop.store.session.study_id
    if refused:
        loop.blocked_reason = said
        because = _REFUSALS.get(status, "the service refused the request")
        console.print(
            f"[yellow]That is a refusal, not a hiccup: {because}. Waiting will not "
            "change it, so nothing more will be sent until you say something.[/]\n"
            "[yellow]Your work is safe -- every job is still running on the instance "
            "and every file is mirrored here.[/]"
        )
        view.info("waiting for you - the model service refused the last call")
    elif loop.api_failures >= 2:
        # Twice in a row is not a blip. Say plainly what is happening and what to do,
        # rather than repeating "the thread is intact" while nothing gets through --
        # which is exactly what a live session did, to a very frustrated user.
        console.print(
            f"[yellow]The model API has failed {loop.api_failures} times in a row. "
            "The usual cause is a rate limit or usage cap on your model API key, not "
            "your work -- every job and file is safe on the instance and mirrored "
            "here.[/]\n[yellow]Wait a minute and try again, or leave with /exit and "
            f"resume in a fresh, smaller thread: [bold]openreynolds --study {study}[/][/]"
        )
    else:
        view.info("the thread is intact - say something to continue, or /exit")
    return False


def _run_interactive(
    loop: Loop,
    backend: Backend,
    store: Store,
    view: View,
    browser: Browser,
    reader: Any,
    live: LiveMirror | None = None,
    progress: Any = None,
    concierge: Any = None,
) -> None:
    while True:
        if store.live_jobs():
            if progress is not None:
                progress.idle()
            wake = watch(
                backend, store, view, reader,
                narrate_every_s=loop.cfg.narrate_every_s,
                progress=progress,
            )
            if wake.kind == "eof":
                return
            if wake.kind in ("job", "narrate"):
                if loop.blocked_reason:
                    # The service is refusing calls for a reason waiting does not fix.
                    # A job ending is a fact worth keeping in the thread for whenever
                    # this resumes; progress chatter is not, and neither is a turn --
                    # it would be refused exactly as the last ninety were.
                    if wake.kind == "job":
                        loop.inform(wake.text)
                    continue
                loop.inform(wake.text)
            elif wake.kind == "user":
                # A person speaking is the one thing that can change a refusal:
                # they have topped the account up, fixed the key, or want to hear
                # the failure again. Either way, try.
                loop.blocked_reason = None
                # A long solve is exactly when someone wants to leave, or to ask what
                # is happening without setting the whole thing off again. The desk
                # answers the question now; the model still hears it at the next wake.
                if concierge is not None and commands.parse(wake.text).kind in (
                    commands.SAY, commands.ASIDE
                ):
                    concierge.ask(commands.parse(wake.text).text)
                spoken = _apply(
                    commands.parse(wake.text), loop, view, browser, store, progress
                )
                if spoken is QUIT:
                    return
                if spoken is None:
                    continue
            else:
                continue
            from_prompt = False
        else:
            if progress is not None:
                progress.begin("waiting")
            view.prompt()
            line = reader.get()
            if line is None:
                return
            loop.blocked_reason = None
            spoken = _apply(commands.parse(line), loop, view, browser, store, progress)
            if spoken is QUIT:
                return
            if spoken is None:
                continue
            from_prompt = True

        completed = _run_turn(loop, view)
        # If a turn typed at the prompt failed, the desk still answers -- the reason
        # someone asks "are you still working?" is usually that the agent went quiet,
        # and a failing turn is exactly that. (Mid-solve messages already reach the
        # desk above; this covers the idle prompt, which did not.)
        if not completed and from_prompt and concierge is not None and isinstance(spoken, str):
            concierge.ask(spoken)
        # After the turn rather than before it: whatever the agent just made is the
        # thing worth having, and a turn that failed still leaves whatever its jobs
        # wrote. Unchanged files are not asked for again, so this is cheap -- and it
        # is asked for, not waited for: the next thing this thread does is read what
        # the user typed, and a sync that takes twenty minutes must not stand in
        # front of that (see LiveMirror.catch_up).
        if live is not None:
            live.catch_up()
        else:
            _mirror(browser, view)
        if completed and loop.needs_refresh:
            loop.refresh(situation(store, backend))


def _run_one_shot(
    loop: Loop,
    backend: Backend,
    store: Store,
    prompt: str,
    view: View,
    reader: Any,
    max_wait_minutes: float = 0.0,
    live: LiveMirror | None = None,
    progress: Any = None,
) -> str:
    """Run until the model is done and no jobs remain, and say how it went.

    There is nobody here to answer a question, so if the model ends its turn wanting
    one, this waits on the job instead -- possibly for hours. `--max-wait` bounds that.
    Stopping only ends the waiting: the job carries on out on the instance and the
    study resumes.

    The answer is one of "ok", "failed" (the model API would not complete a turn)
    and "timeout" (`--max-wait` ran out with a job still running). A script driving
    `-p` has nothing but the exit code to go on, and for a long while all three
    exited 0 -- so a scheduled run whose every turn was refused by a rate limit
    looked, to the scheduler, exactly like one that had finished.
    """
    loop.say(prompt)
    if not _run_turn(loop, view):
        return "failed"
    if live is not None:
        live.catch_up()

    deadline = time.monotonic() + max_wait_minutes * 60 if max_wait_minutes else None
    while store.live_jobs():
        if progress is not None:
            progress.idle()
        wake = watch(backend, store, view, reader, deadline=deadline, progress=progress)
        if wake.kind == "timeout":
            view.info(
                f"stopped waiting after {max_wait_minutes:g} min; the job is still "
                f"running - resume with --study {store.session.study_id}"
            )
            return "timeout"
        if wake.kind != "job":
            break
        loop.inform(wake.text)
        if not _run_turn(loop, view):
            return "failed"
        if live is not None:
            live.catch_up()
        if loop.needs_refresh:
            loop.refresh(situation(store, backend))
    return "ok"


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
