"""The local Docker workspace (`backend/docker.py`) and the switches that choose it.

The pure parts run everywhere. The ones that need a container run only where Docker
and the image are present (`OPENREYNOLDS_TEST_IMAGE`, or the default image already
pulled), and mesh rather than solve: blockMesh on the cavity tutorial.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from openreynolds import cli, prompt
from openreynolds.backend.base import WORKSPACE_ROOT, BackendError
from openreynolds.backend import docker as docker_module
from openreynolds.backend.docker import DEFAULT_IMAGE, DockerBackend
from openreynolds.config import Config

REPO = Path(__file__).resolve().parents[1]


posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="symbolic links and a shell-script docker are POSIX")


# -- choosing it ---------------------------------------------------------------


@pytest.mark.parametrize("value, kind", [
    ("", None), ("0", None), ("false", None),
    ("1", "native"), ("native", "native"), ("docker", "docker"), ("Docker", "docker"),
])
def test_which_local_workspace_the_environment_asks_for(monkeypatch, value, kind):
    monkeypatch.setenv("OPENREYNOLDS_LOCAL", value)
    assert cli.local_workspace() == kind


def test_a_local_workspace_needs_only_the_models_key():
    """The open-source agent on a laptop: Docker and a model key, no service account."""
    cfg = Config(foamd_url="", foamd_api_key="", provider="anthropic", llm_api_key="sk-x")
    assert cfg.missing() == ["FOAMD_URL", "FOAMD_API_KEY"]
    assert cfg.missing(local=True) == []


def test_the_services_model_still_needs_the_services_key_locally():
    cfg = Config(foamd_url="", foamd_api_key="", provider="reynolds")
    assert "FOAMD_API_KEY" in cfg.missing(local=True)


# -- what the model is told ------------------------------------------------------


def test_every_local_swap_still_finds_its_sentence():
    """A swap whose hosted sentence was reworded would silently leave the hosted claim
    (billed, root, network volume) in the local prompt."""
    for hosted, local in prompt.LOCAL_SWAPS:
        assert hosted in prompt.SYSTEM_PROMPT
        assert hosted not in prompt.LOCAL_PROMPT
        assert local in prompt.LOCAL_PROMPT


def test_the_local_prompt_makes_no_hosted_claims():
    text = prompt.system_prompt("openfoam", local=True)
    assert "billed for all of them" not in text
    assert "runs as root" not in text
    assert "network filesystem" not in text
    assert prompt.system_prompt("openfoam") == prompt.SYSTEM_PROMPT
    assert prompt.system_prompt("felix", local=True) == prompt.FELIX_PROMPT


# -- the container, without one ----------------------------------------------------


def test_the_container_is_sandboxed_and_mounts_the_workspace_at_work(tmp_path):
    backend = DockerBackend(root=tmp_path, image="example/image:1")
    args = backend.run_args()
    assert args[args.index("--network") + 1] == "none"
    assert args[args.index("-v") + 1] == f"{tmp_path.resolve()}:{WORKSPACE_ROOT}"
    assert args[-4:] == ["--entrypoint", "sleep", "example/image:1", "infinity"]
    assert backend.workspace_root == WORKSPACE_ROOT


def test_root_in_the_container_gets_the_codestream_shim(tmp_path, monkeypatch):
    """Docker Desktop on Windows runs the container as root, where OpenFOAM refuses
    to compile #codeStream unless the image's shim is preloaded."""
    monkeypatch.delattr("os.getuid", raising=False)
    args = DockerBackend(root=tmp_path, image="x:1").run_args()
    assert "--user" not in args
    assert f"LD_PRELOAD={docker_module.ROOT_SHIM}" in args
    assert docker_module.ROOT_SHIM in (REPO / "docker/workspace/Dockerfile").read_text()


def test_the_default_image_is_the_one_this_repository_builds():
    """The workflow publishes `:<the FROM line's OpenFOAM version>`, and that is the
    tag DEFAULT_IMAGE names."""
    dockerfile = (REPO / "docker/workspace/Dockerfile").read_text()
    base = next(line for line in dockerfile.splitlines() if line.startswith("FROM "))
    assert "@sha256:" in base, "the base is pinned by digest"
    version = base.split(":")[1].split("@")[0]
    assert DEFAULT_IMAGE == f"ghcr.io/inviscidai/reynolds-workspace:{version}"
    workflow = (REPO / ".github/workflows/workspace-image.yml").read_text()
    assert f'VERSION: "{version}"' in workflow


def test_the_default_image_is_the_published_one(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENREYNOLDS_IMAGE", raising=False)
    assert DockerBackend(root=tmp_path).image == DEFAULT_IMAGE
    monkeypatch.setenv("OPENREYNOLDS_IMAGE", "mine:1")
    assert DockerBackend(root=tmp_path).image == "mine:1"


def test_a_missing_docker_is_said_plainly(tmp_path):
    backend = DockerBackend(root=tmp_path, docker=str(tmp_path / "no-docker-here"))
    with pytest.raises(BackendError) as caught:
        backend.exec("true")
    assert caught.value.code == "docker_missing"


def test_one_workspace_directory_is_one_container(tmp_path):
    a = DockerBackend(root=tmp_path / "a")
    assert DockerBackend(root=tmp_path / "a").container == a.container
    assert DockerBackend(root=tmp_path / "b").container != a.container


def test_a_job_id_cannot_name_a_path(tmp_path):
    backend = DockerBackend(root=tmp_path)
    with pytest.raises(BackendError):
        backend.job_status("../../etc")


@posix_only
def test_a_docker_that_cannot_reach_its_daemon_is_said_plainly(tmp_path):
    fake = tmp_path / "docker"
    fake.write_text("#!/bin/sh\necho 'permission denied while trying to connect to the "
                    "Docker daemon socket at unix:///var/run/docker.sock' >&2\nexit 1\n")
    fake.chmod(0o755)
    backend = DockerBackend(root=tmp_path / "w", docker=str(fake))
    with pytest.raises(BackendError) as caught:
        backend.exec("true")
    assert caught.value.code == "docker_unavailable"
    assert "docker group" in caught.value.message


@posix_only
def test_a_link_out_of_the_workspace_is_not_followed_on_this_machine(tmp_path):
    """A link the model makes in the container is read here, on the host, where `/`
    is this machine's: writing through it wrote anywhere the person can write."""
    backend = DockerBackend(root=tmp_path / "w")
    case = backend.root / "s1"
    (case / "real").mkdir(parents=True)
    (case / "real" / "f").write_text("hi")
    (case / "root").symlink_to("/")
    (case / "up").symlink_to("../..")
    (case / "abs").symlink_to("/work/s1/real")
    (case / "rel").symlink_to("real")
    (case / "loop").symlink_to("loop")
    for path in ["/work/s1/root/etc/passwd", "/work/s1/up/x", "/work/s1/loop/x"]:
        with pytest.raises(BackendError) as caught:
            backend.get_file(path)
        assert caught.value.code == "outside_workspace", path
    with pytest.raises(BackendError):
        backend.put_file(f"/work/s1/root{tmp_path}/escaped", b"x")
    assert not (tmp_path / "escaped").exists()
    with pytest.raises(BackendError):
        backend.stat("/work/s1/root")
    assert backend.get_file("/work/s1/abs/f") == b"hi", "a link to /work is the workspace"
    assert backend.get_file("/work/s1/rel/f") == b"hi"


@posix_only
def test_a_tree_copied_out_keeps_its_links_as_links(tmp_path):
    backend = DockerBackend(root=tmp_path / "w")
    tree = backend.root / "s1" / "t"
    tree.mkdir(parents=True)
    (tree / "a").write_text("a")
    (tree / "host").symlink_to("/")
    got = backend.get_tree(["/work/s1/t"], tmp_path / "mirror")
    assert [p.name for p in got] == ["a"]
    assert (tmp_path / "mirror" / "t" / "host").is_symlink(), "copied as a link, not as /"
    assert backend.get_tree(["/work/s1/t/host", "/work/s1/t/a"], tmp_path / "m2") == [
        tmp_path / "m2" / "a"], "a link out, named on its own, is skipped like a missing file"


@posix_only
def test_a_tree_copied_in_does_not_write_through_a_link(tmp_path):
    backend = DockerBackend(root=tmp_path / "w")
    (backend.root / "s1").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (backend.root / "s1" / "case").symlink_to(outside)
    upload = tmp_path / "upload"
    (upload / "sub").mkdir(parents=True)
    (upload / "sub" / "f").write_text("x")
    (upload / "sub" / "f").chmod(0o755)
    with pytest.raises(BackendError):
        backend.put_tree(upload, "/work/s1/case")
    assert list(outside.iterdir()) == []
    backend.put_tree(upload, "/work/s1/fine")
    assert (backend.root / "s1" / "fine" / "sub" / "f").read_text() == "x"
    assert (backend.root / "s1" / "fine" / "sub" / "f").stat().st_mode & 0o111, "still runnable"


@posix_only
def test_a_job_whose_files_became_links_is_refused(tmp_path):
    backend = DockerBackend(root=tmp_path / "w")
    job = "0" * 32
    folder = backend.root / ".logs" / "jobs" / job
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text("{}")
    (folder / "ended_at").symlink_to(tmp_path / "victim")
    with pytest.raises(BackendError) as caught:
        backend.job_status(job)
    assert caught.value.code == "outside_workspace"
    assert not (tmp_path / "victim").exists()


# -- the container, with one -------------------------------------------------------


def _image() -> str | None:
    import os

    if not shutil.which("docker"):
        return None
    image = os.environ.get("OPENREYNOLDS_TEST_IMAGE") or DEFAULT_IMAGE
    found = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    return image if found.returncode == 0 else None


needs_image = pytest.mark.skipif(_image() is None, reason="no Docker, or the image is not here")


@pytest.fixture
def live(tmp_path):
    backend = DockerBackend(root=tmp_path / "work", image=_image())
    yield backend
    subprocess.run(["docker", "rm", "-f", backend.container], capture_output=True)


@needs_image
def test_a_command_sees_work_and_openfoam(live):
    result = live.exec("pwd && echo $WM_PROJECT_VERSION && touch made-here", cwd="/work/s1")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[:2] == ["/work/s1", "v2512"]
    made = live.root / "s1" / "made-here"
    assert made.is_file(), "a file written in the container is in the host directory"
    assert made.stat().st_uid == Path(live.root).stat().st_uid, "owned by the person, not root"


@needs_image
def test_the_container_has_no_network(live):
    result = live.exec("python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 3)\"")
    assert result.exit_code != 0


@needs_image
def test_a_command_past_its_time_is_stopped_inside_the_container(live):
    result = live.exec("sleep 30", timeout_s=2)
    assert result.exit_code == 124
    assert "timed out" in result.output
    left = live.exec("pgrep -ax sleep | grep -v infinity || true")
    assert left.output.strip() == "", "the command did not outlive its timeout"


@needs_image
def test_the_container_itself_is_not_another_sessions_work(live):
    """It said "another session is working on this workspace: docker-init" at the end
    of every local session, and left the workspace up for nobody."""
    live.exec("true")
    assert cli._neighbour_work(live, "/work/s1") == []


@needs_image
def test_blockmesh_runs_on_the_cavity(live):
    live.exec("cp -r $FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity .", cwd="/work/s1")
    result = live.exec("blockMesh", cwd="/work/s1/cavity", timeout_s=120)
    assert result.exit_code == 0, result.output[-500:]
    assert (live.root / "s1" / "cavity" / "constant" / "polyMesh" / "points").is_file()


@needs_image
def test_a_job_runs_ends_and_is_still_known_to_a_new_session(live):
    job = live.job_start("echo hello; sleep 1; echo bye", cwd="/work/s1", name="greet")
    assert live.job_status(job).running
    deadline = time.monotonic() + 30
    while live.job_status(job).running and time.monotonic() < deadline:
        time.sleep(0.2)
    done = live.job_status(job)
    assert (done.status, done.end_reason, done.exit_code) == ("exited", "completed", 0)
    assert "hello" in live.job_tail(job)[0]
    again = DockerBackend(root=live.root, image=live.image)
    assert again.job_status(job).end_reason == "completed", "a resumed session can see it"


@needs_image
def test_a_job_and_what_it_started_are_killed_together(live):
    job = live.job_start("sleep 300 & sleep 300; wait", cwd="/work/s1")
    status = live.job_kill(job)
    assert (status.status, status.end_reason) == ("killed", "killed_by_client")
    assert live.exec("pgrep -ax sleep | grep -v infinity || true").output.strip() == ""


@needs_image
def test_kill_on_stops_a_job_at_the_line(live):
    job = live.job_start("for i in $(seq 1 100); do echo step $i; sleep 0.2; done",
                         cwd="/work/s1", kill_on=[r"step 3\b"])
    deadline = time.monotonic() + 30
    while live.job_status(job).running and time.monotonic() < deadline:
        time.sleep(0.2)
    status = live.job_status(job)
    assert status.end_reason == "kill_on_match"
    assert status.killed_by == "step 3"


@needs_image
def test_shutting_down_removes_the_container_and_keeps_the_files(live):
    live.exec("echo kept > note.txt", cwd="/work/s1")
    live.shutdown()
    assert live._state() is None
    assert (live.root / "s1" / "note.txt").read_text() == "kept\n"
    assert live.exec("cat note.txt", cwd="/work/s1").output == "kept\n", "a fresh one comes up"


@needs_image
def test_a_container_stopped_under_the_session_comes_back(live):
    """Docker restarting, the laptop sleeping, `docker stop`: every command after it
    failed with "container ... is not running" for the rest of the session."""
    live.exec("true")
    subprocess.run(["docker", "stop", "-t", "1", live.container], capture_output=True)
    assert live.exec("echo back").output == "back\n"
    subprocess.run(["docker", "rm", "-f", live.container], capture_output=True)
    job = live.job_start("echo again", cwd="/work/s1")
    assert live.job_status(job).name is None


@needs_image
def test_a_job_whose_container_went_away_is_said_to_have_been_killed(live):
    job = live.job_start("sleep 300", cwd="/work/s1")
    subprocess.run(["docker", "rm", "-f", live.container], capture_output=True)
    status = live.job_status(job)
    assert (status.status, status.end_reason) == ("exited", "killed_externally")


@needs_image
def test_a_missing_image_names_itself_and_the_way_round(tmp_path):
    backend = DockerBackend(root=tmp_path / "w", image="openreynolds-test/no-such-image:0")
    with pytest.raises(BackendError) as caught:
        backend.exec("true")
    assert caught.value.code == "image_missing"
    assert "openreynolds-test/no-such-image:0" in caught.value.message
    assert "OPENREYNOLDS_IMAGE" in caught.value.message


@needs_image
def test_a_parallel_mesh_check_runs_as_this_user(live):
    """mpirun as a user the image may have no name for, which is anyone's but one."""
    setup = (
        "cp -r $FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity . && cd cavity && "
        "blockMesh > /dev/null && printf 'FoamFile{version 2.0;format ascii;class "
        "dictionary;object decomposeParDict;}\\nnumberOfSubdomains 2;\\nmethod scotch;\\n' "
        "> system/decomposeParDict && decomposePar -force > /dev/null && "
        "mpirun -np 2 checkMesh -parallel"
    )
    result = live.exec(setup, cwd="/work/mp", timeout_s=300)
    assert result.exit_code == 0, result.output[-500:]
    assert "Mesh OK" in result.output


def test_a_resume_hint_names_the_local_workspace(monkeypatch):
    monkeypatch.setenv("OPENREYNOLDS_LOCAL", "docker")
    assert cli._local_flag() == " --local"
    monkeypatch.delenv("OPENREYNOLDS_LOCAL")
    assert cli._local_flag() == ""
