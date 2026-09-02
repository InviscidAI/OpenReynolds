#!/usr/bin/env python3
"""Is the shock-capturing solver on this instance, and what does a shell need to run it?

HiSA is an implicit density-based OpenFOAM solver (AUSM+up flux, dual-time stepping,
GMRES under LU-SGS). Newer images bake it in, at the OpenFOAM site directories that the
sourced bashrc already puts on PATH and LD_LIBRARY_PATH, so on those instances it runs
with no exports at all. Before that it was built from source onto the workspace volume
(2026-08-30, during the ONERA M6 replication where `rhoSimpleFoam` converged beautifully
and showed no shock at any of seven span stations; on the same 1.79M-cell mesh HiSA
resolved the lambda structure and landed drag within 0.3% of a published band), and a
volume build is still the route onto an instance running an older image.

So this looks in the image's site directories first and at the volume second, and says
which of the two it found. Whether it is here at all remains a fact about *this*
instance, and guessing wrong is expensive in both directions: assuming it is absent
means re-deriving a density-based dictionary set by hand, and assuming it is present
means discovering otherwise from the middle of a job log.

A volume build needs four environment variables that OpenFOAM's own `bashrc` does not
set, and `mpirun` needs them forwarded with `-x` -- ranks that start without them die on
a missing `libhisa*.so`, which reads like a solver crash and is not one. An image build
needs none of that. Either way this prints what a shell needs rather than describing it,
and prints the path to the ONERA M6 case HiSA ships, which is by a wide margin the
fastest route to a correct dictionary set given that `foamToC` and `foamInfo` are absent
from this image.

This reports and changes nothing: it exports nothing, writes nothing, and installs
nothing, and the reading can be wrong -- a binary it cannot analyse is reported as
unchecked rather than as working or as broken.

    python3 hisa_env.py                     # present or absent, exports, example path
    python3 hisa_env.py --json              # the same facts as JSON
    python3 hisa_env.py --user-dir /work/OpenFOAM/user-v2512 --source /work/hisa/hisa

    eval "$(python3 hisa_env.py --exports)" # if you want them in this shell
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Where the 2026-08-30 build landed. Both are overridable, because a later build may
# well choose somewhere else and this script should not be the reason it is not found.
DEFAULT_USER_DIR = "/work/OpenFOAM/user-v2512"
DEFAULT_SOURCE = "/work/hisa/hisa"

# Where newer images bake it: the OpenFOAM site directories, which the sourced bashrc
# puts on PATH and LD_LIBRARY_PATH with no exports needed. A shell that has sourced the
# environment carries the real answer in $FOAM_SITE_APPBIN/$FOAM_SITE_LIBBIN; the
# literal paths are the v2512 image's, for reading this from a shell that has not.
DEFAULT_SITE_APPBIN = (
    "/usr/lib/openfoam/openfoam2512/site/2512/platforms/linux64GccDPInt32Opt/bin"
)
DEFAULT_SITE_LIBBIN = (
    "/usr/lib/openfoam/openfoam2512/site/2512/platforms/linux64GccDPInt32Opt/lib"
)
# The image keeps HiSA's worked ONERA M6 case here; the build tree itself is cleaned
# out of the image, so this is a dictionaries-only tree rather than a full source.
DEFAULT_IMAGE_SOURCE = "/opt/hisa"

# WM_OPTIONS for the ESI image. A build with a different precision or label size lands
# in a differently named directory, which is why the platform directory is looked up
# rather than assumed when the environment already names one.
DEFAULT_PLATFORM = "linux64GccDPInt32Opt"

EXAMPLE_CASE = "examples/oneraM6/simulation"


def platform_dir(user_dir: Path) -> str:
    """The `platforms/<WM_OPTIONS>` name to use.

    `$WM_OPTIONS` is set by the sourced OpenFOAM environment and is the authority when
    it is there. When it is not -- reading this from outside the container, say -- the
    single platform directory that exists under `user_dir` is a better answer than a
    hard-coded string, and the hard-coded string is the last resort.
    """
    from_env = os.environ.get("WM_OPTIONS")
    if from_env:
        return from_env
    platforms = user_dir / "platforms"
    if platforms.is_dir():
        found = sorted(p.name for p in platforms.iterdir() if p.is_dir())
        if len(found) == 1:
            return found[0]
    return DEFAULT_PLATFORM


def locations(user_dir: Path, source: Path) -> dict[str, Path]:
    platform = platform_dir(user_dir)
    appbin = user_dir / "platforms" / platform / "bin"
    libbin = user_dir / "platforms" / platform / "lib"
    return {
        "user_dir": user_dir,
        "appbin": appbin,
        "libbin": libbin,
        "binary": appbin / "hisa",
        "source": source,
        "example": source / EXAMPLE_CASE,
    }


def site_locations() -> dict[str, Path]:
    """Where an image build would be. The sourced environment is the authority when it
    is there; the literal v2512 paths are for a shell that has not sourced it."""
    appbin = Path(os.environ.get("FOAM_SITE_APPBIN") or DEFAULT_SITE_APPBIN)
    libbin = Path(os.environ.get("FOAM_SITE_LIBBIN") or DEFAULT_SITE_LIBBIN)
    source = Path(DEFAULT_IMAGE_SOURCE)
    return {
        "appbin": appbin,
        "libbin": libbin,
        "binary": appbin / "hisa",
        "source": source,
        "example": source / EXAMPLE_CASE,
    }


def export_lines(where: dict[str, Path]) -> list[str]:
    """The four lines every shell that runs HiSA needs, in dependency order."""
    return [
        f"export WM_PROJECT_USER_DIR={where['user_dir'].as_posix()}",
        f"export FOAM_USER_APPBIN={where['appbin'].as_posix()}",
        f"export FOAM_USER_LIBBIN={where['libbin'].as_posix()}",
        "export LD_LIBRARY_PATH=$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH",
        "export PATH=$FOAM_USER_APPBIN:$PATH",
    ]


def mpirun_prefix() -> str:
    """MPI ranks inherit nothing, so the same variables go across with `-x`."""
    return (
        "mpirun -x WM_PROJECT_USER_DIR -x FOAM_USER_APPBIN -x FOAM_USER_LIBBIN "
        "-x LD_LIBRARY_PATH -np <N> hisa -parallel"
    )


# ----------------------------------------------------------------- probing ---
#
# Nothing from here to `probe` is about HiSA. It is the general question every tool
# built onto the volume rather than into the image asks in the same shape -- is the
# binary here, and will it start? -- and the next such tool should import these
# rather than paste them.
#
# They live in this file rather than in a `openreynolds/toolbox/_probe.py` for two
# reasons. The toolbox is a directory of loose scripts that get synced to `/work`
# and copied around one at a time, so a private module is a file that is useless on
# its own and a script that breaks when it travels without it; every other file in
# here answers `--help` and runs. And this file is held to standard-library-only
# imports by its own test, which a sibling import would end. `from hisa_env import
# probe_binary` costs a future per-tool module one line and no new file.


def find_binary(name: str, *directories: Path, search_path: bool = False) -> Path | None:
    """Where a tool actually is, if it is anywhere. `None` rather than a raise.

    The named directories are tried in order, and `$PATH` last and only when asked
    for, because "is it at the prefix it was built into" and "is it on the `$PATH`
    of this shell" are different questions: a tool whose environment has been
    sourced answers yes to the second and a fresh shell answers no, and neither
    answer says whether the build is on the volume.
    """
    for directory in directories:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    if search_path:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def link_check(binary: Path, libbin: Path | None = None) -> tuple[str, list[str]]:
    """`ldd` over the binary, with an extra library directory on the search path.

    Three outcomes, kept apart because they mean different things. `ok` -- it resolved
    every shared object. `missing` -- there is no binary at all, or `ldd` named at
    least one shared object it could not find, which is the signature of a binary that
    is present and will still refuse to start, and the usual cause is
    `LD_LIBRARY_PATH` without `FOAM_USER_LIBBIN` on it. `unchecked` -- there is a
    binary, but there is no `ldd` here or it declined to analyse the file. Unchecked is
    not a failure: a build can be perfectly good and unverifiable from where this is
    running.

    An absent binary used to come back `unchecked`, which put "there is nothing here"
    and "there is something here and it cannot be analysed" in the same bucket -- the
    one distinction the three outcomes exist to make. It is `missing` now.
    """
    if not binary.is_file():
        return "missing", []

    env = dict(os.environ)
    extra = [str(libbin)] if libbin is not None else []
    search = extra + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    if search:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(search)

    try:
        proc = subprocess.run(
            ["ldd", str(binary)],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return "unchecked", []

    output = f"{proc.stdout}\n{proc.stderr}"
    missing = sorted(
        {
            line.split("=>")[0].strip()
            for line in output.splitlines()
            if "not found" in line and "=>" in line
        }
    )
    if missing:
        return "missing", missing
    if "=>" in output:
        return "ok", []
    return "unchecked", []


def probe_binary(
    name: str,
    *directories: Path,
    libbin: Path | None = None,
    search_path: bool = False,
) -> dict[str, Any]:
    """One tool's presence and linkage as plain data, for any tool on the volume.

    `path` is where it was found, or -- when it was found nowhere -- the first place
    that was looked, because "not at the place you would expect it" is more use to
    whoever reads this than an empty string.
    """
    found = find_binary(name, *directories, search_path=search_path)
    binary = found or (directories[0] / name if directories else Path(name))
    linkage, missing = link_check(binary, libbin)
    return {
        "name": name,
        "path": binary,
        "present": found is not None,
        "linkage": linkage,
        "missing_libraries": missing,
    }


# --------------------------------------------------------------- HiSA again ---


def probe(user_dir: Path, source: Path) -> dict[str, Any]:
    """Everything this script knows, as plain data. `main` only formats it.

    The image's site directories are looked at first, because an image build is on
    every instance and needs no environment; the volume locations second, because an
    older image's instance may still carry the 2026-08-30 build there. `location`
    says which of the two answered.
    """
    site = site_locations()
    where = locations(user_dir, source)

    in_image = False
    tool = probe_binary("hisa", site["appbin"], libbin=site["libbin"])
    if tool["present"]:
        in_image = True
    else:
        # `$PATH` is deliberately not searched: a shell that has already sourced the
        # exports below would find hisa there, and this is asked in order to write them.
        tool = probe_binary("hisa", where["appbin"], libbin=where["libbin"])
    binary = tool["path"]
    present = tool["present"]
    linkage, missing = tool["linkage"], tool["missing_libraries"]

    if not present:
        reason = (
            f"no hisa binary at {site['binary'].as_posix()} (image) or "
            f"{binary.as_posix()} (volume) -- this instance does not have it, "
            "or it was built somewhere else (--user-dir)"
        )
    elif linkage == "missing":
        reason = "the binary is there and its shared objects do not resolve: " + ", ".join(
            missing
        )
    elif in_image:
        reason = f"hisa is at {binary.as_posix()}, baked into the image"
    else:
        reason = f"hisa is at {binary.as_posix()}"

    # An image build is already on the sourced PATH and library path, so the export
    # lines are the volume build's need, not a general one.
    example = site["example"] if site["example"].is_dir() else where["example"]
    found_source = site["source"] if in_image else where["source"]
    return {
        "available": present and linkage != "missing",
        "location": "image" if in_image else "volume",
        "binary": binary.as_posix(),
        "binary_present": present,
        "linkage": linkage,
        "missing_libraries": missing,
        "reason": reason,
        "platform": platform_dir(user_dir),
        "user_dir": where["user_dir"].as_posix(),
        "appbin": (site["appbin"] if in_image else where["appbin"]).as_posix(),
        "libbin": (site["libbin"] if in_image else where["libbin"]).as_posix(),
        "source": found_source.as_posix(),
        "example": example.as_posix(),
        "example_present": example.is_dir(),
        "exports": [] if in_image else export_lines(where),
        "mpirun": "mpirun -np <N> hisa -parallel" if in_image else mpirun_prefix(),
    }


def report(found: dict[str, Any]) -> str:
    lines = []
    state = "available" if found["available"] else "not available"
    lines.append(f"# hisa: {state}")
    lines.append(f"  {found['reason']}")
    if found["binary_present"]:
        note = {
            "ok": "ldd resolved every shared object",
            "missing": "ldd could not resolve: " + ", ".join(found["missing_libraries"]),
            "unchecked": "not link-checked (no usable ldd here)",
        }[found["linkage"]]
        lines.append(f"  {note}")

    lines.append("")
    lines.append("# what a shell needs")
    if found["exports"]:
        lines.extend(f"  {line}" for line in found["exports"])
        lines.append("")
        lines.append("  in parallel, the same variables have to cross into the ranks:")
        lines.append(f"  {found['mpirun']}")
    else:
        lines.append(
            "  nothing to export: the image's site directories are on PATH and the "
            "library path once the OpenFOAM environment is sourced"
        )
        lines.append(f"  in parallel: {found['mpirun']}")

    lines.append("")
    lines.append("# a worked case to copy dictionaries from")
    if found["example_present"]:
        lines.append(f"  {found['example']}")
        lines.append(
            "  its system/ and constant/ are a correct HiSA dictionary set; "
            "the freestream is the part that changes"
        )
    else:
        lines.append(f"  {found['example']}: not there")
        lines.append(f"  the source tree it comes with lives at {found['source']}")

    if not found["available"]:
        lines.append("")
        lines.append("# if it is genuinely absent")
        lines.append(
            "  newer images ship it; an instance on an older image can carry a source "
            "build against v2512 from gitlab.com/hisa/hisa onto the volume, roughly "
            "half an hour of compile"
        )
        lines.append(
            "  rhoCentralFoam is in the image and captures shocks too, at an explicit "
            "timestep set by the smallest cell -- on a layered mesh that is the prism "
            "layer, not the shock cell"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--user-dir",
        type=Path,
        default=Path(os.environ.get("WM_PROJECT_USER_DIR") or DEFAULT_USER_DIR),
        help=f"OpenFOAM user directory the build landed in (default {DEFAULT_USER_DIR})",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(DEFAULT_SOURCE),
        help=f"HiSA source tree, which carries the examples (default {DEFAULT_SOURCE})",
    )
    parser.add_argument("--json", action="store_true", help="the same facts as JSON")
    parser.add_argument(
        "--exports",
        action="store_true",
        help="only the export lines, for `eval $(...)`",
    )
    args = parser.parse_args(argv)

    found = probe(args.user_dir, args.source)

    if args.exports:
        print("\n".join(found["exports"]))
    elif args.json:
        print(json.dumps(found, indent=2))
    else:
        print(report(found))

    # Absence is an answer, not an error: a study that wanted to know gets told, and
    # nothing here decides what to do about it.
    return 0


if __name__ == "__main__":
    sys.exit(main())
