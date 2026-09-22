#!/usr/bin/env python3
"""The patch lists in a snappyHexMeshDict, written from what is on disk.

A snappy case names every patch three times over -- once in `geometry`, once in
`castellatedMeshControls/features`, once in `refinementSurfaces` -- and a fourth
time in `surfaceFeatureExtractDict`. The four lists have to agree, by name, with
each other and with `constant/triSurface/`. At three patches that is easy and at
twenty it is not: a surface left out of `refinementSurfaces` is still meshed, its
faces land in whatever patch snappy defaults to, they take that patch's boundary
condition, and the run finishes. A wrong answer rather than an error.

So this enumerates. It reads `constant/triSurface/` and the `patches.json` manifest
beside it and fills regions marked in a dictionary:

    // PATCH-ENTRIES-BEGIN geometry
    // PATCH-ENTRIES-END

`--insert` rewrites what is between the markers and leaves the rest of the file
alone, so it is re-runnable: re-export the patches, run it again, the dictionary
follows. Running it twice with the same files on disk produces the same bytes.

It measures and it writes the lists. It decides nothing else: the refinement level
is whatever `--level` says, the patch type comes from the manifest's `role`, and a
surface with no manifest entry is reported as such rather than guessed about. What
belongs in the mesh, and whether the manifest matches the disk at all, are
questions for `cad_audit.py`; whether the `locationInMesh` is inside anything is
`domain_probe.py`'s.

    python3 patch_entries.py <case> --print
    python3 patch_entries.py <case> --insert system/snappyHexMeshDict
    python3 patch_entries.py <case> --insert system/snappyHexMeshDict \\
                                    --insert system/surfaceFeatureExtractDict
    python3 patch_entries.py <case> --insert system/snappyHexMeshDict --level "2 2"
    python3 patch_entries.py <case> --json

Exit code is 0 whatever is found, and 2 when the inputs are refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight  # noqa: E402  (sibling script, not a package)

Finding = preflight.Finding

SECTIONS = ("geometry", "features", "refinementSurfaces", "extract")
"""The marked regions this fills. The first three are snappyHexMeshDict's; `extract`
is surfaceFeatureExtractDict's per-surface blocks."""

BEGIN = "// PATCH-ENTRIES-BEGIN"
END = "// PATCH-ENTRIES-END"

ROLE_TYPES = {
    "inlet": "patch",
    "outlet": "patch",
    "wall": "wall",
    "symmetry": "symmetryPlane",
    "interface": "patch",
}
"""I3's `role` to the OpenFOAM patch type it becomes. A role this does not know, and
a surface with no manifest entry at all, becomes `wall` -- the type that carries a
no-slip condition, so a patch that was silently mistyped is over-constrained rather
than left open, and the `manifest` finding says which ones they were."""

DEFAULT_TYPE = "wall"


class Refused(Exception):
    """The inputs are not the ones this script reads. Exit 2, say what to pass."""


# ------------------------------------------------------------------ reading --


def surface_dir(case: Path) -> Path:
    """Where the per-patch STLs are, by preflight's list of the places they live."""
    for relative in preflight.SURFACE_DIRS:
        candidate = case / relative
        if candidate.is_dir():
            return candidate
    raise Refused(
        f"refused: no surface directory under {case.as_posix()} -- looked for "
        + ", ".join(preflight.SURFACE_DIRS)
    )


def read_manifest(directory: Path) -> dict[str, Any]:
    """I3's `patches.json`, or an empty manifest. Absent is not an error here: the
    enumeration is of what is on disk, and the manifest only adds the role."""
    path = directory / "patches.json"
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as error:
        raise Refused(f"refused: {path.as_posix()} is not readable JSON: {error}")
    if not isinstance(loaded, dict):
        raise Refused(f"refused: {path.as_posix()} is not a manifest object")
    return loaded


def patches(case: Path, directory: Path | None = None) -> list[dict[str, Any]]:
    """One record per surface file on disk, sorted by name.

    The file list is the authority: a manifest entry with no file contributes
    nothing here, because there is nothing to hand snappy. `cad_audit.py` is where
    that disagreement is a finding.
    """
    directory = directory or surface_dir(case)
    manifest = read_manifest(directory)
    by_file: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("patches", []) or []:
        if isinstance(entry, dict) and entry.get("file"):
            by_file[str(entry["file"])] = entry

    found: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in preflight.SURFACE_SUFFIXES:
            continue
        entry = by_file.get(path.name, {})
        role = str(entry.get("role", "") or "")
        triangles = preflight.read_triangles(path)
        found.append({
            "name": str(entry.get("name") or path.stem),
            "file": path.name,
            "role": role,
            "type": ROLE_TYPES.get(role.lower(), DEFAULT_TYPE),
            "in_manifest": bool(entry),
            "triangles": 0 if triangles is None else int(len(triangles)),
        })
    if not found:
        raise Refused(
            f"refused: no surface files in {directory.as_posix()} -- expected one "
            f"{' or '.join(preflight.SURFACE_SUFFIXES)} per patch, plus patches.json"
        )
    found.sort(key=lambda record: record["name"])
    return found


# ----------------------------------------------------------------- fragments --


def fragment(section: str, found: list[dict[str, Any]], level: str = "0 0",
             feature_level: int = 0, included_angle: int = 150) -> list[str]:
    """The lines for one section, unindented. `indent` puts them where they go."""
    if section not in SECTIONS:
        raise Refused(f"refused: unknown section '{section}'. Known: "
                      + ", ".join(SECTIONS))
    lines: list[str] = []
    for record in found:
        name, file = record["name"], record["file"]
        if section == "geometry":
            lines += [
                f"{file}",
                "{",
                "    type triSurfaceMesh;",
                f"    name {name};",
                "}",
            ]
        elif section == "features":
            stem = file.rsplit(".", 1)[0]
            lines.append(f'{{ file "{stem}.eMesh"; level {feature_level}; }}')
        elif section == "refinementSurfaces":
            lines += [
                f"{name}",
                "{",
                f"    level ({level});",
                f"    patchInfo {{ type {record['type']}; }}"
                + ("" if record["in_manifest"] else "   // no manifest entry"),
                "}",
            ]
        else:  # extract
            lines += [
                f"{file}",
                "{",
                "    extractionMethod    extractFromSurface;",
                f"    includedAngle       {included_angle};",
                "    geometricTestOnly   yes;",
                "    writeObj            no;",
                "}",
            ]
    return lines


def indent(lines: list[str], prefix: str) -> list[str]:
    return [f"{prefix}{line}" if line else "" for line in lines]


# ------------------------------------------------------------------ writing --


def sections_in(text: str) -> list[str]:
    """The marked sections a dictionary carries, in the order they appear."""
    found = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(BEGIN):
            found.append(stripped[len(BEGIN):].strip())
    return found


def rewrite(text: str, found: list[dict[str, Any]], level: str = "0 0",
            feature_level: int = 0, included_angle: int = 150) -> str:
    """Every marked region in `text`, refilled. Everything else is untouched.

    An unbalanced marker pair is refused rather than repaired: a BEGIN with no END
    means writing to the end of the file, which is how a dictionary gets truncated.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped.startswith(BEGIN):
            out.append(line)
            index += 1
            continue
        section = stripped[len(BEGIN):].strip()
        if section not in SECTIONS:
            raise Refused(
                f"refused: marker '{BEGIN} {section}' names no section this fills. "
                "Known: " + ", ".join(SECTIONS)
            )
        prefix = line[: len(line) - len(line.lstrip())]
        close = None
        for ahead in range(index + 1, len(lines)):
            if lines[ahead].strip().startswith(END):
                close = ahead
                break
        if close is None:
            raise Refused(
                f"refused: '{BEGIN} {section}' has no matching '{END}' -- the region "
                "to rewrite has no end, and this will not write to the end of the file"
            )
        out.append(line)
        body = indent(
            fragment(section, found, level, feature_level, included_angle), prefix
        )
        out.extend(f"{entry}\n" for entry in body)
        out.append(lines[close])
        index = close + 1
    return "".join(out)


def insert(path: Path, found: list[dict[str, Any]], level: str = "0 0",
           feature_level: int = 0, included_angle: int = 150) -> list[str]:
    """Rewrite one dictionary in place. Returns the sections it filled."""
    if not path.is_file():
        raise Refused(f"refused: {path.as_posix()} is not a file")
    text = path.read_text(encoding="utf-8", errors="replace")
    filled = sections_in(text)
    if not filled:
        raise Refused(
            f"refused: {path.as_posix()} carries no '{BEGIN} <section>' marker, so "
            "there is nothing to fill. The shipped dictionaries in "
            "templates/snappy/ carry them."
        )
    rewritten = rewrite(text, found, level, feature_level, included_angle)
    if rewritten != text:
        path.write_text(rewritten, encoding="utf-8")
    return filled


# ------------------------------------------------------------------ findings --


def findings(found: list[dict[str, Any]], filled: dict[str, list[str]]) -> list[Finding]:
    """What was enumerated, as evidence. None of these blocks anything."""
    out = [
        Finding(
            "patch-entries",
            "ok",
            f"{preflight.count_phrase(len(found), 'surface')} on disk: "
            + ", ".join(f"{r['name']} ({r['triangles']:,} tri)" for r in found),
            "each one appears once in every section filled, because the section is "
            "written from this list rather than typed alongside it",
        )
    ]
    unnamed = [record["name"] for record in found if not record["in_manifest"]]
    if unnamed:
        out.append(Finding(
            "manifest",
            "warn",
            f"no patches.json entry for {', '.join(unnamed)}",
            f"their patch type fell back to {DEFAULT_TYPE}, which is a guess about "
            "what the surface is for rather than a reading of it",
            "give each one a role in constant/triSurface/patches.json, or set the "
            "patchInfo type by hand after the insert; cad_audit.py checks the "
            "manifest against the disk properly",
        ))
    for path, sections in filled.items():
        out.append(Finding(
            "insert", "ok", f"{path}: filled {', '.join(sections)}",
            "the marked regions were rewritten; nothing outside them was touched",
        ))
    return out


def envelope(found: list[dict[str, Any]], filled: dict[str, list[str]]) -> dict[str, Any]:
    register = findings(found, filled)
    return {
        "script": "patch_entries",
        "ok": preflight.worst_status(register) != "fail",
        "findings": [finding.as_dict() for finding in register],
        "measured": {
            "patches": found,
            "sections": SECTIONS,
            "inserted": filled,
        },
    }


def report(found: list[dict[str, Any]], filled: dict[str, list[str]]) -> str:
    lines = [f"# patch entries ({len(found)} surfaces)"]
    for record in found:
        lines.append(
            f"  {record['name']:<24} {record['file']:<24} "
            f"type {record['type']:<14} {record['triangles']:>9,} triangles"
            + ("" if record["in_manifest"] else "   (no manifest entry)")
        )
    for path, sections in filled.items():
        lines.append(f"  filled {', '.join(sections)} in {path}")
    lines.append("")
    lines.append(
        "  The lists are what is on disk. Whether what is on disk is the domain you "
        "meant is cad_audit.py's question, and where the fluid is is domain_probe.py's."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------- CLI --


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path, help="the case holding constant/triSurface/")
    parser.add_argument(
        "--insert", action="append", default=[], metavar="DICT",
        help="a dictionary whose marked regions to rewrite in place; repeatable. "
             "Relative paths are read under the case",
    )
    parser.add_argument(
        "--print", dest="show", action="append", default=[], metavar="SECTION",
        help=f"print one section's fragment; repeatable. One of {', '.join(SECTIONS)}, "
             "or omit the value for all of them",
        nargs="?", const="",
    )
    parser.add_argument(
        "--level", default="0 0",
        help="the (min max) surface refinement level written into refinementSurfaces "
             "(default: '0 0', the background size). A re-run rewrites it, so pass it "
             "again rather than editing the dictionary and re-inserting",
    )
    parser.add_argument("--feature-level", type=int, default=0,
                        help="refinement level on the feature edges (default: 0)")
    parser.add_argument("--included-angle", type=int, default=150,
                        help="surfaceFeatureExtract's includedAngle (default: 150)")
    parser.add_argument("--json", action="store_true", help="the same facts as JSON")
    args = parser.parse_args(argv)

    try:
        found = patches(args.case)
        filled: dict[str, list[str]] = {}
        for name in args.insert:
            path = Path(name)
            if not path.is_absolute():
                path = args.case / path
            filled[path.as_posix()] = insert(
                path, found, args.level, args.feature_level, args.included_angle
            )
    except Refused as refusal:
        print(str(refusal), file=sys.stderr)
        return 2

    if args.show:
        wanted = [name for name in args.show if name] or list(SECTIONS)
        for section in wanted:
            if section not in SECTIONS:
                print(f"refused: unknown section '{section}'. Known: "
                      + ", ".join(SECTIONS), file=sys.stderr)
                return 2
        for section in wanted:
            if len(wanted) > 1:
                print(f"// PATCH-ENTRIES-BEGIN {section}")
            print("\n".join(fragment(
                section, found, args.level, args.feature_level, args.included_angle
            )))
            if len(wanted) > 1:
                print(END)
        return 0

    if args.json:
        print(json.dumps(envelope(found, filled), indent=2))
    else:
        print(report(found, filled))
    return 0


if __name__ == "__main__":
    sys.exit(main())
