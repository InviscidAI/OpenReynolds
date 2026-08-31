#!/usr/bin/env python3
"""checkMesh output -> a compact metric table.

Extracts the numbers and the patch table and prints them together. It attaches no
verdicts and applies no thresholds: which of these matter depends on the case, the
schemes, and where the quantity of interest lives, and that judgement is yours.

    python3 mesh_digest.py log.checkMesh
    checkMesh 2>&1 | python3 mesh_digest.py -
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
"""Deliberately does not swallow a trailing sentence period, which checkMesh emits
directly after several of its numbers."""

COUNTS = re.compile(
    r"^\s*(cells|faces|points|internal faces|boundary patches|"
    r"hexahedra|prisms|wedges|pyramids|tet wedges|tetrahedra|polyhedra):\s+(\d+)"
)
NON_ORTHO = re.compile(rf"non-orthogonality Max:\s*({NUM})\s*average:\s*({NUM})")
SKEWNESS = re.compile(rf"Max skewness = ({NUM})")
ASPECT = re.compile(rf"Max aspect ratio = ({NUM})")
OPENNESS = re.compile(rf"Max cell openness = ({NUM})")
FACE_AREA = re.compile(rf"Minimum face area = ({NUM})\.? Maximum face area = ({NUM})")
CELL_VOL = re.compile(rf"Min volume = ({NUM})\.? Max volume = ({NUM})")
DETERMINANT = re.compile(rf"Minimum face determinant = ({NUM})")
BOUNDING_BOX = re.compile(r"Overall domain bounding box \((.+?)\) \((.+?)\)")
PATCH_ROW = re.compile(r"^\s{4}(\w[\w.\-]*)\s+(\d+)\s+(\d+)\s+(.*)$")
FAILED = re.compile(r"\*\*\*(.+)$")
WARNING = re.compile(r"^\s*\*+\s*(.+)$")
"""checkMesh's advisories, which carry ONE star -- `*Number of severely non-orthogonal
(> 70 degrees) faces: 74.` -- not the two this pattern used to require. Between a regex
that could not match them and a call site that never ran it, a mesh with 74 faces at
86.3 degrees was summarised as "Mesh OK". Three-star failures are matched by `FAILED`
first and never reach here."""


def parse(text: str) -> dict:
    data: dict = {"counts": {}, "patches": [], "failures": [], "topology": []}
    in_patches = False

    for line in text.splitlines():
        match = COUNTS.match(line)
        if match:
            data["counts"][match.group(1)] = int(match.group(2))
            continue

        for key, pattern in (
            ("non_ortho", NON_ORTHO),
            ("skewness", SKEWNESS),
            ("aspect_ratio", ASPECT),
            ("openness", OPENNESS),
            ("face_area", FACE_AREA),
            ("cell_volume", CELL_VOL),
            ("face_determinant", DETERMINANT),
            ("bounding_box", BOUNDING_BOX),
        ):
            match = pattern.search(line)
            if match:
                groups = match.groups()
                data[key] = groups[0] if len(groups) == 1 else groups

        if line.strip().startswith("Patch") and "Faces" in line:
            in_patches = True
            continue
        if in_patches:
            match = PATCH_ROW.match(line)
            if match:
                data["patches"].append(match.groups())
                continue
            if line.strip() == "":
                in_patches = False

        match = FAILED.search(line)
        if match:
            data["failures"].append(match.group(1).strip())
            continue
        match = WARNING.match(line)
        if match:
            # checkMesh emits its severe-non-orthogonality and high-skewness advisories
            # with ONE star, and only its hard failures with three. This regex existed
            # and was never called, so a mesh with 74 severely non-orthogonal faces at
            # 86.3 degrees was reported as "Mesh OK" -- which it is, in checkMesh's own
            # words, and which is not the same as fit to solve on.
            data.setdefault("warnings", []).append(match.group(1).strip())
            continue
        if "Mesh OK" in line:
            data["mesh_ok_line"] = line.strip()

    return data


def report(data: dict) -> str:
    lines = ["# checkMesh"]

    if data["counts"]:
        lines.append("\n## counts")
        for key, value in data["counts"].items():
            lines.append(f"  {key:<18}{value:>14,}")

    metrics = [
        ("non-orthogonality max / avg", data.get("non_ortho")),
        ("max skewness", data.get("skewness")),
        ("max aspect ratio", data.get("aspect_ratio")),
        ("max cell openness", data.get("openness")),
        ("min / max face area", data.get("face_area")),
        ("min / max cell volume", data.get("cell_volume")),
        ("min face determinant", data.get("face_determinant")),
    ]
    present = [(label, value) for label, value in metrics if value]
    if present:
        lines.append("\n## metrics")
        for label, value in present:
            shown = " / ".join(value) if isinstance(value, tuple) else value
            lines.append(f"  {label:<30}{shown}")

    if data.get("bounding_box"):
        low, high = data["bounding_box"]
        lines.append(f"\ndomain bounding box: ({low}) to ({high})")

    if data["patches"]:
        lines.append("\n## patches")
        lines.append(f"  {'name':<24}{'faces':>10}{'points':>10}  type/notes")
        for name, faces, points, rest in data["patches"]:
            lines.append(f"  {name:<24}{int(faces):>10,}{int(points):>10,}  {rest.strip()}")

    if data["failures"]:
        lines.append("\n## lines checkMesh flagged with ***")
        for item in data["failures"]:
            lines.append(f"  {item}")

    if data.get("warnings"):
        lines.append("\n## lines checkMesh flagged with * (advisory)")
        for item in data["warnings"]:
            lines.append(f"  {item}")

    if data.get("mesh_ok_line"):
        # Printed alongside the advisories rather than instead of them. "Mesh OK" is
        # checkMesh's verdict on its own hard checks, and a mesh can carry it while
        # reporting 74 severely non-orthogonal faces -- which is what a reader takes
        # "Mesh OK" to have ruled out.
        lines.append(f"\n{data['mesh_ok_line']}")
        if data.get("warnings"):
            lines.append("  (checkMesh's own hard checks passed; the advisories above "
                         "are what it still wants looked at)")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="checkMesh log, or - for stdin")
    args = parser.parse_args()
    text = sys.stdin.read() if args.log == "-" else Path(args.log).read_text(errors="replace")
    print(report(parse(text)))


if __name__ == "__main__":
    main()
