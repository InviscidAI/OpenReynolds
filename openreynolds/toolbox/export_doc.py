#!/usr/bin/env python3
"""A study's write-up as LaTeX or Word, from the markdown it already has.

    python3 export_doc.py results.md --tex report.tex
    python3 export_doc.py results.md --docx report.docx
    python3 export_doc.py <case> --tex report.tex          # finds results.md under it

`results.py` writes `results.md` and `report.py` assembles a PDF. Neither is a file
somebody can edit: a reviewer who wants to add a paragraph, a supervisor who marks up a
draft and a journal that wants a `.tex` source all need the text, not a rendering of it.
This writes those two formats and nothing else.

**No pandoc and no `python-docx`.** Neither is on the workspace image and neither is
worth adding for this: LaTeX is text, and a `.docx` is a zip of XML that `zipfile` and
`xml.etree` write between them. So this runs on a laptop with nothing installed, which
is the same promise the rest of the toolbox makes.

What it handles, because it is what `results.md` contains: ATX headings, paragraphs,
bullet and numbered lists, fenced code, pipe tables, images, block quotes, horizontal
rules, and inline `code`, **bold**, *italic* and links. What it does not handle it
**says** rather than dropping: nested lists deeper than one level, HTML blocks, footnotes
and reference-style links are reported by line number and passed through as plain text,
so a document is never silently different from its source.

The `.docx` it writes is a minimal Word document: `[Content_Types].xml`, the package and
document relationships, and `word/document.xml`. Images are embedded as their own parts
with the relationship Word needs. It opens in Word, LibreOffice and Google Docs; what it
does not carry is a theme, a style gallery or a table of contents, because those are a
template's job and this is a converter.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

# -- what a markdown line is ---------------------------------------------------

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
NUMBER = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")
FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)\s*$")
RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
QUOTE = re.compile(r"^\s*>\s?(.*)$")
TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
CODE_SPAN = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*([^*]+)\*\*")
ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")

# Reported by line rather than dropped, because a document that is quietly different
# from its source is worse than one that says what it could not carry.
UNSUPPORTED = (
    (re.compile(r"^\s*<[a-zA-Z/!]"), "an HTML block"),
    (re.compile(r"\[\^[^\]]+\]"), "a footnote"),
    (re.compile(r"^\s*\[[^\]]+\]:\s"), "a reference-style link definition"),
)


class Refused(Exception):
    """Said rather than guessed at."""


def find_markdown(source: Path) -> Path:
    """The markdown file, or the one `results.py` wrote under a case directory."""
    if source.is_file():
        return source
    if not source.exists():
        raise Refused(f"refused: {source} does not exist")
    named = source / "results.md"
    if named.is_file():
        return named
    found = sorted(source.glob("*.md")) + sorted(source.glob("*/results.md"))
    if len(found) == 1:
        return found[0]
    if not found:
        raise Refused(
            f"refused: {source} is a directory and holds no markdown. `results.py` "
            f"writes results.md; name a file instead.")
    raise Refused(
        f"refused: {source} holds {len(found)} markdown files and none is results.md. "
        f"Name one: " + ", ".join(q.name for q in found[:6]))


def blocks(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Markdown as a list of blocks, and what could not be carried.

    One pass, line by line, because `results.md` is generated and regular. The notes
    are returned rather than printed so the caller decides where they go.
    """
    out: list[dict[str, Any]] = []
    notes: list[str] = []
    lines = text.replace("\r\n", "\n").split("\n")
    n = 0
    while n < len(lines):
        line = lines[n]
        fence = FENCE.match(line)
        if fence:
            closer, body = fence.group(2)[0], []
            n += 1
            while n < len(lines) and not (lines[n].strip().startswith(closer * 3)):
                body.append(lines[n])
                n += 1
            n += 1
            out.append({"kind": "code", "lines": body, "language": fence.group(3)})
            continue

        for pattern, what in UNSUPPORTED:
            if pattern.search(line):
                notes.append(f"line {n + 1}: {what}, carried as plain text")
                break

        if not line.strip():
            n += 1
            continue
        if RULE.match(line):
            out.append({"kind": "rule"})
            n += 1
            continue
        head = HEADING.match(line)
        if head:
            out.append({"kind": "heading", "level": len(head.group(1)),
                        "text": head.group(2).strip()})
            n += 1
            continue
        if line.strip().startswith("|") and n + 1 < len(lines) and TABLE_SEP.match(lines[n + 1]):
            rows = []
            while n < len(lines) and lines[n].strip().startswith("|"):
                if not TABLE_SEP.match(lines[n]):
                    cells = [c.strip() for c in lines[n].strip().strip("|").split("|")]
                    rows.append(cells)
                n += 1
            out.append({"kind": "table", "rows": rows})
            continue
        quote = QUOTE.match(line)
        if quote:
            body = []
            while n < len(lines) and QUOTE.match(lines[n]):
                body.append(QUOTE.match(lines[n]).group(1))
                n += 1
            out.append({"kind": "quote", "text": " ".join(body).strip()})
            continue
        item = BULLET.match(line) or NUMBER.match(line)
        if item:
            ordered = NUMBER.match(line) is not None
            items = []
            while n < len(lines):
                hit = BULLET.match(lines[n]) or NUMBER.match(lines[n])
                if not hit:
                    break
                # The unsupported check at the top of the outer loop never sees these:
                # they are consumed here. A nested item is flattened to one level, so it
                # is reported from the place that knows it was nested.
                if len(hit.group(1)) >= 2:
                    notes.append(f"line {n + 1}: a nested list item, flattened to one "
                                 f"level")
                items.append(hit.group(2).strip())
                n += 1
            out.append({"kind": "list", "ordered": ordered, "items": items})
            continue
        # A paragraph runs to the next blank line or block opener.
        body = []
        while n < len(lines) and lines[n].strip():
            if (HEADING.match(lines[n]) or FENCE.match(lines[n]) or RULE.match(lines[n])
                    or BULLET.match(lines[n]) or NUMBER.match(lines[n])
                    or QUOTE.match(lines[n]) or lines[n].strip().startswith("|")):
                if body:
                    break
            body.append(lines[n])
            n += 1
        joined = " ".join(x.strip() for x in body).strip()
        only_image = IMAGE.fullmatch(joined)
        if only_image:
            out.append({"kind": "image", "alt": only_image.group(1),
                        "path": only_image.group(2)})
        elif joined:
            out.append({"kind": "paragraph", "text": joined})
    return out, notes


# -- LaTeX ---------------------------------------------------------------------

TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
BS_CHAR = chr(92)
"""The backslash, named once: a generated LaTeX writer is mostly backslashes and doubling them in every
f-string is how one gets lost."""

TEX_SECTIONS = ("section", "subsection", "subsubsection", "paragraph", "subparagraph")


def tex_escape(text: str) -> str:
    return "".join(TEX_ESCAPES.get(ch, ch) for ch in text)


def tex_inline(text: str) -> str:
    """Inline markup to LaTeX, code spans first so their contents are not escaped twice."""
    spans: list[str] = []

    def hold(match: re.Match) -> str:
        spans.append(match.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = CODE_SPAN.sub(hold, text)
    text = IMAGE.sub(lambda m: m.group(1), text)
    # hyperref reads \href's first argument with verbatim-like catcodes, so
    # `_`, `&` and `~` are safe inside the URL. `%` is not -- it would comment out the
    # rest of the line -- and neither is `#` or a backslash. Those three are escaped
    # and the rest is left alone. Not verified by compiling: there is no LaTeX
    # toolchain on the machine this was written on, so it rests on hyperref's
    # documented behaviour, and `tests/test_toolbox_export_doc.py` pins the three.
    def href(match):
        url = match.group(2)
        for ch, with_ in ((BS_CHAR, BS_CHAR * 2), ("%", BS_CHAR + "%"), ("#", BS_CHAR + "#")):
            url = url.replace(ch, with_)
        return f"{BS_CHAR}href{{{url}}}{{{match.group(1)}}}"

    text = LINK.sub(href, text)
    # Escape, but keep the \href we just wrote and the placeholders.
    pieces = re.split(r"(\\href\{[^}]*\}\{[^}]*\})", text)
    text = "".join(p if p.startswith("\\href{") else tex_escape(p) for p in pieces)
    text = BOLD.sub(lambda m: f"\\textbf{{{m.group(1)}}}", text)
    text = ITALIC.sub(lambda m: f"\\emph{{{m.group(1)}}}", text)
    for index, held in enumerate(spans):
        text = text.replace(f"\x00{index}\x00", f"\\texttt{{{tex_escape(held)}}}")
    return text


def to_tex(parsed: list[dict[str, Any]], title: str) -> str:
    body: list[str] = []
    for block in parsed:
        kind = block["kind"]
        if kind == "heading":
            level = min(block["level"], len(TEX_SECTIONS)) - 1
            body.append(f"\\{TEX_SECTIONS[level]}{{{tex_inline(block['text'])}}}")
        elif kind == "paragraph":
            body.append(tex_inline(block["text"]))
        elif kind == "code":
            body.append("\\begin{verbatim}\n" + "\n".join(block["lines"]) + "\n\\end{verbatim}")
        elif kind == "list":
            env = "enumerate" if block["ordered"] else "itemize"
            items = "\n".join(f"  \\item {tex_inline(x)}" for x in block["items"])
            body.append(f"\\begin{{{env}}}\n{items}\n\\end{{{env}}}")
        elif kind == "quote":
            body.append("\\begin{quote}\n" + tex_inline(block["text"]) + "\n\\end{quote}")
        elif kind == "rule":
            body.append("\\hrulefill")
        elif kind == "image":
            body.append("\\begin{figure}[htbp]\n  \\centering\n"
                        f"  \\includegraphics[width=\\linewidth]{{{block['path']}}}\n"
                        f"  \\caption{{{tex_inline(block['alt'])}}}\n\\end{{figure}}")
        elif kind == "table":
            rows = block["rows"]
            width = max(len(r) for r in rows)
            spec = "l" * width
            lines = [f"\\begin{{tabular}}{{{spec}}}", "\\hline"]
            for index, row in enumerate(rows):
                padded = row + [""] * (width - len(row))
                lines.append(" & ".join(tex_inline(c) for c in padded) + r" \\")
                if index == 0:
                    lines.append("\\hline")
            lines += ["\\hline", "\\end{tabular}"]
            body.append("\n".join(lines))
    return (
        "\\documentclass[11pt,a4paper]{article}\n"
        "\\usepackage[utf8]{inputenc}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n"
        "\\usepackage[margin=25mm]{geometry}\n"
        f"\\title{{{tex_escape(title)}}}\n\\date{{}}\n"
        "\\begin{document}\n\\maketitle\n\n" + "\n\n".join(body) + "\n\n\\end{document}\n")


# -- Word ----------------------------------------------------------------------

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Default Extension="jpeg" ContentType="image/jpeg"/>
<Default Extension="jpg" ContentType="image/jpeg"/>
<Default Extension="gif" ContentType="image/gif"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
WP = ('xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')


def runs(text: str) -> str:
    """Inline markup as Word runs. Bold, italic and code become run properties."""
    out: list[str] = []
    token = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|(?<!\*)\*[^*]+\*(?!\*))")
    text = IMAGE.sub(lambda m: m.group(1), text)
    text = LINK.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)
    for piece in token.split(text):
        if not piece:
            continue
        props, body = "", piece
        if piece.startswith("`") and piece.endswith("`") and len(piece) > 1:
            props = '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
            body = piece[1:-1]
        elif piece.startswith("**") and piece.endswith("**"):
            props, body = "<w:b/>", piece[2:-2]
        elif piece.startswith("*") and piece.endswith("*"):
            props, body = "<w:i/>", piece[1:-1]
        rpr = f"<w:rPr>{props}</w:rPr>" if props else ""
        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{xml_escape(body)}</w:t></w:r>')
    return "".join(out) or '<w:r><w:t xml:space="preserve"></w:t></w:r>'


def to_docx(parsed: list[dict[str, Any]], title: str, target: Path, base: Path) -> dict[str, Any]:
    """Write the package. Images that exist are embedded; ones that do not are named."""
    body: list[str] = [
        f'<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr>{runs(title)}</w:p>']
    images: list[tuple[str, Path]] = []
    missing: list[str] = []

    for block in parsed:
        kind = block["kind"]
        if kind == "heading":
            level = min(block["level"], 6)
            body.append(f'<w:p><w:pPr><w:pStyle w:val="Heading{level}"/></w:pPr>'
                        f'{runs(block["text"])}</w:p>')
        elif kind == "paragraph":
            body.append(f"<w:p>{runs(block['text'])}</w:p>")
        elif kind == "code":
            for line in block["lines"]:
                body.append(
                    '<w:p><w:pPr><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
                    "</w:rPr></w:pPr>"
                    '<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/></w:rPr>'
                    f'<w:t xml:space="preserve">{xml_escape(line)}</w:t></w:r></w:p>')
        elif kind == "list":
            style = "ListParagraph"
            for index, item in enumerate(block["items"], start=1):
                marker = f"{index}. " if block["ordered"] else "\u2022 "
                body.append(f'<w:p><w:pPr><w:pStyle w:val="{style}"/>'
                            '<w:ind w:left="720"/></w:pPr>'
                            f'<w:r><w:t xml:space="preserve">{marker}</w:t></w:r>'
                            f"{runs(item)}</w:p>")
        elif kind == "quote":
            body.append('<w:p><w:pPr><w:ind w:left="720"/><w:rPr><w:i/></w:rPr></w:pPr>'
                        f"{runs(block['text'])}</w:p>")
        elif kind == "rule":
            body.append('<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="6" '
                        'w:space="1" w:color="auto"/></w:pBdr></w:pPr></w:p>')
        elif kind == "image":
            path = (base / block["path"]).resolve()
            if path.is_file():
                rid = f"rIdImg{len(images) + 1}"
                images.append((rid, path))
                body.append(drawing(rid, block["alt"]))
            else:
                missing.append(block["path"])
                body.append(f"<w:p>{runs('[image not found: ' + block['path'] + ']')}</w:p>")
            if block["alt"]:
                body.append('<w:p><w:pPr><w:rPr><w:i/></w:rPr></w:pPr>'
                            f"{runs(block['alt'])}</w:p>")
        elif kind == "table":
            rows = block["rows"]
            width = max(len(r) for r in rows)
            cells = []
            for index, row in enumerate(rows):
                padded = row + [""] * (width - len(row))
                tcs = []
                for cell in padded:
                    inner = runs(f"**{cell}**" if index == 0 and cell else cell)
                    tcs.append("<w:tc><w:tcPr><w:tcW w:w='0' w:type='auto'/></w:tcPr>"
                               f"<w:p>{inner}</w:p></w:tc>")
                cells.append("<w:tr>" + "".join(tcs) + "</w:tr>")
            body.append(
                "<w:tbl><w:tblPr><w:tblStyle w:val='TableGrid'/>"
                "<w:tblW w:w='0' w:type='auto'/><w:tblBorders>"
                + "".join(f"<w:{side} w:val='single' w:sz='4' w:space='0' w:color='auto'/>"
                          for side in ("top", "left", "bottom", "right",
                                       "insideH", "insideV"))
                + "</w:tblBorders></w:tblPr>" + "".join(cells) + "</w:tbl>")

    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                f"<w:document {W} {WP}><w:body>" + "".join(body)
                + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
                  '<w:pgMar w:top="1418" w:right="1418" w:bottom="1418" w:left="1418"/>'
                  "</w:sectPr></w:body></w:document>")

    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for rid, path in images:
        rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
                    f'officeDocument/2006/relationships/image" '
                    f'Target="media/{path.name}"/>')
    rels.append("</Relationships>")

    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", ROOT_RELS)
        zf.writestr("word/document.xml", document)
        zf.writestr("word/_rels/document.xml.rels", "\n".join(rels))
        for _, path in images:
            zf.write(path, f"word/media/{path.name}")
    return {"images": len(images), "missing": missing}


def drawing(rid: str, alt: str) -> str:
    """One inline image, sized to the text width at 96 dpi in EMU."""
    cx, cy = 5486400, 3657600          # 6 in by 4 in, which fits the margins above
    return (
        "<w:p><w:r><w:drawing>"
        f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:docPr id="1" name="Picture" descr="{xml_escape(alt)}"/>'
        "<a:graphic><a:graphicData uri=\"http://schemas.openxmlformats.org/"
        "drawingml/2006/picture\"><pic:pic>"
        '<pic:nvPicPr><pic:cNvPr id="0" name="Picture"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch>'
        "</pic:blipFill>"
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        "</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A study's write-up as LaTeX or Word, from the markdown it has.",
        epilog="Neither pandoc nor python-docx is needed or used: LaTeX is text and a "
               ".docx is a zip of XML, so this runs where nothing is installed.")
    parser.add_argument("source", type=Path,
                        help="a markdown file, or a case directory holding results.md")
    parser.add_argument("--tex", type=Path, default=None, help="write LaTeX here")
    parser.add_argument("--docx", type=Path, default=None, help="write a .docx here")
    parser.add_argument("--title", default="",
                        help="the document title (default: the first heading, else the "
                             "file's name)")
    args = parser.parse_args(argv)

    if args.tex is None and args.docx is None:
        parser.error("name an output: --tex, --docx, or both")

    try:
        path = find_markdown(args.source)
    except Refused as exc:
        print(exc)
        return 2

    text = path.read_text(encoding="utf-8")
    parsed, notes = blocks(text)
    if not parsed:
        print(f"refused: {path} holds no markdown blocks this can carry")
        return 2

    title = args.title
    if not title:
        head = next((b["text"] for b in parsed if b["kind"] == "heading"), "")
        title = head or path.stem

    print(f"read {path}: {len(parsed)} block(s)")
    counts: dict[str, int] = {}
    for block in parsed:
        counts[block["kind"]] = counts.get(block["kind"], 0) + 1
    print("  " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))

    if args.tex:
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        args.tex.write_text(to_tex(parsed, title), encoding="utf-8")
        print(f"wrote {args.tex} ({args.tex.stat().st_size} bytes)")
        print("  pdflatex it, or hand the .tex to a journal; graphicx and hyperref are "
              "the only packages it needs")
    if args.docx:
        found = to_docx(parsed, title, args.docx, path.parent)
        print(f"wrote {args.docx} ({args.docx.stat().st_size} bytes, "
              f"{found['images']} image(s) embedded)")
        for name in found["missing"]:
            print(f"  image not found and named in the text instead: {name}")

    for note in notes:
        print(f"  not carried as markup: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
