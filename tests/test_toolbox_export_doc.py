"""`export_doc.py`: a study's markdown as LaTeX and as Word, with neither pandoc nor
python-docx.

"Word and LaTeX export" was a roadmap item the triage recorded as "absent; no pandoc, no
docx writer" and nothing was ever built for it. Neither dependency is on the workspace
image and neither is worth adding: LaTeX is text and a `.docx` is a zip of XML.

What the tests hold, in order of how much they would cost to get wrong:

  * the `.docx` is a valid OOXML package -- the four required parts, every XML part
    parsing, exactly one `sectPr`. A file Word refuses to open is worse than no file;
  * the `.tex` has no unescaped special outside `verbatim` and `tabular`, because one
    bare `%` comments out the rest of the line and one bare `_` is "Missing $ inserted".
    The first version of the writer had four, inside `\\href` URLs;
  * every block kind survives the round trip, so a table does not quietly become a
    paragraph;
  * what cannot be carried is reported by line, not dropped.

Neither output is compiled or opened here. There is no LaTeX toolchain on the machine
this was written on and no Word, so the checks are structural and the limit is stated
rather than papered over: a document that parses is not the same as a document that
renders well.
"""
from __future__ import annotations

import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

TOOLBOX = Path(__file__).resolve().parents[1] / "openreynolds" / "toolbox"
SCRIPT = TOOLBOX / "export_doc.py"
BS = chr(92)
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

SAMPLE = """# A study of something

The headline is **4.509** against a reference, measured on `simpleFoam` at Re 1600.
A link to [the benchmark](../thermal_stress/README.md) and one with a
[percent sign](http://example.com/a%20b#frag) in it.

## What was measured

| quantity | ours | reference |
|---|---|---|
| Nusselt | 3.677 | 3.6568 |
| drag, 50% span | 1.2 | 1.3 |

- the first thing, with an underscore_in_it
- the second thing

1. ordered one
2. ordered two

> A quotation with a $ and a # in it.

```
verbatim: 100% & _raw_ $text$ #hash
```

---

![A figure that does not exist](missing.png)
"""


def run(*args: str) -> subprocess.CompletedProcess[str]:
    """The script as a user runs it: a plain interpreter, no package import."""
    return subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=300)


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    md = tmp_path / "results.md"
    md.write_text(SAMPLE, encoding="utf-8")
    return md


# -- the package has to be one Word will open ---------------------------------

def test_the_docx_is_a_valid_ooxml_package(sample, tmp_path):
    out = tmp_path / "out.docx"
    done = run(str(sample), "--docx", str(out))
    assert done.returncode == 0, done.stdout + done.stderr
    assert out.is_file()

    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None, "the zip is corrupt"
        names = set(zf.namelist())
        for required in ("[Content_Types].xml", "_rels/.rels", "word/document.xml",
                         "word/_rels/document.xml.rels"):
            assert required in names, f"{required} is missing; Word will refuse the file"
        for part in names:
            if part.endswith((".xml", ".rels")):
                ET.fromstring(zf.read(part))        # raises if malformed
        document = zf.read("word/document.xml").decode("utf-8")

    assert document.count("<w:sectPr>") == 1, "a body takes exactly one sectPr"
    root = ET.fromstring(document)
    assert sum(1 for _ in root.iter(W + "p")) > 5
    assert sum(1 for _ in root.iter(W + "tbl")) == 1


def test_the_docx_carries_the_table_as_a_table(sample, tmp_path):
    """A table that arrives as a paragraph has lost the thing it was for."""
    out = tmp_path / "out.docx"
    run(str(sample), "--docx", str(out))
    with zipfile.ZipFile(out) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    rows = list(root.iter(W + "tr"))
    assert len(rows) == 3, "a header and two body rows"
    cells = list(rows[0].iter(W + "tc"))
    assert len(cells) == 3
    text = "".join(t.text or "" for t in rows[1].iter(W + "t"))
    assert "3.677" in text and "3.6568" in text


def test_the_docx_escapes_xml_rather_than_breaking_it(sample, tmp_path):
    """`&` and `<` in the source must arrive as entities, not as markup."""
    source = tmp_path / "sharp.md"
    source.write_text("# T\n\nAn & ampersand, a < bracket and a > bracket.\n",
                      encoding="utf-8")
    out = tmp_path / "sharp.docx"
    assert run(str(source), "--docx", str(out)).returncode == 0
    with zipfile.ZipFile(out) as zf:
        document = zf.read("word/document.xml").decode("utf-8")
    ET.fromstring(document)
    assert "&amp;" in document
    assert "An & amp" not in document


# -- the LaTeX has to be something that compiles -------------------------------

def bare(text: str, char: str) -> list[str]:
    return re.findall(r"(?<!" + BS + BS + r")" + re.escape(char), text)


def outside_verbatim(tex: str) -> str:
    body = re.sub(BS + BS + r"begin\{verbatim\}.*?" + BS + BS + r"end\{verbatim\}",
                  "", tex, flags=re.S)
    # `\href`'s first argument is read with verbatim-like catcodes by hyperref, so `_`,
    # `&` and `~` are safe there; `%` and `#` are checked separately below.
    return re.sub(BS + BS + r"href\{[^}]*\}", "", body)


def test_the_tex_has_no_unescaped_special(sample, tmp_path):
    """One bare `%` comments out the rest of the line and one bare `_` is "Missing $
    inserted". The first version of this writer emitted four, all inside href URLs."""
    out = tmp_path / "out.tex"
    assert run(str(sample), "--tex", str(out)).returncode == 0
    tex = out.read_text(encoding="utf-8")
    body = outside_verbatim(tex)
    no_tabular = re.sub(BS + BS + r"begin\{tabular\}.*?" + BS + BS + r"end\{tabular\}",
                        "", body, flags=re.S)
    assert not bare(body, "_"), "a bare underscore is Missing $ inserted"
    assert not bare(body, "%"), "a bare percent comments out the rest of the line"
    assert not bare(body, "#")
    assert not bare(body, "$")
    assert not bare(no_tabular, "&"), "& is only legal inside tabular"


def test_href_urls_escape_what_hyperref_cannot_take(sample, tmp_path):
    """`%` and `#` are not safe even in href's first argument."""
    out = tmp_path / "out.tex"
    run(str(sample), "--tex", str(out))
    tex = out.read_text(encoding="utf-8")
    urls = re.findall(BS + BS + r"href\{([^}]*)\}", tex)
    assert urls, "the sample has links; none reached the output"
    for url in urls:
        assert not bare(url, "%"), f"unescaped % in {url}"
        assert not bare(url, "#"), f"unescaped # in {url}"


def test_the_tex_environments_balance(sample, tmp_path):
    out = tmp_path / "out.tex"
    run(str(sample), "--tex", str(out))
    tex = out.read_text(encoding="utf-8")
    opens = sorted(re.findall(BS + BS + r"begin\{(" + r"\w+)\}", tex))
    closes = sorted(re.findall(BS + BS + r"end\{(" + r"\w+)\}", tex))
    assert opens == closes, f"unbalanced: {opens} against {closes}"
    assert tex.startswith(BS + "documentclass")
    assert tex.rstrip().endswith(BS + "end{document}")


def test_the_tex_keeps_the_verbatim_block_verbatim(sample, tmp_path):
    """A fenced block is machine output. Escaping inside it would change it."""
    out = tmp_path / "out.tex"
    run(str(sample), "--tex", str(out))
    tex = out.read_text(encoding="utf-8")
    inside = re.search(BS + BS + r"begin\{verbatim\}(.*?)" + BS + BS + r"end\{verbatim\}",
                       tex, re.S).group(1)
    assert "100% & _raw_ $text$ #hash" in inside, "the fenced block was escaped"


# -- the blocks, and the refusals ---------------------------------------------

def test_every_block_kind_survives(sample, tmp_path):
    done = run(str(sample), "--tex", str(tmp_path / "o.tex"))
    for kind in ("heading", "paragraph", "table", "list", "code", "image"):
        assert kind in done.stdout, f"{kind} was not counted: {done.stdout}"


def test_a_case_directory_finds_its_results_md(sample, tmp_path):
    """`results.py` writes results.md, so a case directory is enough."""
    done = run(str(tmp_path), "--tex", str(tmp_path / "o.tex"))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "results.md" in done.stdout


def test_a_missing_image_is_named_rather_than_dropped(sample, tmp_path):
    out = tmp_path / "out.docx"
    done = run(str(sample), "--docx", str(out))
    assert "missing.png" in done.stdout
    with zipfile.ZipFile(out) as zf:
        document = zf.read("word/document.xml").decode("utf-8")
    assert "image not found" in document, "the reader must see the gap too"


def test_an_image_that_exists_is_embedded(sample, tmp_path):
    """A one-pixel PNG, so the part and its relationship are both exercised."""
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6300010000050001"
        "0d0a2db40000000049454e44ae426082")
    (tmp_path / "there.png").write_bytes(png)
    source = tmp_path / "withimage.md"
    source.write_text("# T\n\n![A real one](there.png)\n", encoding="utf-8")
    out = tmp_path / "withimage.docx"
    done = run(str(source), "--docx", str(out))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1 image(s) embedded" in done.stdout
    with zipfile.ZipFile(out) as zf:
        assert "word/media/there.png" in zf.namelist()
        rels = zf.read("word/_rels/document.xml.rels").decode("utf-8")
        assert "media/there.png" in rels, "the part is there but nothing points at it"
        ET.fromstring(zf.read("word/document.xml"))


def test_no_output_asked_for_is_refused(sample):
    done = run(str(sample))
    assert done.returncode != 0
    assert "--tex" in done.stderr or "--tex" in done.stdout


def test_a_directory_with_no_markdown_is_refused_by_name(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    done = run(str(empty), "--tex", str(tmp_path / "o.tex"))
    assert done.returncode == 2
    assert "refused" in done.stdout and "results.md" in done.stdout


def test_a_path_that_does_not_exist_is_refused_by_name(tmp_path):
    done = run(str(tmp_path / "nope.md"), "--tex", str(tmp_path / "o.tex"))
    assert done.returncode == 2
    assert "refused" in done.stdout and "does not exist" in done.stdout


def test_what_cannot_be_carried_is_reported_by_line(tmp_path):
    source = tmp_path / "hard.md"
    source.write_text("# T\n\n- top\n  - nested one\n\n<div>html</div>\n",
                      encoding="utf-8")
    done = run(str(source), "--tex", str(tmp_path / "o.tex"))
    assert done.returncode == 0
    assert "not carried as markup" in done.stdout
    assert "nested list item" in done.stdout, done.stdout
    assert "HTML block" in done.stdout
    assert re.search(r"line \d+:", done.stdout), "a note without a line is not actionable"


def test_the_title_is_the_first_heading_unless_given(sample, tmp_path):
    out = tmp_path / "out.tex"
    run(str(sample), "--tex", str(out))
    assert "A study of something" in out.read_text(encoding="utf-8")
    other = tmp_path / "other.tex"
    run(str(sample), "--tex", str(other), "--title", "Something Else")
    assert "Something Else" in other.read_text(encoding="utf-8")


def test_both_formats_in_one_call(sample, tmp_path):
    tex, docx = tmp_path / "b.tex", tmp_path / "b.docx"
    done = run(str(sample), "--tex", str(tex), "--docx", str(docx))
    assert done.returncode == 0
    assert tex.is_file() and docx.is_file()
    with zipfile.ZipFile(docx) as zf:
        ET.fromstring(zf.read("word/document.xml"))
