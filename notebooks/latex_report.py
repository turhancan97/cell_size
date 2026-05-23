"""Shared LaTeX/PDF build helpers for biology and combined reports."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
PREAMBLE_PATH = NOTEBOOKS_DIR / "latex" / "preamble.tex"


def _find_tool(name: str) -> str | None:
    """Resolve pandoc/xelatex/tectonic on PATH or next to the active Python."""
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).resolve().parent / name
    return str(candidate) if candidate.is_file() else None


def _latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def _render_inline_latex(text: str) -> str:
    out = re.sub(
        r"\*\*([^\*]+)\*\*",
        lambda m: "\\textbf{" + _latex_escape(m.group(1)) + "}",
        text,
    )
    out = re.sub(
        r"`([^`]+)`",
        lambda m: "\\texttt{" + _latex_escape(m.group(1)) + "}",
        out,
    )
    out = re.sub(r"\\\((.+?)\\\)", r"$\1$", out)
    if "\\textbf{" in out or "\\texttt{" in out:
        return out
    return _latex_escape(out)


def _render_table_latex(lines: list[str]) -> str:
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    rows = []
    for raw in lines[2:]:
        rows.append([c.strip() for c in raw.strip().strip("|").split("|")])
    col_spec = "l" * len(header)
    parts = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(_render_inline_latex(c) for c in header) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        parts.append(" & ".join(_render_inline_latex(c) for c in row) + r" \\")
    parts.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(parts)


def _strip_yaml_front_matter(md_text: str) -> tuple[str, str | None]:
    if md_text.startswith("---\n"):
        end = md_text.find("\n---\n", 4)
        if end != -1:
            yaml_block = md_text[4:end]
            body = md_text[end + 5 :]
            m = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', yaml_block, re.M)
            return body, m.group(1) if m else None
    return md_text, None


def markdown_to_latex(md_text: str) -> str:
    """Minimal Markdown → LaTeX body (fallback when pandoc is unavailable)."""
    lines = md_text.splitlines()
    out: list[str] = []
    i, n = 0, len(lines)
    paragraph: list[str] = []

    def flush_para() -> None:
        if paragraph:
            out.append(_render_inline_latex(" ".join(paragraph)) + "\n")
            paragraph.clear()

    while i < n:
        line = lines[i]
        if line.startswith("# "):
            flush_para()
            out.append(f"\\section{{{_render_inline_latex(line[2:].strip())}}}\n")
            i += 1
            continue
        if line.startswith("## "):
            flush_para()
            out.append(f"\\subsection{{{_render_inline_latex(line[3:].strip())}}}\n")
            i += 1
            continue
        if line.startswith("### "):
            flush_para()
            out.append(f"\\subsubsection{{{_render_inline_latex(line[4:].strip())}}}\n")
            i += 1
            continue
        img = re.match(r"^!\[([^\]]*)\]\(([^\)]+)\)\s*$", line)
        if img:
            flush_para()
            path = img.group(2).lstrip("./")
            cap = _latex_escape(img.group(1) or path)
            out.append(
                "\\begin{figure}[H]\n\\centering\n"
                f"\\includegraphics[width=0.92\\linewidth]{{{path}}}\n"
                f"\\caption{{{cap}}}\n\\end{{figure}}\n"
            )
            i += 1
            continue
        if line.strip().startswith("|") and i + 1 < n and re.match(r"^\s*\|?\s*-", lines[i + 1]):
            flush_para()
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            out.append(_render_table_latex(tbl) + "\n")
            continue
        if re.match(r"^\s*[-*]\s+", line):
            flush_para()
            items: list[str] = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            out.append("\\begin{itemize}\n")
            for it in items:
                out.append(f"  \\item {_render_inline_latex(it)}\n")
            out.append("\\end{itemize}\n")
            continue
        if line.strip() == "":
            flush_para()
            i += 1
            continue
        paragraph.append(line)
        i += 1
    flush_para()
    return "".join(out)


def unicode_to_latex(text: str) -> str:
    """Convert Unicode symbols that PDF engines may not render into LaTeX commands."""
    text = text.replace(r"\textasciitilde±", r"$\approx\pm$")
    text = text.replace(r"\textasciitilde{}", r"$\approx$")

    reps = [
        ("µm²", r"$\mu$m$^2$"),
        ("µm", r"$\mu$m"),
        ("R²", r"R$^2$"),
        ("≤", r"$\leq$"),
        ("≥", r"$\geq$"),
        ("±", r"$\pm$"),
        ("÷", r"$\div$"),
        ("×", r"$\times$"),
        ("→", r"$\rightarrow$"),
        ("≈", r"$\approx$"),
        ("–", r"--"),
        ("—", r"---"),
    ]
    for src, dst in reps:
        text = text.replace(src, dst)

    text = text.replace(r"\$\approx\pm\$", r"\(\approx\pm\)")
    text = text.replace(r"\$\leq\$", r"\(\leq\)")
    text = text.replace(r"\$\geq\$", r"\(\geq\)")
    text = text.replace(r"\$\pm\$", r"\(\pm\)")
    text = text.replace(r"\$\div\$", r"\(\div\)")
    text = text.replace(r"\$\times\$", r"\(\times\)")
    text = text.replace(r"\$\rightarrow\$", r"\(\rightarrow\)")
    text = text.replace(r"\$\approx\$", r"\(\approx\)")

    return text


def title_page_heading(title: str) -> str:
    """Render title on two lines when separated by an em dash."""
    for sep in (" --- ", " — ", " – ", " - "):
        if sep in title:
            line1, line2 = title.split(sep, 1)
            return (
                f"{{\\Huge\\bfseries\\color{{ReportAccent}} {line1.strip()}\\par}}\n"
                "\\vspace{0.35em}\n"
                f"{{\\Huge\\bfseries\\color{{ReportAccent}} {line2.strip()}\\par}}"
            )
    return f"{{\\Huge\\bfseries\\color{{ReportAccent}} {title}\\par}}"


def polish_tex(
    tex_path: Path,
    *,
    subtitle: str = "Morphometric analysis report",
    summary_section: bool = False,
) -> None:
    """Apply scientific layout polish after pandoc conversion."""
    text = tex_path.read_text(encoding="utf-8")

    text = text.replace(
        r"\setcounter{secnumdepth}{-\maxdimen} % remove section numbering",
        r"% section numbering enabled in preamble",
    )

    title_m = re.search(r"\\title\{([^}]+)\}", text)
    date_m = re.search(r"\\date\{([^}]*)\}", text)
    title = title_m.group(1) if title_m else "Report"
    report_date = date_m.group(1).strip() if date_m and date_m.group(1).strip() else r"\today"

    title_page = (
        "\\begin{titlepage}\n"
        "\\thispagestyle{empty}\n"
        "\\centering\n"
        "\\vspace*{2.2cm}\n"
        "{\\color{ReportAccent}\\rule{\\textwidth}{2pt}\\par}\n"
        "\\vspace{1.4cm}\n"
        f"{title_page_heading(title)}\n"
        "\\vspace{1.2cm}\n"
        f"{{\\large\\color{{ReportMuted}} {subtitle}\\par}}\n"
        "\\vfill\n"
        f"{{\\large {report_date}\\par}}\n"
        "\\vspace{2cm}\n"
        "{\\color{ReportAccent}\\rule{\\textwidth}{2pt}\\par}\n"
        "\\end{titlepage}\n"
        "\\setcounter{page}{1}\n"
    )
    text = text.replace("\\maketitle\n", title_page, 1)
    if "\\maketitle" in text:
        text = text.replace("\\maketitle", title_page, 1)

    def _wrap_box(match: re.Match[str]) -> str:
        return match.group(1) + "\\begin{abstractbox}\n" + match.group(2).strip() + "\n\\end{abstractbox}\n"

    for section_name in ("Abstract", "Summary"):
        if f"\\section{{{section_name}}}" in text:
            text = re.sub(
                rf"(\\section\{{{section_name}\}}.*?\n)(.*?)(?=\n\\section\{{)",
                _wrap_box,
                text,
                count=1,
                flags=re.DOTALL,
            )
            break

    text = text.replace("\\begin{figure}", "\\begin{figure}[H]")
    text = re.sub(
        r"\\pandocbounded\{\\includegraphics(\[[^\]]*\])?\{([^}]+)\}\}",
        r"\\includegraphics[width=0.98\\linewidth]{\2}",
        text,
    )

    if "\\section{Discussion}" in text and "\\begin{document}" in text:
        prefix, rest = text.split("\\begin{document}", 1)
        body, suffix = rest.split("\\section{Discussion}", 1)

        def _wrap_keystats_itemize(match: re.Match[str]) -> str:
            inner = match.group(1)
            if "\\begin{itemize}" in inner:
                return match.group(0)
            return (
                "\\begin{keystats}\n\\begin{itemize}"
                + inner
                + "\\end{itemize}\n\\end{keystats}"
            )

        body = re.sub(
            r"\\begin\{itemize\}(.*?)\\end\{itemize\}",
            _wrap_keystats_itemize,
            body,
            flags=re.DOTALL,
        )
        text = prefix + "\\begin{document}" + body + "\\section{Discussion}" + suffix

    text = re.sub(
        r"(\\caption\{Classification yield\}.*?\\end\{figure\}\n\n)(.*?)(\n\nMorphometry reflects)",
        r"\1\\begin{keystats}\n\2\n\\end{keystats}\3",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"(\\caption\{Per-frog nucleus area\}.*?\\end\{figure\}\n\n)(Per-frog mean cell area spans.*?)(\n\n\\section\{)",
        r"\1\\begin{keystats}\n\2\n\\end{keystats}\3",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"(\{\\def\\LTcaptype\{none\}.*?)\\begin\{longtable\}",
        r"\1\\begin{scitable}\n\\begin{longtable}",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\end\{longtable\}\n(\})",
        r"\\end{longtable}\n\\end{scitable}\n\1",
        text,
    )

    _six_col_old = (
        "\\begin{longtable}[]{@{}\n"
        "  >{\\raggedright\\arraybackslash}p{(\\linewidth - 10\\tabcolsep) * \\real{0.1667}}\n"
        "  >{\\raggedright\\arraybackslash}p{(\\linewidth - 10\\tabcolsep) * \\real{0.1667}}\n"
        "  >{\\raggedright\\arraybackslash}p{(\\linewidth - 10\\tabcolsep) * \\real{0.1667}}\n"
        "  >{\\raggedright\\arraybackslash}p{(\\linewidth - 10\\tabcolsep) * \\real{0.1667}}\n"
        "  >{\\raggedright\\arraybackslash}p{(\\linewidth - 10\\tabcolsep) * \\real{0.1667}}\n"
        "  >{\\raggedright\\arraybackslash}p{(\\linewidth - 10\\tabcolsep) * \\real{0.1667}}@{}}"
    )
    _six_col_new = (
        "\\begin{longtable}[]{@{}\n"
        "  >{\\raggedright\\arraybackslash}p{0.34\\linewidth}\n"
        "  >{\\centering\\arraybackslash}p{0.09\\linewidth}\n"
        "  >{\\centering\\arraybackslash}p{0.11\\linewidth}\n"
        "  >{\\centering\\arraybackslash}p{0.19\\linewidth}\n"
        "  >{\\centering\\arraybackslash}p{0.12\\linewidth}\n"
        "  >{\\centering\\arraybackslash}p{0.07\\linewidth}@{}}"
    )
    if "Nucleus area vs cell area" in text:
        text = text.replace(_six_col_old, _six_col_new, 1)

    text = unicode_to_latex(text)
    tex_path.write_text(text, encoding="utf-8")


def write_latex(
    md_path: Path,
    tex_path: Path,
    *,
    notebooks_dir: Path | None = None,
    preamble_path: Path | None = None,
    polish_kwargs: dict | None = None,
) -> None:
    """Convert filled Markdown to standalone LaTeX (pandoc preferred, built-in fallback)."""
    notebooks_dir = notebooks_dir or NOTEBOOKS_DIR
    preamble_path = preamble_path or PREAMBLE_PATH
    polish_kwargs = polish_kwargs or {}

    md_text = md_path.read_text(encoding="utf-8")
    if _find_tool("pandoc"):
        pandoc = _find_tool("pandoc")
        cmd = [
            pandoc, str(md_path), "-o", str(tex_path),
            "--from", "markdown", "--to", "latex", "--standalone",
            "--resource-path", str(notebooks_dir),
            "-V", "documentclass=article", "-V", "lang=en",
        ]
        if preamble_path.is_file():
            cmd.extend(["-H", str(preamble_path)])
        print("=" * 70)
        print("Running pandoc")
        print("=" * 70)
        subprocess.run(cmd, check=True, cwd=str(notebooks_dir))
        polish_tex(tex_path, **polish_kwargs)
    else:
        print("=" * 70)
        print("pandoc not found — using built-in Markdown→LaTeX converter")
        print("=" * 70)
        preamble = preamble_path.read_text(encoding="utf-8") if preamble_path.is_file() else ""
        body_md, title = _strip_yaml_front_matter(md_text)
        body = markdown_to_latex(body_md)
        title_block = ""
        if title:
            title_block = f"\\title{{{_latex_escape(title)}}}\n\\maketitle\n"
        doc = (
            "\\documentclass{article}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{longtable}\n"
            "\\usepackage{lmodern}\n"
            f"{preamble}\n"
            "\\begin{document}\n"
            f"{title_block}"
            f"{body}\n"
            "\\end{document}\n"
        )
        tex_path.write_text(doc, encoding="utf-8")
        polish_tex(tex_path, **polish_kwargs)
    print(f"[ok] wrote {tex_path.relative_to(REPO_ROOT)}")


def run_pdf_compile(tex_path: Path, pdf_path: Path) -> bool:
    """Compile .tex → .pdf using xelatex (preferred) or tectonic."""
    out_dir = tex_path.parent

    if _find_tool("xelatex"):
        xelatex = _find_tool("xelatex")
        print("=" * 70)
        print("Running xelatex")
        print("=" * 70)
        for _ in range(2):
            subprocess.run(
                [xelatex, "-interaction=nonstopmode", "-output-directory", str(out_dir), tex_path.name],
                check=True,
                cwd=str(out_dir),
            )
        if pdf_path.is_file():
            print(f"[ok] wrote {pdf_path.relative_to(REPO_ROOT)} ({pdf_path.stat().st_size / 1024:.1f} KB)")
            return True

    if _find_tool("tectonic"):
        tectonic = _find_tool("tectonic")
        print("=" * 70)
        print("Running tectonic")
        print("=" * 70)
        for cmd in (
            [tectonic, "-X", "compile", str(tex_path), "--outdir", str(out_dir)],
            [tectonic, str(tex_path), "--outdir", str(out_dir)],
        ):
            try:
                subprocess.run(cmd, check=True, cwd=str(out_dir))
                break
            except subprocess.CalledProcessError:
                continue
        if pdf_path.is_file():
            print(f"[ok] wrote {pdf_path.relative_to(REPO_ROOT)} ({pdf_path.stat().st_size / 1024:.1f} KB)")
            return True

    print(
        "[warn] No PDF engine found (tried xelatex, tectonic).\n"
        "       conda install -c conda-forge tectonic   # or: texlive-core\n"
        f"       Manual: cd {out_dir} && tectonic -X compile {tex_path.name} --output-directory ."
    )
    return False
