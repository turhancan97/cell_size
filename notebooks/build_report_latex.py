"""Build biology-only LaTeX/PDF report from report_biology.md.template.

End-to-end pipeline:
  1. Regenerate biology figures (biology_plots.write_all)
  2. Fill template with stats from biology_stats.load_biology_stats
  3. Pandoc → LaTeX → XeLaTeX PDF

Usage (from repo root):

    conda activate cell-size
    python notebooks/build_report_latex.py
    python notebooks/build_report_latex.py --no-figures
    python notebooks/build_report_latex.py --no-figures --skip-fill   # compile edited report_biology.md
    python notebooks/build_report_latex.py --tex-only

Requires: optional pandoc + xelatex (TeX Live) for PDF; built-in LaTeX fallback if pandoc absent.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS_DIR))

from biology_plots import write_all as write_biology_figures  # noqa: E402
from biology_stats import fill_template, load_biology_stats  # noqa: E402

TEMPLATE_PATH = NOTEBOOKS_DIR / "report_biology.md.template"
MD_PATH = NOTEBOOKS_DIR / "report_biology.md"
TEX_PATH = NOTEBOOKS_DIR / "report_biology.tex"
PDF_PATH = NOTEBOOKS_DIR / "report_biology.pdf"
PREAMBLE_PATH = NOTEBOOKS_DIR / "latex" / "preamble.tex"
FIGURES_DIR = NOTEBOOKS_DIR / "figures"


def generate_figures() -> None:
    print("=" * 70)
    print("Generating biology figures")
    print("=" * 70)
    write_biology_figures(figures_dir=FIGURES_DIR)


def fill_markdown(template_path: Path, output_path: Path) -> None:
    print("=" * 70)
    print("Loading stats and filling template")
    print("=" * 70)
    stats = load_biology_stats()
    template = template_path.read_text(encoding="utf-8")
    filled = fill_template(template, stats["placeholders"])
    output_path.write_text(filled, encoding="utf-8")
    print(f"[ok] wrote {output_path.relative_to(REPO_ROOT)}")


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


def _unicode_to_latex(text: str) -> str:
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

    # Pandoc may escape inline math when `%` or digits follow `$` immediately.
    text = text.replace(r"\$\approx\pm\$", r"\(\approx\pm\)")
    text = text.replace(r"\$\leq\$", r"\(\leq\)")
    text = text.replace(r"\$\geq\$", r"\(\geq\)")
    text = text.replace(r"\$\pm\$", r"\(\pm\)")
    text = text.replace(r"\$\div\$", r"\(\div\)")
    text = text.replace(r"\$\times\$", r"\(\times\)")
    text = text.replace(r"\$\rightarrow\$", r"\(\rightarrow\)")
    text = text.replace(r"\$\approx\$", r"\(\approx\)")

    return text


def _title_page_heading(title: str) -> str:
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


def _polish_tex(tex_path: Path) -> None:
    """Apply scientific layout polish after pandoc conversion."""
    text = tex_path.read_text(encoding="utf-8")

    # Enable numbered sections (pandoc disables by default).
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
        f"{_title_page_heading(title)}\n"
        "\\vspace{1.2cm}\n"
        "{\\large\\color{ReportMuted} Morphometric analysis report\\par}\n"
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

    def _wrap_abstract(match: re.Match[str]) -> str:
        return match.group(1) + "\\begin{abstractbox}\n" + match.group(2).strip() + "\n\\end{abstractbox}\n"

    text = re.sub(
        r"(\\section\{Abstract\}.*?\n)(.*?)(?=\n\\section\{)",
        _wrap_abstract,
        text,
        count=1,
        flags=re.DOTALL,
    )

    text = text.replace("\\begin{figure}", "\\begin{figure}[H]")
    text = re.sub(
        r"\\pandocbounded\{\\includegraphics(\[[^\]]*\])?\{([^}]+)\}\}",
        r"\\includegraphics[width=0.98\\linewidth]{\2}",
        text,
    )

    if "\\section{Discussion}" in text and "\\begin{document}" in text:
        prefix, rest = text.split("\\begin{document}", 1)
        body, suffix = rest.split("\\section{Discussion}", 1)
        body = re.sub(
            r"\\begin\{itemize\}(.*?)\\end\{itemize\}",
            r"\\begin{keystats}\n\\begin{itemize}\1\\end{itemize}\n\\end{keystats}",
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

    text = _unicode_to_latex(text)

    tex_path.write_text(text, encoding="utf-8")


def write_latex(md_path: Path, tex_path: Path) -> None:
    """Convert filled Markdown to standalone LaTeX (pandoc preferred, built-in fallback)."""
    md_text = md_path.read_text(encoding="utf-8")
    if shutil.which("pandoc"):
        cmd = [
            "pandoc", str(md_path), "-o", str(tex_path),
            "--from", "markdown", "--to", "latex", "--standalone",
            "--resource-path", str(NOTEBOOKS_DIR),
            "-V", "documentclass=article", "-V", "lang=en",
        ]
        if PREAMBLE_PATH.is_file():
            cmd.extend(["-H", str(PREAMBLE_PATH)])
        print("=" * 70)
        print("Running pandoc")
        print("=" * 70)
        subprocess.run(cmd, check=True, cwd=str(NOTEBOOKS_DIR))
        _polish_tex(tex_path)
    else:
        print("=" * 70)
        print("pandoc not found — using built-in Markdown→LaTeX converter")
        print("=" * 70)
        preamble = PREAMBLE_PATH.read_text(encoding="utf-8") if PREAMBLE_PATH.is_file() else ""
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
        _polish_tex(tex_path)
    print(f"[ok] wrote {tex_path.relative_to(REPO_ROOT)}")


def run_pdf_compile(tex_path: Path, pdf_path: Path) -> bool:
    """Compile .tex → .pdf using xelatex (preferred) or tectonic."""
    out_dir = tex_path.parent

    if shutil.which("xelatex"):
        print("=" * 70)
        print("Running xelatex")
        print("=" * 70)
        for _ in range(2):
            subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-output-directory", str(out_dir), tex_path.name],
                check=True,
                cwd=str(out_dir),
            )
        if pdf_path.is_file():
            print(f"[ok] wrote {pdf_path.relative_to(REPO_ROOT)} ({pdf_path.stat().st_size / 1024:.1f} KB)")
            return True

    if shutil.which("tectonic"):
        print("=" * 70)
        print("Running tectonic")
        print("=" * 70)
        # -o / --outdir is the output DIRECTORY, not the PDF path
        for cmd in (
            ["tectonic", "-X", "compile", str(tex_path), "--output-directory", str(out_dir)],
            ["tectonic", str(tex_path), "--outdir", str(out_dir)],
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
        "       Manual: cd notebooks && tectonic -X compile report_biology.tex --output-directory ."
    )
    return False


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build biology LaTeX/PDF report.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure regeneration.")
    parser.add_argument(
        "--skip-fill",
        action="store_true",
        help="Skip template fill; compile existing --output-md (default: report_biology.md).",
    )
    parser.add_argument("--tex-only", action="store_true", help="Stop after .tex (no xelatex).")
    parser.add_argument("-i", "--input", type=Path, default=TEMPLATE_PATH, help="Markdown template.")
    parser.add_argument("-o", "--output-md", type=Path, default=MD_PATH, help="Filled Markdown output.")
    parser.add_argument("--tex", type=Path, default=TEX_PATH, help="LaTeX output path.")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH, help="PDF output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if not args.no_figures:
        generate_figures()

    if args.skip_fill:
        if not args.output_md.resolve().is_file():
            raise FileNotFoundError(
                f"--skip-fill: {args.output_md} not found. "
                "Run without --skip-fill once, or edit that file first."
            )
        print("=" * 70)
        print(f"Skipping template fill — using {args.output_md.relative_to(REPO_ROOT)}")
        print("=" * 70)
    else:
        fill_markdown(args.input.resolve(), args.output_md.resolve())

    write_latex(args.output_md.resolve(), args.tex.resolve())

    if not args.tex_only:
        run_pdf_compile(args.tex.resolve(), args.pdf.resolve())


if __name__ == "__main__":
    main()
