"""Build combined classifier + biology LaTeX/PDF report from report.md.template.

End-to-end pipeline:
  1. Regenerate classifier + Part C biology figures (make_report_figures.generate_all_report_figures)
  2. Fill template with stats from report_stats.load_report_stats
  3. Pandoc → LaTeX → PDF

Usage (from repo root):

    conda activate cell-size
    python notebooks/build_report.py
    python notebooks/build_report.py --no-figures
    python notebooks/build_report.py --no-figures --skip-classifier-inference
    python notebooks/build_report.py --skip-fill          # compile existing report.md
    python notebooks/build_report.py --tex-only

Biology-only extended report: notebooks/build_report_latex.py → report_biology.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS_DIR))

from biology_stats import fill_template  # noqa: E402
from latex_report import run_pdf_compile, write_latex  # noqa: E402
from report_stats import load_report_stats  # noqa: E402

TEMPLATE_PATH = NOTEBOOKS_DIR / "report.md.template"
MD_PATH = NOTEBOOKS_DIR / "report.md"
TEX_PATH = NOTEBOOKS_DIR / "report.tex"
PDF_PATH = NOTEBOOKS_DIR / "report.pdf"


def generate_figures() -> None:
    from make_report_figures import generate_all_report_figures  # noqa: PLC0415

    print("=" * 70)
    print("Generating combined report figures")
    print("=" * 70)
    generate_all_report_figures()


def fill_markdown(template_path: Path, output_path: Path, *, run_classifier_inference: bool = True) -> None:
    print("=" * 70)
    print("Loading stats and filling template")
    print("=" * 70)
    stats = load_report_stats(run_classifier_inference=run_classifier_inference)
    template = template_path.read_text(encoding="utf-8")
    filled = fill_template(template, stats["placeholders"])
    output_path.write_text(filled, encoding="utf-8")
    print(f"[ok] wrote {output_path.relative_to(REPO_ROOT)}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined LaTeX/PDF report.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure regeneration.")
    parser.add_argument(
        "--skip-fill",
        action="store_true",
        help="Skip template fill; compile existing report.md.",
    )
    parser.add_argument(
        "--skip-classifier-inference",
        action="store_true",
        help="Use cached classifier_stats.json when filling template (no torch inference).",
    )
    parser.add_argument("--tex-only", action="store_true", help="Stop after .tex (no PDF compile).")
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
        fill_markdown(
            args.input.resolve(),
            args.output_md.resolve(),
            run_classifier_inference=not args.skip_classifier_inference,
        )

    write_latex(
        args.output_md.resolve(),
        args.tex.resolve(),
        polish_kwargs={
            "subtitle": "Automatic quality filtering and morphometry across frogs",
        },
    )

    if not args.tex_only:
        run_pdf_compile(args.tex.resolve(), args.pdf.resolve())


if __name__ == "__main__":
    main()
