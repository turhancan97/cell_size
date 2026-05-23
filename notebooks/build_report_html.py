"""Render notebooks/report.html from notebooks/report.md.

Produces an HTML file with a modern, light, biologist-friendly layout:

- narrow body column, generous whitespace,
- card-style sections, soft shadows, rounded corners,
- a KPI strip at the top of each major section,
- responsive tables with subtle row striping.

Two image modes are supported:

- ``--embed-images`` (default): figures are inlined as base64 data URIs,
  producing a single self-contained HTML file that can be emailed or
  archived without its figure folder.
- ``--link-images``: figures are referenced via relative ``<img src="...">``
  tags, producing a much lighter HTML file that loads figures from
  ``notebooks/figures/`` at view time.

Usage:
    conda activate cell-size
    python notebooks/build_report_html.py                       # embedded (default)
    python notebooks/build_report_html.py --link-images         # lightweight
    python notebooks/build_report_html.py --link-images -o report_linked.html
"""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MD_PATH = REPO_ROOT / "notebooks" / "report.md"
HTML_PATH = REPO_ROOT / "notebooks" / "report.html"
FIGURES_DIR = REPO_ROOT / "notebooks" / "figures"

# Module-level config set by main(); affects how _embed_image renders <img>.
EMBED_IMAGES = True
HTML_DIR = HTML_PATH.parent


# ---------------------------------------------------------------------------
# Tiny, targeted Markdown → HTML renderer (purpose-built for report.md)
# ---------------------------------------------------------------------------


def _embed_image(rel_path: str) -> str:
    """Render an ``<img>`` tag for ``rel_path`` (relative to ``notebooks/``).

    Uses a base64 data URI if ``EMBED_IMAGES`` is true; otherwise emits
    a relative link so the HTML file can stay light and load from the
    figures folder at view time.
    """
    path = (REPO_ROOT / "notebooks" / rel_path).resolve()
    if not path.is_file():
        return f'<div class="missing-image">Missing image: {html.escape(rel_path)}</div>'

    alt = html.escape(path.stem)
    if EMBED_IMAGES:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        src = f"data:{mime};base64,{data}"
    else:
        # Relative path from the HTML output file to the figure on disk.
        try:
            rel_src = path.relative_to(HTML_DIR.resolve())
        except ValueError:
            import os
            rel_src = Path(os.path.relpath(path, start=HTML_DIR.resolve()))
        src = rel_src.as_posix()
        src = html.escape(src, quote=True)

    return (
        f'<figure class="report-figure">'
        f'<img src="{src}" alt="{alt}" loading="lazy"/>'
        f'</figure>'
    )


def _render_inline(text: str) -> str:
    """Render inline Markdown (bold, italic, code, links) into HTML."""
    # escape first
    out = html.escape(text, quote=False)
    # inline code
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    # bold **...**
    out = re.sub(r"\*\*([^\*]+)\*\*", r"<strong>\1</strong>", out)
    # italic *...* (avoid matching ** and inside code — heuristic is fine here)
    out = re.sub(r"(?<!\*)\*([^\*\n]+)\*(?!\*)", r"<em>\1</em>", out)
    # links [text](url)
    out = re.sub(
        r"\[([^\]]+)\]\(([^\)]+)\)",
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
        out,
    )
    return out


def _render_table(lines: list[str]) -> str:
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    # second line is the separator; skip it
    body_rows = []
    for raw in lines[2:]:
        cols = [c.strip() for c in raw.strip().strip("|").split("|")]
        body_rows.append(cols)

    thead = "<thead><tr>" + "".join(f"<th>{_render_inline(c)}</th>" for c in header) + "</tr></thead>"
    tbody = "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{_render_inline(c)}</td>" for c in row) + "</tr>"
        for row in body_rows
    ) + "</tbody>"
    return f'<div class="table-wrap"><table class="report-table">{thead}{tbody}</table></div>'


def markdown_to_html(md_text: str) -> str:
    lines = md_text.splitlines()
    out_parts: list[str] = []
    i = 0
    n = len(lines)

    def flush_paragraph(buf: list[str]) -> None:
        if not buf:
            return
        para = " ".join(_render_inline(x) for x in buf).strip()
        if para:
            out_parts.append(f"<p>{para}</p>")
        buf.clear()

    paragraph_buf: list[str] = []

    while i < n:
        line = lines[i]

        # Horizontal rule ------------------------------------------------
        if line.strip() == "---":
            flush_paragraph(paragraph_buf)
            out_parts.append('<hr class="report-hr"/>')
            i += 1
            continue

        # Headings -------------------------------------------------------
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush_paragraph(paragraph_buf)
            level = len(m.group(1))
            content = _render_inline(m.group(2).strip())
            out_parts.append(f"<h{level}>{content}</h{level}>")
            i += 1
            continue

        # Blockquote -----------------------------------------------------
        if line.startswith("> "):
            flush_paragraph(paragraph_buf)
            buf = []
            while i < n and lines[i].startswith("> "):
                buf.append(lines[i][2:])
                i += 1
            quoted = " ".join(_render_inline(x) for x in buf)
            out_parts.append(f'<blockquote class="callout">{quoted}</blockquote>')
            continue

        # Image-only line (stand-alone figure) ---------------------------
        img_match = re.match(r"^!\[([^\]]*)\]\(([^\)]+)\)\s*$", line)
        if img_match:
            flush_paragraph(paragraph_buf)
            out_parts.append(_embed_image(img_match.group(2)))
            i += 1
            # Next paragraph starting with *Figure.* becomes a <figcaption>
            if i < n and lines[i].strip().startswith("*") and lines[i].strip().endswith("*"):
                cap = lines[i].strip().strip("*")
                out_parts.append(f'<p class="figure-caption">{_render_inline(cap)}</p>')
                i += 1
            continue

        # Table ----------------------------------------------------------
        if line.strip().startswith("|") and i + 1 < n and re.match(r"^\s*\|?\s*-", lines[i + 1]):
            flush_paragraph(paragraph_buf)
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out_parts.append(_render_table(table_lines))
            continue

        # Unordered list -------------------------------------------------
        if re.match(r"^\s*[-*]\s+", line):
            flush_paragraph(paragraph_buf)
            items: list[list[str]] = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append([re.sub(r"^\s*[-*]\s+", "", lines[i])])
                i += 1
                # continuation lines (indented)
                while i < n and lines[i].startswith("  ") and lines[i].strip():
                    items[-1].append(lines[i].strip())
                    i += 1
            li_html = "".join(
                f"<li>{_render_inline(' '.join(it))}</li>" for it in items
            )
            out_parts.append(f"<ul class='report-list'>{li_html}</ul>")
            continue

        # Ordered list ---------------------------------------------------
        if re.match(r"^\s*\d+\.\s+", line):
            flush_paragraph(paragraph_buf)
            items2: list[list[str]] = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                items2.append([re.sub(r"^\s*\d+\.\s+", "", lines[i])])
                i += 1
                while i < n and lines[i].startswith("   ") and lines[i].strip():
                    items2[-1].append(lines[i].strip())
                    i += 1
            ol_html = "".join(
                f"<li>{_render_inline(' '.join(it))}</li>" for it in items2
            )
            out_parts.append(f"<ol class='report-list'>{ol_html}</ol>")
            continue

        # Blank line flushes paragraph
        if line.strip() == "":
            flush_paragraph(paragraph_buf)
            i += 1
            continue

        # Plain paragraph line
        paragraph_buf.append(line)
        i += 1

    flush_paragraph(paragraph_buf)
    return "\n".join(out_parts)


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #f8fafc;
  --surface: #ffffff;
  --ink: #0f172a;
  --ink-soft: #334155;
  --muted: #64748b;
  --line: #e2e8f0;
  --accent: #2563eb;       /* Deep professional blue */
  --accent-hover: #1d4ed8;
  --accent-soft: #eff6ff;
  --warn: #ea580c;
  --warn-soft: #fff7ed;
  --success: #059669;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
  --radius: 16px;
  --radius-sm: 8px;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink-soft);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 16px;
  line-height: 1.7;
  -webkit-font-smoothing: antialiased;
}

.wrap {
  max-width: 980px;
  margin: 0 auto;
  padding: 56px 24px 96px;
}

/* Header Styling - Modern Dark Gradient */
header.report-header {
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  border-radius: var(--radius);
  padding: 48px 56px;
  box-shadow: var(--shadow-lg);
  margin-bottom: 40px;
  color: #ffffff;
}

header.report-header h1 {
  margin: 0 0 12px 0;
  font-size: 36px;
  font-weight: 700;
  letter-spacing: -0.025em;
  color: #ffffff;
}

header.report-header .subtitle {
  margin: 0;
  color: #94a3b8;
  font-size: 16px;
  font-weight: 400;
  letter-spacing: 0.01em;
}

/* Main Cards */
section.card {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 48px 56px;
  box-shadow: var(--shadow);
  margin-bottom: 32px;
  border: 1px solid var(--line);
}

/* Typography */
h1, h2, h3, h4 {
  color: var(--ink);
  letter-spacing: -0.015em;
  font-weight: 600;
}

h1 { font-size: 28px; margin: 0 0 20px; }

h2 {
  font-size: 22px;
  margin: 40px 0 20px;
  padding-bottom: 12px;
  border-bottom: 2px solid var(--line);
}

h3 {
  font-size: 18px;
  margin: 28px 0 12px;
  color: var(--ink);
  font-weight: 600;
}

h4 {
  font-size: 14px;
  margin: 20px 0 8px;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
}

p { margin: 0 0 18px; color: var(--ink-soft); }

a {
  color: var(--accent);
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: border-color 0.2s ease, color 0.2s ease;
}

a:hover {
  color: var(--accent-hover);
  border-bottom: 1px solid var(--accent-hover);
}

/* Code Blocks */
code {
  background: var(--bg);
  color: #db2777; /* Subtle pink/magenta for code */
  padding: 3px 6px;
  border-radius: 6px;
  border: 1px solid var(--line);
  font-size: 0.88em;
  font-family: "SF Mono", Monaco, Consolas, "Liberation Mono", monospace;
}

strong { color: var(--ink); font-weight: 600; }

hr.report-hr {
  border: 0;
  border-top: 1px solid var(--line);
  margin: 48px 0;
}

/* Callouts */
blockquote.callout {
  margin: 24px 0;
  padding: 18px 24px;
  border-left: 4px solid var(--warn);
  background: var(--warn-soft);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  color: #9a3412;
  font-size: 15px;
  line-height: 1.6;
}

blockquote.callout em { color: #9a3412; font-weight: 500; }

/* Lists */
ul.report-list, ol.report-list {
  padding-left: 24px;
  margin: 0 0 20px;
}

ul.report-list li, ol.report-list li {
  margin: 8px 0;
  padding-left: 4px;
}

/* Tables */
.table-wrap {
  overflow-x: auto;
  margin: 24px 0 32px;
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--line);
}

table.report-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14.5px;
  background: var(--surface);
  text-align: left;
}

table.report-table thead th {
  background: var(--bg);
  color: var(--muted);
  padding: 14px 18px;
  font-weight: 600;
  text-transform: uppercase;
  font-size: 12px;
  letter-spacing: 0.05em;
  border-bottom: 2px solid var(--line);
}

table.report-table tbody td {
  padding: 14px 18px;
  border-bottom: 1px solid var(--line);
  color: var(--ink-soft);
}

table.report-table tbody tr {
  transition: background-color 0.15s ease;
}

table.report-table tbody tr:hover {
  background-color: var(--accent-soft);
}

table.report-table tbody tr:last-child td { border-bottom: none; }

/* Figures & Images */
figure.report-figure {
  margin: 32px 0 16px;
  padding: 24px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-sm);
  text-align: center;
}

figure.report-figure img {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  display: block;
  margin: 0 auto;
}

p.figure-caption {
  font-size: 14px;
  color: var(--muted);
  font-style: italic;
  margin: 12px 0 24px;
  text-align: center;
}

.missing-image {
  padding: 24px;
  background: var(--warn-soft);
  color: var(--warn);
  border-radius: var(--radius-sm);
  font-family: "SF Mono", monospace;
  font-size: 13px;
  border: 1px dashed var(--warn);
}

/* Footer */
footer.report-footer {
  margin-top: 48px;
  text-align: center;
  color: var(--muted);
  font-size: 13px;
}

footer.report-footer code {
  background: transparent;
  border: none;
  color: var(--muted);
}

/* Responsive */
@media (max-width: 768px) {
  .wrap { padding: 32px 16px 64px; }
  header.report-header { padding: 32px 24px; }
  section.card { padding: 32px 24px; }
  h1 { font-size: 26px; }
  h2 { font-size: 20px; }
  body { font-size: 15px; }
}
"""


def build_html(body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Cell Classifier and Cell-Size Report</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="wrap">
    <header class="report-header">
      <h1>Cell Classifier and Cell-Size Report</h1>
      <p class="subtitle">Automatic quality filtering of segmented cells and size analysis across frogs.</p>
    </header>
    <section class="card">
      {body_html}
    </section>
    <footer class="report-footer">
      Generated from <code>notebooks/report.md</code> via <code>notebooks/build_report_html.py</code>.
    </footer>
  </div>
</body>
</html>
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render notebooks/report.md to HTML, with figures either embedded "
            "as base64 data URIs (self-contained) or linked from the figures "
            "folder (lightweight)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--embed-images",
        dest="embed",
        action="store_true",
        help="Inline every figure as a base64 data URI (self-contained HTML).",
    )
    mode.add_argument(
        "--link-images",
        dest="embed",
        action="store_false",
        help="Reference figures via relative <img src=...> (lighter HTML).",
    )
    parser.set_defaults(embed=True)
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help=(
            "Output HTML path. Defaults to notebooks/report.html for embedded "
            "mode and notebooks/report_linked.html for linked mode."
        ),
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=MD_PATH,
        help="Input Markdown file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    global EMBED_IMAGES, HTML_DIR
    EMBED_IMAGES = bool(args.embed)

    output_path = args.output
    if output_path is None:
        output_path = HTML_PATH if EMBED_IMAGES else HTML_PATH.with_name("report_linked.html")
    output_path = output_path.resolve()
    HTML_DIR = output_path.parent
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    md_text = Path(args.input).read_text(encoding="utf-8")
    # Strip the first title line (we render it in the header).
    lines = md_text.splitlines()
    if lines and lines[0].startswith("# "):
        md_text = "\n".join(lines[1:]).lstrip("\n")
    body_html = markdown_to_html(md_text)
    final_html = build_html(body_html)
    output_path.write_text(final_html, encoding="utf-8")

    size_kb = output_path.stat().st_size / 1024
    try:
        shown = output_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = output_path
    mode = "embedded base64" if EMBED_IMAGES else "linked figures/ folder"
    print(f"[ok] wrote {shown}  ({size_kb:,.1f} KB, {mode})")


if __name__ == "__main__":
    main()
