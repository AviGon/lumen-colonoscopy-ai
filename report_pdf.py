"""Turn an agent-written report_DRAFT.md into a printable A4 PDF with the endoscopic image."""
import io
import re
from pathlib import Path

import markdown
from xhtml2pdf import pisa

CSS = """
@page {
  size: a4; margin: 1.8cm 1.8cm 2cm 1.8cm;
  @frame footer { -pdf-frame-content: footer; bottom: 0.8cm; margin-left: 1.8cm; margin-right: 1.8cm; height: 0.8cm; }
}
body { font-family: Helvetica; font-size: 10pt; line-height: 1.4; color: #1b1f24; }
h1 { font-size: 17pt; margin: 0 0 4pt 0; color: #12305e; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt 0; padding-bottom: 2pt; border-bottom: 1px solid #c9d1dc; color: #12305e; }
h3 { font-size: 11pt; margin: 12pt 0 4pt 0; color: #12305e; }
p { margin: 0 0 6pt 0; }
table { width: 100%; margin: 4pt 0 8pt 0; }
td, th { padding: 3pt 5pt; border: 0.5px solid #c9d1dc; vertical-align: top; text-align: left; }
th { background-color: #eef2f7; }
ul, ol { margin: 0 0 6pt 0; }
li { margin-bottom: 2pt; }
.figure { text-align: center; margin: 8pt 0 4pt 0; }
.caption { text-align: center; font-size: 8.5pt; color: #5d6673; margin-bottom: 8pt; }
#footer { font-size: 8pt; color: #5d6673; text-align: center; }
"""


def build_pdf(md_text: str, image: Path | None, caption: str) -> bytes:
    md_text = re.sub(r"<!--.*?-->", "", md_text, flags=re.S).strip()
    md_text = re.sub(r"^(\s*)- \[ \]", r"\1- [ &nbsp; ]", md_text, flags=re.M)  # checkbox -> printable box
    md_text = re.sub(r"_{3,}", lambda m: "\\_" * len(m.group()), md_text)  # fill-in lines, not emphasis
    html = markdown.markdown(md_text, extensions=["tables", "sane_lists"])
    html = re.sub(r"<table>\s*<thead>\s*<tr>\s*<th>\s*</th>\s*<th>\s*</th>\s*</tr>\s*</thead>", "<table>", html)  # blank header rows

    if image and image.exists():
        fig = (f'<div class="figure"><img src="{image.as_posix()}" width="440"></div>'
               f'<div class="caption">{caption}</div>')
        anchor = re.search(r"<h2>\s*Description\s*</h2>", html)
        html = html[:anchor.start()] + fig + html[anchor.start():] if anchor else fig + html

    doc = (f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}"
           "<div id='footer'>Lumen · Colonoscopy AI. AI-assisted draft for physician review. Not a diagnosis. "
           "Page <pdf:pagenumber> of <pdf:pagecount></div></body></html>")
    out = io.BytesIO()
    result = pisa.CreatePDF(doc, dest=out, encoding="utf-8")
    if result.err:
        raise RuntimeError("PDF generation failed")
    return out.getvalue()
