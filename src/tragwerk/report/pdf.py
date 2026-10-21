"""Print an HTML report to PDF with a headless Chromium (optional dependency)."""
from __future__ import annotations

from pathlib import Path


def html_to_pdf(html_text: str, out_path: str | Path, title: str = "") -> Path:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PDF export needs the optional dependency: pip install tragwerk[pdf] && playwright install chromium") from exc
    out = Path(out_path)
    footer = ("<div style='font-size:8px;color:#6b7280;width:100%;padding:0 14mm;display:flex;justify-content:space-between;"
              "font-family:Arial,sans-serif'><span>" + title.replace("<", "&lt;") +
              "</span><span>page <span class='pageNumber'></span> of <span class='totalPages'></span></span></div>")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_text, wait_until="load")
        page.emulate_media(media="print")
        page.pdf(path=str(out), format="A4", print_background=True, display_header_footer=True,
                 header_template="<div></div>", footer_template=footer,
                 margin={"top": "16mm", "bottom": "18mm", "left": "14mm", "right": "14mm"})
        browser.close()
    return out
