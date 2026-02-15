from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PdfPage:
    page: int  # 1-based
    text: str


def extract_pages(path: str | Path) -> list[PdfPage]:
    p = Path(path)
    # Prefer PyMuPDF for better extraction.
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(p))
        out: list[PdfPage] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            out.append(PdfPage(page=i + 1, text=_clean(text)))
        doc.close()
        return out
    except Exception:
        pass

    # Fallback to pypdf
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(p))
    out2: list[PdfPage] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        out2.append(PdfPage(page=i + 1, text=_clean(text)))
    return out2


def _clean(text: str) -> str:
    # Keep it conservative; just normalize line endings and strip trailing spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(lines).strip()
