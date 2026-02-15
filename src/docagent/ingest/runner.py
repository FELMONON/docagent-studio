from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..index import sqlite_store
from . import markdown as md
from . import pdf
from .utils import relpath, sha256_file


SUPPORTED_TEXT_EXTS = {".md", ".markdown", ".txt"}


@dataclass(frozen=True)
class IngestOptions:
    input_dir: Path
    notion_root: Path | None = None
    max_chunk_chars: int = 2500
    overlap: int = 200


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        yield p


def ingest_into_db(
    *,
    conn,
    db_path: str,
    options: IngestOptions,
) -> dict[str, Any]:
    sqlite_store.init_db(conn)

    docs_seen = 0
    docs_changed = 0
    chunks_inserted = 0

    for path in iter_files(options.input_dir):
        ext = path.suffix.lower()
        if ext not in SUPPORTED_TEXT_EXTS and ext != ".pdf":
            continue

        docs_seen += 1

        rp = relpath(path, options.input_dir)
        stype = _source_type_for(path, options)
        title = path.stem
        digest = sha256_file(path)

        doc_id, changed = sqlite_store.upsert_document(
            conn,
            source_type=stype,
            path=rp,
            title=title,
            sha256=digest,
            metadata={"abs_path": str(path)},
        )
        if changed:
            docs_changed += 1

            if ext == ".pdf":
                chunks = _chunks_from_pdf(path, rel=rp, doc_id=doc_id, options=options)
            else:
                chunks = _chunks_from_markdown(path, rel=rp, doc_id=doc_id, stype=stype, options=options)

            if chunks:
                sqlite_store.insert_chunks(conn, chunks)
                chunks_inserted += len(chunks)

        # Commit periodically to keep memory stable
        if docs_seen % 20 == 0:
            conn.commit()

    conn.commit()

    return {
        "documents_seen": docs_seen,
        "documents_changed": docs_changed,
        "chunks_inserted": chunks_inserted,
    }


def _source_type_for(path: Path, options: IngestOptions) -> str:
    if options.notion_root is not None:
        try:
            path.resolve().relative_to(options.notion_root.resolve())
            return "notion"
        except Exception:
            pass
    return "pdf" if path.suffix.lower() == ".pdf" else "md"


def _chunks_from_pdf(path: Path, *, rel: str, doc_id: int, options: IngestOptions) -> list[dict[str, Any]]:
    pages = pdf.extract_pages(path)
    out: list[dict[str, Any]] = []
    for page in pages:
        if not page.text.strip():
            continue

        chunks = md.chunk_text(page.text, max_chars=options.max_chunk_chars, overlap=options.overlap)
        for idx, chunk in enumerate(chunks):
            suffix = f".c{idx+1}" if len(chunks) > 1 else ""
            source_ref = f"pdf:{rel}#p{page.page}{suffix}"
            out.append(
                {
                    "doc_id": doc_id,
                    "source_ref": source_ref,
                    "heading": None,
                    "page_start": page.page,
                    "page_end": page.page,
                    "text": chunk,
                    "metadata": {"page": page.page, "chunk": idx + 1, "chunks_on_page": len(chunks)},
                }
            )
    return out


def _chunks_from_markdown(
    path: Path,
    *,
    rel: str,
    doc_id: int,
    stype: str,
    options: IngestOptions,
) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = md.split_sections(text)

    out: list[dict[str, Any]] = []
    for sec in sections:
        chunks = md.chunk_text(sec.text, max_chars=options.max_chunk_chars, overlap=options.overlap)
        for idx, chunk in enumerate(chunks):
            suffix = f".c{idx+1}" if len(chunks) > 1 else ""
            source_ref = f"{stype}:{rel}#L{sec.start_line}{suffix}"
            out.append(
                {
                    "doc_id": doc_id,
                    "source_ref": source_ref,
                    "heading": sec.heading_path or None,
                    "page_start": None,
                    "page_end": None,
                    "text": chunk,
                    "metadata": {
                        "start_line": sec.start_line,
                        "heading_path": sec.heading_path,
                        "chunk": idx + 1,
                        "chunks_in_section": len(chunks),
                    },
                }
            )
    return out
