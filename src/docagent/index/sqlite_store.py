from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Document:
    doc_id: int
    source_type: str
    path: str
    title: str
    sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    doc_id: int
    source_ref: str
    text: str
    heading: str | None
    page_start: int | None
    page_end: int | None
    metadata: dict[str, Any]


def connect(db_path: str | os.PathLike[str]) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
          doc_id INTEGER PRIMARY KEY,
          source_type TEXT NOT NULL,
          path TEXT NOT NULL UNIQUE,
          title TEXT NOT NULL,
          sha256 TEXT NOT NULL,
          metadata_json TEXT NOT NULL,
          created_at INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
          chunk_id INTEGER PRIMARY KEY,
          doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
          source_ref TEXT NOT NULL,
          heading TEXT,
          page_start INTEGER,
          page_end INTEGER,
          text TEXT NOT NULL,
          metadata_json TEXT NOT NULL
        );
        """
    )

    # FTS5 table is rebuilt by `rebuild_fts()`.
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(text, chunk_id UNINDEXED);
        """
    )

    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    conn.commit()


def upsert_document(
    conn: sqlite3.Connection,
    *,
    source_type: str,
    path: str,
    title: str,
    sha256: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[int, bool]:
    """Insert/update document.

    Returns: (doc_id, changed)
    """
    metadata = metadata or {}

    row = conn.execute(
        "SELECT doc_id, sha256 FROM documents WHERE path = ?",
        (path,),
    ).fetchone()
    now = int(time.time())

    if row is None:
        cur = conn.execute(
            """
            INSERT INTO documents(source_type, path, title, sha256, metadata_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (source_type, path, title, sha256, json.dumps(metadata, ensure_ascii=True), now),
        )
        return int(cur.lastrowid), True

    doc_id = int(row["doc_id"])
    if row["sha256"] == sha256:
        # No change
        return doc_id, False

    conn.execute(
        """
        UPDATE documents
        SET source_type=?, title=?, sha256=?, metadata_json=?
        WHERE doc_id=?
        """,
        (source_type, title, sha256, json.dumps(metadata, ensure_ascii=True), doc_id),
    )
    # Replace chunks
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    return doc_id, True


def insert_chunks(conn: sqlite3.Connection, chunks: Iterable[dict[str, Any]]) -> None:
    conn.executemany(
        """
        INSERT INTO chunks(
          doc_id, source_ref, heading, page_start, page_end, text, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                int(c["doc_id"]),
                str(c["source_ref"]),
                c.get("heading"),
                c.get("page_start"),
                c.get("page_end"),
                str(c["text"]),
                json.dumps(c.get("metadata", {}), ensure_ascii=True),
            )
            for c in chunks
        ],
    )


def rebuild_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM chunks_fts")
    conn.execute(
        """
        INSERT INTO chunks_fts(rowid, text, chunk_id)
        SELECT chunk_id, text, chunk_id FROM chunks;
        """
    )
    conn.commit()


def iter_chunks(conn: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    cur = conn.execute(
        "SELECT chunk_id, doc_id, source_ref, heading, page_start, page_end, text, metadata_json FROM chunks ORDER BY chunk_id"
    )
    yield from cur


def get_chunks_by_ids(conn: sqlite3.Connection, chunk_ids: list[int]) -> list[sqlite3.Row]:
    if not chunk_ids:
        return []
    placeholders = ",".join(["?"] * len(chunk_ids))
    cur = conn.execute(
        f"SELECT chunk_id, doc_id, source_ref, heading, page_start, page_end, text, metadata_json FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    )
    rows = list(cur.fetchall())
    by_id = {int(r["chunk_id"]): r for r in rows}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


def get_chunks_by_source_ref(conn: sqlite3.Connection, source_ref: str) -> list[sqlite3.Row]:
    cur = conn.execute(
        "SELECT chunk_id, doc_id, source_ref, heading, page_start, page_end, text, metadata_json FROM chunks WHERE source_ref = ? ORDER BY chunk_id",
        (str(source_ref),),
    )
    return list(cur.fetchall())


def fts_search(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[tuple[int, float]]:
    """Return [(chunk_id, score)] sorted best-first.

    FTS5 bm25() is smaller-is-better; we convert to a positive score.
    """
    q = query.strip()
    if not q:
        return []

    cur = conn.execute(
        """
        SELECT chunk_id, bm25(chunks_fts) AS bm25
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY bm25 ASC
        LIMIT ?
        """,
        (q, int(limit)),
    )
    out: list[tuple[int, float]] = []
    for row in cur.fetchall():
        cid = int(row["chunk_id"])
        bm25 = float(row["bm25"])
        score = 1.0 / (1.0 + max(bm25, 0.0))
        out.append((cid, score))
    return out
