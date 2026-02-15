from __future__ import annotations

import sqlite3


def init_graph(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entities (
          entity_id INTEGER PRIMARY KEY,
          name TEXT NOT NULL,
          name_norm TEXT NOT NULL UNIQUE,
          mention_count INTEGER NOT NULL,
          chunk_count INTEGER NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name_norm ON entities(name_norm);")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_entities (
          chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
          entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
          PRIMARY KEY (chunk_id, entity_id)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity_id);")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entity_edges (
          entity_id_a INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
          entity_id_b INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
          weight INTEGER NOT NULL,
          PRIMARY KEY (entity_id_a, entity_id_b)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_edges_a ON entity_edges(entity_id_a);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_edges_b ON entity_edges(entity_id_b);")

    conn.commit()


def clear_graph(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM entity_edges;")
    conn.execute("DELETE FROM chunk_entities;")
    conn.execute("DELETE FROM entities;")
    conn.commit()


def upsert_entity(conn: sqlite3.Connection, *, name: str, name_norm: str) -> int:
    row = conn.execute("SELECT entity_id FROM entities WHERE name_norm = ?", (name_norm,)).fetchone()
    if row is not None:
        return int(row["entity_id"])

    cur = conn.execute(
        "INSERT INTO entities(name, name_norm, mention_count, chunk_count) VALUES(?, ?, 0, 0)",
        (name, name_norm),
    )
    return int(cur.lastrowid)


def add_chunk_entity(conn: sqlite3.Connection, *, chunk_id: int, entity_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO chunk_entities(chunk_id, entity_id) VALUES(?, ?)",
        (int(chunk_id), int(entity_id)),
    )


def bump_entity_counts(conn: sqlite3.Connection, *, entity_id: int, mention_count: int, chunk_inc: int) -> None:
    conn.execute(
        """
        UPDATE entities
        SET mention_count = mention_count + ?,
            chunk_count = chunk_count + ?
        WHERE entity_id = ?
        """,
        (int(mention_count), int(chunk_inc), int(entity_id)),
    )


def upsert_edge(conn: sqlite3.Connection, *, a: int, b: int, weight: int) -> None:
    a = int(a)
    b = int(b)
    if a == b:
        return
    if a > b:
        a, b = b, a
    conn.execute(
        """
        INSERT INTO entity_edges(entity_id_a, entity_id_b, weight)
        VALUES (?, ?, ?)
        ON CONFLICT(entity_id_a, entity_id_b) DO UPDATE SET weight = weight + excluded.weight
        """,
        (a, b, int(weight)),
    )


def get_entity_matches(conn: sqlite3.Connection, term_norm: str, *, limit: int = 10):
    like = f"%{term_norm}%"
    return conn.execute(
        """
        SELECT entity_id, name, name_norm, chunk_count
        FROM entities
        WHERE name_norm LIKE ?
        ORDER BY chunk_count DESC
        LIMIT ?
        """,
        (like, int(limit)),
    ).fetchall()


def get_neighbors(conn: sqlite3.Connection, entity_id: int, *, limit: int = 10):
    # For undirected edges, query both sides.
    eid = int(entity_id)
    return conn.execute(
        """
        SELECT
          CASE
            WHEN entity_id_a = ? THEN entity_id_b
            ELSE entity_id_a
          END AS neighbor_id,
          weight
        FROM entity_edges
        WHERE entity_id_a = ? OR entity_id_b = ?
        ORDER BY weight DESC
        LIMIT ?
        """,
        (eid, eid, eid, int(limit)),
    ).fetchall()


def get_entity_by_id(conn: sqlite3.Connection, entity_id: int):
    return conn.execute(
        "SELECT entity_id, name, name_norm, mention_count, chunk_count FROM entities WHERE entity_id = ?",
        (int(entity_id),),
    ).fetchone()


def get_top_chunks_for_entity(conn: sqlite3.Connection, entity_id: int, *, limit: int = 5):
    return conn.execute(
        """
        SELECT c.chunk_id, c.source_ref, c.heading, c.text
        FROM chunk_entities ce
        JOIN chunks c ON c.chunk_id = ce.chunk_id
        WHERE ce.entity_id = ?
        ORDER BY c.chunk_id DESC
        LIMIT ?
        """,
        (int(entity_id), int(limit)),
    ).fetchall()

