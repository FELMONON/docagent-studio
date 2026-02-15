from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import sqlite3

from ..index import sqlite_store
from .extract import extract_entities
from .sqlite_graph import (
    add_chunk_entity,
    bump_entity_counts,
    clear_graph,
    init_graph,
    upsert_edge,
    upsert_entity,
)


def build_graph(
    *,
    conn: sqlite3.Connection,
    clear: bool = True,
    min_chars: int = 3,
    max_per_chunk: int = 25,
    flush_edges_every: int = 50_000,
) -> dict[str, Any]:
    """Build an entity co-occurrence graph from existing chunks."""
    sqlite_store.init_db(conn)
    init_graph(conn)
    if clear:
        clear_graph(conn)

    # Cache name_norm -> entity_id to avoid round-trips.
    ent_cache: dict[str, int] = {}

    edge_counts: dict[tuple[int, int], int] = defaultdict(int)

    chunks_seen = 0
    chunks_with_entities = 0
    entities_linked = 0
    edges_upserted = 0

    def flush_edges() -> int:
        nonlocal edges_upserted
        if not edge_counts:
            return 0
        for (a, b), w in list(edge_counts.items()):
            upsert_edge(conn, a=a, b=b, weight=w)
            edges_upserted += 1
        edge_counts.clear()
        conn.commit()
        return edges_upserted

    for row in sqlite_store.iter_chunks(conn):
        chunks_seen += 1
        chunk_id = int(row["chunk_id"])
        text = str(row["text"])

        ents = extract_entities(text, min_chars=min_chars, max_per_chunk=max_per_chunk)
        if not ents:
            continue

        chunks_with_entities += 1
        chunk_entity_ids: list[int] = []

        for name_norm, (display, count) in ents.items():
            eid = ent_cache.get(name_norm)
            if eid is None:
                eid = upsert_entity(conn, name=display, name_norm=name_norm)
                ent_cache[name_norm] = eid

            add_chunk_entity(conn, chunk_id=chunk_id, entity_id=eid)
            bump_entity_counts(conn, entity_id=eid, mention_count=count, chunk_inc=1)

            chunk_entity_ids.append(eid)
            entities_linked += 1

        # Add co-occurrence edges within the chunk.
        uniq = sorted(set(chunk_entity_ids))
        for a, b in combinations(uniq, 2):
            edge_counts[(a, b)] += 1

        if len(edge_counts) >= flush_edges_every:
            flush_edges()

        if chunks_seen % 500 == 0:
            conn.commit()

    flush_edges()
    conn.commit()

    return {
        "chunks_seen": chunks_seen,
        "chunks_with_entities": chunks_with_entities,
        "entities_linked": entities_linked,
        "edges_upserted": edges_upserted,
        "unique_entities": conn.execute("SELECT COUNT(*) AS n FROM entities").fetchone()["n"],
        "unique_edges": conn.execute("SELECT COUNT(*) AS n FROM entity_edges").fetchone()["n"],
    }

