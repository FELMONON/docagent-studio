from __future__ import annotations

import sqlite3
from typing import Any

from .extract import extract_entities, extract_query_terms, norm_entity
from .sqlite_graph import (
    get_entity_by_id,
    get_entity_matches,
    get_neighbors,
    get_top_chunks_for_entity,
)


def query_graph(
    *,
    conn: sqlite3.Connection,
    query: str,
    entity_limit: int = 5,
    neighbor_limit: int = 8,
    chunk_limit: int = 5,
) -> dict[str, Any]:
    ents = extract_entities(query, max_per_chunk=entity_limit)
    terms = list(ents.keys())
    if not terms:
        terms = [norm_entity(t) for t in extract_query_terms(query, max_terms=entity_limit)]

    matches = []
    for t in terms:
        matches.extend(list(get_entity_matches(conn, t, limit=entity_limit)))

    # Dedup by entity_id, keep highest chunk_count.
    by_id: dict[int, Any] = {}
    for r in matches:
        eid = int(r["entity_id"])
        prev = by_id.get(eid)
        if prev is None or int(r["chunk_count"]) > int(prev["chunk_count"]):
            by_id[eid] = r

    top = sorted(by_id.values(), key=lambda r: int(r["chunk_count"]), reverse=True)[:entity_limit]
    out_entities = []

    for r in top:
        eid = int(r["entity_id"])
        ent = get_entity_by_id(conn, eid)
        if ent is None:
            continue

        neighbors = []
        for n in get_neighbors(conn, eid, limit=neighbor_limit):
            nid = int(n["neighbor_id"])
            nrow = get_entity_by_id(conn, nid)
            if nrow is None:
                continue
            neighbors.append({"entity": dict(nrow), "weight": int(n["weight"])})

        chunks = []
        for c in get_top_chunks_for_entity(conn, eid, limit=chunk_limit):
            chunks.append(
                {
                    "chunk_id": int(c["chunk_id"]),
                    "source_ref": str(c["source_ref"]),
                    "heading": (str(c["heading"]) if c["heading"] is not None else None),
                    "preview": " ".join(str(c["text"]).split())[:220],
                }
            )

        out_entities.append(
            {
                "entity": dict(ent),
                "neighbors": neighbors,
                "chunks": chunks,
            }
        )

    return {"terms": terms, "entities": out_entities}

