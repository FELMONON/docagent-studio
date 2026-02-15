"""Knowledge-graph utilities (lightweight GraphRAG-style indexing).

This module builds an entity co-occurrence graph from chunks already stored in
the SQLite DB. It's local-first and intentionally simple: extraction is
heuristic by default so it works offline and fast on small machines.
"""

