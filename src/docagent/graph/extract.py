from __future__ import annotations

import re

# Very small, local-first entity extraction:
# - Captures multi-word Capitalized sequences: "Carl Jung", "New York"
# - Captures all-caps acronyms: "NLP", "USA"
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z]{2,}(?:-[A-Z]{2,})*|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b"
)

# Filter out common titlecase words that are rarely meaningful entities alone.
_STOP = {
    "A",
    "An",
    "And",
    "Are",
    "As",
    "At",
    "Be",
    "But",
    "By",
    "Can",
    "Do",
    "For",
    "From",
    "He",
    "Her",
    "His",
    "I",
    "If",
    "In",
    "Into",
    "Is",
    "It",
    "Its",
    "Me",
    "My",
    "No",
    "Not",
    "Of",
    "On",
    "Or",
    "Our",
    "She",
    "So",
    "That",
    "The",
    "Their",
    "There",
    "These",
    "They",
    "This",
    "Those",
    "To",
    "We",
    "Were",
    "What",
    "When",
    "Where",
    "Who",
    "Why",
    "With",
    "You",
    "Your",
}


def norm_entity(name: str) -> str:
    # Normalize for stable matching.
    return re.sub(r"\s+", " ", name.strip()).lower()


def extract_entities(
    text: str,
    *,
    min_chars: int = 3,
    max_per_chunk: int = 25,
) -> dict[str, tuple[str, int]]:
    """Return {name_norm: (display_name, count_in_text)}.

    The extraction is deliberately heuristic; it is good enough to build an
    "entity index" for exploration and graph-based retrieval boosts.
    """
    counts: dict[str, tuple[str, int]] = {}
    for m in _ENTITY_RE.finditer(text):
        raw = m.group(0).strip()
        if len(raw) < min_chars:
            continue
        if raw in _STOP:
            continue

        n = norm_entity(raw)
        if n in _STOP:
            continue

        prev = counts.get(n)
        if prev is None:
            counts[n] = (raw, 1)
        else:
            # Keep first seen as display form; increment count.
            counts[n] = (prev[0], prev[1] + 1)

        if len(counts) >= max_per_chunk:
            break

    return counts


def extract_query_terms(text: str, *, max_terms: int = 8) -> list[str]:
    """Fallback term extraction when no entities are present."""
    toks = re.findall(r"[A-Za-z0-9_]+", text.lower())
    out = []
    for t in toks:
        if len(t) < 3:
            continue
        if t in {s.lower() for s in _STOP}:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= max_terms:
            break
    return out

