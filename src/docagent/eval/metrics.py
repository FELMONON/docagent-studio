from __future__ import annotations

import re
from dataclasses import dataclass


_CIT_RE = re.compile(r"\[([^\[\]]{3,200})\]")


@dataclass(frozen=True)
class EvalRow:
    question: str
    answer: str | None
    required_sources: list[str]


def extract_citations(text: str) -> list[str]:
    cits: list[str] = []
    for m in _CIT_RE.finditer(text or ""):
        raw = m.group(1).strip()
        # Heuristic: keep only our source-ref looking tokens.
        if raw.startswith("pdf:") or raw.startswith("md:") or raw.startswith("notion:"):
            cits.append(raw)
    return cits


def recall_at_k(required: list[str], retrieved: list[str]) -> float:
    if not required:
        return 1.0
    s = set(retrieved)
    hit = any(r in s for r in required)
    return 1.0 if hit else 0.0


def coverage(required: list[str], cited: list[str]) -> float:
    if not required:
        return 1.0
    req = set(required)
    cit = set(cited)
    return len(req & cit) / max(1, len(req))
