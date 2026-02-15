from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from ..chat.rag import answer_question
from ..index.retriever import HybridRetriever
from .metrics import coverage, extract_citations, recall_at_k


@dataclass(frozen=True)
class EvalSummary:
    n: int
    retrieval_recall: float
    citation_coverage: float | None


def load_eval_set(path: str | Path) -> list[dict]:
    p = Path(path)
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def run_eval(
    *,
    retriever: HybridRetriever,
    eval_path: str | Path,
    k: int,
    generate: bool,
    llm=None,
) -> EvalSummary:
    console = Console()
    rows = load_eval_set(eval_path)

    retrieval_scores = []
    coverage_scores = []

    for i, row in enumerate(rows, start=1):
        q = str(row["question"])
        required = list(row.get("required_sources") or [])

        hits = retriever.retrieve(q, k=k)
        retrieved_sources = [h.source_ref for h in hits]
        retrieval_scores.append(recall_at_k(required, retrieved_sources))

        if generate:
            if llm is None:
                raise ValueError("generate=True requires an llm client")
            ans = answer_question(retriever=retriever, llm=llm, question=q, k=k)
            cited = extract_citations(ans.text)
            coverage_scores.append(coverage(required, cited))

        if i % 10 == 0:
            console.print(f"Evaluated {i}/{len(rows)}")

    n = len(rows)
    retrieval_recall = sum(retrieval_scores) / max(1, n)
    citation_cov = (sum(coverage_scores) / max(1, len(coverage_scores))) if generate else None

    return EvalSummary(n=n, retrieval_recall=retrieval_recall, citation_coverage=citation_cov)
