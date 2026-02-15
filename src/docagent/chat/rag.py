from __future__ import annotations

import re
from dataclasses import dataclass

from ..index.retriever import HybridRetriever, RetrievedChunk
from .llm import ChatMessage, OllamaChatClient


SYSTEM_PROMPT = (
    "You are a personal knowledge base assistant.\n"
    "\n"
    "Rules:\n"
    "- Use ONLY the provided SOURCE blocks as ground truth.\n"
    "- Do not use or mention external sources (websites, papers, authors) unless they appear in the SOURCE text.\n"
    "- Do not include URLs.\n"
    "- Every sentence must end with one or more citations in square brackets using the exact source id, e.g. [pdf:notes.pdf#p12].\n"
    '- If the answer is not in the sources, say: "I couldn\'t find that in the provided documents."'
)


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[str]


def answer_question(
    *,
    retriever: HybridRetriever,
    llm: OllamaChatClient,
    question: str,
    k: int = 8,
) -> Answer:
    hits = retriever.retrieve(question, k=k)
    sources = []
    blocks = []

    for h in hits:
        sources.append(h.source_ref)
        blocks.append(f"SOURCE {h.source_ref}\n{h.text}")

    context = "\n\n---\n\n".join(blocks) if blocks else "(no sources retrieved)"

    allowed_sources = sorted(set(sources))

    user_prompt = (
        f"Question:\n{question.strip()}\n\n"
        f"Sources:\n{context}\n\n"
        "Write a concise answer in plain text.\n"
        "No headings. No 'Sources:' section. No links.\n"
        "Citations must be in square brackets, at the end of each sentence.\n"
        "Example format:\n"
        "Attachment theory says early caregiver relationships shape later relationships. [md:notes.md#L3]\n"
        "An internal working model is a mental representation of self and others. [md:notes.md#L7]\n"
        f"Allowed SOURCE_ID values: {', '.join(allowed_sources) if allowed_sources else '(none)'}"
    )

    msgs = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]

    msg = _strip_sources_section(llm.chat(msgs))
    if _is_grounded(msg, allowed_sources):
        return Answer(text=msg.strip(), sources=sources)

    # Self-correction pass if the model violates the citation/grounding rules.
    fix_prompt = (
        "Rewrite your answer to comply with the Rules exactly.\n"
        "- Use ONLY these SOURCE_ID citations: "
        f"{', '.join(allowed_sources) if allowed_sources else '(none)'}\n"
        "- Remove any URLs.\n"
        "- Do not mention any external sources.\n"
        "- Do not add a bibliography.\n"
        "- Every sentence must end with citations like [SOURCE_ID]."
    )
    msg2 = _strip_sources_section(llm.chat(msgs + [ChatMessage(role="user", content=fix_prompt)]))
    if _is_grounded(msg2, allowed_sources):
        return Answer(text=msg2.strip(), sources=sources)

    # Reliable fallback: return excerpts with citations rather than a potentially
    # ungrounded answer (small local models can struggle with citation style).
    return Answer(text=_fallback_answer(question, hits), sources=sources)


_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def _is_grounded(text: str, allowed_sources: list[str]) -> bool:
    if "http://" in text or "https://" in text:
        return False
    cites = [m.group(1).strip() for m in _CITATION_RE.finditer(text)]
    if not cites:
        return False
    if not allowed_sources:
        return False
    allowed = set(allowed_sources)
    # Any citation must be one of the retrieved source ids.
    if any(c not in allowed for c in cites):
        return False
    return True


def _fallback_answer(question: str, hits: list[RetrievedChunk]) -> str:
    if not hits:
        return "I couldn't find that in the provided documents."

    # Extractive fallback: answer by quoting the most relevant snippets with citations.
    lines: list[str] = []
    for h in hits[: min(3, len(hits))]:
        excerpt = " ".join(h.text.strip().split())
        if len(excerpt) > 360:
            excerpt = excerpt[:360].rstrip() + "..."
        lines.append(f"- {excerpt} [{h.source_ref}]")
    return "\n".join(lines)


def _strip_sources_section(text: str) -> str:
    """Remove model-added trailing 'Sources:' blocks.

    We already return sources separately from the CLI and require inline
    citations, so a trailing bibliography-style list is redundant and often
    violates the prompt rules.
    """
    lines = text.splitlines()
    if not lines:
        return text

    # Find the last occurrence of a "Sources:" heading and drop it + following bullets.
    last_idx = -1
    for i, line in enumerate(lines):
        if line.strip().lower() in {"sources:", "source:", "citations:", "references:"}:
            last_idx = i
    if last_idx == -1:
        return text

    tail = [ln.strip() for ln in lines[last_idx + 1 :]]
    if not tail:
        return "\n".join(lines[:last_idx]).rstrip()

    # Only strip if the tail looks like a list of ids.
    ok = True
    for ln in tail:
        if not ln:
            continue
        if ln.startswith("- "):
            continue
        # Allow bare ids without '- ' prefix.
        if ":" in ln and ("#p" in ln or "#l" in ln.lower()):
            continue
        ok = False
        break

    if not ok:
        return text

    return "\n".join(lines[:last_idx]).rstrip()
