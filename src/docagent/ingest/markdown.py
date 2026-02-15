from __future__ import annotations

import re
from dataclasses import dataclass


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")


@dataclass(frozen=True)
class MdSection:
    heading_path: str
    start_line: int  # 1-based
    text: str


def split_sections(markdown_text: str) -> list[MdSection]:
    lines = markdown_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_heading_path = ""
    current_start_line = 1

    out: list[MdSection] = []

    def flush(end_line: int) -> None:
        nonlocal current_lines, current_heading_path, current_start_line
        text = "\n".join(current_lines).strip()
        if text:
            out.append(
                MdSection(
                    heading_path=current_heading_path,
                    start_line=current_start_line,
                    text=text,
                )
            )
        current_lines = []
        current_start_line = end_line

    for idx, line in enumerate(lines, start=1):
        m = _HEADING_RE.match(line)
        if m:
            # New section
            flush(idx)
            level = len(m.group(1))
            title = m.group(2).strip()

            # Adjust heading stack
            if level <= 0:
                level = 1
            if len(heading_stack) >= level:
                heading_stack = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(title)

            current_heading_path = " > ".join([h for h in heading_stack if h])
            current_start_line = idx
            continue

        current_lines.append(line)

    flush(len(lines) + 1)
    return out


def chunk_text(text: str, max_chars: int = 2500, overlap: int = 200) -> list[str]:
    t = text.strip()
    if len(t) <= max_chars:
        return [t] if t else []

    # Prefer splitting on blank lines.
    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []

    def buf_len() -> int:
        return sum(len(x) for x in buf) + max(0, len(buf) - 1) * 2

    for p in paras:
        if not buf:
            buf.append(p)
            continue

        if buf_len() + 2 + len(p) <= max_chars:
            buf.append(p)
            continue

        chunks.append("\n\n".join(buf).strip())

        # Overlap by tail chars
        tail = chunks[-1][-overlap:] if overlap > 0 else ""
        buf = [tail, p] if tail else [p]

    if buf:
        chunks.append("\n\n".join(buf).strip())

    # Final cleanup
    return [c for c in chunks if c.strip()]
