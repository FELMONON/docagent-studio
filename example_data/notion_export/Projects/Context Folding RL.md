# Context Folding RL

Context folding is a strategy where an agent writes code to manage context limits by chunking and delegating sub-tasks to helper calls.

## Why It Matters

It enables working with inputs larger than a model's context window by reducing the "read everything" requirement.

## Practical Pattern

- Split large documents into chunks.
- Run extraction or summarization over chunks.
- Aggregate results and verify before answering.

