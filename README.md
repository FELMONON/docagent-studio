# DocAgent Studio

**Local-first RAG with verifiable citations — offline, no API keys, designed for knowledge workers.**

DocAgent Studio is a retrieval-augmented generation engine that runs entirely on your machine. It ingests your PDFs, Markdown, and Notion exports into a local SQLite database, retrieves context using hybrid lexical + vector search, and generates answers with verifiable, source-referenced citations — all without sending a single byte to the cloud.

---

## Why not LangChain / LlamaIndex?

| | DocAgent Studio | Typical RAG frameworks |
|---|---|---|
| **Runs offline** | Yes — SQLite + local embeddings + Ollama | Usually requires cloud APIs |
| **Verifiable citations** | Every sentence cites a traceable `source_ref` (e.g. `md:notes.md#L9`, `pdf:paper.pdf#p3`) that you can inspect with `docagent show` | Citations are often just "chunk text" with no stable reference |
| **Retrieval** | Hybrid — SQLite FTS5 (lexical) fused with cosine-similarity embeddings via tunable alpha | Typically vector-only or requires external services |
| **No wrapper tax** | Single `pip install`, zero config, no YAML chains, no API keys | Framework overhead, plugin systems, cloud key management |
| **Self-correcting grounding** | LLM output is validated against retrieved sources; ungrounded answers trigger a correction pass or fall back to extractive quotes | Trust the model output as-is |

DocAgent Studio is not a framework — it is a **complete, opinionated RAG engine** built for people who want to search their own documents and get answers they can verify.

---

## Key Features

### Hybrid Retrieval (SQLite FTS5 + Embeddings)
Queries hit both a full-text search index (FTS5, inside the same SQLite DB) and a local embedding index (fastembed + NumPy cosine similarity). Scores are fused with a configurable alpha weight, so you get the precision of keyword matching and the recall of semantic search in a single pass.

### Verifiable Citations with Source-Ref Tracking
Every chunk in the database has a stable `source_ref` — e.g. `pdf:report.pdf#p7` or `md:notes.md#L42`. The LLM is required to cite these refs inline. You can verify any citation instantly:
```bash
docagent show --db ./data/docs.db --source-ref "md:notes.md#L9"
```

### GraphRAG with Entity Co-occurrence
Build a lightweight knowledge graph from your corpus. Entities are extracted from chunks and linked by co-occurrence, enabling graph-based exploration of your documents:
```bash
docagent graph build --db ./data/docs.db
docagent graph query --db ./data/docs.db "Attachment"
```

### Built-in Evaluation Metrics
Measure retrieval recall and citation coverage against a ground-truth eval set — no external tools needed:
```bash
docagent eval --db ./data/docs.db --eval ./eval/sample_eval.jsonl
```

### Production CLI
A single `docagent` command covers the full workflow: `ingest`, `index`, `ask`, `search`, `show`, `eval`, `graph`, `stats`, `doctor`, and `serve`.

---

## Architecture

```
                         +----------------+
                         |  User Query    |
                         +-------+--------+
                                 |
                    +------------+------------+
                    |                         |
              +-----v------+         +-------v-------+
              |  FTS5       |         |  Embeddings   |
              |  (lexical)  |         |  (vector)     |
              +-----+-------+         +-------+-------+
                    |                         |
                    +------------+------------+
                                 |
                      +----------v----------+
                      |  Score Fusion       |
                      |  a*vec + (1-a)*lex  |
                      +----------+----------+
                                 |
                      +----------v----------+
                      |  Top-K Chunks       |
                      |  w/ source_refs     |
                      +----------+----------+
                                 |
                      +----------v----------+
                      |  Ollama LLM         |
                      |  (local inference)  |
                      +----------+----------+
                                 |
                      +----------v----------+
                      |  Grounding Check    |
                      |  + Self-Correction  |
                      +----------+----------+
                                 |
                      +----------v----------+
                      |  Cited Answer       |
                      |  [source_ref]       |
                      +---------------------+
```

All data lives in a single SQLite database + two NumPy sidecar files. No external services required.

---

## Quick Start

### 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

For the web UI: `pip install -e '.[web]'`

### 2. Ingest your documents

```bash
docagent ingest --input /path/to/your/docs --db ./data/docs.db
```

Supports `*.pdf`, `*.md`, `*.markdown`, `*.txt`. For Notion exports, unzip the Markdown export and point `--input` at the folder.

### 3. Build the search index

```bash
docagent index --db ./data/docs.db
```

### 4. Ask a question

```bash
ollama pull llama3.2:1b  # one-time setup
docagent ask --db ./data/docs.db "What did I write about attachment theory?"
```

### 5. Verify a citation

```bash
docagent show --db ./data/docs.db --source-ref "md:notes.md#L9"
```

---

## Web UI

```bash
docagent serve --db ./data/docs.db
# Open http://127.0.0.1:8000
```

---

## CLI Reference

| Command | Description |
|---|---|
| `docagent ingest` | Ingest PDFs + Markdown into a local SQLite DB |
| `docagent index` | Build FTS5 + embedding indexes |
| `docagent ask` | Ask a question and get a cited answer |
| `docagent search` | Debug retrieval — show top-K chunks with scores |
| `docagent show` | Inspect a specific chunk by `source_ref` or `chunk_id` |
| `docagent eval` | Evaluate retrieval recall and citation coverage |
| `docagent graph build` | Build entity co-occurrence graph |
| `docagent graph query` | Explore entities and neighbors |
| `docagent stats` | Show corpus statistics |
| `docagent doctor` | Check local dependencies and print fixes |
| `docagent serve` | Launch the web UI |
| `docagent make-trainset` | Export instruction JSONL for LoRA/SFT |
| `docagent make-trainset-dir` | Export train/valid/test splits for MLX LoRA |

---

## Debugging

```bash
docagent doctor --db ./data/docs.db    # check Ollama + DB health
docagent search --db ./data/docs.db "secure base" --k 5   # inspect retrieval
```

---

## Knowledge Graph (GraphRAG)

```bash
docagent graph build --db ./data/docs.db
docagent graph query --db ./data/docs.db "Attachment"
```

Builds a lightweight entity co-occurrence graph stored in SQLite. Entities are extracted via NLP heuristics and linked by shared chunk presence.

---

## Evaluation

Create an eval set as JSONL:

```json
{"question":"...","answer":"...","required_sources":["pdf:foo.pdf#p12"]}
```

Run:

```bash
docagent eval --db ./data/docs.db --eval ./eval/sample_eval.jsonl
```

A starter eval set is included: `eval/sample_eval.jsonl`.

---

## Training Data Export (Optional)

Export instruction datasets for LoRA/SFT fine-tuning:

```bash
# Single JSONL file
docagent make-trainset --db ./data/docs.db --out ./train.jsonl --n 500

# Train/valid/test splits (e.g., for MLX LoRA)
docagent make-trainset-dir --db ./data/docs.db --out-dir ./data/trainset --n 2000
```

Example MLX LoRA fine-tuning on Apple Silicon:

```bash
mlx_lm.lora \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --train --data ./data/trainset \
  --iters 300 --batch-size 1 --learning-rate 1e-4 \
  --adapter-path ./data/adapters/docagent-lora \
  --grad-checkpoint
```

---

## Design Decisions

- **SQLite as the single source of truth** — chunks, FTS index, documents table, and graph all live in one `.db` file. Portable, inspectable, no server.
- **Brute-force NumPy cosine similarity** — fast enough for personal corpora (thousands of chunks). No FAISS/Chroma dependency.
- **Self-correcting grounding loop** — if the LLM produces ungrounded citations or URLs, a correction pass fires automatically. If that also fails, the system returns extractive quotes with citations rather than hallucinated text.
- **Runs on a MacBook Air M2 with 8 GB RAM.**

---

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

---

## Paper

See `docs/paper.md`.

---

## License

MIT
