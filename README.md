# Personal Docs Agent (Local-First RAG + Citations)

A local-first assistant for your **PDF + Notion-export + Markdown** knowledge base.

What you get:
- Ingest PDFs and Markdown (including Notion Markdown exports)
- Structure-aware chunking (headings + pages)
- Hybrid retrieval: SQLite FTS5 (lexical) + local embeddings (vector)
- Answers with **verifiable citations**
- Simple offline evaluation metrics (retrieval recall, citation coverage)
- Optional dataset export for LoRA/SFT (e.g., MLX fine-tuning)

## Quickstart

### 1) Create a virtualenv and install

```bash
cd personal-docs-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

To use the web UI:

```bash
pip install -e '.[web]'
```

### 2) Ingest docs into a local DB

```bash
docagent ingest --input /path/to/your/docs --db ./data/docs.db
```

Supported inputs:
- `*.pdf`
- `*.md`, `*.markdown`, `*.txt`

For Notion: export to **Markdown** (zip), unzip it, and point `--input` to that folder.

### 3) Build the search index

```bash
docagent index --db ./data/docs.db
```

This creates:
- SQLite FTS index inside the DB
- Embedding files next to the DB: `docs.db.embeddings.npy`, `docs.db.chunk_ids.npy`

### 4) Ask questions (local)

Default is Ollama on `http://localhost:11434`:

```bash
ollama pull llama3.2:1b
```

```bash
docagent ask --db ./data/docs.db "What did I write about attachment theory?"
```

Tips:
- If citations look broken, make sure your terminal supports plain brackets (we print with Rich markup disabled).
- Set `DOCAGENT_OLLAMA_TEMPERATURE=0.2` to reduce rambling / improve determinism.

## Web UI (DocAgent Studio)

Run:

```bash
docagent serve --db ./data/docs.db
```

Then open `http://127.0.0.1:8000`.

### Live (Camera + Voice)

In the UI, open the **Live** tab:
- **Camera**: Start camera, take a snapshot (stays local in your browser).
- **Voice**: Dictation uses your browser's SpeechRecognition (works best in Chromium-based browsers). "Speak" uses your browser's TTS.
- **Ask Snapshot**: requires a vision-capable Ollama model (example: `ollama pull llava:latest`) and sends the snapshot to `POST /api/vision`.

## Debugging

Check local dependencies:

```bash
docagent doctor --db ./data/docs.db
```

Inspect retrieval:

```bash
docagent search --db ./data/docs.db "secure base" --k 5
```

Show a specific chunk:

```bash
docagent show --db ./data/docs.db --source-ref "md:notes.md#L9"
```

## Knowledge Graph (GraphRAG-Style)

Build a lightweight entity co-occurrence graph:

```bash
docagent graph build --db ./data/docs.db
```

Explore entities and neighbors:

```bash
docagent graph query --db ./data/docs.db "Attachment"
```

## Evaluation

Create an eval set as JSONL:

```json
{"question":"...","answer":"...","required_sources":["pdf:foo.pdf#p12"]}
```

Run:

```bash
docagent eval --db ./data/docs.db --eval ./eval_set.jsonl
```

This repo includes a small starter set: `eval/sample_eval.jsonl`.

## Training (optional)

Export a simple instruction dataset (JSONL) for LoRA/SFT:

```bash
docagent make-trainset --db ./data/docs.db --out ./train.jsonl --n 500
```

For MLX LoRA fine-tuning on Apple Silicon, export a split dataset directory:

```bash
docagent make-trainset-dir --db ./data/docs.db --out-dir ./data/trainset --n 2000
```

Then fine-tune with MLX (example):

```bash
mlx_lm.lora \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --train \
  --data ./data/trainset \
  --iters 300 \
  --batch-size 1 \
  --learning-rate 1e-4 \
  --adapter-path ./data/adapters/docagent-lora \
  --grad-checkpoint
```

You can then run evaluation (`docagent eval --generate ...`) and swap the model used by `docagent ask` (Ollama model name) or add an MLX backend.

## Notes

- This project is designed to run on a MacBook Air M2 w/ 8GB RAM.
- Vector search is brute-force NumPy cosine similarity, which is fast enough for personal doc corpora.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Paper

See `docs/paper.md`.
