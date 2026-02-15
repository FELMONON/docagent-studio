#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke test for the full pipeline (ingest -> index -> eval).
# Optional: add `--generate` to the eval command if Ollama is running.

DB_PATH="${1:-./data/smoke/docs.db}"

mkdir -p "$(dirname "$DB_PATH")"

docagent ingest --input ./example_data --notion-root ./example_data/notion_export --db "$DB_PATH"
docagent index --db "$DB_PATH"
docagent eval --db "$DB_PATH" --eval ./eval/sample_eval.jsonl --k 8

echo "OK: smoke complete"

