# Eval Sets

Eval sets are JSONL files with at least:

```json
{"question":"...","required_sources":["md:notes.md#L9"]}
```

Run:

```bash
docagent eval --db ./data/docs.db --eval ./eval/sample_eval.jsonl --k 8
docagent eval --db ./data/docs.db --eval ./eval/sample_eval.jsonl --k 8 --generate
```

Notes:
- `required_sources` should match the `source_ref` values created during ingest.
- Use `docagent search` and `docagent show` to discover correct source refs.

