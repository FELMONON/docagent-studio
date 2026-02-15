# Contributing

This repo is optimized for a simple local-first workflow.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[web]"
```

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Project Layout

- `src/docagent/`: core library + CLI
- `src/docagent/web/`: FastAPI web UI (DocAgent Studio)
- `docs/`: project paper and notes
- `eval/`: example eval sets

## Pull Requests

- Keep changes small and focused.
- Add or update tests for bug fixes.
- Avoid committing corpora or generated DB/index files (see `.gitignore`).

