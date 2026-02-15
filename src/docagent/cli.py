from __future__ import annotations

import json
import os
import random
import sqlite3
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .chat.llm import OllamaChatClient
from .chat.rag import answer_question
from .config import Settings
from .eval.run_eval import run_eval
from .index import build as index_build
from .index import sqlite_store
from .index import vector_store
from .index.embedder import Embedder
from .index.retriever import HybridRetriever
from .ingest.runner import IngestOptions, ingest_into_db


app = typer.Typer(add_completion=False, help="Personal Docs Agent: local-first RAG with citations.")
console = Console()

graph_app = typer.Typer(add_completion=False, help="Knowledge graph utilities (GraphRAG-style).")
app.add_typer(graph_app, name="graph")


@app.command()
def ingest(
    input: Path = typer.Option(..., "--input", exists=True, file_okay=False, dir_okay=True),
    db: Path = typer.Option(..., "--db", help="SQLite DB path to create/update"),
    notion_root: Path | None = typer.Option(None, "--notion-root", help="Mark markdown under this path as notion:"),
    max_chunk_chars: int = typer.Option(2500, help="Max chars per chunk"),
    overlap: int = typer.Option(200, help="Overlap chars between chunks"),
):
    """Ingest PDF + Markdown into a SQLite DB."""
    opts = IngestOptions(
        input_dir=input,
        notion_root=notion_root,
        max_chunk_chars=max_chunk_chars,
        overlap=overlap,
    )

    conn = sqlite_store.connect(db)
    try:
        res = ingest_into_db(conn=conn, db_path=str(db), options=opts)
    finally:
        conn.close()

    console.print(f"Documents seen: {res['documents_seen']}")
    console.print(f"Documents changed: {res['documents_changed']}")
    console.print(f"Chunks inserted: {res['chunks_inserted']}")
    console.print("Next: run `docagent index --db ...` to build search indexes.")


@app.command()
def index(
    db: Path = typer.Option(..., "--db", exists=True, file_okay=True, dir_okay=False),
    embed_model: str | None = typer.Option(None, help="Embedding model name (fastembed)"),
    batch_size: int = typer.Option(64, help="Embedding batch size"),
):
    """Build FTS + embeddings for retrieval."""
    settings = Settings()
    model = embed_model or settings.embed_model

    conn = sqlite_store.connect(db)
    try:
        sqlite_store.init_db(conn)
        result = index_build.build_indexes(conn=conn, db_path=str(db), embed_model=model, batch_size=batch_size)
    finally:
        conn.close()

    console.print(f"Indexed {result.num_chunks} chunks (dim={result.embedding_dim})")


@app.command()
def ask(
    db: Path = typer.Option(..., "--db", exists=True, file_okay=True, dir_okay=False),
    question: str = typer.Argument(...),
    k: int = typer.Option(8, help="Top-k chunks to retrieve"),
    model: str | None = typer.Option(None, "--model", help="Ollama model name"),
    base_url: str | None = typer.Option(None, "--base-url", help="Ollama base URL"),
    temperature: float | None = typer.Option(None, "--temperature", help="Ollama temperature (lower is more deterministic)"),
):
    """Ask a question and get a cited answer (defaults to Ollama)."""
    settings = Settings()
    ollama_model = model or settings.ollama_model
    ollama_url = base_url or settings.ollama_base_url

    conn = sqlite_store.connect(db)
    try:
        sqlite_store.init_db(conn)
        try:
            vindex = vector_store.load(db)
        except FileNotFoundError as e:
            console.print(str(e), style="red")
            console.print("Run: `docagent index --db ...`", style="yellow")
            raise typer.Exit(code=2)
        embedder = Embedder(settings.embed_model)
        retriever = HybridRetriever(conn=conn, db_path=str(db), embedder=embedder, vector_index=vindex)

        llm = OllamaChatClient(
            base_url=ollama_url,
            model=ollama_model,
            options={"temperature": float(temperature if temperature is not None else settings.ollama_temperature)},
        )
        ans = answer_question(retriever=retriever, llm=llm, question=question, k=k)
    finally:
        conn.close()

    # Rich treats [..] as markup by default; our citation format uses brackets.
    console.print(ans.text, markup=False)
    if ans.sources:
        console.print("\nSources:", markup=False)
        for s in ans.sources:
            console.print(f"- {s}", markup=False)


@app.command()
def doctor(
    db: Path | None = typer.Option(None, "--db", help="Optional DB path to check"),
    model: str | None = typer.Option(None, "--model", help="Ollama model name to check"),
    base_url: str | None = typer.Option(None, "--base-url", help="Ollama base URL"),
):
    """Check local dependencies (DB/index + Ollama) and print actionable fixes."""
    settings = Settings()
    ollama_model = model or settings.ollama_model
    ollama_url = (base_url or settings.ollama_base_url).rstrip("/")

    ok = True

    console.print("Ollama:")
    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        models = [m.get("name") for m in (data.get("models") or []) if isinstance(m, dict)]
        if not models:
            console.print(f"- Server reachable at {ollama_url} but no models are installed.", style="yellow")
            console.print(f"  Fix: `ollama pull {ollama_model}`", style="yellow")
            ok = False
        else:
            console.print(f"- Server reachable at {ollama_url} ({len(models)} model(s) installed).", style="green")
            if ollama_model not in models:
                console.print(f"- Missing model: {ollama_model}", style="yellow")
                console.print(f"  Fix: `ollama pull {ollama_model}`", style="yellow")
                ok = False
            else:
                console.print(f"- Model OK: {ollama_model}", style="green")
    except Exception as e:
        console.print(f"- Not reachable at {ollama_url}: {e}", style="red")
        console.print("  Fix: start Ollama (`ollama serve`) then retry.", style="yellow")
        ok = False

    if db is not None:
        console.print("\nDB/Index:")
        if not db.exists():
            console.print(f"- Missing DB: {db}", style="red")
            console.print("  Fix: run `docagent ingest --input ... --db ...`", style="yellow")
            ok = False
        else:
            conn = sqlite_store.connect(db)
            try:
                sqlite_store.init_db(conn)
                doc_n = int(conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"])
                chunk_n = int(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])
                console.print(f"- Documents: {doc_n}", style="green" if doc_n > 0 else "yellow")
                console.print(f"- Chunks: {chunk_n}", style="green" if chunk_n > 0 else "yellow")
                if chunk_n == 0:
                    console.print("  Fix: re-run ingest; no chunks were indexed.", style="yellow")
                    ok = False

                fts_n = int(conn.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()["n"])
                if fts_n == 0 and chunk_n > 0:
                    console.print("- FTS index empty.", style="yellow")
                    console.print("  Fix: run `docagent index --db ...`", style="yellow")
                    ok = False
                else:
                    console.print(f"- FTS rows: {fts_n}", style="green")
            finally:
                conn.close()

            ids_path, emb_path = vector_store.paths_for_db(db)
            if not ids_path.exists() or not emb_path.exists():
                console.print("- Vector index missing.", style="yellow")
                console.print("  Fix: run `docagent index --db ...`", style="yellow")
                ok = False
            else:
                console.print(f"- Vector index OK: {ids_path.name}, {emb_path.name}", style="green")

    if not ok:
        raise typer.Exit(code=1)


@app.command()
def search(
    db: Path = typer.Option(..., "--db", exists=True, file_okay=True, dir_okay=False),
    query: str = typer.Argument(...),
    k: int = typer.Option(10, help="Top-k chunks to show"),
    show_text: bool = typer.Option(False, "--show-text", help="Also print full chunk text"),
):
    """Debug retrieval: show the top retrieved chunks with scores."""
    settings = Settings()

    conn = sqlite_store.connect(db)
    try:
        sqlite_store.init_db(conn)
        vindex = vector_store.load(db)
        embedder = Embedder(settings.embed_model)
        retriever = HybridRetriever(conn=conn, db_path=str(db), embedder=embedder, vector_index=vindex)
        hits = retriever.retrieve(query, k=int(k))
    finally:
        conn.close()

    table = Table(title=f"Top {k} Chunks")
    table.add_column("#", justify="right", width=4)
    table.add_column("score", justify="right", width=8)
    table.add_column("source")
    table.add_column("preview")

    for i, h in enumerate(hits, start=1):
        preview = " ".join(h.text.split())
        if len(preview) > 220:
            preview = preview[:220].rstrip() + "..."
        table.add_row(
            Text(str(i)),
            Text(f"{h.score:.3f}"),
            Text(h.source_ref),
            Text(preview),
        )

    console.print(table)

    if show_text:
        for h in hits:
            console.print("\n" + "=" * 80, markup=False)
            console.print(h.source_ref, markup=False, style="bold")
            console.print(h.text, markup=False)


@app.command()
def show(
    db: Path = typer.Option(..., "--db", exists=True, file_okay=True, dir_okay=False),
    source_ref: str | None = typer.Option(None, "--source-ref", help="Exact source ref to show"),
    chunk_id: int | None = typer.Option(None, "--chunk-id", help="Chunk id to show"),
):
    """Show a stored chunk by source_ref or chunk_id."""
    if source_ref is None and chunk_id is None:
        raise typer.BadParameter("Provide --source-ref or --chunk-id")

    conn = sqlite_store.connect(db)
    try:
        sqlite_store.init_db(conn)
        rows = []
        if chunk_id is not None:
            r = conn.execute(
                "SELECT chunk_id, source_ref, heading, page_start, page_end, text FROM chunks WHERE chunk_id = ?",
                (int(chunk_id),),
            ).fetchone()
            if r is not None:
                rows = [r]
        else:
            rows = sqlite_store.get_chunks_by_source_ref(conn, str(source_ref))
    finally:
        conn.close()

    if not rows:
        console.print("No matching chunks found.", style="yellow")
        raise typer.Exit(code=2)

    for r in rows:
        console.print("=" * 80, markup=False)
        console.print(f"chunk_id: {r['chunk_id']}", markup=False)
        console.print(f"source_ref: {r['source_ref']}", markup=False)
        if r["heading"] is not None:
            console.print(f"heading: {r['heading']}", markup=False)
        if r["page_start"] is not None:
            console.print(f"pages: {r['page_start']}-{r['page_end']}", markup=False)
        console.print("")
        console.print(str(r["text"]), markup=False)


@app.command()
def serve(
    db: Path = typer.Option(Path("./data/docs.db"), "--db", help="Default DB path for the server"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev only)"),
):
    """Run the DocAgent Studio web UI (FastAPI)."""
    try:
        import uvicorn
    except Exception:
        console.print("Missing web dependencies. Install: `pip install -e '.[web]'`", style="red")
        raise typer.Exit(code=2)

    from .web.server import create_app

    app_ = create_app(default_db_path=str(db))
    uvicorn.run(app_, host=host, port=int(port), reload=bool(reload))


@app.command()
def stats(
    db: Path = typer.Option(..., "--db", exists=True, file_okay=True, dir_okay=False),
):
    """Show corpus stats."""
    conn = sqlite_store.connect(db)
    try:
        doc_n = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunk_n = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        by_type = conn.execute("SELECT source_type, COUNT(*) AS n FROM documents GROUP BY source_type").fetchall()
    finally:
        conn.close()

    table = Table(title="Docagent Stats")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Documents", str(doc_n))
    table.add_row("Chunks", str(chunk_n))
    console.print(table)

    if by_type:
        t2 = Table(title="Documents by Type")
        t2.add_column("source_type")
        t2.add_column("count")
        for r in by_type:
            t2.add_row(str(r["source_type"]), str(r["n"]))
        console.print(t2)


@app.command("make-trainset")
def make_trainset(
    db: Path = typer.Option(..., "--db", exists=True, file_okay=True, dir_okay=False),
    out: Path = typer.Option(..., "--out", help="Output JSONL path"),
    n: int = typer.Option(500, help="Number of examples"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Export a simple instruction JSONL for LoRA/SFT (citation style)."""
    rng = random.Random(seed)

    conn = sqlite_store.connect(db)
    try:
        rows = list(conn.execute("SELECT chunk_id, source_ref, heading, text FROM chunks WHERE LENGTH(text) > 200"))
    finally:
        conn.close()

    if not rows:
        raise typer.BadParameter("No chunks found. Run ingest first.")

    out.parent.mkdir(parents=True, exist_ok=True)

    sys_msg = (
        "You are a personal knowledge base assistant. "
        "Given sources with ids, answer the user's question using only those sources and cite using [SOURCE_ID]."
    )

    with out.open("w", encoding="utf-8") as f:
        for _ in range(int(n)):
            row = rng.choice(rows)
            source_ref = str(row["source_ref"])
            heading = (str(row["heading"]) if row["heading"] is not None else "").strip()
            text = str(row["text"]).strip()

            question = _make_question(heading)
            answer = _extractive_answer(text, source_ref)

            user = f"Question: {question}\n\nSources:\nSOURCE {source_ref}\n{text}\n"

            record = {
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    console.print(f"Wrote {n} examples to {out}")


@app.command("make-trainset-dir")
def make_trainset_dir(
    db: Path = typer.Option(..., "--db", exists=True, file_okay=True, dir_okay=False),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output directory containing train/valid/test.jsonl"),
    n: int = typer.Option(2000, help="Number of examples"),
    seed: int = typer.Option(42, help="Random seed"),
    valid_frac: float = typer.Option(0.1, help="Validation fraction (0-1)"),
    test_frac: float = typer.Option(0.0, help="Test fraction (0-1)"),
):
    """Export train/valid/test.jsonl (chat format) suitable for MLX LoRA fine-tuning."""
    rng = random.Random(seed)
    valid_frac = float(valid_frac)
    test_frac = float(test_frac)

    if valid_frac <= 0.0 or valid_frac >= 1.0:
        raise typer.BadParameter("--valid-frac must be between 0 and 1 (non-zero).")
    if test_frac < 0.0 or test_frac >= 1.0:
        raise typer.BadParameter("--test-frac must be between 0 and 1.")
    if valid_frac + test_frac >= 1.0:
        raise typer.BadParameter("--valid-frac + --test-frac must be < 1.0.")

    conn = sqlite_store.connect(db)
    try:
        rows = list(conn.execute("SELECT chunk_id, source_ref, heading, text FROM chunks WHERE LENGTH(text) > 200"))
    finally:
        conn.close()

    if not rows:
        raise typer.BadParameter("No chunks found. Run ingest first.")

    sys_msg = (
        "You are a personal knowledge base assistant. "
        "Given sources with ids, answer the user's question using only those sources and cite using [SOURCE_ID]."
    )

    records = []
    for _ in range(int(n)):
        row = rng.choice(rows)
        source_ref = str(row["source_ref"])
        heading = (str(row["heading"]) if row["heading"] is not None else "").strip()
        text = str(row["text"]).strip()

        question = _make_question(heading)
        answer = _extractive_answer(text, source_ref)
        user = f"Question: {question}\n\nSources:\nSOURCE {source_ref}\n{text}\n"
        records.append(
            {
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ]
            }
        )

    rng.shuffle(records)
    n_total = len(records)
    n_valid = max(1, int(round(n_total * valid_frac)))
    n_test = int(round(n_total * test_frac))
    n_train = n_total - n_valid - n_test
    if n_train <= 0:
        raise typer.BadParameter("Not enough samples left for train split; reduce valid/test fractions or increase n.")

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": records[:n_train],
        "valid": records[n_train : n_train + n_valid],
        "test": records[n_train + n_valid :],
    }

    for split_name, split_records in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in split_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    console.print(f"Wrote {n_train} train, {n_valid} valid, {n_test} test examples to {out_dir}")


@app.command()
def eval(
    db: Path = typer.Option(..., "--db", exists=True, file_okay=True, dir_okay=False),
    eval: Path = typer.Option(..., "--eval", exists=True, file_okay=True, dir_okay=False),
    k: int = typer.Option(8, help="Top-k retrieval"),
    generate: bool = typer.Option(False, help="Also generate answers (needs Ollama running)"),
    model: str | None = typer.Option(None, "--model", help="Ollama model"),
    base_url: str | None = typer.Option(None, "--base-url", help="Ollama base URL"),
    temperature: float | None = typer.Option(None, "--temperature", help="Ollama temperature"),
):
    """Evaluate retrieval (and optionally end-to-end citations)."""
    settings = Settings()

    conn = sqlite_store.connect(db)
    try:
        vindex = vector_store.load(db)
        embedder = Embedder(settings.embed_model)
        retriever = HybridRetriever(conn=conn, db_path=str(db), embedder=embedder, vector_index=vindex)

        llm = None
        if generate:
            llm = OllamaChatClient(
                base_url=(base_url or settings.ollama_base_url),
                model=(model or settings.ollama_model),
                options={"temperature": float(temperature if temperature is not None else settings.ollama_temperature)},
            )

        summary = run_eval(retriever=retriever, eval_path=eval, k=k, generate=generate, llm=llm)
    finally:
        conn.close()

    console.print(f"Examples: {summary.n}")
    console.print(f"Retrieval recall@{k}: {summary.retrieval_recall:.3f}")
    if summary.citation_coverage is not None:
        console.print(f"Citation coverage: {summary.citation_coverage:.3f}")


@graph_app.command("build")
def graph_build(
    db: Path = typer.Option(..., "--db", exists=True, file_okay=True, dir_okay=False),
    clear: bool = typer.Option(True, "--clear/--no-clear", help="Clear any existing graph tables first"),
    min_chars: int = typer.Option(3, help="Minimum entity length"),
    max_per_chunk: int = typer.Option(25, help="Max entities extracted per chunk"),
):
    """Build an entity co-occurrence graph from your chunks."""
    from .graph.build import build_graph

    conn = sqlite_store.connect(db)
    try:
        res = build_graph(conn=conn, clear=bool(clear), min_chars=int(min_chars), max_per_chunk=int(max_per_chunk))
    finally:
        conn.close()

    for k, v in res.items():
        console.print(f"{k}: {v}", markup=False)


@graph_app.command("query")
def graph_query(
    db: Path = typer.Option(..., "--db", exists=True, file_okay=True, dir_okay=False),
    query: str = typer.Argument(...),
    entity_limit: int = typer.Option(5, help="Max entities to return"),
    neighbor_limit: int = typer.Option(8, help="Max neighbors per entity"),
    chunk_limit: int = typer.Option(5, help="Max example chunks per entity"),
):
    """Explore the graph: match entities, show neighbors, show example chunks."""
    from .graph.query import query_graph

    conn = sqlite_store.connect(db)
    try:
        res = query_graph(
            conn=conn,
            query=query,
            entity_limit=int(entity_limit),
            neighbor_limit=int(neighbor_limit),
            chunk_limit=int(chunk_limit),
        )
    finally:
        conn.close()

    console.print(f"terms: {', '.join(res['terms'])}", markup=False)
    for item in res["entities"]:
        ent = item["entity"]
        console.print("\n" + "=" * 80, markup=False)
        console.print(f"{ent['name']} (chunks={ent['chunk_count']}, mentions={ent['mention_count']})", markup=False, style="bold")
        if item["neighbors"]:
            console.print("neighbors:", markup=False)
            for n in item["neighbors"]:
                console.print(f"- {n['entity']['name']} (w={n['weight']})", markup=False)
        if item["chunks"]:
            console.print("chunks:", markup=False)
            for c in item["chunks"]:
                console.print(f"- {c['source_ref']} :: {c['preview']}", markup=False)


def _make_question(heading: str) -> str:
    if heading:
        return f"What are the key points in the section '{heading}'?"
    return "Summarize the key points in the provided excerpt."


def _extractive_answer(text: str, source_ref: str) -> str:
    # Simple: take first 2 sentences-ish.
    snippet = _first_sentences(text, n=2, max_chars=500)
    return f"{snippet} [{source_ref}]"


def _first_sentences(text: str, n: int, max_chars: int) -> str:
    t = " ".join(text.split())
    if len(t) <= max_chars:
        return t

    # naive sentence split
    parts = []
    buf = ""
    for ch in t:
        buf += ch
        if ch in ".!?":
            parts.append(buf.strip())
            buf = ""
            if len(parts) >= n:
                break
    if not parts:
        return t[:max_chars].rstrip()

    out = " ".join(parts)
    return out[:max_chars].rstrip()


if __name__ == "__main__":
    app()
