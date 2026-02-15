import time
import zipfile
from pathlib import Path
from typing import Any

import sqlite3


def create_app(*, default_db_path: str | None = None):
    # Lazy import so core CLI works without web deps.
    from fastapi import FastAPI, File, Form, UploadFile, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    from ..chat.llm import ChatMessage, LLMError, OllamaChatClient
    from ..chat.rag import answer_question
    from ..config import Settings
    from ..graph.build import build_graph
    from ..graph.query import query_graph
    from ..index import build as index_build
    from ..index import sqlite_store, vector_store
    from ..index.embedder import Embedder
    from ..index.retriever import HybridRetriever
    from ..ingest.runner import IngestOptions, ingest_into_db

    settings = Settings()
    db_default = default_db_path or getattr(settings, "db_path", "./data/docs.db")

    base = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(base / "templates"))

    app = FastAPI(title="DocAgent Studio", version="0.1.1")
    app.mount("/static", StaticFiles(directory=str(base / "static")), name="static")

    def _open_db(db_path: str) -> sqlite3.Connection:
        conn = sqlite_store.connect(db_path)
        sqlite_store.init_db(conn)
        return conn

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "db_path": db_default,
                "embed_model": settings.embed_model,
                "ollama_base_url": settings.ollama_base_url,
                "ollama_model": settings.ollama_model,
                "ollama_temperature": settings.ollama_temperature,
            },
        )

    @app.get("/api/health")
    def health(base_url: str | None = None):
        import httpx

        url = (base_url or settings.ollama_base_url).rstrip("/")
        out: dict[str, Any] = {"ollama_base_url": url, "ollama_ok": False, "models": []}
        try:
            r = httpx.get(f"{url}/api/tags", timeout=3.0)
            r.raise_for_status()
            data = r.json()
            models = [m.get("name") for m in (data.get("models") or []) if isinstance(m, dict)]
            out["models"] = models
            out["ollama_ok"] = True
        except Exception as e:
            out["error"] = str(e)
        return out

    @app.post("/api/ingest")
    async def ingest(
        db_path: str = Form(...),
        input_dir: str = Form(""),
        notion_root: str = Form(""),
        max_chunk_chars: int = Form(2500),
        overlap: int = Form(200),
        files: list[UploadFile] = File(default=[]),
    ):
        # Option A: ingest from a local path (server-side).
        if input_dir.strip():
            in_dir = Path(input_dir).expanduser().resolve()
            if not in_dir.exists() or not in_dir.is_dir():
                return JSONResponse({"ok": False, "error": f"input_dir not found: {in_dir}"}, status_code=400)
            nr = Path(notion_root).expanduser().resolve() if notion_root.strip() else None
            opts = IngestOptions(
                input_dir=in_dir,
                notion_root=nr,
                max_chunk_chars=int(max_chunk_chars),
                overlap=int(overlap),
            )
        else:
            # Option B: ingest uploaded files (PDF/MD/zip).
            if not files:
                return JSONResponse({"ok": False, "error": "Provide input_dir or upload at least one file."}, status_code=400)

            upload_root = Path("./data/uploads").resolve()
            upload_dir = upload_root / time.strftime("%Y%m%d-%H%M%S")
            upload_dir.mkdir(parents=True, exist_ok=True)

            for f in files:
                name = Path(f.filename or "upload.bin").name
                dest = upload_dir / name
                data = await f.read()
                dest.write_bytes(data)

                if dest.suffix.lower() == ".zip":
                    try:
                        with zipfile.ZipFile(dest, "r") as zf:
                            zf.extractall(upload_dir)
                    except Exception as e:
                        return JSONResponse({"ok": False, "error": f"Failed to unzip {name}: {e}"}, status_code=400)

            nr = Path(notion_root).expanduser().resolve() if notion_root.strip() else None
            opts = IngestOptions(
                input_dir=upload_dir,
                notion_root=nr,
                max_chunk_chars=int(max_chunk_chars),
                overlap=int(overlap),
            )

        conn = _open_db(db_path)
        try:
            res = ingest_into_db(conn=conn, db_path=str(db_path), options=opts)
            conn.commit()
        finally:
            conn.close()

        return {"ok": True, **res}

    @app.post("/api/index")
    def index(
        payload: dict[str, Any],
    ):
        db_path = str(payload.get("db_path") or db_default)
        embed_model = str(payload.get("embed_model") or settings.embed_model)
        batch_size = int(payload.get("batch_size") or 64)

        conn = _open_db(db_path)
        try:
            res = index_build.build_indexes(conn=conn, db_path=db_path, embed_model=embed_model, batch_size=batch_size)
        finally:
            conn.close()

        return {"ok": True, "num_chunks": res.num_chunks, "embedding_dim": res.embedding_dim}

    @app.post("/api/stats")
    def stats(payload: dict[str, Any]):
        db_path = str(payload.get("db_path") or db_default)
        conn = _open_db(db_path)
        try:
            doc_n = int(conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"])
            chunk_n = int(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])
            fts_n = int(conn.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()["n"])
        finally:
            conn.close()
        ids_path, emb_path = vector_store.paths_for_db(db_path)
        return {
            "ok": True,
            "documents": doc_n,
            "chunks": chunk_n,
            "fts_rows": fts_n,
            "vector_index": {"chunk_ids": str(ids_path), "embeddings": str(emb_path), "exists": ids_path.exists() and emb_path.exists()},
        }

    @app.post("/api/search")
    def search(payload: dict[str, Any]):
        db_path = str(payload.get("db_path") or db_default)
        query = str(payload.get("query") or "").strip()
        k = int(payload.get("k") or 8)
        if not query:
            return JSONResponse({"ok": False, "error": "query is required"}, status_code=400)

        conn = _open_db(db_path)
        try:
            vindex = vector_store.load(db_path)
            embedder = Embedder(settings.embed_model)
            retriever = HybridRetriever(conn=conn, db_path=db_path, embedder=embedder, vector_index=vindex)
            hits = retriever.retrieve(query, k=k)
        finally:
            conn.close()

        out = []
        for h in hits:
            preview = " ".join(h.text.split())
            if len(preview) > 260:
                preview = preview[:260].rstrip() + "..."
            out.append({"source_ref": h.source_ref, "score": h.score, "preview": preview, "chunk_id": h.chunk_id})
        return {"ok": True, "hits": out}

    @app.get("/api/chunk")
    def chunk(db_path: str, source_ref: str | None = None, chunk_id: int | None = None):
        if not source_ref and chunk_id is None:
            return JSONResponse({"ok": False, "error": "Provide source_ref or chunk_id"}, status_code=400)

        conn = _open_db(db_path)
        try:
            if chunk_id is not None:
                r = conn.execute(
                    "SELECT chunk_id, source_ref, heading, page_start, page_end, text FROM chunks WHERE chunk_id=?",
                    (int(chunk_id),),
                ).fetchone()
                rows = [r] if r is not None else []
            else:
                rows = sqlite_store.get_chunks_by_source_ref(conn, str(source_ref))
        finally:
            conn.close()

        if not rows:
            return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)

        r0 = rows[0]
        return {
            "ok": True,
            "chunk": {
                "chunk_id": int(r0["chunk_id"]),
                "source_ref": str(r0["source_ref"]),
                "heading": (str(r0["heading"]) if r0["heading"] is not None else None),
                "page_start": r0["page_start"],
                "page_end": r0["page_end"],
                "text": str(r0["text"]),
            },
        }

    @app.post("/api/ask")
    def ask(payload: dict[str, Any]):
        db_path = str(payload.get("db_path") or db_default)
        question = str(payload.get("question") or "").strip()
        k = int(payload.get("k") or 8)
        model = str(payload.get("model") or settings.ollama_model)
        base_url = str(payload.get("base_url") or settings.ollama_base_url)
        temperature = float(payload.get("temperature") if payload.get("temperature") is not None else settings.ollama_temperature)
        if not question:
            return JSONResponse({"ok": False, "error": "question is required"}, status_code=400)

        conn = _open_db(db_path)
        try:
            vindex = vector_store.load(db_path)
            embedder = Embedder(settings.embed_model)
            retriever = HybridRetriever(conn=conn, db_path=db_path, embedder=embedder, vector_index=vindex)
            llm = OllamaChatClient(base_url=base_url, model=model, options={"temperature": temperature})
            ans = answer_question(retriever=retriever, llm=llm, question=question, k=k)
        except FileNotFoundError as e:
            return JSONResponse({"ok": False, "error": str(e), "hint": "Run indexing first."}, status_code=400)
        except LLMError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
        finally:
            conn.close()

        return {"ok": True, "answer": ans.text, "sources": ans.sources}

    @app.post("/api/vision")
    def vision(payload: dict[str, Any]):
        question = str(payload.get("question") or "").strip()
        model = str(payload.get("model") or settings.ollama_model)
        base_url = str(payload.get("base_url") or settings.ollama_base_url)
        temperature = float(payload.get("temperature") if payload.get("temperature") is not None else settings.ollama_temperature)
        image_b64 = payload.get("image_b64")

        if not question:
            return JSONResponse({"ok": False, "error": "question is required"}, status_code=400)
        if not isinstance(image_b64, str) or not image_b64.strip():
            return JSONResponse({"ok": False, "error": "image_b64 is required (base64 jpeg/png bytes, without data: prefix)"}, status_code=400)

        try:
            llm = OllamaChatClient(base_url=base_url, model=model, options={"temperature": temperature})
            msgs = [
                ChatMessage(
                    role="system",
                    content="You are a helpful assistant. Answer the user's question about the provided image. If you are unsure, say so.",
                ),
                ChatMessage(role="user", content=question, images=[image_b64.strip()]),
            ]
            ans = llm.chat(msgs)
        except LLMError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=502)

        return {"ok": True, "answer": ans}

    @app.post("/api/graph/build")
    def graph_build(payload: dict[str, Any]):
        db_path = str(payload.get("db_path") or db_default)
        clear = bool(payload.get("clear", True))
        min_chars = int(payload.get("min_chars") or 3)
        max_per_chunk = int(payload.get("max_per_chunk") or 25)

        conn = _open_db(db_path)
        try:
            res = build_graph(conn=conn, clear=clear, min_chars=min_chars, max_per_chunk=max_per_chunk)
        finally:
            conn.close()
        return {"ok": True, "stats": res}

    @app.get("/api/graph/query")
    def graph_query(db_path: str, q: str, entity_limit: int = 5):
        conn = _open_db(db_path)
        try:
            res = query_graph(conn=conn, query=q, entity_limit=int(entity_limit))
        finally:
            conn.close()
        return {"ok": True, "result": res}

    return app
