from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Default DB path used by the web UI and CLI defaults.
    db_path: str = os.getenv("DOCAGENT_DB_PATH", "./data/docs.db")

    # Embeddings
    embed_model: str = os.getenv("DOCAGENT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

    # Ollama
    ollama_base_url: str = os.getenv("DOCAGENT_OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("DOCAGENT_OLLAMA_MODEL", "llama3.2:1b")
    ollama_temperature: float = float(os.getenv("DOCAGENT_OLLAMA_TEMPERATURE", "0.2"))
