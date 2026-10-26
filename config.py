from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RSC_DATA_DIR", BASE_DIR / "data"))
DUCKDB_PATH = DATA_DIR / "market.duckdb"
QDRANT_PATH = DATA_DIR / "qdrant"
SEC_DIR = DATA_DIR / "sec"

for _p in (DATA_DIR, QDRANT_PATH, SEC_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# SEC EDGAR requires a declared User-Agent (name + email) on every request.
SEC_COMPANY_NAME = os.getenv("SEC_COMPANY_NAME", "RetailStockCopilot")
SEC_EMAIL = os.getenv("SEC_EMAIL", "you@example.com")

PRICE_HISTORY_PERIOD = "5y"

# Embeddings / vector store
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QDRANT_COLLECTION = "tenk_chunks"
CHUNK_WORDS = 280        # ~370 tokens, safely inside bge-small's 512-token window
CHUNK_OVERLAP = 50
MIN_CHUNK_WORDS = 30

# Local LLM
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_LLM = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_NUM_CTX = 8192    # Ollama's default context is small; RAG prompts would be silently truncated

COLORS = {
    "bg": "#0B0E14",
    "panel": "#121722",
    "grid": "#1A2234",
    "amber": "#FF9E00",
    "green": "#00E676",
    "red": "#FF1744",
    "text": "#C9D1D9",
    "muted": "#6B7A99",
    "cyan": "#29B6F6",
}