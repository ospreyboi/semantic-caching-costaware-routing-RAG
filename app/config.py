"""Central config. Everything is overridable via environment variables (or a .env file)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Corpus
DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT / "data" / "docs"))
CORPUS_VERSION = os.getenv("CORPUS_VERSION", "v1")  # bump whenever docs are re-ingested

# Chunking (tokens counted with tiktoken cl100k_base)
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 75))

# Embeddings + vector store
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
QDRANT_PATH = os.getenv("QDRANT_PATH", str(ROOT / "qdrant_data"))  # local, file-based Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "")  # set to use a Qdrant server instead of local files
COLLECTION = os.getenv("COLLECTION", "docs")
TOP_K = int(os.getenv("TOP_K", 5))

# LLMs (any LiteLLM model string: "gemini/gemini-2.5-flash", "gpt-4o-mini", "groq/llama-3.1-8b-instant", ...)
LLM_MODEL = os.getenv("LLM_MODEL", "groq/openai/gpt-oss-120b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", 2000))

# Fallback price table (USD per 1M tokens) used when LiteLLM doesn't know a model's price.
PRICES = {
    "groq/openai/gpt-oss-120b": {"in": 0.15, "out": 0.60},
    "groq/openai/gpt-oss-20b": {"in": 0.075, "out": 0.30},
    "gemini/gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    "gemini/gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "groq/llama-3.1-8b-instant": {"in": 0.05, "out": 0.08},
    "groq/llama-3.3-70b-versatile": {"in": 0.59, "out": 0.79},
}
