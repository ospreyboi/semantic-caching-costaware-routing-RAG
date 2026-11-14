"""Embedding model + Qdrant client, both loaded once and shared."""
from functools import lru_cache

from qdrant_client import QdrantClient

from app import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    vecs = _model().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=len(texts) > 200)
    return [v.tolist() for v in vecs]


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    return QdrantClient(path=config.QDRANT_PATH)
