"""Load markdown docs -> clean -> chunk (~500 tokens, overlap) -> embed -> store in Qdrant.

Run:  python -m app.ingest
"""
import re
import uuid
from dataclasses import dataclass
from functools import lru_cache

import tiktoken
from qdrant_client.models import Distance, PointStruct, VectorParams

from app import config
from app.store import embed, get_client

@lru_cache(maxsize=1)
def _enc():
    return tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    doc_id: str      # relative path, e.g. "tutorial/query-params.md" (used as the citation + eval key)
    title: str
    chunk_index: int
    text: str


def clean_markdown(raw: str) -> str:
    text = re.sub(r"\s*\{\s*#[\w-]+\s*\}", "", raw)                      # heading anchors: "## Foo { #foo }"
    text = re.sub(r"\{\*\s*([^*]+?)\s*\*\}", r"[code example: \1]", text)  # mkdocs code includes
    text = re.sub(r"<[^>]+>", "", text)                                   # html tags (keep inner text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def doc_title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, flags=re.M)
    return m.group(1).strip() if m else fallback


def chunk_text(text: str, size: int = config.CHUNK_TOKENS, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    tokens = _enc().encode(text)
    if len(tokens) <= size:
        return [text]
    chunks, step = [], size - overlap
    for start in range(0, len(tokens), step):
        window = tokens[start:start + size]
        chunks.append(_enc().decode(window))
        if start + size >= len(tokens):
            break
    return chunks


def load_chunks() -> list[Chunk]:
    out: list[Chunk] = []
    for path in sorted(config.DOCS_DIR.rglob("*.md")):
        doc_id = path.relative_to(config.DOCS_DIR).as_posix()
        text = clean_markdown(path.read_text(encoding="utf-8"))
        if not text:
            continue
        title = doc_title(text, doc_id)
        for i, piece in enumerate(chunk_text(text)):
            out.append(Chunk(doc_id, title, i, piece))
    return out


def ingest() -> int:
    chunks = load_chunks()
    # Prefix the doc title so short chunks keep their topic when embedded.
    vectors = embed([f"{c.title}\n{c.text}" for c in chunks])

    client = get_client()
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        config.COLLECTION,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.doc_id}#{c.chunk_index}")),
            vector=v,
            payload={
                "doc_id": c.doc_id,
                "title": c.title,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "corpus_version": config.CORPUS_VERSION,
            },
        )
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(points), 256):
        client.upsert(config.COLLECTION, points[i:i + 256])
    n_docs = len({c.doc_id for c in chunks})
    print(f"Ingested {len(chunks)} chunks from {n_docs} docs into '{config.COLLECTION}' "
          f"(corpus_version={config.CORPUS_VERSION})")
    client.close()
    return len(chunks)


if __name__ == "__main__":
    ingest()
