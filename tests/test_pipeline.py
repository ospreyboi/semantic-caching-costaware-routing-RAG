"""Fast tests that need no model downloads or API keys.  Run: pytest -q"""
import json
from pathlib import Path

from app.ingest import chunk_text, clean_markdown, load_chunks, _enc
from app.rag import cited_sources

ROOT = Path(__file__).resolve().parent.parent


def test_clean_markdown_strips_anchors_and_includes():
    raw = "# Title { #title }\n\n{* ../../docs_src/x.py hl[3] *}\n\nSome <dfn title='x'>text</dfn>."
    out = clean_markdown(raw)
    assert "{ #title }" not in out
    assert "[code example: ../../docs_src/x.py hl[3]]" in out
    assert "<dfn" not in out and "text" in out


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(3000))
    chunks = chunk_text(text, size=500, overlap=75)
    assert len(chunks) > 1
    assert all(len(_enc().encode(c)) <= 500 for c in chunks)
    # overlap: the end of chunk 0 reappears at the start of chunk 1
    tail = _enc().decode(_enc().encode(chunks[0])[-20:])
    assert tail.strip()[:10] in chunks[1]


def test_corpus_loads():
    chunks = load_chunks()
    assert len({c.doc_id for c in chunks}) >= 100


def test_citation_mapping():
    chunks = [{"doc_id": "a.md"}, {"doc_id": "b.md"}, {"doc_id": "a.md"}]
    assert cited_sources("Foo [1] bar [3][2] baz [9]", chunks) == ["a.md", "b.md"]


def test_eval_set_is_grounded():
    docs = ROOT / "data" / "docs"
    for line in open(ROOT / "eval" / "eval_set.jsonl", encoding="utf-8"):
        item = json.loads(line)
        if item["source_doc"]:
            assert item["evidence"] in (docs / item["source_doc"]).read_text(encoding="utf-8"), item["id"]
