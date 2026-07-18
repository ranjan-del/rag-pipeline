"""End-to-end + unit tests for the RAG pipeline.

Everything here runs OFFLINE: the in-memory vector store, the hashing embedder,
and the deterministic LLM fallback need no network, no model download, and no
API key. That is the whole point — a reviewer can clone and run these tests with
only the default dependencies installed.

Run from the repo root:
    pytest -q
"""

import numpy as np

from app.generate.answer import answer_question
from app.generate.citation import build_citations
from app.ingest.chunk import chunk_text
from app.ingest.embed import HashingEmbedder, embed_chunks
from app.ingest.extract import extract_text
from app.main import Pipeline
from app.retrieve.retriever import Retriever
from app.store.vector_store import InMemoryVectorStore

SAMPLE = (
    "Photosynthesis is how plants convert sunlight into chemical energy. "
    "Chlorophyll in the leaves absorbs light. "
    "The Eiffel Tower is a landmark located in Paris, France. "
    "It was completed in 1889 for the World's Fair."
)


# --- Stage 1: extract ------------------------------------------------------

def test_extract_text_produces_blocks():
    blocks = extract_text("hello world", source="doc.txt")
    assert blocks == [{"source": "doc.txt", "page": 1, "text": "hello world"}]


# --- Stage 2: chunk --------------------------------------------------------

def test_chunk_metadata_and_overlap():
    blocks = extract_text("A" * 1000, source="doc.txt")
    chunks = chunk_text(blocks, chunk_size=300, overlap=50)
    # 1000 chars, step 250 -> windows at 0,250,500,750 => 4 chunks.
    assert len(chunks) == 4
    # Metadata is present and ids are unique.
    assert {c["metadata"]["source"] for c in chunks} == {"doc.txt"}
    assert len(chunks) == len({c["id"] for c in chunks})
    # Overlap: chunk 1 starts 250 chars in, before chunk 0 ends at 300.
    assert chunks[1]["metadata"]["char_start"] == 250
    assert chunks[0]["metadata"]["char_end"] == 300


def test_chunk_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text(extract_text("x"), chunk_size=100, overlap=100)


# --- Stage 3: embed --------------------------------------------------------

def test_embeddings_are_normalized_and_deterministic():
    emb = HashingEmbedder(dim=64)
    v1 = emb.embed_one("photosynthesis in plants")
    v2 = emb.embed_one("photosynthesis in plants")
    # Deterministic across calls.
    assert np.allclose(v1, v2)
    # L2-normalized (unit length) for non-empty text.
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-5
    assert v1.shape == (64,)


# --- Stage 4 + 5: store + retrieve ----------------------------------------

def test_retrieval_returns_relevant_chunk():
    blocks = extract_text(SAMPLE, source="facts.txt")
    chunks = chunk_text(blocks, chunk_size=80, overlap=20)
    embedder = HashingEmbedder(dim=128)
    embed_chunks(chunks, embedder)

    store = InMemoryVectorStore(dim=embedder.dim)
    store.upsert(chunks)
    assert len(store) == len(chunks)

    retriever = Retriever(store, embedder)
    results = retriever.retrieve("Where is the Eiffel Tower located?", top_k=3)

    assert results, "expected at least one retrieved chunk"
    # The top result should mention the Eiffel Tower / Paris, not photosynthesis.
    top_text = results[0]["text"].lower()
    assert "eiffel" in top_text or "paris" in top_text
    # Scores are cosine similarities, descending.
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_store_search_returns_empty():
    store = InMemoryVectorStore(dim=16)
    assert store.search(np.zeros(16), top_k=5) == []


# --- Stage 6/7/8: answer + citations (offline fallback) --------------------

def test_answer_includes_citations_offline(monkeypatch):
    # Guarantee the offline path even if a key is present in the environment.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    blocks = extract_text(SAMPLE, source="facts.txt")
    chunks = chunk_text(blocks, chunk_size=80, overlap=20)
    embedder = HashingEmbedder(dim=128)
    embed_chunks(chunks, embedder)
    store = InMemoryVectorStore(dim=embedder.dim)
    store.upsert(chunks)
    retriever = Retriever(store, embedder)

    result = answer_question("What is photosynthesis?", retriever, top_k=3)

    assert result["answer"], "answer should not be empty"
    assert result["citations"], "expected citations back to source chunks"
    first = result["citations"][0]
    # Citations carry traceable metadata.
    assert first["marker"] == "[1]"
    assert first["source"] == "facts.txt"
    assert first["page"] == 1
    assert first["chunk_id"] is not None
    # Offline answer references retrieved context.
    assert "[1]" in result["answer"]


def test_build_citations_shape():
    chunk = {
        "id": "d-p1-c0",
        "text": "x" * 300,
        "metadata": {"source": "d", "page": 1},
        "score": 0.5,
    }
    cites = build_citations([chunk], max_snippet=50)
    assert cites[0]["snippet"].endswith("…")  # truncated + ellipsis
    assert len(cites[0]["snippet"]) <= 51


# --- Full pipeline via the Pipeline orchestrator ---------------------------

def test_full_pipeline_end_to_end(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    pipeline = Pipeline(chunk_size=80, overlap=20)
    added = pipeline.ingest_text(SAMPLE, source="facts.txt")
    assert added > 0
    assert len(pipeline.store) == added

    result = pipeline.query("When was the Eiffel Tower completed?", top_k=3)
    assert result["citations"]
    # Retrieval should surface the year-1889 chunk near the top.
    joined = " ".join(r["text"] for r in result["retrieved"]).lower()
    assert "1889" in joined
