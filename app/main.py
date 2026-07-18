"""Application entrypoint — FastAPI API + CLI demo.

Wires all pipeline stages together behind a small object (``Pipeline``) and
exposes them over HTTP:

- ``POST /ingest`` — ingest raw text OR an uploaded file (extract -> chunk ->
  embed -> store).
- ``POST /query``  — answer a question (retrieve -> context -> LLM -> citations).
- ``GET  /health`` — liveness + how many chunks are currently stored.

Running ``python -m app.main`` (or ``python app/main.py``) executes a small
end-to-end demo that ingests a sample document and answers a question — a quick
way to see the whole flow without starting a server.

Everything runs OFFLINE by default: the in-memory vector store, the hashing
embedder, and the deterministic LLM fallback need no network or API key.

MEMORY.md checklist:
- [x] Upload + PDF text extraction (POST /ingest)
- [x] Retriever -> context -> LLM answer with citations (POST /query)
- [x] `docker compose up` runs the pipeline + vector DB + UI/API
"""

from __future__ import annotations

import os
import tempfile

from pydantic import BaseModel

from app.generate.answer import answer_question
from app.ingest.chunk import chunk_text
from app.ingest.embed import embed_chunks, get_embedder
from app.ingest.extract import extract_pdf, extract_text
from app.retrieve.retriever import Retriever
from app.store.vector_store import InMemoryVectorStore


# --- API request schemas (module level so FastAPI recognizes them as bodies) ---
class IngestTextRequest(BaseModel):
    text: str
    source: str = "text-input"


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Pipeline:
    """Holds the shared pipeline state: embedder, vector store, retriever.

    A single instance is created per process and reused across requests so that
    everything ingested stays queryable. In a real system you'd persist the
    store; here it lives in memory, which is perfect for a teaching demo.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50, top_k: int = 5):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

        # The embedder and store must agree on vector dimension, so we build
        # them together and share the embedder with the retriever.
        self.embedder = get_embedder()  # offline hashing embedder by default
        self.store = InMemoryVectorStore(dim=self.embedder.dim)
        self.retriever = Retriever(self.store, self.embedder)

    def ingest_blocks(self, blocks: list[dict]) -> int:
        """Run chunk -> embed -> store for already-extracted text blocks.

        Returns the number of chunks added.
        """
        chunks = chunk_text(blocks, chunk_size=self.chunk_size, overlap=self.overlap)
        embed_chunks(chunks, self.embedder)
        self.store.upsert(chunks)
        return len(chunks)

    def ingest_text(self, text: str, source: str = "text-input") -> int:
        """Ingest a plain-text document."""
        return self.ingest_blocks(extract_text(text, source=source))

    def ingest_pdf(self, pdf_path: str) -> int:
        """Ingest a PDF file from disk."""
        return self.ingest_blocks(extract_pdf(pdf_path))

    def query(self, question: str, top_k: int | None = None) -> dict:
        """Answer a question end-to-end."""
        return answer_question(question, self.retriever, top_k=top_k or self.top_k)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

def create_app() -> "FastAPI":  # noqa: F821 (FastAPI imported lazily below)
    """Build and return the FastAPI application.

    FastAPI/pydantic are imported lazily so the pipeline modules and tests can
    be used without the web framework installed.
    """
    from fastapi import FastAPI, File, UploadFile

    app = FastAPI(
        title="rag-pipeline",
        version="1.0.0",
        description="Educational RAG pipeline: PDF -> grounded answer with citations.",
    )

    # One shared pipeline for the whole process.
    pipeline = Pipeline()

    # ---- Routes ----
    @app.get("/health")
    def health() -> dict:
        """Liveness check + how many chunks are stored."""
        return {"status": "ok", "chunks_indexed": len(pipeline.store)}

    @app.post("/ingest")
    def ingest(payload: IngestTextRequest) -> dict:
        """Ingest a plain-text document (extract -> chunk -> embed -> store)."""
        added = pipeline.ingest_text(payload.text, source=payload.source)
        return {"ingested_chunks": added, "total_chunks": len(pipeline.store)}

    @app.post("/ingest/file")
    async def ingest_file(file: UploadFile = File(...)) -> dict:
        """Ingest an uploaded file.

        PDFs are parsed with pypdf; anything else is treated as UTF-8 text. The
        upload is written to a temp file so pypdf can read it by path.
        """
        raw = await file.read()
        filename = file.filename or "upload"

        if filename.lower().endswith(".pdf"):
            # pypdf needs a path, so spill the bytes to a temp file.
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                added = pipeline.ingest_pdf(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            added = pipeline.ingest_text(raw.decode("utf-8", errors="replace"), source=filename)

        return {"ingested_chunks": added, "total_chunks": len(pipeline.store)}

    @app.post("/query")
    def query(payload: QueryRequest) -> dict:
        """Answer a question with citations back to source chunks."""
        result = pipeline.query(payload.question, top_k=payload.top_k)
        # Drop the bulky raw chunks from the API response (keep citations).
        return {
            "question": result["question"],
            "answer": result["answer"],
            "citations": result["citations"],
        }

    return app


# Module-level ASGI app for `uvicorn app.main:app`. Guarded so importing this
# module (e.g. in tests) doesn't require FastAPI to be installed.
try:  # pragma: no cover - trivial import guard
    app = create_app()
except Exception:  # pragma: no cover
    app = None


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

SAMPLE_DOC = """
Retrieval-Augmented Generation (RAG) combines a retriever with a language model.
The retriever searches a knowledge base for passages relevant to a question.
Those passages are then passed to the language model as context.
This grounds the model's answer in real source documents and enables citations.

Embeddings turn text into vectors so that similar text has similar vectors.
A vector store indexes these vectors and supports fast similarity search.
Cosine similarity is a common way to measure how close two vectors are.
Chunking splits a long document into smaller pieces before embedding.
"""


def main() -> None:
    """Run a one-shot end-to-end demo of the pipeline."""
    pipeline = Pipeline()

    n = pipeline.ingest_text(SAMPLE_DOC, source="rag-intro.txt")
    print(f"Ingested {n} chunks from the sample document.\n")

    question = "What does the retriever do in RAG?"
    result = pipeline.query(question)

    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Citations:")
    for c in result["citations"]:
        print(f"  {c['marker']} {c['source']} (page {c['page']}, score {c['score']}): {c['snippet']}")


if __name__ == "__main__":
    main()
