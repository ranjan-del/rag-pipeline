"""Stage 5 — Retriever.

Given a user question, embed it with the SAME embedder used at ingest time, run
a similarity search over the vector store, and assemble the top chunks into a
context string that respects a budget (the "context window").

Using the same embedder for the query and the documents is essential: the two
vectors only live in the same space if they were produced by the same function.

MEMORY.md checklist:
- [x] Retriever (similarity search) -> context assembly (respect context window)
Learning outcomes: Vector search, Similarity, Context window
"""

from __future__ import annotations

import numpy as np


class Retriever:
    """Query the vector store and build LLM-ready context."""

    def __init__(self, vector_store, embedder, max_context_chars: int = 4000):
        """
        Args:
            vector_store: any object implementing the ``VectorStore`` interface.
            embedder: the SAME embedder used to build the stored vectors.
            max_context_chars: character budget for assembled context. We budget
                in characters (not tokens) to keep the default path dependency
                free; ~4 chars per token is the usual rule of thumb, so 4000
                chars is roughly a 1k-token window.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.max_context_chars = max_context_chars

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """Return the ``top_k`` chunks most relevant to the question.

        Each returned chunk carries its ``score`` and ``metadata`` so callers can
        both rank and cite it.
        """
        query_vector: np.ndarray = self.embedder.embed_one(question)
        return self.vector_store.search(query_vector, top_k=top_k)

    def assemble_context(self, chunks: list[dict]) -> str:
        """Concatenate retrieved chunks into a single context string.

        Chunks are added in ranked order until the character budget is reached,
        so the most relevant material survives when we have to truncate. Each
        chunk is prefixed with a ``[n]`` marker matching its position, which the
        LLM prompt and the citation layer both reference.
        """
        parts: list[str] = []
        used = 0
        for i, chunk in enumerate(chunks, start=1):
            block = f"[{i}] {chunk['text']}"
            # Stop before blowing the budget, but always include at least one
            # chunk so the model has something to work with.
            if used + len(block) > self.max_context_chars and parts:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)
