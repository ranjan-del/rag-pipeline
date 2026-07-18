"""Stage 8 — Citation.

Map the retrieved chunks the answer was grounded in back to their source
metadata, so every answer is traceable to the original document (source + page +
chunk). Each citation's number matches the ``[n]`` marker used in the assembled
context and the LLM prompt.

MEMORY.md checklist:
- [x] LLM answer generation with citations back to source chunks
Learning outcomes: Metadata
"""

from __future__ import annotations


def build_citations(retrieved_chunks: list[dict], max_snippet: int = 160) -> list[dict]:
    """Produce citation entries for the retrieved chunks, in rank order.

    We cite by retrieval rank (``[1]``, ``[2]`` ...) rather than trying to parse
    which markers the answer text used. For a teaching pipeline this is the
    honest, predictable choice: the citations describe exactly the context the
    model was given.

    Args:
        retrieved_chunks: the ranked chunks from the retriever (each with
            ``id``, ``text``, ``metadata``, and ``score``).
        max_snippet: how many characters of the chunk text to include as a
            preview snippet.

    Returns:
        A list of citation dicts::

            {
                "marker": "[1]",
                "chunk_id": str,
                "source": str,
                "page": int,
                "score": float,
                "snippet": str,
            }
    """
    citations: list[dict] = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        text = chunk.get("text", "")
        snippet = text[:max_snippet].strip()
        if len(text) > max_snippet:
            snippet += "…"

        citations.append(
            {
                "marker": f"[{i}]",
                "chunk_id": chunk.get("id"),
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "score": round(float(chunk.get("score", 0.0)), 4),
                "snippet": snippet,
            }
        )
    return citations
