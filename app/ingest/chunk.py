"""Stage 2 — Chunk.

Split extracted text into chunks and attach metadata (source, page, chunk index,
character span). Making the chunking strategy explicit and visible is a core
learning outcome of this project, so the sizing is fully parameterised and every
chunk carries the metadata needed to trace it back to the original document.

Why chunk at all? Embeddings and LLM context windows are finite. We break the
document into overlapping windows so that (a) each vector represents a small,
coherent span of text, and (b) a fact that straddles a boundary still appears
whole in at least one chunk thanks to the overlap.

MEMORY.md checklist:
- [x] Chunking (with metadata) — make strategy configurable/visible
Learning outcomes: Chunking, Metadata
"""

from __future__ import annotations


def chunk_text(
    blocks: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Split per-source text blocks into overlapping, character-based chunks.

    This is a *fixed-size sliding window* over characters — the simplest strategy
    to reason about, which is exactly what we want for a teaching repo. The
    window advances by ``chunk_size - overlap`` each step so consecutive chunks
    share ``overlap`` characters of context.

    Args:
        blocks: output of the extract stage — ``[{"source", "page", "text"}]``.
        chunk_size: maximum characters per chunk. Must be > 0.
        overlap: characters shared between consecutive chunks. Must be
            ``0 <= overlap < chunk_size`` (otherwise the window never advances).

    Returns:
        A list of chunk dicts, each shaped::

            {
                "id": "<source>-p<page>-c<index>",
                "text": "...",
                "metadata": {
                    "source": str,
                    "page": int,
                    "chunk_index": int,   # global, across all blocks
                    "char_start": int,    # offset within this block's text
                    "char_end": int,
                },
            }
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    step = chunk_size - overlap  # how far the window slides each iteration
    chunks: list[dict] = []
    chunk_index = 0  # global index so every chunk id across the doc is unique

    for block in blocks:
        text = block["text"]
        source = block["source"]
        page = block["page"]

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()

            # Skip windows that are only whitespace (can happen at the tail).
            if piece:
                chunks.append(
                    {
                        "id": f"{source}-p{page}-c{chunk_index}",
                        "text": piece,
                        "metadata": {
                            "source": source,
                            "page": page,
                            "chunk_index": chunk_index,
                            "char_start": start,
                            "char_end": end,
                        },
                    }
                )
                chunk_index += 1

            # If we've reached the end of this block, stop (avoids an extra empty
            # window when len(text) is an exact multiple of the step).
            if end == len(text):
                break
            start += step

    return chunks
