"""Stage 2 — Chunk.

Split extracted text into chunks and attach metadata (source page, chunk index,
offsets). Chunking strategy is meant to be configurable and visible — that is a
core learning outcome of this project.

MEMORY.md checklist:
- [ ] Chunking (with metadata) — make strategy configurable/visible
Learning outcomes: Chunking, Metadata
"""


def chunk_text(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Split per-page text into overlapping chunks with metadata.

    Returns a list of {"id", "text", "metadata": {page, chunk_index, ...}}.

    TODO: MEMORY.md — implement a configurable chunking strategy (fixed /
    recursive / semantic) and record metadata for each chunk.
    """
    raise NotImplementedError("chunk_text is a scaffold stub")
