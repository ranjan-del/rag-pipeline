"""Stage 8 — Citation.

Map the generated answer back to the specific source chunks it was grounded in,
so every answer is traceable to the original document (page + chunk).

MEMORY.md checklist:
- [ ] LLM answer generation with citations back to source chunks
Learning outcomes: Metadata
"""


def build_citations(answer: str, retrieved_chunks: list[dict]) -> list[dict]:
    """Produce citation entries linking the answer to its source chunks.

    Returns a list of {"chunk_id", "page", "snippet"} references.

    TODO: MEMORY.md — attribute the answer back to the retrieved chunks' metadata.
    """
    raise NotImplementedError("build_citations is a scaffold stub")
