"""Stage 3 — Embedding.

Encode text chunks into dense vectors for semantic search.

MEMORY.md checklist:
- [ ] Embeddings + vector database (store + query)
Learning outcomes: Embeddings
"""

# TODO: from sentence_transformers import SentenceTransformer


def embed_chunks(chunks: list[dict], model_name: str = "all-MiniLM-L6-v2") -> list[dict]:
    """Attach an embedding vector to each chunk.

    Returns the chunks enriched with a "vector" field.

    TODO: MEMORY.md — load the embedding model and encode each chunk's text.
    """
    raise NotImplementedError("embed_chunks is a scaffold stub")
