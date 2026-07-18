"""Stage 4 — Vector Database.

Store chunk vectors (plus metadata) and run similarity queries against them.
Backed by Qdrant.

MEMORY.md checklist:
- [ ] Embeddings + vector database (store + query)
Learning outcomes: Vector search, Similarity
"""

# TODO: from qdrant_client import QdrantClient


class VectorStore:
    """Thin wrapper around the Qdrant vector database.

    TODO: MEMORY.md — connect to Qdrant, ensure the collection exists, and expose
    upsert + similarity-search operations.
    """

    def __init__(self, host: str = "qdrant", port: int = 6333, collection: str = "documents"):
        # TODO: instantiate QdrantClient and ensure the collection exists.
        raise NotImplementedError("VectorStore is a scaffold stub")

    def upsert(self, chunks: list[dict]) -> None:
        """Store embedded chunks (vector + metadata) in the collection."""
        # TODO: MEMORY.md — upsert vectors + payload into Qdrant.
        raise NotImplementedError

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """Return the top_k most similar chunks for a query vector."""
        # TODO: MEMORY.md — run a similarity search and return scored chunks.
        raise NotImplementedError
