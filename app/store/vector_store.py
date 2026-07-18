"""Stage 4 — Vector Database.

Store chunk vectors (plus metadata) and run similarity queries against them.

The DEFAULT backend is a pure in-memory NumPy store: it keeps all vectors in one
matrix and ranks by cosine similarity with a single matrix-vector product. It
needs no server, no network, and no external dependency beyond NumPy — perfect
for learning and for offline tests.

The ``VectorStore`` protocol keeps the interface small (``upsert`` / ``search`` /
``__len__``) so a heavier backend such as Qdrant can be dropped in later without
touching the retriever. A commented ``QdrantVectorStore`` sketch is included to
show where that would go; it is intentionally NOT imported by the default path.

MEMORY.md checklist:
- [x] Embeddings + vector database (store + query)
Learning outcomes: Vector search, Similarity
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    """Minimal interface the rest of the pipeline depends on."""

    def upsert(self, chunks: list[dict]) -> None: ...

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]: ...

    def __len__(self) -> int: ...


class InMemoryVectorStore:
    """In-memory cosine-similarity store (DEFAULT backend).

    Vectors are expected to be L2-normalized by the embedder, so cosine
    similarity reduces to a dot product. We stack all chunk vectors into one
    ``(n, dim)`` matrix; a query is then ranked with ``matrix @ query`` — one
    fast, vectorized operation instead of a Python loop.
    """

    def __init__(self, dim: int | None = None):
        # dim can be left None and inferred from the first upsert.
        self.dim = dim
        self._vectors: np.ndarray | None = None  # (n, dim) matrix of chunk vectors
        self._chunks: list[dict] = []  # parallel list holding text + metadata

    def upsert(self, chunks: list[dict]) -> None:
        """Add embedded chunks to the store.

        Each chunk must already carry a ``"vector"`` (1-D ``np.ndarray``) from the
        embed stage. The vector is stored in the matrix; the chunk (minus its
        bulky vector) is kept alongside for returning metadata on search.
        """
        if not chunks:
            return

        new_vectors = np.vstack([np.asarray(c["vector"], dtype=np.float32) for c in chunks])

        if self.dim is None:
            self.dim = new_vectors.shape[1]
        elif new_vectors.shape[1] != self.dim:
            raise ValueError(
                f"vector dim {new_vectors.shape[1]} does not match store dim {self.dim}"
            )

        # Store a copy of each chunk WITHOUT the vector to keep payloads lean and
        # avoid duplicating the (already-in-matrix) array in every record.
        for chunk in chunks:
            record = {k: v for k, v in chunk.items() if k != "vector"}
            self._chunks.append(record)

        self._vectors = (
            new_vectors if self._vectors is None else np.vstack([self._vectors, new_vectors])
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return the ``top_k`` most similar chunks for a query vector.

        Args:
            query_vector: an L2-normalized 1-D vector (from ``embedder.embed_one``).
            top_k: how many results to return.

        Returns:
            A list of chunk records (id, text, metadata) each with an added
            ``"score"`` (cosine similarity in ``[-1, 1]``), ordered most-similar
            first.
        """
        if self._vectors is None or len(self._chunks) == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        # Cosine similarity == dot product because everything is L2-normalized.
        scores = self._vectors @ query  # shape (n,)

        k = min(top_k, len(scores))
        # argpartition finds the top-k cheaply, then we sort just those k.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results: list[dict] = []
        for idx in top_idx:
            record = dict(self._chunks[idx])  # shallow copy so callers can't mutate ours
            record["score"] = float(scores[idx])
            results.append(record)
        return results

    def __len__(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# OPTIONAL heavier backend — Qdrant. Kept as a reference implementation so the
# docker-compose Qdrant service has a home. It is NOT imported anywhere on the
# default path, so tests never need a running Qdrant. Uncomment + install
# ``qdrant-client`` to use it.
# ---------------------------------------------------------------------------
#
# class QdrantVectorStore:
#     """Qdrant-backed store implementing the same VectorStore interface."""
#
#     def __init__(self, dim: int, host: str = "localhost", port: int = 6333,
#                  collection: str = "documents"):
#         from qdrant_client import QdrantClient
#         from qdrant_client.models import Distance, VectorParams
#
#         self.collection = collection
#         self._client = QdrantClient(host=host, port=port)
#         self._client.recreate_collection(
#             collection_name=collection,
#             vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
#         )
#         self._auto_id = 0
#
#     def upsert(self, chunks: list[dict]) -> None:
#         from qdrant_client.models import PointStruct
#         points = []
#         for c in chunks:
#             payload = {k: v for k, v in c.items() if k != "vector"}
#             points.append(PointStruct(id=self._auto_id, vector=list(c["vector"]),
#                                       payload=payload))
#             self._auto_id += 1
#         self._client.upsert(collection_name=self.collection, points=points)
#
#     def search(self, query_vector, top_k: int = 5) -> list[dict]:
#         hits = self._client.search(collection_name=self.collection,
#                                    query_vector=list(query_vector), limit=top_k)
#         return [{**h.payload, "score": h.score} for h in hits]
#
#     def __len__(self) -> int:
#         return self._client.count(self.collection).count
