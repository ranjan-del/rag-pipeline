"""Stage 3 — Embedding.

Encode text into dense vectors so that semantically similar text lands close
together in vector space (that closeness is what "retrieval" exploits).

Design goal for this repo: the DEFAULT path must run offline with no model
downloads and no network. So the default ``HashingEmbedder`` builds a
deterministic bag-of-words vector using feature hashing — no learned weights, no
files to fetch. It is not as smart as a real sentence transformer, but it is
100% reproducible and good enough to demonstrate (and test) that relevant chunks
score higher than irrelevant ones.

A ``SentenceTransformerEmbedder`` is provided as an OPTIONAL upgrade behind a
lazy import; it is never needed for the default path or the test-suite.

MEMORY.md checklist:
- [x] Embeddings + vector database (store + query)
Learning outcomes: Embeddings
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

# A simple word tokenizer: lowercase alphanumeric runs. Good enough for a
# bag-of-words model and keeps the behaviour easy to predict.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    """Structural interface every embedder implements.

    Keeping this as a ``Protocol`` means the retriever/store just need "something
    with ``dim`` and ``embed`` / ``embed_one``" — they don't care which concrete
    embedder is used, so swapping in sentence-transformers later is trivial.
    """

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim), L2-normalized
        ...

    def embed_one(self, text: str) -> np.ndarray:  # (dim,), L2-normalized
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length.

    We normalize so that a dot product between two vectors equals their cosine
    similarity. That lets the vector store use a plain matrix multiply for
    ranking (see ``store/vector_store.py``).
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid divide-by-zero for empty/all-stopword text: leave those rows as-is
    # (all zeros), which yields a similarity of 0 against everything.
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Deterministic bag-of-words embedder via the hashing trick (DEFAULT).

    Each token is hashed into one of ``dim`` buckets and its count is added to
    that bucket. The resulting count vector is L2-normalized. Two texts that
    share many words end up with vectors pointing in a similar direction, so
    cosine similarity ranks word-overlap — enough to make retrieval meaningful
    and, crucially, fully offline and reproducible.

    Why the hashing trick instead of a fixed vocabulary? It gives a fixed-size
    vector without having to build or ship a vocabulary file, and it handles
    unseen words for free.
    """

    def __init__(self, dim: int = 256):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _hash(self, token: str) -> int:
        # Python's built-in hash() is salted per-process, which would make
        # embeddings non-reproducible across runs. Use a stable hash instead.
        import hashlib

        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % self.dim

    def embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in _tokenize(text):
            vector[self._hash(token)] += 1.0
        return _l2_normalize(vector.reshape(1, -1))[0]

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.vstack([self.embed_one(t) for t in texts])
        return matrix.astype(np.float32)


class SentenceTransformerEmbedder:
    """Optional real-model embedder (requires ``sentence-transformers``).

    Not used by the default path or tests — it needs a one-time model download.
    Provided so learners can compare hashed bag-of-words against true semantic
    embeddings by flipping a flag.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Lazy import so the dependency is only required when this class is used.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return _l2_normalize(vectors.astype(np.float32))

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def get_embedder(use_sentence_transformers: bool = False, dim: int = 256) -> Embedder:
    """Factory: return the default offline embedder, or the optional real one.

    Args:
        use_sentence_transformers: if True, use the model-backed embedder
            (requires the optional dependency + a download). Defaults to False
            so the standard path stays offline.
        dim: vector size for the default ``HashingEmbedder``.
    """
    if use_sentence_transformers:
        return SentenceTransformerEmbedder()
    return HashingEmbedder(dim=dim)


def embed_chunks(chunks: list[dict], embedder: Embedder | None = None) -> list[dict]:
    """Attach an embedding vector to each chunk (in place) and return them.

    Args:
        chunks: chunk dicts from the chunk stage.
        embedder: any object implementing the ``Embedder`` protocol. Defaults to
            the offline ``HashingEmbedder``.

    Returns:
        The same chunks, each with a new ``"vector"`` key holding a 1-D
        ``np.ndarray``.
    """
    if embedder is None:
        embedder = HashingEmbedder()
    if not chunks:
        return chunks

    vectors = embedder.embed([c["text"] for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector
    return chunks
