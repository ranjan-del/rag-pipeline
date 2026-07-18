"""Stage 5 — Retriever.

Embed the user's question, run similarity search over the vector store, and
assemble the retrieved chunks into a context string that respects the LLM's
context window.

MEMORY.md checklist:
- [ ] Retriever (similarity search) -> context assembly (respect context window)
Learning outcomes: Vector search, Similarity, Context window
"""


class Retriever:
    """Query the vector store and build LLM-ready context.

    TODO: MEMORY.md — embed the query, search the vector store, and pack the most
    relevant chunks into the context window.
    """

    def __init__(self, vector_store, embedder=None, max_context_tokens: int = 3000):
        # TODO: hold references to the vector store + embedding function.
        raise NotImplementedError("Retriever is a scaffold stub")

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """Return the top_k chunks most relevant to the question."""
        # TODO: MEMORY.md — embed the question and similarity-search the store.
        raise NotImplementedError

    def assemble_context(self, chunks: list[dict]) -> str:
        """Concatenate chunks into a context string within the token budget."""
        # TODO: MEMORY.md — respect max_context_tokens while assembling context.
        raise NotImplementedError
