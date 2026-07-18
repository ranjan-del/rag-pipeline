"""Stage 7 — Answer.

Orchestrate the final query flow: retrieve context, call the LLM (or its offline
fallback), and package the answer together with its citations. This is the
single function the API and CLI call to answer a question end-to-end.

MEMORY.md checklist:
- [x] LLM answer generation with citations back to source chunks
"""

from __future__ import annotations

from app.generate import citation, llm


def answer_question(question: str, retriever, top_k: int = 5) -> dict:
    """Run the full query pipeline and return an answer with citations.

    Steps:
        1. Retrieve the top_k most relevant chunks (embed query -> vector search).
        2. Assemble them into a context string within the budget.
        3. Generate a grounded answer (Claude, or the deterministic fallback).
        4. Build citations that map back to each chunk's source metadata.

    Args:
        question: the user's question.
        retriever: a ``Retriever`` instance wired to the vector store + embedder.
        top_k: how many chunks to retrieve.

    Returns:
        ``{"question", "answer", "citations", "retrieved"}`` where ``retrieved``
        is the raw ranked chunks (handy for debugging/teaching).
    """
    # 1. Retrieve.
    retrieved = retriever.retrieve(question, top_k=top_k)

    # 2. Assemble context (respecting the retriever's character budget).
    context = retriever.assemble_context(retrieved)

    # 3. Generate the answer. Pass the chunks so the offline fallback can build
    #    an extractive answer without an API key.
    answer_text = llm.generate(question, context, chunks=retrieved)

    # 4. Attribute the answer back to its sources.
    citations = citation.build_citations(retrieved)

    return {
        "question": question,
        "answer": answer_text,
        "citations": citations,
        "retrieved": retrieved,
    }
