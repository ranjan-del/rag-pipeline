"""Stage 7 — Answer.

Orchestrate the final query flow: retrieve context, call the LLM, and package the
answer together with its citations.

MEMORY.md checklist:
- [ ] LLM answer generation with citations back to source chunks
"""


def answer_question(question: str, retriever, top_k: int = 5) -> dict:
    """Run the full query pipeline and return an answer with citations.

    Returns {"answer": str, "citations": [...], "sources": [...]}.

    TODO: MEMORY.md — retrieve -> assemble context -> LLM -> attach citations.
    """
    raise NotImplementedError("answer_question is a scaffold stub")
