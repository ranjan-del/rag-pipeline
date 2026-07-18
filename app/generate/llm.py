"""Stage 6 — LLM.

Call the language model with the question and the retrieved context to produce a
grounded answer.

MEMORY.md checklist:
- [ ] LLM answer generation with citations back to source chunks
"""


def build_prompt(question: str, context: str) -> str:
    """Build a grounded prompt instructing the model to answer only from context.

    TODO: MEMORY.md — craft a prompt that requires citing source chunks.
    """
    raise NotImplementedError("build_prompt is a scaffold stub")


def generate(question: str, context: str, model: str | None = None) -> str:
    """Send the grounded prompt to the LLM and return the raw answer text.

    TODO: MEMORY.md — call the configured LLM provider and return its response.
    """
    raise NotImplementedError("generate is a scaffold stub")
