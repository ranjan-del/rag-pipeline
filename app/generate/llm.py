"""Stage 6 — LLM.

Call a language model with the question and the retrieved context to produce a
grounded answer.

Design goal for this repo: the pipeline must RUN OFFLINE with no API key. So:

- The DEFAULT and preferred backend is Anthropic Claude (the ``anthropic`` SDK,
  model ``claude-sonnet-5``, key from the ``ANTHROPIC_API_KEY`` env var).
- If no key is set (or the SDK/network is unavailable), we fall back to a
  deterministic *extractive* answer: we stitch together the top retrieved chunks
  and label it clearly as an offline stub. This keeps the whole pipeline — and
  the tests — working with zero external dependencies.

Never hardcode secrets: the key is read from the environment only.

MEMORY.md checklist:
- [x] LLM answer generation with citations back to source chunks
"""

from __future__ import annotations

import os

# Model id per the project blueprint. Kept as a constant so it is easy to find
# and change. Claude Sonnet 5 is a good default for grounded Q&A.
CLAUDE_MODEL = "claude-sonnet-5"


def build_prompt(question: str, context: str) -> str:
    """Build a grounded prompt that instructs the model to answer only from context.

    The ``[n]`` markers in the context (added by the retriever) let the model
    cite its sources by number, which the citation stage maps back to chunk
    metadata.
    """
    return (
        "You are a helpful assistant. Answer the question using ONLY the "
        "context below. Each context passage is prefixed with a number like "
        "[1]. Cite the passages you use by their number, e.g. [1]. If the "
        "answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def _extractive_fallback(question: str, chunks: list[dict]) -> str:
    """Deterministic offline answer: stitch the top retrieved chunks together.

    This is intentionally simple and reproducible. It is NOT a real generated
    answer — it just surfaces the most relevant retrieved text with citation
    markers so the pipeline produces something useful (and testable) without an
    LLM. Each chunk keeps the ``[n]`` marker matching its rank.
    """
    if not chunks:
        return "I don't know — no relevant context was found."

    # Take up to the top 3 chunks so the stub answer stays focused.
    parts = []
    for i, chunk in enumerate(chunks[:3], start=1):
        snippet = chunk["text"].strip()
        parts.append(f"[{i}] {snippet}")

    stitched = " ".join(parts)
    return (
        "(offline extractive answer — set ANTHROPIC_API_KEY for a generated "
        f"response)\n\nBased on the retrieved context: {stitched}"
    )


def generate(
    question: str,
    context: str,
    chunks: list[dict] | None = None,
    model: str | None = None,
) -> str:
    """Produce an answer for ``question`` grounded in ``context``.

    Tries Anthropic Claude first; falls back to the deterministic extractive
    answer when no API key is set or the call fails for any reason.

    Args:
        question: the user's question.
        context: the assembled context string (with ``[n]`` markers).
        chunks: the retrieved chunks, used to build the offline fallback answer.
        model: override the Claude model id (defaults to ``CLAUDE_MODEL``).

    Returns:
        The answer text.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # No key -> go straight to the offline path. This is the default in tests.
    if not api_key:
        return _extractive_fallback(question, chunks or [])

    try:
        # Lazy import so the SDK is only needed on the online path.
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the env
        response = client.messages.create(
            model=model or CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": build_prompt(question, context)}],
        )
        # Concatenate all text blocks in the response.
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
    except Exception:
        # Any failure (missing SDK, network error, auth error) degrades to the
        # deterministic fallback so the pipeline never hard-crashes.
        return _extractive_fallback(question, chunks or [])
