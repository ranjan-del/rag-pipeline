"""Stage 1 — Extract.

Turn a source document into raw, per-source text blocks.

Two entry points are provided so the rest of the pipeline never has to care
where the text came from:

- ``extract_text`` : accept already-plain text (a string). This is what the
                     offline tests and the ``/ingest`` text path use — no PDF and
                     no binary fixtures needed.
- ``extract_pdf``  : read a PDF from disk with ``pypdf`` (one block per page, so
                     page numbers survive for later citations).

Both return the SAME shape: a list of "blocks", where each block is
``{"source": str, "page": int, "text": str}``. Keeping ``source`` + ``page`` on
every block is what makes downstream citations possible.

MEMORY.md checklist:
- [x] Upload + PDF text extraction
"""

from __future__ import annotations


def extract_text(text: str, source: str = "text-input") -> list[dict]:
    """Wrap a plain-text string in the pipeline's block format.

    Why this exists: the whole pipeline (chunk -> embed -> store -> retrieve)
    only ever sees "blocks", so accepting raw text here means tests and callers
    can drive the system end-to-end without a real PDF.

    We treat the whole string as a single "page" (page 1). Callers that already
    have page boundaries can call this once per page instead.

    Args:
        text: the raw document text.
        source: a human-readable name for where the text came from (surfaced in
            citations).

    Returns:
        A one-element list ``[{"source", "page", "text"}]``.
    """
    return [{"source": source, "page": 1, "text": text}]


def extract_pdf(pdf_path: str) -> list[dict]:
    """Extract text from a PDF, one block per page.

    Page numbers are 1-based to match how a human reads a PDF. ``pypdf`` is
    imported lazily so importing this module (and running the plain-text path)
    never requires the dependency to be installed.

    Args:
        pdf_path: filesystem path to the PDF.

    Returns:
        A list of ``{"source", "page", "text"}`` blocks, one per non-empty page.
    """
    # Lazy import: only the PDF path needs pypdf. The offline/default path and
    # the test-suite do not, so we keep the import out of module scope.
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    source = pdf_path.rsplit("/", 1)[-1]  # basename, e.g. "paper.pdf"

    blocks: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        # extract_text() can return None for image-only pages; normalise to "".
        page_text = (page.extract_text() or "").strip()
        if page_text:
            blocks.append({"source": source, "page": page_number, "text": page_text})
    return blocks
