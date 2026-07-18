"""Stage 1 — Extract.

Turn an uploaded PDF into raw text (per page, so page numbers survive for
citations later).

MEMORY.md checklist:
- [ ] Upload + PDF text extraction
"""

# TODO: from pypdf import PdfReader


def extract_text(pdf_path: str) -> list[dict]:
    """Extract text from a PDF, one entry per page.

    Returns a list of {"page": int, "text": str}. Keeping page numbers here is
    what makes downstream citations possible.

    TODO: MEMORY.md — read the PDF with pypdf and collect per-page text.
    """
    raise NotImplementedError("extract_text is a scaffold stub")
