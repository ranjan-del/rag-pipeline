"""rag-pipeline application package.

Educational, end-to-end RAG pipeline:
    Upload PDF -> Extract -> Chunk -> Embedding -> Vector DB
    -> Retriever -> LLM -> Answer -> Citation

Each stage lives in its own module so the flow of data is explicit.
See MEMORY.md for the authoritative spec and checklist.
"""

__version__ = "0.1.0"
