"""Ingest stage: PDF -> text -> chunks -> embeddings.

Modules:
- extract.py : PDF file -> raw text
- chunk.py   : raw text -> chunks (with metadata)
- embed.py   : chunks -> embedding vectors
"""
