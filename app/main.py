"""Application entrypoint (FastAPI / CLI stub).

Wires the RAG stages together and exposes them over an API.
This is scaffolding only — no real feature logic yet.

MEMORY.md checklist covered here:
- [ ] Upload + PDF text extraction (POST /ingest)
- [ ] Retriever -> context -> LLM answer with citations (POST /query)
- [ ] `docker compose up` runs the pipeline + vector DB + UI/API
"""

# TODO: import FastAPI and build the app once dependencies are installed.
# from fastapi import FastAPI, UploadFile
#
# app = FastAPI(title="rag-pipeline", version="0.1.0")


def create_app():
    """Build and return the FastAPI application.

    TODO: instantiate FastAPI, register routes, wire pipeline stages.
    """
    # TODO: MEMORY.md — construct the app and mount routers.
    raise NotImplementedError("create_app is a scaffold stub")


# TODO: POST /ingest — accept a PDF upload, then run extract -> chunk -> embed -> store.
# TODO: POST /query — run retriever -> context assembly -> LLM -> answer + citations.
# TODO: GET  /health — return service/vector-db health.


def main() -> None:
    """CLI entrypoint for running the pipeline locally.

    TODO: parse args and launch uvicorn, or run a one-shot ingest/query flow.
    """
    # TODO: MEMORY.md — start the API server (uvicorn) or a CLI flow.
    raise NotImplementedError("main is a scaffold stub")


if __name__ == "__main__":
    main()
