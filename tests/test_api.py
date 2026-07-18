"""API tests for the FastAPI app.

Uses FastAPI's TestClient (backed by httpx) so no server needs to be running.
All offline — no API key required.
"""

from fastapi.testclient import TestClient

from app.main import create_app


def make_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return TestClient(create_app())


def test_health(monkeypatch):
    client = make_client(monkeypatch)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] == 0


def test_ingest_then_query(monkeypatch):
    client = make_client(monkeypatch)

    doc = (
        "The mitochondrion is the powerhouse of the cell. "
        "It generates most of the cell's supply of ATP. "
        "The Great Wall of China is over 13,000 miles long."
    )
    resp = client.post("/ingest", json={"text": doc, "source": "bio.txt"})
    assert resp.status_code == 200
    assert resp.json()["ingested_chunks"] > 0

    resp = client.post("/query", json={"question": "What does the mitochondrion do?", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["citations"]
    assert body["citations"][0]["source"] == "bio.txt"
