import os

os.environ["LLM_PROVIDER"] = "groq"

from fastapi.testclient import TestClient

from app.main import app


def test_api_endpoints():
    with TestClient(app) as client:
        assert client.get("/health").json()["row_count"] == 500
        assert client.get("/tickets?page=1&page_size=2").json()["total"] == 500
        assert client.get("/anomalies?days=7").status_code == 200
        response = client.post(
            "/query", json={"question": "How many tickets are currently open?"}
        )
        assert response.status_code == 200
        assert response.json()["raw_result"] == {"count": 111}
