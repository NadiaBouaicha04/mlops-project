from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_check():
    """Test du endpoint /"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
