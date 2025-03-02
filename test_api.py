import pytest
from fastapi.testclient import TestClient
from api import app  # Assurez-vous que c'est le bon chemin vers ton API

client = TestClient(app)


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction():
    response = client.post("/predict", json={"features": [1.0] * 14})
    assert response.status_code == 200
    assert response.json()["prediction"] in ["Churn", "No Churn"]
