from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_check():
    """Test du endpoint /"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    """Test du endpoint /predict avec des données valides"""
    sample_data = {"features": [1.0] * 19}  # Exemple avec 14 features
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["Churn", "No Churn"]

