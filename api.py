"""
API pour la prédiction du churn avec FastAPI.
"""

from typing import List, Dict
import numpy as np
import mlflow.pyfunc
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Charger le modèle depuis MLflow Model Registry
MODEL_NAME: str = "ChurnPredictionModel"
MODEL_VERSION: str = "1"

try:
    MODEL = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/1")  
    print(
        f"✅ Modèle {MODEL_NAME} (version {MODEL_VERSION}) " "chargé depuis MLflow Model Registry"
    )
except mlflow.exceptions.MlflowException as error:
    print(f"❌ Erreur MLflow : {error}")
    print("🔄 Tentative de chargement direct du modèle depuis models/")

    try:
        MODEL = joblib.load("models/churn_model.pkl")
        print("✅ Modèle chargé avec succès depuis models/churn_model.pkl")
    except FileNotFoundError:
        print("❌ Erreur : Aucun modèle trouvé.")
        MODEL = None


class ChurnInput(BaseModel):
    """Classe pour structurer les données d'entrée."""

    features: List[float]


@app.post("/predict")
def predict(data: ChurnInput) -> Dict[str, str]:
    """
    Prend une liste de features et retourne une prédiction de churn.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="❌ Modèle non chargé.")

    input_data = np.array(data.features).reshape(1, -1)
    prediction = MODEL.predict(input_data)

    return {"prediction": "Churn" if prediction[0] == 1 else "No Churn"}


@app.get("/")
def health_check() -> Dict[str, str]:
    """
    Endpoint de vérification du statut de l'API.
    """
    return {"status": "ok"}
