"""
Pipeline de Machine Learning pour la prédiction de churn.
"""

from typing import Tuple
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


def prepare_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge et prépare les données pour l'entraînement.

    Args:
        data_path (str): Chemin du fichier CSV contenant les données.

    Returns:
        Tuple contenant x_train, x_test, y_train, y_test.
    """
    data = pd.read_csv(data_path)

    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        encoder = LabelEncoder()
        for col in categorical_cols:
            data[col] = encoder.fit_transform(data[col])

    x_features = data.drop(["Churn"], axis=1)
    y_labels = data["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    joblib.dump(scaler, "scaler.joblib")

    return x_train_scaled, x_test_scaled, y_train, y_test


def train_model(x_train: np.ndarray, y_train: np.ndarray, model_type: str = "RandomForest"):
    """
    Entraîne un modèle de machine learning.

    Args:
        x_train (np.ndarray): Features d'entraînement.
        y_train (np.ndarray): Labels d'entraînement.
        model_type (str): Type de modèle à entraîner.

    Returns:
        Modèle entraîné.
    """
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel="linear", probability=True),
        "LogisticRegression": LogisticRegression(),
    }

    if model_type not in models:
        raise ValueError(f"Modèle non reconnu. Choisissez parmi {list(models.keys())}")

    model = models[model_type]
    model.fit(x_train, y_train)

    return model


def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    """
    Évalue le modèle et retourne accuracy et F1-score.

    Args:
        model: Modèle entraîné.
        x_test (np.ndarray): Features de test.
        y_test (np.ndarray): Labels de test.

    Returns:
        Tuple contenant l'accuracy et le F1-score.
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    return accuracy, f1


def save_model(model, model_path: str):
    """
    Sauvegarde le modèle entraîné.

    Args:
        model: Modèle entraîné.
        model_path (str): Chemin du fichier pour sauvegarder le modèle.
    """
    joblib.dump(model, model_path)
