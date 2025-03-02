"""
Pipeline de préparation des données, d'entraînement et d'évaluation du modèle.
"""

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
from sklearn.base import ClassifierMixin
from typing import Tuple


def prepare_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Charge et prépare les données pour l'entraînement."""
    data = pd.read_csv(data_path)

    # Encodage des variables catégorielles
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"Encodage des colonnes catégorielles : {categorical_cols}")
        encoder = LabelEncoder()
        for col in categorical_cols:
            data[col] = encoder.fit_transform(data[col])

    # Séparation en features et target
    features = data.drop(['Churn'], axis=1).values
    target = data['Churn'].values

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    # Normalisation des données
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Sauvegarde du scaler
    joblib.dump(scaler, 'scaler.joblib')

    return x_train_scaled, x_test_scaled, y_train, y_test


def train_model(x_train: np.ndarray, y_train: np.ndarray, model_type: str = "RandomForest") -> ClassifierMixin:
    """Entraîne un modèle de machine learning et le retourne."""
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel='linear', probability=True),
        "LogisticRegression": LogisticRegression()
    }

    if model_type not in models:
        raise ValueError(f"Modèle non reconnu. Choisissez parmi {list(models.keys())}")

    model = models[model_type]
    model.fit(x_train, y_train)

    return model


def evaluate_model(model: ClassifierMixin, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    """Évalue le modèle et retourne accuracy et f1-score."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    return accuracy, f1


def save_model(model: ClassifierMixin, model_path: str) -> None:
    """Sauvegarde le modèle entraîné."""
    joblib.dump(model, model_path)
    print(f" Modèle sauvegardé sous {model_path}")

