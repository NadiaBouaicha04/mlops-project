import pytest
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model
import joblib
import os


@pytest.fixture(scope="module")
def data():
    # Prépare les données et retourne les ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = prepare_data()
    return x_train, x_test, y_train, y_test


@pytest.fixture(scope="module")
def model(data):
    x_train, x_test, y_train, y_test = data
    # Entraîne le modèle et retourne l'objet model
    model = train_model(x_train, y_train, model_name="Random Forest")
    return model


def test_model_training(model, data):
    """Vérifie que le modèle peut être entraîné sans erreur"""
    assert model is not None, "Le modèle n'a pas été entraîné correctement"


def test_model_accuracy(model, data):
    """Vérifie que l'accuracy du modèle est supérieure à un seuil donné"""
    x_train, x_test, y_train, y_test = data
    y_pred = model.predict(x_test)

    accuracy = (y_pred == y_test).mean()
    assert accuracy > 0.8, f"Accuracy est trop faible : {accuracy}"


def test_classification_report(model, data):
    """Vérifie que le rapport de classification contient les bonnes métriques"""
    from sklearn.metrics import classification_report

    x_train, x_test, y_train, y_test = data
    y_pred = model.predict(x_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    assert "0" in report, "La classe 0 n'est pas présente dans le rapport"
    assert "1" in report, "La classe 1 n'est pas présente dans le rapport"


def test_confusion_matrix(model, data):
    """Vérifie que la matrice de confusion est bien générée"""
    from sklearn.metrics import confusion_matrix

    x_train, x_test, y_train, y_test = data
    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    assert cm.shape == (2, 2), f"Matrice de confusion inattendue : {cm.shape}"


def test_model_saving_and_loading(model):
    """Vérifie la sauvegarde et le chargement du modèle"""
    model_path = "models/churn_model.pkl"

    # Sauvegarder le modèle
    save_model(model, model_path)
    assert os.path.exists(model_path), "Le modèle n'a pas été sauvegardé correctement"

    # Charger le modèle
    loaded_model = load_model(model_path)
    assert loaded_model is not None, "Le modèle chargé est invalide"

    # Vérifier que le modèle chargé fonctionne
    assert model.predict([[0.0, 0.0, 0.0]]) == loaded_model.predict(
        [[0.0, 0.0, 0.0]]
    ), "Le modèle chargé ne correspond pas au modèle original"


if __name__ == "__main__":
    pytest.main()
