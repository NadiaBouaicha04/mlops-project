import numpy as np
from model_pipeline import prepare_data, train_model, evaluate_model, save_model
import os

def test_prepare_data():
    """Test la préparation des données"""
    x_train, x_test, y_train, y_test = prepare_data("data/Churn_Modelling.csv")
    assert x_train.shape[0] > 0
    assert x_test.shape[0] > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_train_model():
    """Test l'entraînement du modèle"""
    x_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    model = train_model(x_train, y_train, model_type="RandomForest")
    assert model is not None

