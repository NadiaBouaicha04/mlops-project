"""
Script principal pour entraîner et sauvegarder un modèle de Machine Learning.
"""

import argparse
import mlflow
import mlflow.sklearn
from model_pipeline import prepare_data, train_model, evaluate_model, save_model

mlflow.set_tracking_uri("http://localhost:5000")

try:
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    print("✅ Connexion à MLflow réussie ! Expériences disponibles :")
    for exp in experiments:
        print(f" - ID: {exp.experiment_id}, Nom: {exp.name}")
except mlflow.exceptions.MlflowException as error:
    print(f"❌ Erreur de connexion à MLflow : {error}")


def main():
    """
    Fonction principale pour entraîner un modèle et l'enregistrer dans MLflow.
    """
    parser = argparse.ArgumentParser(description="Pipeline ML pour la prédiction de churn.")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle.")
    parser.add_argument("--data_path", type=str, help="Chemin vers le fichier de données.")
    parser.add_argument("--model_path", type=str, help="Chemin pour sauvegarder le modèle.")
    args = parser.parse_args()

    if args.train:
        x_train, x_test, y_train, y_test = prepare_data(args.data_path)
        model = train_model(x_train, y_train)

        accuracy, f1 = evaluate_model(model, x_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        save_model(model, args.model_path)
        print(f"✅ Modèle enregistré avec MLflow (Accuracy: {accuracy}, F1-Score: {f1})")


if __name__ == "__main__":
    main()
