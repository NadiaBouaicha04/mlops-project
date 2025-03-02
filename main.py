"""
Script principal pour l'entraînement et l'enregistrement du modèle avec MLflow.
"""

import argparse
import mlflow
import mlflow.sklearn
from model_pipeline import prepare_data, train_model, evaluate_model, save_model
from sklearn.metrics import f1_score
from typing import Optional

mlflow.set_tracking_uri("http://localhost:5000")

try:
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    print("Connexion à MLflow réussie ! Expériences disponibles :")
    for exp in experiments:
        print(f" - ID: {exp.experiment_id}, Nom: {exp.name}")
except mlflow.exceptions.MlflowException as error:
    print(" Erreur de connexion à MLflow :", error)


def main() -> None:
    """Pipeline de machine learning pour la prédiction du churn."""
    parser = argparse.ArgumentParser(description="Pipeline de Machine Learning pour la prédiction de churn.")
    parser.add_argument('--train', action='store_true', help="Entraîner le modèle.")
    parser.add_argument('--data_path', type=str, help="Chemin vers le fichier de données.")
    parser.add_argument('--model_path', type=str, help="Chemin pour sauvegarder le modèle.")
    args = parser.parse_args()

    with mlflow.start_run():
        if args.train:
            x_train, x_test, y_train, y_test = prepare_data(args.data_path)
            model = train_model(x_train, y_train)

            # Obtenir les prédictions du modèle
            y_pred = model.predict(x_test)

            # Calculer la métrique f1_score
            f1 = f1_score(y_test, y_pred, average="macro")

            # Logger accuracy et f1_score dans MLflow
            accuracy, f1 = evaluate_model(model, x_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)

            # Sauvegarder le modèle
            save_model(model, args.model_path)
            print(f" Modèle enregistré avec MLflow (Accuracy: {accuracy}, F1-Score: {f1})")

            # Vérification avant d'accéder à `info.run_id`
            active_run = mlflow.active_run()
            if active_run is not None:
                model_uri = f"runs:/{active_run.info.run_id}/churn_model"
                print(f" Modèle enregistré avec MLflow : {model_uri}")

                # Ajout au Model Registry
                model_name: str = "ChurnPredictionModel"
                mlflow.register_model(model_uri=model_uri, name=model_name)
                print(f" Modèle ajouté au Model Registry sous le nom {model_name}")
            else:
                print(" Erreur : Aucun run MLflow actif.")


if __name__ == "__main__":
    main()

