import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import platform

# 📌 Utilisation d'un backend sans interface graphique (remplace TkAgg)
matplotlib.use("Agg")

# Connexion à MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.tracking.MlflowClient()

# Récupérer l'expérience par son nom
experiment = client.get_experiment_by_name("Default")  # Remplace par le bon nom si nécessaire
experiment_id = experiment.experiment_id if experiment else "0"

# Récupérer tous les runs de l'expérience
runs = client.search_runs(experiment_ids=[experiment_id])

# Vérifier s'il y a des métriques disponibles
if not runs:
    print("⚠️ Aucun run trouvé dans MLflow.")
else:
    print(f"📊 {len(runs)} runs trouvés !")

    # Extraire les métriques
    run_ids = []
    accuracies = []
    f1_scores = []
    timestamps = []

    for run in runs:
        metrics = run.data.metrics  # 📌 Vérifie que tes métriques existent bien dans MLflow UI
        if "accuracy" in metrics and "f1_score" in metrics:
            run_ids.append(run.info.run_id)
            accuracies.append(metrics["accuracy"])
            f1_scores.append(metrics["f1_score"])
            timestamps.append(run.info.start_time)

    if len(accuracies) > 0:
        # 📌 Création d'un DataFrame pour affichage
        df = pd.DataFrame(
            {
                "Run ID": run_ids,
                "Timestamp": timestamps,
                "Accuracy": accuracies,
                "F1-Score": f1_scores,
            }
        )
        df = df.sort_values("Timestamp")

        # 📊 Création du graphique
        plt.figure(figsize=(10, 5))
        plt.plot(df["Timestamp"], df["Accuracy"], marker="o", linestyle="-", label="Accuracy")
        plt.plot(df["Timestamp"], df["F1-Score"], marker="s", linestyle="--", label="F1-Score")
        plt.xlabel("Temps")
        plt.ylabel("Score")
        plt.title("Évolution de l'Accuracy et du F1-Score")
        plt.legend()
        plt.grid()
        plt.xticks(rotation=45)

        # 📌 Sauvegarde du graphique au lieu d'afficher (MLflow ne capture pas plt.show())
        plot_path = "monitoring_plot.png"
        plt.savefig(plot_path)
        print(f"📊 Graphique enregistré sous '{plot_path}'")

        # 📌 Enregistrement du graphique dans MLflow
        with mlflow.start_run():
            mlflow.log_artifact(plot_path)

        # 📌 Ouvrir l'image si possible
        if os.path.exists(plot_path):
            try:
                if platform.system() == "Linux":
                    os.system(f"xdg-open {plot_path}")
                elif platform.system() == "Darwin":  # MacOS
                    os.system(f"open {plot_path}")
                elif platform.system() == "Windows":
                    os.system(f"start {plot_path}")
            except Exception as e:
                print(f"⚠️ Impossible d'ouvrir l'image automatiquement : {e}")

    else:
        print("⚠️ Aucune métrique `accuracy` ou `f1_score` trouvée.")
