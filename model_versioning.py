from mlflow.tracking import MlflowClient

model_name = "ChurnPredictionModel"
client = MlflowClient()


# Voir les versions enregistrées
def list_versions():
    print(f" Versions enregistrées pour {model_name} :")
    for mv in client.search_model_versions(f"name='{model_name}'"):
        print(f" Version: {mv.version}, Statut: {mv.current_stage}, Source: {mv.source}")


# Passer une version en Production
def promote_version(version):
    client.transition_model_version_stage(name=model_name, version=version, stage="Production")
    print(f" Modèle {model_name} version {version} est maintenant en production !")


# Supprimer une version (si besoin)
def delete_version(version):
    client.delete_model_version(name=model_name, version=version)
    print(f" Version {version} supprimée !")


if __name__ == "__main__":
    list_versions()
    promote_version("1")  # Mets la version 1 en production
