# ==============================
# Variables
# ==============================
PYTHON = python
VENV = venv
MLFLOW_PORT = 5000
EXPERIMENT_NAME = churn_experiment

IMAGE_FASTAPI = nadiabouaicha/fastapi-churn-v2
IMAGE_STREAMLIT = nadiabouaicha/streamlit-churn-v2
IMAGE_MLFLOW = nadiabouaicha/mlflow-server-v2

CONTAINER_FASTAPI = fastapi-container-v2
CONTAINER_STREAMLIT = streamlit-container-v2
CONTAINER_MLFLOW = mlflow-container-v2

NETWORK_NAME = churn-network

# ==============================
# Création de l'environnement virtuel
# ==============================
venv:
	@echo "Création de l'environnement virtuel..."
	$(PYTHON) -m venv $(VENV)
	@echo "Activation de l'environnement virtuel et installation des dépendances..."
	. $(VENV)/bin/activate; pip install --upgrade pip && pip install -r requirements.txt

# ==============================
# Lancer MLflow UI dans un container Docker
# ==============================
mlflow-docker:
	@echo "Démarrage du serveur MLflow dans Docker..."
	sudo docker rm -f $(CONTAINER_MLFLOW) || true
	sudo docker run -d --name $(CONTAINER_MLFLOW) \
		-p $(MLFLOW_PORT):5000 \
		-v /mlruns:/mlflow \
		-e MLFLOW_TRACKING_URI=http://0.0.0.0:5000 \
		--network $(NETWORK_NAME) \
		$(IMAGE_MLFLOW)

# ==============================
# Construire les images Docker
# ==============================
docker-build:
	@echo "Construction des images Docker..."
	sudo docker build -t $(IMAGE_MLFLOW) -f Dockerfile.mlflow .
	sudo docker build -t $(IMAGE_FASTAPI) -f Dockerfile .
	sudo docker build -t $(IMAGE_STREAMLIT) -f Dockerfile.streamlit .

# ==============================
# Lancer les containers
# ==============================
docker-run:
	@echo "Lancement des containers Docker..."
	sudo docker network create $(NETWORK_NAME) || true
	sudo docker run -d --name $(CONTAINER_FASTAPI) --network $(NETWORK_NAME) -p 8000:8000 $(IMAGE_FASTAPI)
	sudo docker run -d --name $(CONTAINER_STREAMLIT) --network $(NETWORK_NAME) -p 8501:8501 $(IMAGE_STREAMLIT)

# ==============================
# Exécuter l'entraînement du modèle
# ==============================
train:
	@echo "Entraînement du modèle et enregistrement dans MLflow..."
	. $(VENV)/bin/activate; python main.py --train --data_path "data/Churn_Modelling.csv" --model_path "models/churn_model.pkl"

# ==============================
# Pousser les images sur Docker Hub
# ==============================
docker-push:
	@echo "Poussée des images Docker sur Docker Hub..."
	sudo docker push $(IMAGE_MLFLOW)
	sudo docker push $(IMAGE_FASTAPI)
	sudo docker push $(IMAGE_STREAMLIT)

# ==============================
# Nettoyer les containers et images Docker
# ==============================
clean:
	@echo "Suppression des containers et images Docker..."
	sudo docker rm -f $(CONTAINER_FASTAPI) $(CONTAINER_STREAMLIT) $(CONTAINER_MLFLOW) || true
	sudo docker rmi -f $(IMAGE_FASTAPI) $(IMAGE_STREAMLIT) $(IMAGE_MLFLOW) || true
	sudo docker network rm $(NETWORK_NAME) || true
	sudo docker system prune -f

# ==============================
# Exécuter toutes les étapes en une seule commande
# ==============================
all: venv docker-build mlflow-docker docker-run train docker-push
	@echo " Déploiement complet terminé ! "
	
# Analyse qualité du code
lint:
	@echo " Analyse du code avec pylint..."
	pylint api.py model_pipeline.py main.py

format:
	@echo "🖊️ Formatage du code avec black..."
	black .

type-check:
	@echo " Vérification des types avec mypy..."
	mypy api.py model_pipeline.py main.py

# Exécuter les tests unitaires
test:
	@echo " Exécution des tests avec pytest..."
	pytest --cov=api --cov-report=term-missing

ci-checks: lint format type-check test
	@echo " Tous les checks CI sont passés !"


