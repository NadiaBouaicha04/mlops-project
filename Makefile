# Déclaration des variables
PYTHON=python3
ENV_NAME=mlops_env
REQUIREMENTS=requirements.txt
DATA_PATH=Churn_Modelling.csv
MODEL_PATH=models/churn_model.pkl

# 1. Configuration de l'environnement
setup:
	@echo "Création de l'environnement virtuel et installation des dépendances..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@. $(ENV_NAME)/bin/activate && pip install --upgrade pip
	@. $(ENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS)

# 2. Préparation des données
prepare:
	@echo "Préparation des données..."
	@. $(ENV_NAME)/bin/activate && python3 main.py --prepare --data_path $(DATA_PATH)

# 3. Entraînement du modèle
train:
	@echo "Entraînement du modèle..."
	@. $(ENV_NAME)/bin/activate && python3 main.py --train --data_path $(DATA_PATH)

# 4. Évaluation du modèle
evaluate:
	@echo "Évaluation du modèle..."
	@. $(ENV_NAME)/bin/activate && python3 main.py --evaluate --data_path $(DATA_PATH) --model_path $(MODEL_PATH)

# 5. Sauvegarde du modèle
save:
	@echo "Sauvegarde du modèle..."
	@. $(ENV_NAME)/bin/activate && python3 main.py --save --data_path $(DATA_PATH) --model_path $(MODEL_PATH)

# 6. Charger le modèle
load:
	@echo "Chargement du modèle..."
	@. $(ENV_NAME)/bin/activate && python3 main.py --load --model_path $(MODEL_PATH)

# 7. Nettoyer les fichiers générés
clean:
	@echo "Suppression des fichiers temporaires et artefacts..."
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf *.pkl
	rm -rf $(ENV_NAME)

# Démarrer le serveur Jupyter Notebook
.PHONY: notebook
notebook:
	@echo "Démarrage de Jupyter Notebook..."
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		bash -c "source $(ENV_NAME)/bin/activate && jupyter notebook"; \
	else \
		jupyter notebook; \
	fi

# 8. Test automatique de l'ensemble du processus (make all)
.PHONY: test
test: setup prepare train evaluate save load clean
	@echo "Test automatique passé avec succès !"

# 9. Cible pour exécuter tout dans l'ordre (all)
all: test

