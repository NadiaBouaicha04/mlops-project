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

# 2. Vérification des dépendances
deps:
	@echo "Installation des dépendances..."
	@if [ -f $(REQUIREMENTS) ]; then . $(ENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS); else echo "Fichier requirements.txt non trouvé"; fi

# 3. Vérification du style de code avec flake8
lint:
	@echo "Vérification du style de code..."
	@. $(ENV_NAME)/bin/activate && flake8 --max-line-length=100 .

# 4. Préparation des données
prepare:
	@echo "Préparation des données..."
	@. $(ENV_NAME)/bin/activate && python3 main.py --prepare --data_path $(DATA_PATH)

# 5. Entraînement du modèle
train:
	@echo "Entraînement du modèle..."
	@. $(ENV_NAME)/bin/activate && python3 main.py --train --data_path $(DATA_PATH)

# 6. Évaluation du modèle
evaluate:
	@echo "Évaluation du modèle..."
	@. $(ENV_NAME)/bin/activate && python3 main.py --evaluate --data_path $(DATA_PATH) --model_path $(MODEL_PATH)

# 7. Sauvegarde du modèle
save:
	@echo "Sauvegarde du modèle..."
	@. $(ENV_NAME)/bin/activate && python3 main.py --save --data_path $(DATA_PATH) --model_path $(MODEL_PATH)

# 8. Charger le modèle avec vérification
load:
	@echo "Chargement du modèle..."
	@if [ -f $(MODEL_PATH) ]; then \
		. $(ENV_NAME)/bin/activate && python3 main.py --load --model_path $(MODEL_PATH); \
	else \
		echo "Modèle non trouvé ! Entraînez-le d'abord."; \
	fi

# 9. Exécution des tests unitaires
test:
	@echo "Lancement des tests..."
	@. $(ENV_NAME)/bin/activate && pytest tests/

# 10. Nettoyer les fichiers générés
clean:
	@echo "Suppression des fichiers temporaires et artefacts..."
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf *.pkl
	rm -rf $(ENV_NAME)

# 11. Démarrer le serveur Jupyter Notebook
.PHONY: notebook
notebook:
	@echo "Démarrage de Jupyter Notebook..."
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		bash -c "source $(ENV_NAME)/bin/activate && jupyter notebook"; \
	else \
		jupyter notebook; \
	fi

# 12. Test automatique de l'ensemble du processus
.PHONY: test_pipeline
test_pipeline: setup deps lint prepare train evaluate save load test clean
	@echo "Test automatique passé avec succès !"

# 13. Exécution complète
test_all: test_pipeline

