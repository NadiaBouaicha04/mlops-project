import sys
import os

# Ajoute le dossier racine du projet au PYTHONPATH pour que les tests trouvent les modules
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
