"""Chemins de données et de modèles, indépendants du répertoire courant."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "cart_models"
