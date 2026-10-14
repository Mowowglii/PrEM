"""Helpers pour les fichiers téléversés et leurs noms."""

import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO

from .calculateur import creer, entrainer_deja_existant


def nom_fichier_modele(name: str) -> str:
    """Valide un nom de modèle et ajoute l'extension attendue si nécessaire."""
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Le nom du modèle doit être un nom de fichier, sans chemin.")
    return name if name.endswith(".pkl") else f"{name}.pkl"


def nom_fichier_dataset(name: str) -> str:
    """Valide un nom de dataset CSV et ajoute l'extension attendue."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Le nom du dataset doit être un nom de fichier, sans chemin.")
    suffix = Path(name).suffix.lower()
    if suffix not in {"", ".csv"}:
        raise ValueError("Le dataset doit être un fichier CSV.")
    return name if suffix == ".csv" else f"{name}.csv"


def entrainer_fichier_televerse(
    fichier_csv: BinaryIO,
    max_depth: int | None,
    models_dir: Path,
) -> dict[str, object]:
    """Entraîne un nouveau modèle depuis un fichier téléversé temporaire."""
    with tempfile.TemporaryDirectory(prefix="prem-") as temporary_directory:
        dataset_path = Path(temporary_directory) / "dataset.csv"
        with dataset_path.open("wb") as destination:
            shutil.copyfileobj(fichier_csv, destination)
        return creer(dataset_path, max_depth=max_depth, models_dir=models_dir)


def entrainer_modele_televerse(
    fichier_csv: BinaryIO,
    nom_modele: str,
    models_dir: Path,
) -> dict[str, object]:
    """Réentraîne un modèle existant depuis un fichier téléversé temporaire."""
    with tempfile.TemporaryDirectory(prefix="prem-") as temporary_directory:
        dataset_path = Path(temporary_directory) / "dataset.csv"
        with dataset_path.open("wb") as destination:
            shutil.copyfileobj(fichier_csv, destination)
        return entrainer_deja_existant(dataset_path, nom_modele, models_dir)


def enregistrer_dataset(
    nom_fichier: str,
    contenu: bytes,
    datasets_dir: Path,
) -> str:
    """Enregistre un dataset sans écraser un fichier existant."""
    datasets_dir.mkdir(parents=True, exist_ok=True)
    destination = datasets_dir / nom_fichier
    if destination.exists():
        raise FileExistsError(f"Le dataset existe déjà : {nom_fichier}")
    destination.write_bytes(contenu)
    return nom_fichier
