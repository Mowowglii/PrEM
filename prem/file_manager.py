
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO
from .calculateur import entrainer
from .api import status, MODELS_DIR, HTTPException

def nom_fichier_modele(name: str) -> str:
    """Valide un nom de modèle et ajoute l'extension attendue si nécessaire."""
    if Path(name).name != name or name in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le nom du modèle doit être un nom de fichier, sans chemin.",
        )
    return name if name.endswith(".pkl") else f"{name}.pkl"


def entrainer_fichier_televerse(
    fichier_csv: BinaryIO,
    max_depth: int | None,
) -> dict[str, object]:
    """Copie le fichier temporairement puis entraîne le modèle de façon synchrone."""
    with tempfile.TemporaryDirectory(prefix="prem-") as temporary_directory:
        dataset_path = Path(temporary_directory) / "dataset.csv"
        with dataset_path.open("wb") as destination:
            shutil.copyfileobj(fichier_csv, destination)
        return entrainer(dataset_path, max_depth=max_depth, models_dir=MODELS_DIR)