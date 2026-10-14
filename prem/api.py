"""API FastAPI pour entraîner, consulter et utiliser les modèles CART."""

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Path as PathParameter, UploadFile, status
from pandas.errors import ParserError

from calculateur import DatasetInvalide, entrainer
from comestible import Comestible
from decideur import Prediction, predire
from model_reader import ModelReader

DATASETS_DIR = Path("data")
MODELS_DIR = Path("cart_models")

app = FastAPI(
    title="PrEM",
    description="API de prédiction de la variation hebdomadaire du prix de l’électricité.",
)


def _nom_fichier_modele(name: str) -> str:
    """Valide un nom de modèle et ajoute l'extension attendue si nécessaire."""
    if Path(name).name != name or name in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le nom du modèle doit être un nom de fichier, sans chemin.",
        )
    return name if name.endswith(".pkl") else f"{name}.pkl"


@app.post("/models", status_code=status.HTTP_201_CREATED)
def create_model(
    csv: Annotated[UploadFile, File(description="Fichier CSV d'entraînement")],
    max_depth: Annotated[int | None, Form()] = None,
) -> dict[str, object]:
    """Entraîne puis sauvegarde un modèle depuis un CSV envoyé en multipart."""
    if not csv.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Aucun jeu de données n'a été fourni.",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="prem-") as temporary_directory:
            dataset_path = Path(temporary_directory) / "dataset.csv"
            with dataset_path.open("wb") as destination:
                shutil.copyfileobj(csv.file, destination)
            return entrainer(dataset_path, max_depth=max_depth, models_dir=MODELS_DIR)
    except (DatasetInvalide, ValueError, ParserError, UnicodeDecodeError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error


@app.get("/models")
def get_models() -> list[str]:
    """Renvoie les noms des modèles entraînés."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(model.name for model in MODELS_DIR.glob("*.pkl") if model.is_file())


@app.get("/datasets")
def get_datasets() -> list[str]:
    """Renvoie les noms des fichiers CSV disponibles dans le dossier des données."""
    if not DATASETS_DIR.is_dir():
        return []
    return sorted(dataset.name for dataset in DATASETS_DIR.glob("*.csv") if dataset.is_file())


@app.get("/models/{name}/features")
def get_features_of_model(
    name: Annotated[str, PathParameter(title="Nom du modèle dont on cherche les variables")],
) -> dict[str, int | list[str]]:
    """Renvoie les variables explicatives du modèle, classées par type."""
    filename = _nom_fichier_modele(name)
    if not (MODELS_DIR / filename).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle introuvable : {filename}",
        )
    try:
        return ModelReader(filename).get_infos()
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error


@app.post("/models/{name}/predict")
@app.get("/models/{name}/predict")
def predict(
    name: Annotated[str, PathParameter(title="Nom du modèle à utiliser")],
    input: Comestible,
) -> Prediction:
    """Prédit la variation de prix et renvoie le chemin suivi dans l'arbre."""
    filename = _nom_fichier_modele(name)
    try:
        return predire(input.vers_dict(), filename, MODELS_DIR)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
