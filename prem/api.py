"""API FastAPI pour entraîner, consulter et utiliser les modèles CART."""
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Path as PathParameter, UploadFile, status
from pandas.errors import ParserError
from starlette.concurrency import run_in_threadpool

from .file_manager import entrainer_fichier_televerse, nom_fichier_modele
from .calculateur import DatasetInvalide
from .comestible import Comestible
from .decideur import Prediction, predire
from .model_reader import ModelReader

DATASETS_DIR = Path("data")
MODELS_DIR = Path("cart_models")

app = FastAPI(
    title="PrEM",
    description="API de prédiction de la variation hebdomadaire du prix de l’électricité.",
)

@app.post("/models", status_code=status.HTTP_201_CREATED)
async def create_model(
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
        return await run_in_threadpool(
            entrainer_fichier_televerse,
            csv.file,
            max_depth,
        )
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
def get_datasets_name() -> list[str]:
    """Renvoie les noms des fichiers CSV disponibles dans le dossier des données."""
    if not DATASETS_DIR.is_dir():
        return []
    return [dataset.name for dataset in DATASETS_DIR.glob("*.csv") if dataset.is_file()]

@app.get("/datasets/{name}")
async def get_dataset() -> dict[str, str]:
    """Renvoie le fichier (dataset) en .csv portant ce nom dans la requête"""

@app.post("/datasets")
async def create_dataset() -> str:
    """Créer le dataset, sauvegarde le fichier dans le dossier des datasets et renvoie le nom du fichier.csv téléversé"""

@app.post("/models/{name}/train")
async def train_model(
    name: Annotated[str, PathParameter(title="Nom du modèle que l'on souhaite entrainer")],
    csv : Annotated[UploadFile, File(description="Fichier csv d'entrainement")]
) -> dict[str, object]:
    """Entrainer un modèle déjà existant et Récupérer un rapport d'entrainement"""


@app.get("/models/{name}/features")
def get_features_of_model(
    name: Annotated[str, PathParameter(title="Nom du modèle dont on cherche les variables")],
) -> dict[str, int | list[str]]:
    """Renvoie les variables explicatives du modèle, classées par type."""
    filename = nom_fichier_modele(name)
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


@app.get("/models/{name}/tree")
async def get_tree_of_model(
    name: Annotated[str, PathParameter(title="Nom du modèle dont on affiche l'arbre")],
) -> dict[str, str]:
    """Renvoie l'image PNG de l'arbre en Base64 dans une réponse JSON."""
    filename = nom_fichier_modele(name)
    if not (MODELS_DIR / filename).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle introuvable : {filename}",
        )
    try:
            reader = ModelReader(filename)
            return await run_in_threadpool(reader.get_tree_image)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error


@app.post("/models/{name}/predict")
async def predict(
    name: Annotated[str, PathParameter(title="Nom du modèle à utiliser")],
    input: Comestible,
) -> Prediction:
    """Prédit sans bloquer la boucle d'événements du serveur."""
    filename = nom_fichier_modele(name)
    try:
        return await run_in_threadpool(
            predire,
            input.get_dict(),
            filename,
            MODELS_DIR,
        )
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


@app.delete("/models/{name}")
async def delete_model(
    name: Annotated[str, PathParameter(title="Nom du modèle à supprimer")]
) -> dict[str, str]:
    """Supprime le Model dont le nom est en requête et Renvoie le fichier .pkl supprimé"""
    
@app.delete("/datasets/{name}")
async def delete_dataset(
    name: Annotated[str, PathParameter(title="Nom du dataset à supprimer")]
) -> dict[str, str]:
    """Supprimer le DataSet dont le nom est en requête et Renvoie le fichier .csv supprimé"""