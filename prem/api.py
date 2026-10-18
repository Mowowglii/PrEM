"""API FastAPI pour entraîner, consulter et utiliser les modèles CART."""
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Path as PathParameter,
    Response,
    UploadFile,
    status,
)
from pandas.errors import ParserError

from .file_manager import (
    enregistrer_dataset,
    entrainer_fichier_televerse,
    entrainer_modele_televerse,
    nom_fichier_dataset,
    nom_fichier_modele,
)
from .calculateur import DatasetInvalide
from .comestible import Comestible
from .decideur import Prediction, predire
from .model_reader import ModelReader
from .paths import DATASETS_DIR, MODELS_DIR

app = FastAPI(
    title="PrEM",
    description="API de prédiction de la variation hebdomadaire du prix de l’électricité.",
)


def _nom_modele_requete(name: str) -> str:
    try:
        return nom_fichier_modele(name)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error


def _nom_dataset_requete(name: str) -> str:
    try:
        return nom_fichier_dataset(name)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error


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
        return entrainer_fichier_televerse(
            csv.file,
            max_depth,
            MODELS_DIR,
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
def get_dataset(
    name: Annotated[str, PathParameter(title="Nom du dataset à récupérer")],
) -> dict[str, str]:
    """Renvoie le nom et le contenu du fichier CSV demandé."""
    filename = _nom_dataset_requete(name)
    dataset_path = DATASETS_DIR / filename
    if not dataset_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset introuvable : {filename}",
        )
    try:
        contenu = dataset_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Le dataset {filename} n'est pas encodé en UTF-8.",
        ) from error
    return {"name": filename, "content": contenu}


@app.post("/datasets")
def create_dataset(
    csv: Annotated[UploadFile, File(description="Fichier CSV à sauvegarder")],
) -> str:
    """Sauvegarde le CSV téléversé et renvoie son nom."""
    if not csv.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Aucun dataset n'a été fourni.",
        )
    filename = _nom_dataset_requete(csv.filename)
    contenu = csv.file.read()
    try:
        return enregistrer_dataset(
            filename,
            contenu,
            DATASETS_DIR,
        )
    except FileExistsError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error


@app.post("/models/{name}/train")
def train_model(
    name: Annotated[str, PathParameter(title="Nom du modèle que l'on souhaite entrainer")],
    csv: Annotated[UploadFile, File(description="Fichier CSV d'entraînement")],
) -> dict[str, object]:
    """Réentraîne un modèle existant avec le CSV téléversé."""
    filename = _nom_modele_requete(name)
    if not (MODELS_DIR / filename).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle introuvable : {filename}",
        )
    if not csv.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Aucun jeu de données n'a été fourni.",
        )
    try:
        return entrainer_modele_televerse(
            csv.file,
            filename,
            MODELS_DIR,
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except (DatasetInvalide, ValueError, ParserError, UnicodeDecodeError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error


@app.get("/models/{name}/features")
def get_features_of_model(
    name: Annotated[str, PathParameter(title="Nom du modèle dont on cherche les variables")],
) -> dict[str, int | list[str]]:
    """Renvoie les variables explicatives du modèle, classées par type."""
    filename = _nom_modele_requete(name)
    if not (MODELS_DIR / filename).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle introuvable : {filename}",
        )
    try:
        return ModelReader(filename, MODELS_DIR).get_infos()
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error


@app.get("/models/{name}/tree", response_class=Response)
def get_tree_of_model(
    name: Annotated[str, PathParameter(title="Nom du modèle dont on affiche l'arbre")],
) -> Response:
    """Renvoie l'arbre de décision sous forme d'une image PNG."""
    filename = _nom_modele_requete(name)
    if not (MODELS_DIR / filename).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle introuvable : {filename}",
        )
    try:
        reader = ModelReader(filename, MODELS_DIR)
        return Response(content=reader.get_tree_image(), media_type="image/png")
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error


@app.post("/models/{name}/predict")
def predict(
    name: Annotated[str, PathParameter(title="Nom du modèle à utiliser")],
    input: Comestible,
) -> Prediction:
    """Prédit dans le threadpool géré par FastAPI pour cette route synchrone."""
    try:
        filename = _nom_modele_requete(name)
        return predire(
            input,
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
def delete_model(
    name: Annotated[str, PathParameter(title="Nom du modèle à supprimer")]
) -> dict[str, str]:
    """Supprime le modèle demandé et renvoie le nom du fichier supprimé."""
    filename = _nom_modele_requete(name)
    chemin = MODELS_DIR / filename
    if not chemin.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle introuvable : {filename}",
        )
    try:
        chemin.unlink()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modèle introuvable : {filename}",
        ) from error
    return {"deleted": filename}


@app.delete("/datasets/{name}")
def delete_dataset(
    name: Annotated[str, PathParameter(title="Nom du dataset à supprimer")]
) -> dict[str, str]:
    """Supprime le dataset demandé et renvoie le nom du fichier supprimé."""
    filename = _nom_dataset_requete(name)
    chemin = DATASETS_DIR / filename
    if not chemin.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset introuvable : {filename}",
        )
    try:
        chemin.unlink()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset introuvable : {filename}",
        ) from error
    return {"deleted": filename}