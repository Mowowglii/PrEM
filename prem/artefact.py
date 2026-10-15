"""Chargement et validation partagés des artefacts de modèles."""

from pathlib import Path
from typing import cast

import joblib  # type: ignore[reportMissingTypeStubs]
from sklearn.pipeline import Pipeline


def charger_artefact(path: Path) -> dict[str, object]:
    """Charge un artefact Joblib et vérifie sa structure de dictionnaire."""
    valeur = cast(object, joblib.load(path))  # type: ignore[reportUnknownMemberType]
    if not isinstance(valeur, dict):
        raise ValueError(f"Artefact de modèle invalide : {path}")
    artefact_brut = cast(dict[object, object], valeur)
    if any(not isinstance(cle, str) for cle in artefact_brut):
        raise ValueError(f"Artefact de modèle invalide : {path}")
    return cast(dict[str, object], artefact_brut)


def charger_modele(path: Path) -> tuple[Pipeline, dict[str, str]]:
    """Charge et valide le pipeline ainsi que les types des variables du modèle."""
    artefact = charger_artefact(path)
    pipeline = artefact.get("pipeline")
    features_brutes = artefact.get("features")
    if not isinstance(pipeline, Pipeline) or not isinstance(features_brutes, dict):
        raise ValueError(f"Artefact de modèle incomplet : {path}")

    features: dict[str, str] = {}
    for nom, type_feature in cast(dict[object, object], features_brutes).items():
        if (
            not isinstance(nom, str)
            or not isinstance(type_feature, str)
            or type_feature not in {"numerique", "categorielle"}
        ):
            raise ValueError(f"Liste de features invalide dans le modèle : {path}")
        features[nom] = type_feature
    return pipeline, features
