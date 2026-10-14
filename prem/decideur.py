"""Charge un modèle CART et explique sa prédiction pour un Comestible."""

from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path
from typing import Mapping, Protocol, cast

import joblib  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

MODELS_DIR: Path = Path("cart_models")


class _StructureArbre(Protocol):
    children_left: NDArray[np.intp]
    children_right: NDArray[np.intp]
    feature: NDArray[np.intp]
    threshold: NDArray[np.float64]
    n_node_samples: NDArray[np.intp]


class _CheminDecisionCreux(Protocol):
    indices: Iterable[int]


@dataclass(frozen=True)
class Prediction:
    """Prédiction du modèle accompagnée du chemin suivi dans l'arbre."""

    model: str
    delta_prix_eur_mwh: float
    chemin: list[str]
    features_utilisees: list[str]
    n_semaines_feuille: int


def _chemin_modele(models_dir: Path, nom_modele: str | None) -> Path:
    if nom_modele is None:
        modeles = list(models_dir.glob("*.pkl"))
        if not modeles:
            raise FileNotFoundError(f"Aucun modèle .pkl trouvé dans {models_dir}")
        return max(modeles, key=lambda chemin: (chemin.stat().st_mtime_ns, chemin.name))

    if Path(nom_modele).name != nom_modele or nom_modele in {".", ".."}:
        raise ValueError("Le nom du modèle doit être un nom de fichier, sans chemin")
    fichier = nom_modele if nom_modele.endswith(".pkl") else f"{nom_modele}.pkl"
    chemin = models_dir / fichier
    if not chemin.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {chemin}")
    return chemin


def _charger_modele(
    chemin: Path,
) -> tuple[Pipeline, dict[str, str]]:
    artefact = cast(object, joblib.load(chemin))  # type: ignore[reportUnknownMemberType]
    if not isinstance(artefact, dict):
        raise ValueError(f"Artefact de modèle invalide : {chemin}")

    donnees_artefact = cast(dict[object, object], artefact)
    pipeline = donnees_artefact.get("pipeline")
    features_brutes = donnees_artefact.get("features")
    if not isinstance(pipeline, Pipeline) or not isinstance(features_brutes, dict):
        raise ValueError(f"Artefact de modèle incomplet : {chemin}")

    features: dict[str, str] = {}
    for nom, type_feature in cast(dict[object, object], features_brutes).items():
        if (
            not isinstance(nom, str)
            or not isinstance(type_feature, str)
            or type_feature not in {"numerique", "categorielle"}
        ):
            raise ValueError(f"Liste de features invalide dans le modèle : {chemin}")
        features[nom] = type_feature
    return pipeline, features


def _nom_feature_origine(nom_transforme: str, features: list[str]) -> tuple[str, str]:
    """Retrouve la feature d'origine et, le cas échéant, sa catégorie encodée."""
    nom_sans_transformeur = nom_transforme.partition("__")[2]
    candidates = [
        feature for feature in features
        if nom_sans_transformeur == feature or nom_sans_transformeur.startswith(f"{feature}_")
    ]
    if not candidates:
        raise ValueError(f"Feature transformée non reconnue : {nom_transforme}")

    # Le plus long préfixe évite de confondre, par exemple, foo et foo_bar.
    origine = max(candidates, key=len)
    suffixe = nom_sans_transformeur[len(origine):].removeprefix("_")
    return origine, suffixe


def _expliquer_chemin(
    preparation: ColumnTransformer,
    arbre: DecisionTreeRegressor,
    ligne: pd.DataFrame,
    features: list[str],
    types_features: dict[str, str],
) -> tuple[list[str], list[str], int]:
    donnees_transformees = preparation.transform(ligne)
    noms_transformes = preparation.get_feature_names_out()
    chemin_decision = cast(
        _CheminDecisionCreux,
        arbre.decision_path(donnees_transformees),
    )
    noeuds_visites = {int(indice) for indice in chemin_decision.indices}
    structure_arbre = cast(_StructureArbre, arbre.tree_)

    conditions: list[str] = []
    features_utilisees: list[str] = []
    noeud = 0
    while structure_arbre.children_left[noeud] != structure_arbre.children_right[noeud]:
        indice_feature = int(structure_arbre.feature[noeud])
        nom_transforme = str(noms_transformes[indice_feature])
        feature, categorie = _nom_feature_origine(nom_transforme, features)
        gauche = int(structure_arbre.children_left[noeud])
        droite = int(structure_arbre.children_right[noeud])

        if gauche in noeuds_visites:
            noeud_suivant = gauche
            operateur = "!=" if types_features[feature] == "categorielle" and categorie else "<="
        elif droite in noeuds_visites:
            noeud_suivant = droite
            operateur = "=" if types_features[feature] == "categorielle" and categorie else ">"
        else:
            raise RuntimeError("Le chemin de décision ne correspond pas à l'arbre chargé")

        if operateur in {"=", "!="}:
            condition = f"{feature} {operateur} {categorie!r}"
        else:
            condition = f"{feature} {operateur} {structure_arbre.threshold[noeud]:g}"
        conditions.append(condition)
        if feature not in features_utilisees:
            features_utilisees.append(feature)
        noeud = noeud_suivant

    return conditions, features_utilisees, int(structure_arbre.n_node_samples[noeud])


def predire(
    comestible: Mapping[str, object],
    model: str | None = None,
    models_dir: Path = MODELS_DIR,
) -> Prediction:
    """Prédit la variation de prix pour un seul dictionnaire de features.

    Les features omises sont présentées au pipeline comme manquantes afin qu'il
    applique ses imputeurs. ``model`` accepte le nom du modèle avec ou sans
    extension ``.pkl`` ; sans nom, le fichier le plus récemment modifié est pris.
    """
    chemin_modele = _chemin_modele(models_dir, model)
    pipeline, types_features = _charger_modele(chemin_modele)
    features = list(types_features)

    inconnues = set(comestible) - set(features)
    if inconnues:
        noms = ", ".join(sorted(map(str, inconnues)))
        raise ValueError(f"Feature(s) inconnue(s) pour le modèle : {noms}")

    valeurs = {feature: comestible.get(feature) for feature in features}
    ligne = pd.DataFrame([valeurs], columns=features)
    for feature, type_feature in types_features.items():
        if type_feature == "numerique":
            valeur = valeurs[feature]
            if valeur is not None and (
                isinstance(valeur, (bool, np.bool_))
                or not isinstance(valeur, (int, float, np.integer, np.floating))
            ):
                raise ValueError(f"La feature « {feature} » doit être numérique ou None")
            ligne[feature] = pd.to_numeric(ligne[feature], errors="raise")
        elif type_feature != "categorielle":
            raise ValueError(f"Type de feature inconnu dans le modèle : {type_feature}")

    etapes = cast(dict[str, object], pipeline.named_steps)
    preparation = etapes.get("prep")
    arbre = etapes.get("arbre")
    if not isinstance(preparation, ColumnTransformer):
        raise ValueError(f"Le préprocesseur du modèle est invalide : {chemin_modele}")
    if not isinstance(arbre, DecisionTreeRegressor):
        raise ValueError(f"Le pipeline du modèle est invalide : {chemin_modele}")

    valeur_predite = float(
        pipeline.predict(ligne)[0]  # type: ignore[reportUnknownMemberType]
    )
    conditions, features_utilisees, n_semaines_feuille = _expliquer_chemin(
        preparation,
        arbre,
        ligne,
        features,
        types_features,
    )
    return Prediction(
        model=chemin_modele.name,
        delta_prix_eur_mwh=valeur_predite,
        chemin=conditions,
        features_utilisees=features_utilisees,
        n_semaines_feuille=n_semaines_feuille,
    )