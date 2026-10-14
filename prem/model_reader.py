import joblib  # type: ignore[reportMissingTypeStubs]
from pathlib import Path
from typing import cast

from sklearn.pipeline import Pipeline


class ModelReader:
    dir = Path("cart_models")

    def __init__(self, name: str) -> None:
        if not ModelReader.dir.joinpath(name).is_file():
            raise ValueError(f"Nom de modèle inconnu : {name}")

        path = ModelReader.dir.joinpath(name)
        artefact = cast(object, joblib.load(path))  # type: ignore[reportUnknownMemberType]
        if not isinstance(artefact, dict):
            raise ValueError(f"Artefact de modèle invalide : {path}")

        donnees_artefact = cast(dict[object, object], artefact)
        pipeline = donnees_artefact.get("pipeline")
        features = donnees_artefact.get("features")
        if not isinstance(pipeline, Pipeline) or not isinstance(features, dict):
            raise ValueError(f"Artefact de modèle incomplet : {path}")

        features_validees: dict[str, str] = {}
        for nom, type_feature in cast(dict[object, object], features).items():
            if (
                not isinstance(nom, str)
                or not isinstance(type_feature, str)
                or type_feature not in {"numerique", "categorielle"}
            ):
                raise ValueError(f"Liste de features invalide dans le modèle : {path}")
            features_validees[nom] = type_feature

        self.model: Pipeline = pipeline
        self.features = features_validees

    def get_infos(self) -> dict[str, int | list[str]]:
        """Renvoie les noms et les nombres de variables explicatives par type."""
        variables_numeriques = [
            nom for nom, type_feature in self.features.items()
            if type_feature == "numerique"
        ]
        variables_qualitatives = [
            nom for nom, type_feature in self.features.items()
            if type_feature == "categorielle"
        ]
        return {
            "nombre_variables_numeriques": len(variables_numeriques),
            "variables_numeriques": variables_numeriques,
            "nombre_variables_qualitatives": len(variables_qualitatives),
            "variables_qualitatives": variables_qualitatives,
        }
        