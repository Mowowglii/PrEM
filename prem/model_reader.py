"""Charge un modèle CART et expose ses informations et son arbre."""

from io import BytesIO
from pathlib import Path
from typing import cast

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor, plot_tree

from .artefact import charger_modele
from .paths import MODELS_DIR


class ModelReader:
    def __init__(self, name: str, models_dir: Path = MODELS_DIR) -> None:
        path = models_dir / name
        if not path.is_file():
            raise ValueError(f"Nom de modèle inconnu : {name}")

        self.model, self.features = charger_modele(path)

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

    def get_tree_image(self) -> bytes:
        """Renvoie les octets PNG de l'arbre de décision."""
        etapes = cast(dict[str, object], self.model.named_steps)
        preparation = etapes.get("prep")
        arbre = etapes.get("arbre")
        if not isinstance(preparation, ColumnTransformer):
            raise ValueError("Le préprocesseur du modèle est invalide")
        if not isinstance(arbre, DecisionTreeRegressor):
            raise ValueError("L'arbre de décision du modèle est invalide")

        noms_features = [
            str(nom) for nom in preparation.get_feature_names_out()
        ]
        figure = Figure(figsize=(20, 12), constrained_layout=True)
        canvas = FigureCanvasAgg(figure)
        axe = figure.subplots()
        plot_tree(
            arbre,
            feature_names=noms_features,
            filled=True,
            rounded=True,
            ax=axe,
        )

        image = BytesIO()
        canvas.print_png(image)  # type: ignore[reportUnknownMemberType]
        return image.getvalue()
