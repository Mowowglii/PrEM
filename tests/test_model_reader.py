"""Tests unitaires du lecteur de modèles."""

import base64
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # type: ignore[reportMissingTypeStubs]
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from prem.model_reader import ModelReader


class TestModelReader(unittest.TestCase):
    def test_get_tree_image_returns_json_safe_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            models_dir = Path(temporary_directory)
            preparation = ColumnTransformer([
                ("num", SimpleImputer(strategy="median"), ["prix"]),
            ])
            pipeline = Pipeline([
                ("prep", preparation),
                ("arbre", DecisionTreeRegressor(random_state=0, max_depth=1)),
            ])
            pipeline.fit(  # type: ignore[reportUnknownMemberType]
                pd.DataFrame({"prix": [1.0, 2.0, 3.0]}),
                [1.0, 2.0, 3.0],
            )
            joblib.dump(  # type: ignore[reportUnknownMemberType]
                {
                    "pipeline": pipeline,
                    "features": {"prix": "numerique"},
                },
                models_dir / "modele.pkl",
            )

            with patch.object(ModelReader, "dir", models_dir):
                image = ModelReader("modele.pkl").get_tree_image()

        self.assertEqual(image["media_type"], "image/png")
        png_data = base64.b64decode(image["data_base64"], validate=True)
        self.assertTrue(png_data.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_get_infos_compte_les_features_numeriques_et_categorielles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            models_dir = Path(temporary_directory)
            joblib.dump(  # type: ignore[reportUnknownMemberType]
                {
                    "pipeline": Pipeline([("arbre", DecisionTreeRegressor())]),
                    "features": {
                        "prix": "numerique",
                        "temperature": "numerique",
                        "zone": "categorielle",
                    },
                    "rapport": {},
                },
                models_dir / "modele.pkl",
            )

            with patch.object(ModelReader, "dir", models_dir):
                reader = ModelReader("modele.pkl")

            self.assertEqual(
                reader.get_infos(),
                {
                    "nombre_variables_numeriques": 2,
                    "variables_numeriques": ["prix", "temperature"],
                    "nombre_variables_qualitatives": 1,
                    "variables_qualitatives": ["zone"],
                },
            )
            self.assertIsInstance(reader.model, Pipeline)

    def test_rejects_artifact_with_invalid_feature_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            models_dir = Path(temporary_directory)
            joblib.dump(  # type: ignore[reportUnknownMemberType]
                {
                    "pipeline": Pipeline([("arbre", DecisionTreeRegressor())]),
                    "features": {"prix": "inconnu"},
                },
                models_dir / "modele.pkl",
            )

            with patch.object(ModelReader, "dir", models_dir):
                with self.assertRaisesRegex(ValueError, "Liste de features invalide"):
                    ModelReader("modele.pkl")

    def test_rejects_unknown_model_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(ModelReader, "dir", Path(temporary_directory)):
                with self.assertRaisesRegex(ValueError, "Nom de modèle inconnu"):
                    ModelReader("inexistant.pkl")


if __name__ == "__main__":
    unittest.main()
