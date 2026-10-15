"""Tests unitaires du lecteur de modèles."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # type: ignore[reportMissingTypeStubs]
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from prem.model_reader import ModelReader


class TestModelReader(unittest.TestCase):
    def test_get_tree_image_returns_png_bytes(self) -> None:
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

            image = ModelReader("modele.pkl", models_dir).get_tree_image()

        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))

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

            reader = ModelReader("modele.pkl", models_dir)

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

            with self.assertRaisesRegex(ValueError, "Liste de features invalide"):
                ModelReader("modele.pkl", models_dir)

    def test_rejects_unknown_model_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "Nom de modèle inconnu"):
                ModelReader("inexistant.pkl", Path(temporary_directory))


if __name__ == "__main__":
    unittest.main()
