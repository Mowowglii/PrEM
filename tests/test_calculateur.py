"""Tests des opérations d'entraînement des modèles."""

import sys
import tempfile
import unittest
from pathlib import Path

import joblib  # type: ignore[reportMissingTypeStubs]
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prem.calculateur import (
    DatasetInvalide,
    construire_pipeline,
    entrainer_deja_existant,
)


class TestEntrainerDejaExistant(unittest.TestCase):
    def test_retrains_and_updates_existing_artifact(self) -> None:
        donnees = pd.DataFrame({
            "semaine": pd.date_range("2025-01-01", periods=60, freq="W"),
            "prix": [float(index % 10) for index in range(60)],
            "zone": ["nord" if index % 2 else "sud" for index in range(60)],
            "delta_prix": [float(index % 10) * 2 for index in range(60)],
        })
        X = donnees.drop(columns=["semaine", "delta_prix"])
        y = donnees["delta_prix"]
        pipeline = construire_pipeline(X)
        pipeline.fit(X, y)  # type: ignore[reportUnknownMemberType]

        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            models_dir = workspace / "models"
            models_dir.mkdir()
            dataset_path = workspace / "dataset.csv"
            donnees.to_csv(dataset_path, index=False)
            model_path = models_dir / "modele.pkl"
            joblib.dump(  # type: ignore[reportUnknownMemberType]
                {
                    "pipeline": pipeline,
                    "features": {"prix": "numerique", "zone": "categorielle"},
                    "rapport": {"model": "modele", "ancienne_cle": True},
                },
                model_path,
            )

            rapport = entrainer_deja_existant(dataset_path, "modele", models_dir)
            artefact = joblib.load(model_path)  # type: ignore[reportUnknownMemberType]

        self.assertEqual(rapport["model"], "modele")
        self.assertEqual(rapport["n_semaines"], 60)
        self.assertIn("mae_entrainement", rapport)
        self.assertNotIn("ancienne_cle", rapport)
        self.assertEqual(artefact["features"], {"prix": "numerique", "zone": "categorielle"})
        self.assertEqual(artefact["rapport"], rapport)

    def test_rejects_dataset_with_different_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            models_dir = workspace / "models"
            models_dir.mkdir()
            model_path = models_dir / "modele.pkl"
            joblib.dump(  # type: ignore[reportUnknownMemberType]
                {
                    "pipeline": Pipeline([("arbre", DecisionTreeRegressor())]),
                    "features": {"prix": "numerique"},
                },
                model_path,
            )
            dataset_path = workspace / "dataset.csv"
            pd.DataFrame({
                "semaine": pd.date_range("2025-01-01", periods=60, freq="W"),
                "autre_prix": range(60),
                "delta_prix": range(60),
            }).to_csv(dataset_path, index=False)

            with self.assertRaisesRegex(DatasetInvalide, "variables du dataset"):
                entrainer_deja_existant(dataset_path, "modele.pkl", models_dir)


if __name__ == "__main__":
    unittest.main()
