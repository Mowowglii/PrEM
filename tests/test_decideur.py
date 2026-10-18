"""Tests du calculateur de prédiction."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib  # type: ignore[reportMissingTypeStubs]
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prem.comestible import Comestible
from prem.decideur import predire


class TestPredire(unittest.TestCase):
    def test_predict_uses_pipeline_and_preserves_tree_explanation(self) -> None:
        preparation = ColumnTransformer([
            ("num", "passthrough", ["prix"]),
        ])
        pipeline = Pipeline([
            ("prep", preparation),
            ("arbre", DecisionTreeRegressor(max_depth=1, random_state=0)),
        ])
        pipeline.fit(  # type: ignore[reportUnknownMemberType]
            pd.DataFrame({"prix": [0.0, 1.0, 2.0, 3.0]}),
            [0.0, 0.0, 10.0, 10.0],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            models_dir = Path(temporary_directory)
            joblib.dump(  # type: ignore[reportUnknownMemberType]
                {
                    "pipeline": pipeline,
                    "features": {"prix": "numerique"},
                },
                models_dir / "modele.pkl",
            )

            with patch("prem.decideur.MODELS_DIR", models_dir):
                prediction = predire(
                    Comestible({"prix": 0.5}), "modele.pkl", models_dir
                )

        self.assertEqual(prediction.delta_prix_eur_mwh, 0.0)
        self.assertEqual(prediction.chemin, ["prix <= 1.5"])
        self.assertEqual(prediction.features_utilisees, ["prix"])
        self.assertEqual(prediction.n_semaines_feuille, 2)


if __name__ == "__main__":
    unittest.main()
