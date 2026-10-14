"""Tests des routes HTTP de l'API."""

import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from httpx import Response as HttpResponse

from prem import api
from prem.calculateur import DatasetInvalide
from prem.decideur import Prediction


class _ModelReaderStub:
    def __init__(self, features: dict[str, int | list[str]]) -> None:
        self.features = features

    def get_infos(self) -> dict[str, int | list[str]]:
        return self.features


class TestApi(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)
        self.models_dir = self.workspace / "cart_models"
        self.datasets_dir = self.workspace / "data"
        self.models_dir.mkdir()
        self.datasets_dir.mkdir()
        self.models_patch = patch.object(api, "MODELS_DIR", self.models_dir)
        self.datasets_patch = patch.object(api, "DATASETS_DIR", self.datasets_dir)
        self.models_patch.start()
        self.datasets_patch.start()

    def _get(self, url: str) -> HttpResponse:
        return cast(
            HttpResponse,
            self.client.get(url),  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )

    def _post(self, url: str, **kwargs: object) -> HttpResponse:
        return cast(
            HttpResponse,
            self.client.post(url, **kwargs),  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )

    def tearDown(self) -> None:
        self.datasets_patch.stop()
        self.models_patch.stop()
        self.temporary_directory.cleanup()

    def test_get_models_and_datasets_returns_file_names(self) -> None:
        (self.models_dir / "modele.pkl").touch()
        (self.models_dir / "ignore.txt").touch()
        (self.datasets_dir / "donnees.csv").touch()
        (self.datasets_dir / "ignore.json").touch()

        self.assertEqual(self._get("/models").json(), ["modele.pkl"])
        self.assertEqual(self._get("/datasets").json(), ["donnees.csv"])

    def test_create_model_saves_upload_temporarily_and_passes_max_depth(self) -> None:
        def fake_entrainer(
            source: Path,
            max_depth: int | None,
            models_dir: Path,
        ) -> dict[str, object]:
            self.assertEqual(source.read_bytes(), b"semaine,delta_prix\n")
            self.assertEqual(max_depth, 3)
            self.assertEqual(models_dir, self.models_dir)
            return {"model": "modele"}

        with patch.object(api, "entrainer", side_effect=fake_entrainer):
            response = self._post(
                "/models",
                files={"csv": ("donnees.csv", b"semaine,delta_prix\n", "text/csv")},
                data={"max_depth": "3"},
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {"model": "modele"})

    def test_create_model_returns_unprocessable_entity_for_invalid_dataset(self) -> None:
        with patch.object(api, "entrainer", side_effect=DatasetInvalide("CSV invalide")):
            response = self._post(
                "/models",
                files={"csv": ("donnees.csv", b"invalid", "text/csv")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "CSV invalide")

    def test_get_features_returns_model_reader_information(self) -> None:
        (self.models_dir / "modele.pkl").touch()
        features: dict[str, int | list[str]] = {
            "nombre_variables_numeriques": 1,
            "variables_numeriques": ["prix"],
            "nombre_variables_qualitatives": 1,
            "variables_qualitatives": ["zone"],
        }
        with patch.object(api, "ModelReader", return_value=_ModelReaderStub(features)):
            response = self._get("/models/modele/features")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), features)

    def test_get_tree_returns_json_safe_image_data(self) -> None:
        (self.models_dir / "modele.pkl").touch()
        image = {
            "media_type": "image/png",
            "data_base64": "iVBORw0KGgo=",
        }

        class _ReaderWithTreeImage(_ModelReaderStub):
            def get_tree_image(self) -> dict[str, str]:
                return image

        with patch.object(
            api,
            "ModelReader",
            return_value=_ReaderWithTreeImage({}),
        ):
            response = self._get("/models/modele/tree")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), image)

    def test_predict_returns_prediction_as_json(self) -> None:
        prediction = Prediction(
            model="modele.pkl",
            delta_prix_eur_mwh=1.5,
            chemin=["prix <= 2"],
            features_utilisees=["prix"],
            n_semaines_feuille=4,
        )

        calls: list[tuple[dict[str, object], str | None, Path]] = []

        def fake_predire(
            features: dict[str, object],
            model: str | None,
            models_dir: Path,
        ) -> Prediction:
            calls.append((features, model, models_dir))
            return prediction

        with patch.object(api, "predire", side_effect=fake_predire):
            response = self._post("/models/modele/predict", json={})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "modele.pkl")
        self.assertEqual(response.json()["delta_prix_eur_mwh"], 1.5)
        self.assertEqual(calls[0][1:], ("modele.pkl", self.models_dir))

    def test_unknown_model_returns_not_found(self) -> None:
        response = self._get("/models/inexistant/features")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
