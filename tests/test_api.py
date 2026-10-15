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
from prem.comestible import Comestible
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

    def _delete(self, url: str) -> HttpResponse:
        return cast(
            HttpResponse,
            self.client.delete(url),  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
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
            source: object,
            max_depth: int | None,
            models_dir: Path,
        ) -> dict[str, object]:
            self.assertEqual(source.read(), b"semaine,delta_prix\n")  # type: ignore[reportUnknownMemberType]
            self.assertEqual(max_depth, 3)
            self.assertEqual(models_dir, self.models_dir)
            return {"model": "modele"}

        with (
            patch.object(api, "entrainer_fichier_televerse", side_effect=fake_entrainer),
        ):
            response = self._post(
                "/models",
                files={"csv": ("donnees.csv", b"semaine,delta_prix\n", "text/csv")},
                data={"max_depth": "3"},
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {"model": "modele"})

    def test_create_model_returns_unprocessable_entity_for_invalid_dataset(self) -> None:
        with patch.object(
            api,
            "entrainer_fichier_televerse",
            side_effect=DatasetInvalide("CSV invalide"),
        ):
            response = self._post(
                "/models",
                files={"csv": ("donnees.csv", b"invalid", "text/csv")},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "CSV invalide")

    def test_create_and_get_dataset(self) -> None:
        contenu = b"semaine,delta_prix\n2026-01-01,1.2\n"
        response = self._post(
            "/datasets",
            files={"csv": ("donnees.csv", contenu, "text/csv")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "donnees.csv")
        dataset = self._get("/datasets/donnees")
        self.assertEqual(
            dataset.json(),
            {"name": "donnees.csv", "content": contenu.decode("utf-8")},
        )

    def test_create_dataset_does_not_overwrite_existing_file(self) -> None:
        (self.datasets_dir / "donnees.csv").write_text("ancien", encoding="utf-8")

        response = self._post(
            "/datasets",
            files={"csv": ("donnees.csv", b"nouveau", "text/csv")},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            (self.datasets_dir / "donnees.csv").read_text(encoding="utf-8"),
            "ancien",
        )

    def test_train_existing_model_uses_uploaded_dataset(self) -> None:
        (self.models_dir / "modele.pkl").touch()
        report: dict[str, object] = {"model": "modele", "mae_entrainement": 1.2}
        calls: list[tuple[str, Path]] = []

        def fake_train(source: object, name: str, models_dir: Path) -> dict[str, object]:
            calls.append((name, models_dir))
            self.assertEqual(source.read(), b"dataset")  # type: ignore[reportUnknownMemberType]
            return report

        with (
            patch.object(api, "entrainer_modele_televerse", side_effect=fake_train),
        ):
            response = self._post(
                "/models/modele/train",
                files={"csv": ("donnees.csv", b"dataset", "text/csv")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), report)
        self.assertEqual(calls, [("modele.pkl", self.models_dir)])

    def test_delete_model_and_dataset(self) -> None:
        (self.models_dir / "modele.pkl").touch()
        (self.datasets_dir / "donnees.csv").touch()

        model_response = self._delete("/models/modele")
        dataset_response = self._delete("/datasets/donnees")

        self.assertEqual(model_response.status_code, 200)
        self.assertEqual(model_response.json(), {"deleted": "modele.pkl"})
        self.assertEqual(dataset_response.status_code, 200)
        self.assertEqual(dataset_response.json(), {"deleted": "donnees.csv"})
        self.assertFalse((self.models_dir / "modele.pkl").exists())
        self.assertFalse((self.datasets_dir / "donnees.csv").exists())

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

    def test_get_tree_returns_png_image(self) -> None:
        (self.models_dir / "modele.pkl").touch()
        image = b"\x89PNG\r\n\x1a\nimage data"

        class _ReaderWithTreeImage(_ModelReaderStub):
            def get_tree_image(self) -> bytes:
                return image

        with patch.object(
            api,
            "ModelReader",
            return_value=_ReaderWithTreeImage({}),
        ):
            response = self._get("/models/modele/tree")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.content, image)

    def test_predict_returns_prediction_as_json(self) -> None:
        (self.models_dir / "modele.pkl").touch()
        prediction = Prediction(
            model="modele.pkl",
            delta_prix_eur_mwh=1.5,
            chemin=["prix <= 2"],
            features_utilisees=["prix"],
            n_semaines_feuille=4,
        )

        infos_features: dict[str, int | list[str]] = {
            "nombre_variables_numeriques": 1,
            "variables_numeriques": ["prix"],
            "nombre_variables_qualitatives": 1,
            "variables_qualitatives": ["zone"],
        }

        calls: list[tuple[dict[str, object], str | None, Path]] = []

        def fake_predire(
            features: dict[str, object],
            model: str | None,
            models_dir: Path,
        ) -> Prediction:
            calls.append((features, model, models_dir))
            return prediction

        with patch.object(
            api,
            "ModelReader",
            return_value=_ModelReaderStub(infos_features),
        ):
            response_features = self._get("/models/modele/features")

        self.assertEqual(response_features.status_code, 200)
        infos_recues = response_features.json()
        features_modele = (
            infos_recues["variables_numeriques"]
            + infos_recues["variables_qualitatives"]
        )
        comestible = Comestible(features_modele)
        comestible.put_value("prix", 1.5)
        comestible.put_value("zone", "France")

        with patch.object(api, "predire", side_effect=fake_predire):
            response = self._post(
                "/models/modele/predict",
                json={"dictio": comestible.get_dict()},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "modele.pkl")
        self.assertEqual(response.json()["delta_prix_eur_mwh"], 1.5)
        self.assertEqual(calls[0][0], {"prix": 1.5, "zone": "France"})
        self.assertEqual(calls[0][1:], ("modele.pkl", self.models_dir))

    def test_unknown_model_returns_not_found(self) -> None:
        response = self._get("/models/inexistant/features")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
