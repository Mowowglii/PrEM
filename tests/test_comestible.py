"""Tests unitaires de la classe Comestible."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prem.comestible import Comestible, ComestibleBuilder


NUMERIC_FEATURES = [
    "delta_prix_m1",
    "var_gaz_ttf",
    "var_co2",
    "ecart_temperature",
    "var_dispo_nucleaire",
    "prod_eolien_solaire",
]
STRING_FEATURES = [
    "conflit_majeur",
    "zone_conflit",
    "escalade",
]


def build_comestible(**values: str | int | float | None) -> Comestible:
    builder = ComestibleBuilder()
    builder.attach_feats(NUMERIC_FEATURES, STRING_FEATURES)
    builder.packing(values)
    comestible = builder.build()
    return comestible


class TestComestible(unittest.TestCase):
    def test_comestibles_sont_compatibles_si_une_valeur_est_absente(self) -> None:
        incomplet = build_comestible(conflit_majeur="oui")
        complet = build_comestible(conflit_majeur="oui", zone_conflit="Ukraine")

        self.assertTrue(incomplet.est_compatible_avec(complet))
        self.assertTrue(complet.est_compatible_avec(incomplet))

    def test_comestibles_sont_compatibles_si_leurs_valeurs_concordent(self) -> None:
        premier = build_comestible(var_gaz_ttf=0, conflit_majeur="non")
        second = build_comestible(var_gaz_ttf=0.0, conflit_majeur="non")

        self.assertTrue(premier.est_compatible_avec(second))

    def test_fusion_complete_les_valeurs_absentes(self) -> None:
        partiel = build_comestible(conflit_majeur="oui")
        complet = build_comestible(conflit_majeur="oui", zone_conflit="Ukraine")

        partiel.fusionner(complet)

        self.assertEqual(partiel, complet)

    def test_fusion_preserve_les_valeurs_nulles_et_les_zero(self) -> None:
        premier = build_comestible(var_gaz_ttf=0)
        second = build_comestible(var_co2=2.5, zone_conflit="Ukraine")

        premier.fusionner(second)

        resultat = build_comestible(
            var_gaz_ttf=0,
            var_co2=2.5,
            zone_conflit="Ukraine",
        )

        self.assertEqual(premier, resultat)

    def test_comestibles_avec_valeurs_contradictoires_sont_incompatibles(self) -> None:
        premier = build_comestible(conflit_majeur="oui", zone_conflit="Ukraine")
        second = build_comestible(
            conflit_majeur="oui",
            zone_conflit="Moyen-Orient",
        )

        self.assertFalse(premier.est_compatible_avec(second))

    def test_fusion_refuse_et_signale_les_features_en_conflit(self) -> None:
        premier = build_comestible(var_gaz_ttf=1.0, zone_conflit="Moyen-Orient")
        second = build_comestible(var_gaz_ttf=2.0, zone_conflit="Ukraine")

        with self.assertRaisesRegex(ValueError, "Incompatibilité avec l'autre Comestible"):
            premier.fusionner(second)


if __name__ == "__main__":
    unittest.main()
