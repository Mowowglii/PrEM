"""Tests unitaires de la classe Comestible."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prem.comestible import Comestible


class TestComestible(unittest.TestCase):
    
    features_list : list[str] = [
                    "delta_prix_m1",
                    "var_gaz_ttf",
                    "var_co2",
                    "ecart_temperature",
                    "var_dispo_nucleaire",
                    "prod_eolien_solaire",
                    "conflit_majeur",
                    "zone_conflit",
                    "escalade"
                ]

    def test_comestibles_sont_compatibles_si_une_valeur_est_absente(self) -> None:
        incomplet = Comestible(self.features_list)
        incomplet.put_value("conflit_majeur", "oui")
        complet = Comestible(self.features_list)
        complet.put_value("conflit_majeur", "oui")
        complet.put_value("zone_conflit", "Ukraine")

        self.assertTrue(incomplet.est_compatible_avec(complet))
        self.assertTrue(complet.est_compatible_avec(incomplet))

    def test_comestibles_sont_compatibles_si_leurs_valeurs_concordent(self) -> None:
        premier = Comestible(self.features_list)
        premier.put_value("var_gaz_ttf", 0)
        premier.put_value("conflit_majeur", "non")
        second = Comestible(self.features_list)
        second.put_value("var_gaz_ttf", 0.0)
        second.put_value("conflit_majeur", "non")

        self.assertTrue(premier.est_compatible_avec(second))

    def test_fusion_complete_les_valeurs_absentes(self) -> None:
        partiel = Comestible(self.features_list)
        partiel.put_value("conflit_majeur", "oui")
        complet = Comestible(self.features_list)
        complet.put_value("conflit_majeur", "oui")
        complet.put_value("zone_conflit", "Ukraine")

        partiel.fusionner(complet)

        self.assertEqual(
            partiel,
            complet,
        )

    def test_fusion_preserve_les_valeurs_nulles_et_les_zero(self) -> None:
        premier = Comestible(self.features_list)
        premier.put_value("var_gaz_ttf", 0)
        second = Comestible(self.features_list)
        second.put_value("var_co2", 2.5)
        second.put_value("zone_conflit", "Ukraine")

        premier.fusionner(second)        

        resultat = Comestible(self.features_list)
        resultat.put_value("var_gaz_ttf", 0)
        resultat.put_value("var_co2", 2.5)
        resultat.put_value("zone_conflit", "Ukraine")


        self.assertEqual(
            premier,
            resultat,
        )

    def test_comestibles_avec_valeurs_contradictoires_sont_incompatibles(self) -> None:
        premier = Comestible(self.features_list)
        premier.put_value("conflit_majeur", "oui")
        premier.put_value("zone_conflit", "Ukraine")
        second = Comestible(self.features_list)
        second.put_value("conflit_majeur", "oui")
        second.put_value("zone_conflit", "Moyen-Orient")

        self.assertFalse(premier.est_compatible_avec(second))

    def test_fusion_refuse_et_signale_les_features_en_conflit(self) -> None:
        premier = Comestible(self.features_list)
        premier.put_value("var_gaz_ttf", 1.0)
        premier.put_value("zone_conflit", "Moyen-Orient")
        second = Comestible(self.features_list)
        second.put_value("var_gaz_ttf", 2.0)
        second.put_value("zone_conflit", "Ukraine")

        with self.assertRaisesRegex(ValueError, "Incompatibilité avec l'autre Comestible"):
            premier.fusionner(second)


if __name__ == "__main__":
    unittest.main()
