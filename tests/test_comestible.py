"""Tests unitaires de la classe Comestible."""

import unittest

from prem.comestible import Comestible


class TestComestible(unittest.TestCase):
    def test_vers_dict_contient_toutes_les_features(self) -> None:
        comestible = Comestible()

        self.assertEqual(
            comestible.vers_dict(),
            {
                "delta_prix_m1": None,
                "var_gaz_ttf": None,
                "var_co2": None,
                "ecart_temperature": None,
                "var_dispo_nucleaire": None,
                "prod_eolien_solaire": None,
                "conflit_majeur": None,
                "zone_conflit": None,
                "escalade": None,
            },
        )

    def test_comestibles_sont_compatibles_si_une_valeur_est_absente(self) -> None:
        incomplet = Comestible(conflit_majeur="oui")
        complet = Comestible(conflit_majeur="oui", zone_conflit="Ukraine")

        self.assertTrue(incomplet.est_compatible_avec(complet))
        self.assertTrue(complet.est_compatible_avec(incomplet))

    def test_comestibles_sont_compatibles_si_leurs_valeurs_concordent(self) -> None:
        premier = Comestible(var_gaz_ttf=0, conflit_majeur="non")
        second = Comestible(var_gaz_ttf=0.0, conflit_majeur="non")

        self.assertTrue(premier.est_compatible_avec(second))

    def test_fusion_complete_les_valeurs_absentes_sans_modifier_les_sources(self) -> None:
        partiel = Comestible(conflit_majeur="oui")
        complet = Comestible(conflit_majeur="oui", zone_conflit="Ukraine")

        fusion = partiel.fusionner(complet)

        self.assertEqual(
            fusion,
            Comestible(conflit_majeur="oui", zone_conflit="Ukraine"),
        )
        self.assertEqual(partiel.zone_conflit, None)
        self.assertEqual(complet.zone_conflit, "Ukraine")

    def test_fusion_preserve_les_valeurs_nulles_et_les_zero(self) -> None:
        premier = Comestible(var_gaz_ttf=0, zone_conflit=None)
        second = Comestible(var_co2=2.5, zone_conflit="Ukraine")

        self.assertEqual(
            premier.fusionner(second),
            Comestible(var_gaz_ttf=0, var_co2=2.5, zone_conflit="Ukraine"),
        )

    def test_comestibles_avec_valeurs_contradictoires_sont_incompatibles(self) -> None:
        premier = Comestible(conflit_majeur="oui", zone_conflit="Ukraine")
        second = Comestible(conflit_majeur="oui", zone_conflit="Moyen-Orient")

        self.assertFalse(premier.est_compatible_avec(second))

    def test_fusion_refuse_et_signale_les_features_en_conflit(self) -> None:
        premier = Comestible(var_gaz_ttf=1.0, zone_conflit="Ukraine")
        second = Comestible(var_gaz_ttf=2.0, zone_conflit="Moyen-Orient")

        with self.assertRaisesRegex(ValueError, "var_gaz_ttf, zone_conflit"):
            premier.fusionner(second)


if __name__ == "__main__":
    unittest.main()
