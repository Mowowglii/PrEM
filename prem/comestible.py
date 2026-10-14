"""Représentation d'un état hebdomadaire des variables explicatives."""

from __future__ import annotations

from dataclasses import dataclass

ValeurFeature = int | float | str | None


@dataclass(frozen=True)
class Comestible:
    """Ligne de features du dataset, sans la semaine ni la cible."""

    delta_prix_m1: int | float | None = None
    var_gaz_ttf: int | float | None = None
    var_co2: int | float | None = None
    ecart_temperature: int | float | None = None
    var_dispo_nucleaire: int | float | None = None
    prod_eolien_solaire: int | float | None = None
    conflit_majeur: str | None = None
    zone_conflit: str | None = None
    escalade: str | None = None

    def vers_dict(self) -> dict[str, ValeurFeature]:
        """Renvoie toutes les features sous la forme attendue par le Décideur."""
        return {
            "delta_prix_m1": self.delta_prix_m1,
            "var_gaz_ttf": self.var_gaz_ttf,
            "var_co2": self.var_co2,
            "ecart_temperature": self.ecart_temperature,
            "var_dispo_nucleaire": self.var_dispo_nucleaire,
            "prod_eolien_solaire": self.prod_eolien_solaire,
            "conflit_majeur": self.conflit_majeur,
            "zone_conflit": self.zone_conflit,
            "escalade": self.escalade,
        }

    def est_compatible_avec(self, autre: Comestible) -> bool:
        """Deux objets sont compatibles si leurs valeurs connues concordent."""
        valeurs_autre = autre.vers_dict()
        return all(
            valeur is None
            or valeurs_autre[feature] is None
            or valeur == valeurs_autre[feature]
            for feature, valeur in self.vers_dict().items()
        )

    def fusionner(self, autre: Comestible) -> Comestible:
        """Fusionne les valeurs connues ; refuse les valeurs contradictoires."""
        valeurs = self.vers_dict()
        valeurs_autre = autre.vers_dict()
        conflits = [
            feature
            for feature, valeur in valeurs.items()
            if valeur is not None
            and valeurs_autre[feature] is not None
            and valeur != valeurs_autre[feature]
        ]
        if conflits:
            noms = ", ".join(conflits)
            raise ValueError(f"Comestibles incompatibles : valeurs contradictoires pour {noms}")

        return Comestible(
            delta_prix_m1=self.delta_prix_m1 if self.delta_prix_m1 is not None else autre.delta_prix_m1,
            var_gaz_ttf=self.var_gaz_ttf if self.var_gaz_ttf is not None else autre.var_gaz_ttf,
            var_co2=self.var_co2 if self.var_co2 is not None else autre.var_co2,
            ecart_temperature=(
                self.ecart_temperature
                if self.ecart_temperature is not None
                else autre.ecart_temperature
            ),
            var_dispo_nucleaire=(
                self.var_dispo_nucleaire
                if self.var_dispo_nucleaire is not None
                else autre.var_dispo_nucleaire
            ),
            prod_eolien_solaire=(
                self.prod_eolien_solaire
                if self.prod_eolien_solaire is not None
                else autre.prod_eolien_solaire
            ),
            conflit_majeur=(
                self.conflit_majeur
                if self.conflit_majeur is not None
                else autre.conflit_majeur
            ),
            zone_conflit=self.zone_conflit if self.zone_conflit is not None else autre.zone_conflit,
            escalade=self.escalade if self.escalade is not None else autre.escalade,
        )