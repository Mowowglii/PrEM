"""Génère un dataset synthétique pour PrEM.

Une ligne = une semaine m. Les variables explicatives décrivent la semaine m-1,
la cible `delta_prix` est la variation du prix moyen day-ahead entre m-1 et m (€/MWh).

Les données sont fictives mais structurées : la cible dépend vraiment des
variables (avec du bruit), pour que l'arbre ait quelque chose à apprendre.
On utilise des variations et des écarts à la normale plutôt que des niveaux :
c'est ce qui fait bouger un prix d'une semaine à l'autre.

Usage :
    python data/generate_dataset.py
    python data/generate_dataset.py --weeks 156 --seed 7 --missing 0.03 --out data/sample.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIQUES = ["var_gaz_ttf", "var_co2", "ecart_temperature",
              "var_dispo_nucleaire", "prod_eolien_solaire"]


def generate(n_weeks: int = 260, seed: int = 42, missing: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    semaines = pd.date_range("2022-01-03", periods=n_weeks, freq="W-MON")

    gaz, co2, dispo = 40.0, 80.0, 75.0
    conflit, zone = False, ""
    delta_precedent = 0.0
    lignes = []

    for semaine in semaines:
        w = semaine.isocalendar().week
        saison = np.cos(2 * np.pi * (w - 3) / 52)  # 1 en janvier, -1 en juillet

        # Conflit : régime persistant, avec des semaines d'escalade
        escalade = False
        if not conflit and rng.random() < 0.04:
            conflit, zone, escalade = True, str(rng.choice(["Ukraine", "Moyen-Orient"])), True
        elif conflit and rng.random() < 0.08:
            conflit, zone = False, ""
        elif conflit and rng.random() < 0.15:
            escalade = True

        # Variables de marché pendant la semaine m-1
        gaz_avant, co2_avant, dispo_avant = gaz, co2, dispo
        choc_gaz = 6.0 if escalade else 0.0
        gaz = float(np.clip(gaz + 0.1 * (40 - gaz) + choc_gaz + rng.normal(0, 2), 15, 150))
        co2 = float(np.clip(co2 + 0.1 * (80 - co2) + rng.normal(0, 2), 40, 120))
        dispo = float(np.clip(dispo + 0.3 * (75 + 8 * saison - dispo) + rng.normal(0, 2), 50, 95))
        ecart_temp = rng.normal(0, 3)  # écart à la normale saisonnière, en °C
        prod_enr = float(np.clip(1.4 + rng.normal(0, 0.3), 0.5, 2.5))  # TWh sur la semaine

        var_gaz, var_co2, var_dispo = gaz - gaz_avant, co2 - co2_avant, dispo - dispo_avant

        # Cible : relations cachées (dont des seuils, que l'arbre sait capter) + bruit
        delta = (
            1.2 * var_gaz
            + 0.3 * var_co2
            + 1.5 * max(-ecart_temp, 0)       # seul le froid fait monter le prix
            - 0.6 * var_dispo
            - 10.0 * (prod_enr - 1.4)
            + (7.0 if escalade else 0.0)
            + (3.0 if escalade and zone == "Ukraine" else 0.0)
            - 1.8                              # recentre la cible autour de 0
            + 0.2 * delta_precedent
            + rng.normal(0, 2.5)
        )

        lignes.append({
            "semaine": semaine.date().isoformat(),
            "delta_prix_m1": round(delta_precedent, 2),
            "var_gaz_ttf": round(var_gaz, 2),
            "var_co2": round(var_co2, 2),
            "ecart_temperature": round(ecart_temp, 1),
            "var_dispo_nucleaire": round(var_dispo, 1),
            "prod_eolien_solaire": round(prod_enr, 2),
            "conflit_majeur": "oui" if conflit else "non",
            "zone_conflit": zone or None,
            "escalade": "oui" if escalade else "non",
            "delta_prix": round(delta, 2),
        })
        delta_precedent = delta

    df = pd.DataFrame(lignes)

    # Quelques trous dans les numériques, pour tester l'imputation du pipeline
    masque = rng.random((len(df), len(NUMERIQUES))) < missing
    df[NUMERIQUES] = df[NUMERIQUES].mask(masque)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset synthétique PrEM")
    parser.add_argument("--weeks", type=int, default=260, help="nombre de semaines (260 = 5 ans)")
    parser.add_argument("--seed", type=int, default=42, help="graine aléatoire")
    parser.add_argument("--missing", type=float, default=0.02, help="part de valeurs manquantes")
    parser.add_argument("--out", type=str, default="sample.csv")
    args = parser.parse_args()

    df = generate(args.weeks, args.seed, args.missing)
    df.to_csv(Path("data/"+args.out), index=False)
    print(f"{len(df)} semaines écrites dans {args.out}")
    print(df["delta_prix"].describe().round(2).to_string())


if __name__ == "__main__":
    main()