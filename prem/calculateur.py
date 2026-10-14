"""Calculateur : entraîne, valide et sauvegarde un arbre CART.

Usage en ligne de commande :
    python -m prem.calculateur data/sample.csv
    python -m prem.calculateur data/sample.csv --max-depth 4
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import cast

import joblib  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

MODELS_DIR = Path("cart_models")
COLONNE_DATE = "semaine"
CIBLE = "delta_prix"
N_SPLITS = 5
MIN_SEMAINES = 60


class DatasetInvalide(ValueError):
    """Levée quand le CSV ne permet pas d'entraîner un modèle (→ 422 côté API)."""


def charger_dataset(source : Path) -> tuple[pd.DataFrame, pd.Series]:
    """Lit le CSV, le trie chronologiquement et sépare X / y."""
    if not(source.is_file() or source.match("*.csv")):
        raise ValueError("Le chemin doit mener à un fichier .csv")
    
    df = pd.read_csv(source)

    for col in (COLONNE_DATE, CIBLE):
        if col not in df.columns:
            raise DatasetInvalide(f"Colonne obligatoire absente : {col}")
    if len(df) < MIN_SEMAINES:
        raise DatasetInvalide(f"{len(df)} semaines, il en faut au moins {MIN_SEMAINES}")

    # Indispensable : TimeSeriesSplit découpe selon l'ordre des lignes, pas selon les dates
    df[COLONNE_DATE] = pd.to_datetime(df[COLONNE_DATE])
    df = df.sort_values(COLONNE_DATE).reset_index(drop=True)
    df = df.dropna(subset=[CIBLE])

    X = df.drop(columns=[COLONNE_DATE, CIBLE])
    y = df[CIBLE]
    return X, y


def construire_pipeline(X: pd.DataFrame) -> Pipeline:
    numeriques = X.select_dtypes(include="number").columns.tolist()
    categorielles = X.select_dtypes(exclude="number").columns.tolist()

    preparation = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeriques),
        ("cat", Pipeline([
            ("imput", SimpleImputer(strategy="constant", fill_value="aucune")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorielles),
    ])

    return Pipeline([
        ("prep", preparation),
        ("arbre", DecisionTreeRegressor(random_state=42)),
    ])


def construire_grille(max_depth: int | None) -> dict[str, list[int]]:
    return {
        "arbre__max_depth": [max_depth] if max_depth else [2, 3, 4, 5, 6],
        "arbre__min_samples_leaf": [3, 5, 10],
    }


def mae_baselines(y: pd.Series, cv: TimeSeriesSplit) -> dict[str, float]:
    """MAE des deux baselines, calculée sur les mêmes plis de test que le modèle."""
    zero : list[np.float64] = []
    persistance : list[np.float64] = []
    for _, test in cv.split(y.to_numpy()):
        y_test = y.iloc[test].to_numpy()
        y_veille = y.shift(1).iloc[test].fillna(0).to_numpy()
        zero.append(np.abs(y_test).mean())
        persistance.append(np.abs(y_test - y_veille).mean())
    return {
        "predire_zero": round(float(np.mean(zero)), 2),
        "semaine_precedente": round(float(np.mean(persistance)), 2),
    }


def entrainer(source : Path, max_depth: int | None = None, models_dir: Path = Path("cart_models")) -> dict[str, object]:
    """Entraîne, valide, sauvegarde. Renvoie le rapport (nom du modèle + métriques)."""
    if max_depth is not None and not 1 <= max_depth <= 20:
        raise DatasetInvalide("max_depth doit être compris entre 1 et 20")

    X, y = charger_dataset(source)
    cv = TimeSeriesSplit(n_splits=N_SPLITS)

    grid: GridSearchCV[Pipeline] = GridSearchCV(
        construire_pipeline(X),
        construire_grille(max_depth),
        cv=cv,
        scoring="neg_mean_absolute_error",
    )
    grid.fit(X, y)  # type: ignore[reportUnknownMemberType]  # refit=True : le meilleur pipeline est réentraîné sur tout X, y

    i = grid.best_index_
    mae_par_pli = [round(float(-grid.cv_results_[f"split{k}_test_score"][i]), 2)
                   for k in range(N_SPLITS)]

    nom = f"cart_{datetime.now():%Y%m%d-%H%M%S}"
    rapport: dict[str, object] = {
        "model": nom,
        "entraine_le": datetime.now().isoformat(timespec="seconds"),
        "n_semaines": len(y),
        "cible": CIBLE,
        "unite": "€/MWh",
        "parametres": {
            k.removeprefix("arbre__"): v
            for k, v in cast(dict[str, object], grid.best_params_).items()  # type: ignore[reportUnknownMemberType]
        },
        "mae_modele": round(float(-grid.best_score_), 2),
        "mae_par_pli": mae_par_pli,
        "mae_baselines": mae_baselines(y, cv),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({ # type: ignore[reportUnknownMemberType]
        "pipeline": grid.best_estimator_,
        "features": {col: ("numerique" if pd.api.types.is_numeric_dtype(X[col]) else "categorielle")
                     for col in X.columns},
        "rapport": rapport,
    }, models_dir / f"{nom}.pkl")

    return rapport


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne un arbre CART PrEM")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()

    rapport = entrainer(args.dataset, args.max_depth)
    for cle, valeur in rapport.items():
        print(f"{cle:>15} : {valeur}")


if __name__ == "__main__":
    main()