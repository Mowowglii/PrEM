# PrEM
Prédiction de la variation hebdomadaire du prix de l’électricité par arbres de décision CART, servis et expliqués via une API FastAPI.

## Installation

Depuis la racine du projet, créez et activez un environnement virtuel, puis installez les dépendances déclarées dans `requirements.txt` :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Pour lancer l'API après l'installation :

```powershell
python -m uvicorn prem.api:app --reload
```

Lancer les tests avec :

```powershell
python -m unittest discover -s tests
```

## API

Démarrer le serveur depuis la racine du projet :

```powershell
uvicorn prem.api:app --reload
```

La documentation interactive est disponible sur `/docs`.

- `GET /models` : liste les modèles entraînés (`.pkl`).
- `GET /datasets` : liste les jeux de données CSV du dossier `data`.
- `GET /datasets/{name}` : renvoie le nom et le contenu UTF-8 d'un dataset.
- `POST /datasets` : sauvegarde un CSV téléversé dans `data` sans écraser un fichier existant.
- `POST /models` : entraîne un modèle à partir d'un CSV envoyé en multipart (`csv` requis, `max_depth` facultatif).
- `POST /models/{name}/train` : réentraîne le pipeline d'un modèle existant à partir d'un CSV dont les variables et types correspondent au modèle.
- `DELETE /models/{name}` et `DELETE /datasets/{name}` : suppriment un modèle ou un dataset.
- `GET /models/{name}/features` : renvoie les variables numériques et qualitatives du modèle.
- `GET /models/{name}/tree` : renvoie directement l'image PNG de l'arbre (`Content-Type: image/png`), affichable dans un navigateur ou dans une balise `<img>`.
- `POST /models/{name}/predict` : prédit avec un corps JSON correspondant aux champs de `Comestible`.

### Parcours de prédiction

Avant de demander une prédiction, récupérez les variables attendues par le modèle avec `GET /models/{name}/features`. Passez ensuite les variables numériques et qualitatives renvoyées à un `ComestibleBuilder`, renseignez les valeurs connues avec `packing`, puis envoyez le `Comestible` construit à `POST /models/{name}/predict` sous la clé `payload`. Les variables non renseignées restent à `None` et seront imputées par le pipeline du modèle.

```python
import requests

from prem.comestible import ComestibleBuilder

base_url = "http://127.0.0.1:8000"
model = "cart_20261014-141920.pkl"

response = requests.get(f"{base_url}/models/{model}/features")
response.raise_for_status()
features = response.json()

builder = ComestibleBuilder()
builder.attach_feats(
    features["variables_numeriques"],
    features["variables_qualitatives"],
)
# Exemple : renseigner uniquement les variables connues
builder.packing({"var_gaz_ttf": 2.0, "escalade": "non"})
comestible = builder.build()

response = requests.post(
    f"{base_url}/models/{model}/predict",
    json={"payload": comestible.get_dict()},
)
response.raise_for_status()
prediction = response.json()
```

`packing` lève une `ValueError` si une clé n'est pas annoncée par `/features`, si une variable numérique ne reçoit pas un `int` ou un `float`, ou si une variable qualitative ne reçoit pas une `str`.
