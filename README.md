# PrEM
Prédiction de la variation hebdomadaire du prix de l’électricité par arbres de décision CART, servis et expliqués via une API FastAPI.

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
- `GET /models/{name}/tree` : renvoie l'arbre de décision sous forme d'un PNG Base64 (`media_type` et `data_base64`).
- `POST /models/{name}/predict` : prédit avec un corps JSON correspondant aux champs de `Comestible`. La route `GET` historique reste disponible.

### Parcours de prédiction

Avant de demander une prédiction, récupérez les variables attendues par le modèle avec `GET /models/{name}/features`. Construisez ensuite un `Comestible` à partir des noms de variables numériques et qualitatives renvoyés, renseignez les valeurs connues, puis envoyez son dictionnaire à `POST /models/{name}/predict`. Les variables laissées à `None` seront imputées par le pipeline du modèle.

```python
import requests

from prem.comestible import Comestible

base_url = "http://127.0.0.1:8000"
model = "cart_20261014-141920.pkl"

features = requests.get(f"{base_url}/models/{model}/features").json()
noms_variables = (
    features["variables_numeriques"]
    + features["variables_qualitatives"]
)

comestible = Comestible(noms_variables)
comestible.put_value("var_gaz_ttf", 2.0)  # Exemple : renseigner une variable connue
comestible.put_value("escalade", "non")

response = requests.post(
    f"{base_url}/models/{model}/predict",
    json={"dictio": comestible.get_dict()},
)
prediction = response.json()
```

N'envoyez que des clés annoncées par `/features` : chaque variable fournie doit appartenir au modèle. Vous pouvez omettre les variables inconnues ou laisser leurs valeurs à `None`.
