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
- `POST /models` : entraîne un modèle à partir d'un CSV envoyé en multipart (`csv` requis, `max_depth` facultatif).
- `GET /models/{name}/features` : renvoie les variables numériques et qualitatives du modèle.
- `POST /models/{name}/predict` : prédit avec un corps JSON correspondant aux champs de `Comestible`. La route `GET` historique reste disponible.
