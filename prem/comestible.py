"""Représentation d'un état hebdomadaire des variables explicatives."""

from __future__ import annotations

from collections import ChainMap
from dataclasses import dataclass
from collections.abc import MutableMapping
from typing import cast, Any

@dataclass
class Comestible:
    """Ligne de features du dataset, sans la semaine ni la cible."""
    payload : dict[str, Any]
    
    def est_compatible_avec(self, autre: Comestible) -> bool:
        """Deux objets sont compatibles si leurs valeurs connues concordent."""
        for feature, value in self.payload.items():
            try :
                otherval = autre.payload[feature]
                if otherval != value and value is not None and otherval is not None:
                    return False
            except KeyError :
                return False
        return True

    def fusionner(self, autre: Comestible) -> None:
        """Fusionne les valeurs connues ; refuse les valeurs contradictoires."""
        if not self.est_compatible_avec(autre):
            raise ValueError("Incompatibilité avec l'autre Comestible")
        
        for key, value in autre.payload.items():
            if self.payload[key] == None and value != None:
                self.payload[key] = value
    
    def get_dict(self) -> dict[str, Any]:
        return self.payload

class ComestibleBuilder:
    """Constructeur de Comestible."""
    def __init__(self):
        self._num_features : dict[str, int | float | None]
        self._str_features : dict[str, str | None]
        self.attributes : MutableMapping[str, Any]

    def attach_feats(self, features_numeric : list[str], features_string : list[str]) -> None:
        self._num_features = {n_features: cast(int | float, None) for n_features in features_numeric}
        self._str_features = {s_features: None for s_features in features_string}
        self.attributes : MutableMapping[str, Any] = ChainMap(self._num_features, self._str_features)
    
    def packing(self, body: dict[str, str | int | float | None]) -> None:
        for key, value in body.items():
            self._put_value(key, value)
    
    def _put_value(self, key : str, value : str | int | float | None) -> None:
        IN_NUM_FEATURES : bool = key in self._num_features.keys()
        IN_STR_FEATURES : bool = key in self._str_features.keys()
        
        if IN_NUM_FEATURES and not(isinstance(value, int) or isinstance(value, float)):
            raise ValueError(f"Le champs {key} doit recevoir int ou bien float")
        elif IN_STR_FEATURES and not isinstance(value, str):
            raise ValueError(f"Le champs {key} doit recevoir str")
        elif IN_NUM_FEATURES or IN_STR_FEATURES:
            self.attributes[key] = value
        else :
            raise ValueError(f"Le champs {key} est inconnu")
        
    def build(self) -> Comestible:
        """Renvoie toutes les features sous la forme attendue par le Décideur."""
        return Comestible(dict(self.attributes))