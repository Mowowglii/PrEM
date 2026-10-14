"""Représentation d'un état hebdomadaire des variables explicatives."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass()
class Comestible:
    """Ligne de features du dataset, sans la semaine ni la cible."""
    dictio : dict[str, object]
    
    def __init__(self, features : list[str]):
        self.dictio : dict[str, object] = {feature: None for feature in features}

    def put_value(self, key : str, value : object) -> None:
        if key not in self.dictio.keys():
            raise KeyError("La Clé entrée ne peut être assignée car non-existante")
        self.dictio[key] = value

    def get_dict(self) -> dict[str, object]:
        """Renvoie toutes les features sous la forme attendue par le Décideur."""
        return self.dictio

    def est_compatible_avec(self, autre: Comestible) -> bool:
        """Deux objets sont compatibles si leurs valeurs connues concordent."""
        for feature, value in self.dictio.items():
            try :
                otherval = autre.get_dict()[feature]
                if otherval != value and value != None and otherval != None:
                    return False
            except KeyError :
                return False
        return True

    def fusionner(self, autre: Comestible) -> None:
        """Fusionne les valeurs connues ; refuse les valeurs contradictoires."""
        if not self.est_compatible_avec(autre):
            raise ValueError("Incompatibilité avec l'autre Comestible")
        
        for key, value in autre.get_dict().items():
            if self.dictio[key] == None and value != None:
                self.put_value(key, value)