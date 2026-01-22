from GASModels.dynamics.seasonality.trigonometric_seasonality import (
    TrigonometricSeasonality,
)
from GASModels.dynamics.seasonality.hs import HSSeasonality
from GASModels.dynamics.seasonality.seasonality import SeasonalityType


class SeasonalityFactory:
    @staticmethod
    def get_seasonality(seas_type: str) -> str:
        if seas_type == SeasonalityType.TRIGONOMETRIC:
            return TrigonometricSeasonality()
        elif seas_type == SeasonalityType.HS:
            return HSSeasonality()
        elif seas_type == SeasonalityType.NONE:
            return None
