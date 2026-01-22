from GASModels.dynamics.seasonality.seasonality import Seasonality, SeasonalityType
import numpy as np


class HSSeasonality(Seasonality):

    def __init__(self):
        super().__init__(SeasonalityType.HS)

    def update(self, components, score, t):

        ## TODO: implement hs seasonality

        pass
