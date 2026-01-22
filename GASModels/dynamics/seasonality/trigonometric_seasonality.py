from GASModels.dynamics.seasonality.seasonality import Seasonality, SeasonalityType
import numpy as np


class TrigonometricSeasonality(Seasonality):

    def __init__(self):
        super().__init__(SeasonalityType.TRIGONOMETRIC)

    def update(self, components, score, t):

        harmonics = components["harmonics"]
        period = components["period"]

        for i, _ in enumerate(components["gamma_"]):
            for j in range(harmonics):
                lambda_ = 2 * np.pi * (j + 1) / period
                gamma_ = (
                    np.cos(lambda_) * components["gamma_"][i][j][t]
                    + np.sin(lambda_[j]) * components["gamma_star"][i][j][t]
                    + score * components["kg"][i]
                )
                gamma_star = (
                    np.cos(lambda_) * components["gamma_star"][i][j][t]
                    - np.sin(lambda_[j]) * components["gamma_"][i][j][t]
                    + score * components["kg"][i]
                )

                components["gamma_"][i][j][t] = gamma_
                components["gamma_star"][i][j][t] = gamma_star

        return components["gamma_"], components["gamma_star"]
