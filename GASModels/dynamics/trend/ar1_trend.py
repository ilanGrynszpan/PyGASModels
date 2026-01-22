from GASModels.dynamics.trend.trend import Trend, TrendType


class AR1Trend(Trend):
    def __init__(self):
        super().__init__(TrendType.AR1)

    def update(self, components, score, t):

        for i, _ in enumerate(components["mu"]):
            components["mu"][i - 1][t] = (
                components["c"][i - 1]
                + components["mu"][i - 1][t - 1] * components["phi"][i - 1]
                + components["kg"][i - 1] * score
            )

        return components["mu"]
