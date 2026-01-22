from GASModels.dynamics.trend.trend import Trend, TrendType


class RWTrend(Trend):
    def __init__(self):
        super().__init__(TrendType.RANDOM_WALK)

    def update(self, components, score, t):

        for i, _ in enumerate(components["mu"]):
            components["mu"][i - 1][t] = (
                components["mu"][i - 1][t - 1] + components["km"][i - 1] * score
            )

        return components["mu"]
