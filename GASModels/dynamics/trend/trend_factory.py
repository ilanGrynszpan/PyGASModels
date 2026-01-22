from GASModels.dynamics.trend.ar1_trend import AR1Trend
from GASModels.dynamics.trend.rw_trend import RWTrend
from GASModels.dynamics.trend.trend import TrendType


class TrendFactory:

    @staticmethod
    def get_trend(trend_type: str):
        if trend_type == TrendType.RANDOM_WALK:
            return RWTrend()
        elif trend_type == TrendType.AR1:
            return AR1Trend()
        elif trend_type == TrendType.NONE:
            return None
