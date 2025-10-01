from GASModels.component import Component
from enum import Enum


class TrendTypes(Enum):
    RANDOM_WALK = "RW"
    AR1 = "AR1"


class Trend(Component):
    BASIC_NAME = "Trend"
    type: TrendTypes = None
    kappa_mu: float = None

    def __init__(self, type: TrendTypes, kappa_mu: float):
        super().__init__(self.BASIC_NAME + "_" + type.value)
        self.type = type
        self.kappa_mu = kappa_mu

    def include_dynamics(self, time_varying, fixed_params, score):
        if self.type == TrendTypes.RANDOM_WALK:
            return self.include_rw_trend(time_varying, score)
        elif self.type == TrendTypes.AR1:
            return self.include_ar1_trend(
                time_varying, fixed_params[0], fixed_params[1], score
            )

    def include_rw_trend(self, mu_prev, score):
        return mu_prev + score * self.kappa_mu

    def include_ar1_trend(self, mu_prev, phi, omega, score):
        return omega + phi * mu_prev + score * self.kappa_mu
