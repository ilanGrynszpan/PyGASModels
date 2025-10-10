from GASModels.components.trend import Trend, TrendTypes
from GASModels.dynamics import Dynamics
from scipy.optimize import minimize


if __name__ == "__main__":
    dynamics = Dynamics(
        distribution="gamma_log_link",
    )
