from GASModels.components.trend import Trend, TrendTypes
from GASModels.dynamics import Dynamics


args = [0.05, 0.02]
components = [Trend(TrendTypes.RANDOM_WALK, args[1])]
# Try different initialization approaches
dynamics = Dynamics(10, components, [0], args)
trend = dynamics.iterate()

print(trend)
