class TrendType:
    RANDOM_WALK = "rw"
    AR1 = "ar1"
    NONE = "none"


class Trend:

    def __init__(self, type: TrendType):
        self.type = type

    def update(self, components, score, t):
        pass
