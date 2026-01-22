class SeasonalityType:
    TRIGONOMETRIC = "trigonometric"
    HS = "hs"
    NONE = "none"


class Seasonality:
    def __init__(self, type: SeasonalityType):
        self.type = type

    def update(self, components, score, t):
        pass
