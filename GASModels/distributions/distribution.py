class Distribution:

    name: str = None
    num_params: int = None

    def __init__(self, name: str, num_params: int = None):
        self.name = name
        self.num_params = num_params

    def __str__(self):
        return self.name

    def get_parameters(self, **kwargs):
        raise NotImplementedError

    def logpdf(self, y, **kwargs):
        raise NotImplementedError

    def score(self, y, **kwargs):
        raise NotImplementedError

    def fisher_information(self, **kwargs):
        raise NotImplementedError
