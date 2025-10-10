class Component:

    name: str = None
    num_params: int = None

    def __init__(self, name, num_params):
        self.name = name
        self.num_params = num_params

    def __str__(self):
        return self.name

    def include_dynamics(self, args):
        raise NotImplementedError
