class Component:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def include_dynamics(self, args):
        raise NotImplementedError
