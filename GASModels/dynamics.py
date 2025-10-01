from GASModels.component import Component
import numpy as np


class Dynamics:

    components: list[Component] = []
    n: int = 0
    fixed_params_indices: list[int] = []
    args: list[float] = []

    def __init__(
        self,
        n: int,
        components: list[Component],
        fixed_params_indices: list[int],
        args: list[float],
    ):
        self.n = n
        self.components = components
        self.fixed_params_indices = fixed_params_indices
        self.args = args

    def iterate(self):

        component_dynamics = np.zeros((len(self.components), self.n))
        score = 2.0

        for i in range(len(self.components)):
            component_dynamics[i, 0] = self.args[i]

        hyperparameters = []
        iterable = self.fixed_params_indices + [len(self.components)]
        for i in range(len(iterable) - 1):
            temp = []
            for j in range(iterable[i], iterable[i + 1]):
                temp.append(self.args[j])
            hyperparameters.append(temp)

        for t in range(1, self.n):
            for i, _ in enumerate(self.components):
                component_dynamics[i, t] = self.components[i].include_dynamics(
                    component_dynamics[i, t - 1], *hyperparameters[i], score
                )

        return component_dynamics

    # components: list[Component] = []
    # n: int = 0
    # n_fixed_params: list[int] = []
    # args: list[float] = []

    # def __init__(
    #     self,
    #     n: int,
    #     n_fixed_params: list[int],
    #     components: list[Component],
    #     args: list[float],
    # ):
    #     self.n = n
    #     self.n_fixed_params = n_fixed_params
    #     self.components = components
    #     self.args = args

    # def iterate(self):
    #     n_components = len(self.components)
    #     component_dynamics = np.zeros((n_components, self.n))
    #     score = 2.0
    #     fixed_params = []

    #     for i in range(n_components):
    #         component_dynamics[i, 0] = self.args[i]

    #     jump = 0
    #     for i in self.n_fixed_params:
    #         hyperparameters = []
    #         for j in range(0, i):
    #             hyperparameters.append(self.args[n_components + jump + j])
    #         jump += i
    #         fixed_params.append(hyperparameters)

    #     for i in range(1, self.n):
    #         for j in range(n_components):
    #             if len(self.n_fixed_params) > 0:
    #                 component_dynamics[j, i] = self.components[j].include_dynamics(
    #                     component_dynamics[j, i - 1], *fixed_params[j], score
    #                 )
    #             else:
    #                 component_dynamics[j, i] = self.components[j].include_dynamics(
    #                     component_dynamics[j, i - 1], score
    #                 )

    #     return component_dynamics
