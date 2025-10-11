from GASModels.component import Component
from GASModels.distributions.distribution import Distribution
import numpy as np


class Dynamics:

    distribution: Distribution = None
    trend: list[int] = []
    seasonality: list[int] = []
    harmonics: int = 0
    period: int = 0
    n: int = 0
    time_varying_indices: list[int] = []
    fixed_params_indices: list[int] = []
    args: list[float] = []

    def __init__(
        self,
        distribution: Distribution,
        n: int,
        trend: list[int],
        seasonality: list[int],
        harmonics: int,
        period: int,
        time_varying_indices: list[int],
        fixed_params_indices: list[int],
        args: list[float],
    ):
        self.distribution = distribution
        self.n = n
        self.trend = trend
        self.seasonality = seasonality
        self.harmonics = harmonics
        self.period = period
        self.time_varying_indices = time_varying_indices
        self.fixed_params_indices = fixed_params_indices
        self.args = args

    def initialize(self):
        pass

    def update_components(self, score, hyperparameters, component_dynamics, t):
        pass

    def fit_in_sample(self, params, y):

        gamma = np.zeros(self.harmonics)
        gamma_star = np.zeros(self.harmonics)

        km = params[0]
        kg = params[1]

        for i in range(self.harmonics):
            gamma[i] = params[i + 2]
            gamma_star[i] = params[i + 2 + self.harmonics]

        print(gamma)

        mu0 = params[self.harmonics * 2 + 2]
        alpha = params[self.harmonics * 2 + 3]

        mu = mu0

        fit_in_sample = np.zeros(self.n)

        for t in range(self.n):

            lambda_t = mu + np.sum(gamma)
            score = self.distribution.score(y[t], alpha=alpha, lambda_=lambda_t)[0]
            fit_in_sample[t] = self.distribution.mean(alpha=alpha, lambda_=lambda_t)
            mu += km * score

            for i in range(self.harmonics):
                lambda_i = (2 * np.pi * i) / self.period
                gamma_t1 = (
                    np.cos(lambda_i) * gamma[i]
                    + np.sin(lambda_i) * gamma_star[i]
                    + kg * score
                )
                gamma_start_t1 = (
                    np.cos(lambda_i) * gamma_star[i]
                    - np.sin(lambda_i) * gamma[i]
                    + kg * score
                )
                gamma[i] = gamma_t1
                gamma_star[i] = gamma_start_t1

        return fit_in_sample

    def objective(self, params, y):

        gamma = np.zeros(self.harmonics)
        gamma_star = np.zeros(self.harmonics)

        km = params[0]
        kg = params[1]

        for i in range(self.harmonics):
            gamma[i] = params[i + 2]
            gamma_star[i] = params[i + 2 + self.harmonics]

        mu0 = params[self.harmonics * 2 + 2]
        alpha = params[self.harmonics * 2 + 3]

        mu_t = mu0
        sum_logpdf = 0

        for _, yt in enumerate(y):

            lambda_t = mu_t + np.sum(gamma)
            logpdf = self.distribution.logpdf(yt, alpha=alpha, lambda_=lambda_t)
            score = self.distribution.score(yt, alpha=alpha, lambda_=lambda_t)[0]

            mu_t += km * score

            for i in range(self.harmonics):
                lambda_i = (2 * np.pi * i) / self.period
                gamma_t1 = (
                    np.cos(lambda_i) * gamma[i]
                    + np.sin(lambda_i) * gamma_star[i]
                    + kg * score
                )
                gamma_start_t1 = (
                    np.cos(lambda_i) * gamma_star[i]
                    - np.sin(lambda_i) * gamma[i]
                    + kg * score
                )
                gamma[i] = gamma_t1
                gamma_star[i] = gamma_start_t1

            sum_logpdf += logpdf

        return sum_logpdf

        # component_dynamics = np.zeros((len(self.components), self.n))
        # fixed_params = np.zeros(len(self.fixed_params_indices))

        # pos_kappas = 0
        # pos_fixed_params = len(self.components)
        # pos_initial_values = pos_fixed_params + len(self.fixed_params_indices)

        # for i in range(len(self.components)):
        #     component_dynamics[i, 0] = self.args[i]

        # for i in range(len(fixed_params)):

        # component_dynamics = np.zeros((len(self.components), self.n))
        # parameter = np.zeros(self.n)
        # score = 2.0

        # for i in range(len(self.components)):
        #     component_dynamics[i, 0] = self.args[i]
        #     parameter[0] += component_dynamics[i, 0]

        # hyperparameters = []
        # iterable = self.fixed_params_indices + [len(self.components)]
        # for i in range(len(iterable) - 1):
        #     temp = []
        #     for j in range(iterable[i], iterable[i + 1]):
        #         temp.append(self.args[j])
        #     hyperparameters.append(temp)

        # for t in range(1, self.n):

        #     score = self.distribution.score(parameter[t - 1], **self.args)

        #     for i, _ in enumerate(self.components):
        #         component_dynamics[i, t] = self.components[i].include_dynamics(
        #             component_dynamics[i, t - 1], *hyperparameters[i], score
        #         )
        #         parameter[t] += component_dynamics[i, t]

        # return component_dynamics, parameter

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
