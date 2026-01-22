from GASModels.component import Component
from GASModels.distributions.distribution import Distribution
import numpy as np
from scipy.fft import fft
from scipy.optimize import curve_fit
from scipy import stats

from GASModels.dynamics.trend.trend_factory import TrendFactory
from GASModels.dynamics.seasonality.seasonality_factory import SeasonalityFactory

from GASModels.dynamics.trend.trend import TrendType
from GASModels.dynamics.seasonality.seasonality import SeasonalityType


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
        trend: list[str],
        seasonality: list[str],
        harmonics: int,
        period: int,
        time_varying_indices: list[int],
        fixed_params_indices: list[int],
        args: list[float],
    ):
        self.distribution = distribution
        self.n = n
        self.trend = [TrendFactory.get_trend(trend_type) for trend_type in trend]
        self.seasonality = [
            SeasonalityFactory.get_seasonality(seas_type) for seas_type in seasonality
        ]
        self.harmonics = harmonics
        self.period = period
        self.time_varying_indices = time_varying_indices
        self.fixed_params_indices = fixed_params_indices
        self.params = None
        self.args = args

    def _get_model_parameters(self, parameter_list):

        array_elems = [
            "km",
            "kg",
            "mu_10",
        ]

        if self.trend == "ar1":
            array_elems.extend(["phi", "c"])

        array_elems.extend(
            [
                *[f"gamma_{h}" for h in range(self.harmonics)],
                *[f"gamma_star_{h}" for h in range(self.harmonics)],
            ]
        )
        time_varying_params = dict.fromkeys(self.time_varying_indices, None)
        for i, param in enumerate(self.time_varying_indices):
            param = {"index": param}
            for j, elem in enumerate(array_elems):
                param[elem] = parameter_list[i * len(array_elems) + j]
            time_varying_params[param["index"]] = param

        fixed_params = dict.fromkeys(self.fixed_params_indices, None)
        for i, param in enumerate(self.fixed_params_indices):
            param = {"index": param}
            param["value"] = parameter_list[
                len(self.time_varying_indices) * len(array_elems) + i
            ]
            fixed_params[param["index"]] = param

        return {
            "time_varying_params": time_varying_params,
            "fixed_params": fixed_params,
        }

    def _initialize_components(self, components, parameter_list):

        time_varying = self._get_model_parameters(parameter_list)["time_varying_params"]

        components["harmonics"] = self.harmonics
        components["period"] = self.period

        for i in list(time_varying.keys()):
            components["mu"][i - 1][0] = time_varying[i]["mu_10"]
            components["km"][i - 1] = time_varying[i]["km"]
            components["kg"][i - 1] = time_varying[i]["kg"]
            for h in range(self.harmonics):
                components["gamma_"][i - 1][h][0] = time_varying[i][f"gamma_{h}"]
                components["gamma_star"][i - 1][h][0] = time_varying[i][
                    f"gamma_star_{h}"
                ]

        return components

    def _initialize_parameters(self, params, parameter_list):

        time_varying = self._get_model_parameters(parameter_list)["time_varying_params"]
        time_invariant = self._get_model_parameters(parameter_list)["fixed_params"]

        for i in list(time_varying.keys()):
            params[i - 1][0] = time_varying[i]["mu_10"] + np.sum(
                time_varying[i][f"gamma_{h}"] for h in range(self.harmonics)
            )

        for i in list(time_invariant.keys()):
            params[i - 1][0] = time_invariant[i]["value"]

        return params

    def _update_components(self, components, score, t):

        components["mu"] = [
            trend.update(components, score, t)
            for trend in self.trend
            if trend.type != TrendType.NONE
        ]
        components["gamma_"], components["gamma_star"] = [
            seasonality.update(components, score, t)
            for seasonality in self.seasonality
            if seasonality.type != SeasonalityType.NONE
        ]

        return components

    def _update_parameters(self, params, components, t):

        params[self.time_varying_indices][t] = components["mu"][
            self.time_varying_indices
        ][t] + np.sum(components["gamma_"][self.time_varying_indices][t], axis=1)

        params[self.fixed_params_indices][t] = params[self.fixed_params_indices][t - 1]

        return params

    def _iterate(self, hyperparams, y, mode):

        components = {}
        components["mu"] = np.zeros((self.distribution.num_params, self.n))
        components["km"] = np.zeros(self.distribution.num_params)
        components["kg"] = np.zeros(self.distribution.num_params)
        components["gamma_"] = np.zeros(
            (self.distribution.num_params, self.harmonics, self.n)
        )
        components["gamma_star"] = np.zeros(
            (self.distribution.num_params, self.harmonics, self.n)
        )
        params = np.zeros((self.distribution.num_params, self.n))

        components = self._initialize_components(components, hyperparams)
        params = self._initialize_parameters(params, hyperparams)

        for t in range(self.n):
            score = self.distribution.score(y[t], params[:, t])
            components = self._update_components(components, score, t)
            params = self._update_parameters(params, components, t)

        if mode == "loglik":
            return [self.distribution.loglik(y[t], params[:, t]) for t in range(self.n)]
        elif mode == "step ahead":
            return [self.distribution.mean(y[t], params[:, t]) for t in range(self.n)]
        elif mode == "pit":
            return [self.distribution.cdf(y[t], params[:, t]) for t in range(self.n)]
        elif mode == "quantile residuals":
            return [
                self.quantile_residuals(self.distribution.cdf(y[t], params[:, t]))
                for t in range(self.n)
            ]

    def objective(self, parameter_list, y):

        params = self._get_model_parameters(parameter_list)
        loglik = self._iterate(params, y, mode="loglik")
        return -loglik

    def quantile_residuals(self, cdf):
        return stats.norm.ppf(cdf)

    # def _estimate_seasonal_components(self, y, harmonics, period):
    #     """Estimate initial seasonal components with regularization for higher harmonics"""
    #     gamma = [0.2] * harmonics
    #     gamma_star = [0.2] * harmonics

    #     return gamma, gamma_star

    # def _seasonal_pattern_regression(self, y, period, harmonics):
    #     """Alternative method using linear regression for seasonal pattern"""
    #     n = len(y)
    #     t = np.arange(n)

    #     # Create seasonal regressors
    #     X = np.ones((n, 1 + 2 * harmonics))  # intercept + 2* harmonics

    #     for i in range(harmonics):
    #         frequency = 2 * np.pi * (i + 1) / period
    #         X[:, 1 + 2 * i] = np.cos(frequency * t)
    #         X[:, 2 + 2 * i] = np.sin(frequency * t)

    #     try:
    #         # Fit regression
    #         coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

    #         # Extract seasonal coefficients (skip intercept)
    #         gamma_0 = np.zeros(harmonics)
    #         gamma_star_0 = np.zeros(harmonics)

    #         for i in range(harmonics):
    #             gamma_0[i] = coeffs[1 + 2 * i] if 1 + 2 * i < len(coeffs) else 0.01
    #             gamma_star_0[i] = coeffs[2 + 2 * i] if 2 + 2 * i < len(coeffs) else 0.01

    #         return gamma_0, gamma_star_0

    #     except:
    #         # Fallback if regression fails
    #         return np.zeros(harmonics) + 0.01, np.zeros(harmonics) + 0.01

    # def get_initial_values(self, y):
    #     """
    #     Estimate initial values that respect the bounds
    #     """
    #     n = len(y)
    #     y_positive = y[y > 0]

    #     if len(y_positive) == 0:
    #         y_positive = np.array([1e-6])

    #     # # 1. Learning rates - within bounds (1, 2000)
    #     # km = 50  # Will be 0.5 after scaling
    #     # kg = 20  # Will be 0.2 after scaling

    #     # # 2. Seasonal components - within bounds (-200, 200)
    #     # gamma_0 = np.zeros(self.harmonics) * 40  # Small values
    #     # gamma_star_0 = np.zeros(self.harmonics) * 40

    #     # # 3. Distribution parameters - CRITICAL: must respect bounds
    #     # mu0 = 10  # Will be 1.0 after scaling (within -1000, 1000)

    #     # # These must be within bounds after scaling:
    #     # delta_0 = 100  # Will be 0.1 after scaling (within -200, 200)
    #     # delta_1 = 100  # Will be 0.05 after scaling (within -100, 100)
    #     # gamma_ = 33  # Will be 1.0 after scaling (within 10, 500)
    #     # xi = 68  # Will be 1.0 after scaling (within 10, 500)
    #     # zeta = 66  # Will be 2.0 after scaling (within 20, 1000) AND zeta > gamma_

    #     # 1. Learning rates - within bounds (1, 2000)
    #     km = 20  # Will be 0.5 after scaling
    #     kg = 20  # Will be 0.2 after scaling

    #     # 2. Seasonal components - within bounds (-200, 200)
    #     gamma_0 = np.zeros(self.harmonics) * 20  # Small values
    #     gamma_star_0 = np.zeros(self.harmonics) * 20

    #     # 3. Distribution parameters - CRITICAL: must respect bounds
    #     mu0 = 10  # Will be 1.0 after scaling (within -1000, 1000)

    #     # These must be within bounds after scaling:
    #     delta_0 = 50  # Will be 0.1 after scaling (within -200, 200)
    #     delta_1 = 50  # Will be 0.05 after scaling (within -100, 100)
    #     gamma_ = 100  # Will be 1.0 after scaling (within 10, 500)
    #     xi = 200  # Will be 1.0 after scaling (within 10, 500)
    #     zeta = 250  # Will be 2.0 after scaling (within 20, 1000) AND zeta > gamma_

    #     # Construct parameter vector
    #     initial_params = np.concatenate(
    #         [
    #             [km, kg],
    #             gamma_0,
    #             gamma_star_0,
    #             [mu0, delta_0, delta_1, gamma_, xi, zeta],
    #         ]
    #     )

    #     return initial_params

    # def initialize(self):
    #     """
    #     Initialize the dynamics model with the given parameters.

    #     This function should be called once before updating the components.

    #     Parameters
    #     ----------
    #     None

    #     Returns
    #     -------
    #     None
    #     """
    #     pass

    # def update_components(self, score, hyperparameters, component_dynamics, t):
    #     pass

    # def fit_in_sample(self, params, y):
    #     gamma = np.zeros(self.harmonics)
    #     gamma_star = np.zeros(self.harmonics)

    #     km = params[0] / 1000
    #     kg = params[1] / 1000

    #     for i in range(self.harmonics):
    #         gamma[i] = params[i + 2] / 100
    #         gamma_star[i] = params[i + 2 + self.harmonics] / 100

    #     mu0 = params[self.harmonics * 2 + 2] / 100
    #     delta_0 = params[self.harmonics * 2 + 3] / 100
    #     delta_1 = params[self.harmonics * 2 + 4] / 100
    #     gamma_ = params[self.harmonics * 2 + 5] / 100
    #     xi = params[self.harmonics * 2 + 6] / 100
    #     zeta = params[self.harmonics * 2 + 7] / 100

    #     # gamma_ = params[self.harmonics * 2 + 3] / 100
    #     # xi = params[self.harmonics * 2 + 4] / 100
    #     # zeta = params[self.harmonics * 2 + 5] / 100

    #     level = np.zeros(self.n)
    #     seas = np.zeros(self.n)

    #     mu = mu0
    #     fit_in_sample = np.zeros(self.n)
    #     mean_gb2 = np.zeros(self.n)
    #     pi = np.zeros(self.n)

    #     for t in range(self.n):
    #         phi = mu + np.sum(gamma)
    #         level[t] = mu
    #         seas[t] = np.sum(gamma)

    #         score = self.distribution.score(
    #             y[t],
    #             delta_0=delta_0,
    #             delta_1=delta_1,
    #             phi=phi,
    #             gamma=gamma_,
    #             xi=xi,
    #             zeta=zeta,
    #         )[0]
    #         mean_gb2[t], fit_in_sample[t] = self.distribution.mean(
    #             delta_0=delta_0,
    #             delta_1=delta_1,
    #             phi=phi,
    #             gamma=gamma_,
    #             xi=xi,
    #             zeta=zeta,
    #         )
    #         mu = mu + km * score
    #         pi[t] = np.exp(delta_0 + delta_1 * phi) / (
    #             1 + np.exp(delta_0 + delta_1 * phi)
    #         )

    #         for i in range(self.harmonics):
    #             lambda_i = (2 * np.pi * (i + 1)) / self.period
    #             gamma_t1 = (
    #                 np.cos(lambda_i) * gamma[i]
    #                 + np.sin(lambda_i) * gamma_star[i]
    #                 + kg * score
    #             )
    #             gamma_start_t1 = (
    #                 np.cos(lambda_i) * gamma_star[i]
    #                 - np.sin(lambda_i) * gamma[i]
    #                 + kg * score
    #             )
    #             gamma[i] = gamma_t1
    #             gamma_star[i] = gamma_start_t1

    #     return mean_gb2, fit_in_sample, pi, level, seas

    # def pit(self, params, y):
    #     gamma = np.zeros(self.harmonics)
    #     gamma_star = np.zeros(self.harmonics)

    #     km = params[0] / 1000
    #     kg = params[1] / 1000

    #     for i in range(self.harmonics):
    #         gamma[i] = params[i + 2] / 100
    #         gamma_star[i] = params[i + 2 + self.harmonics] / 100

    #     mu0 = params[self.harmonics * 2 + 2] / 100
    #     delta_0 = params[self.harmonics * 2 + 3] / 100
    #     delta_1 = params[self.harmonics * 2 + 4] / 100
    #     gamma_ = params[self.harmonics * 2 + 5] / 100
    #     xi = params[self.harmonics * 2 + 6] / 100
    #     zeta = params[self.harmonics * 2 + 7] / 100

    #     # gamma_ = params[self.harmonics * 2 + 3] / 100
    #     # xi = params[self.harmonics * 2 + 4] / 100
    #     # zeta = params[self.harmonics * 2 + 5] / 100

    #     mu = mu0
    #     # The code is accessing an element in the `params` list using an index calculated based on the
    #     # value of `self.harmonics`. The index is calculated as `self.harmonics * 2 + 7`, and the
    #     # corresponding element is divided by 100 and assigned to the variable `zeta`.
    #     pit = np.zeros(self.n)

    #     for t in range(self.n):
    #         phi = mu + np.sum(gamma)

    #         score = self.distribution.score(
    #             y[t],
    #             delta_0=delta_0,
    #             delta_1=delta_1,
    #             phi=phi,
    #             gamma=gamma_,
    #             xi=xi,
    #             zeta=zeta,
    #         )[0]
    #         pit[t] = self.distribution.cdf(
    #             y[t],
    #             delta_0=delta_0,
    #             delta_1=delta_1,
    #             phi=phi,
    #             gamma=gamma_,
    #             xi=xi,
    #             zeta=zeta,
    #         )

    #         mu = mu + km * score

    #         for i in range(self.harmonics):
    #             lambda_i = (2 * np.pi * (i + 1)) / self.period
    #             gamma_t1 = (
    #                 np.cos(lambda_i) * gamma[i]
    #                 + np.sin(lambda_i) * gamma_star[i]
    #                 + kg * score
    #             )
    #             gamma_start_t1 = (
    #                 np.cos(lambda_i) * gamma_star[i]
    #                 - np.sin(lambda_i) * gamma[i]
    #                 + kg * score
    #             )
    #             gamma[i] = gamma_t1
    #             gamma_star[i] = gamma_start_t1

    #     return pit

    # def get_quantile_residuals(self, pit):
    #     """
    #     Calculate quantile residuals from the fitted model.

    #     Quantile residuals are defined as the inverse standard normal CDF
    #     applied to the PIT values: r_t = Φ^{-1}(PIT_t)

    #     Parameters
    #     ----------
    #     params : array_like
    #         Model parameters
    #     y : array_like
    #         Observed time series data

    #     Returns
    #     -------
    #     quantile_residuals : ndarray
    #         Quantile residuals following standard normal distribution
    #         if the model is correctly specified
    #     """

    #     # Transform to quantile residuals using inverse standard normal CDF
    #     quantile_residuals = stats.norm.ppf(pit)

    #     return quantile_residuals

    # def objective(self, params, y):
    #     # Extensive parameter validation FIRST
    #     if np.any(np.isnan(params)) or np.any(np.isinf(params)):
    #         print("Parameters contain NaN or Inf")
    #         return np.inf

    #     # # Check if parameters are within bounds (sanity check)
    #     # for i, (param, (low, high)) in enumerate(zip(params, bounds)):
    #     #     if not (low <= param <= high):
    #     #         print(f"Parameter {i} is out of bounds")
    #     #         return np.inf

    #     try:
    #         # Parameter extraction with extensive clamping
    #         km = params[0] / 1000.0
    #         kg = params[1] / 1000.0

    #         km = np.clip(km, -2.0, 2.0)
    #         kg = np.clip(kg, -2.0, 2.0)

    #         gamma = np.zeros(self.harmonics)
    #         gamma_star = np.zeros(self.harmonics)

    #         for i in range(self.harmonics):
    #             gamma[i] = params[i + 2] / 100.0
    #             gamma_star[i] = params[i + 2 + self.harmonics] / 100.0

    #         mu0 = params[self.harmonics * 2 + 2] / 100.0

    #         delta_0 = params[self.harmonics * 2 + 3] / 100.0
    #         delta_1 = params[self.harmonics * 2 + 4] / 100.0
    #         gamma_ = params[self.harmonics * 2 + 5] / 100.0
    #         xi = params[self.harmonics * 2 + 6] / 100.0
    #         zeta = params[self.harmonics * 2 + 7] / 100.0

    #         # gamma_ = params[self.harmonics * 2 + 3] / 100.0
    #         # xi = params[self.harmonics * 2 + 4] / 100.0
    #         # zeta = params[self.harmonics * 2 + 5] / 100.0

    #         # Verify critical condition
    #         if zeta <= 2 * gamma_:
    #             return 1e6

    #         # if zeta <= 0 or xi <= 0 or gamma_ <= 0:
    #         #     return np.inf

    #     except (ValueError, IndexError) as e:
    #         return np.inf

    #     # Initialize state
    #     mu_t = mu0
    #     sum_logpdf = 0.0
    #     valid_count = 0

    #     # Time iteration with error handling
    #     for t, yt in enumerate(y):
    #         try:
    #             # Calculate phi with bounds
    #             seasonal_sum = np.sum(gamma)
    #             phi = mu_t + seasonal_sum
    #             phi = np.clip(phi, -10, 10)

    #             # Calculate log-likelihood
    #             logpdf = self.distribution.logpdf(
    #                 yt,
    #                 delta_0=delta_0,
    #                 delta_1=delta_1,
    #                 phi=phi,
    #                 gamma=gamma_,
    #                 xi=xi,
    #                 zeta=zeta,
    #             )

    #             if np.isnan(logpdf) or np.isinf(logpdf):
    #                 continue  # Skip but don't fail

    #             sum_logpdf += logpdf
    #             valid_count += 1

    #             # Calculate score
    #             score_vec = self.distribution.score(
    #                 yt,
    #                 delta_0=delta_0,
    #                 delta_1=delta_1,
    #                 phi=phi,
    #                 gamma=gamma_,
    #                 xi=xi,
    #                 zeta=zeta,
    #             )
    #             score = score_vec[0]  # dL_dphi
    #             score = np.clip(score, -50, 50)

    #         except (ValueError, FloatingPointError) as e:
    #             score = 0.0
    #             continue

    #         # Update level
    #         mu_t = mu_t + km * score

    #         # Update seasonal components
    #         for i in range(self.harmonics):
    #             lambda_i = (2 * np.pi * (i + 1)) / self.period

    #             gamma_t1 = (
    #                 np.cos(lambda_i) * gamma[i]
    #                 + np.sin(lambda_i) * gamma_star[i]
    #                 + kg * score
    #             )

    #             gamma_start_t1 = (
    #                 np.cos(lambda_i) * gamma_star[i]
    #                 - np.sin(lambda_i) * gamma[i]
    #                 + kg * score
    #             )

    #             gamma[i] = gamma_t1
    #             gamma_star[i] = gamma_start_t1

    #     # Final validation
    #     if valid_count == 0:
    #         return np.inf

    #     if np.isnan(sum_logpdf) or np.isinf(sum_logpdf):
    #         return np.inf

    #     # Gentle regularization
    #     avg_loglik = sum_logpdf / valid_count
    #     regularization = 0.1 * np.sum(np.square(params / 100.0))
    #     return -avg_loglik + regularization
