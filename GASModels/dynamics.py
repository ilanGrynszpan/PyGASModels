from GASModels.component import Component
from GASModels.distributions.distribution import Distribution
import numpy as np
from scipy.fft import fft
from scipy.optimize import curve_fit


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

    def _estimate_seasonal_components(self, y, harmonics, period):
        """Estimate initial seasonal components with regularization for higher harmonics"""
        gamma = [10] * harmonics
        gamma_star = [10] * harmonics

        return gamma, gamma_star

    def _seasonal_pattern_regression(self, y, period, harmonics):
        """Alternative method using linear regression for seasonal pattern"""
        n = len(y)
        t = np.arange(n)

        # Create seasonal regressors
        X = np.ones((n, 1 + 2 * harmonics))  # intercept + 2* harmonics

        for i in range(harmonics):
            frequency = 2 * np.pi * (i + 1) / period
            X[:, 1 + 2 * i] = np.cos(frequency * t)
            X[:, 2 + 2 * i] = np.sin(frequency * t)

        try:
            # Fit regression
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

            # Extract seasonal coefficients (skip intercept)
            gamma_0 = np.zeros(harmonics)
            gamma_star_0 = np.zeros(harmonics)

            for i in range(harmonics):
                gamma_0[i] = coeffs[1 + 2 * i] if 1 + 2 * i < len(coeffs) else 0.01
                gamma_star_0[i] = coeffs[2 + 2 * i] if 2 + 2 * i < len(coeffs) else 0.01

            return gamma_0, gamma_star_0

        except:
            # Fallback if regression fails
            return np.zeros(harmonics) + 0.01, np.zeros(harmonics) + 0.01

    def get_initial_values(self, y):
        """
        Estimate initial values for Gamma score-driven model with harmonics

        Parameters:
        y: time series data
        """

        n = len(y)

        # 1. Initial mu0 - based on overall level in log space (for Gamma log-link)
        mu0 = 0  # np.log(y[0])

        # 2. Initial parameters - estimate Gamma shape parameter from data
        delta_0 = 50
        delta_1 = 10
        gamma_ = 33.7
        xi = 68.3
        zeta = 64.1

        # 3. Learning rates km, kg - start with small values
        km = 20
        kg = 20

        # 4. Seasonal components gamma_0, gamma_star_0 - use Fourier analysis
        gamma_0, gamma_star_0 = self._estimate_seasonal_components(
            y, self.harmonics, self.period
        )

        # Construct parameter vector for optimization
        initial_params = np.concatenate(
            [
                [km, kg],
                gamma_0,
                gamma_star_0,
                [mu0, delta_0, delta_1, gamma_, xi, zeta],
            ]
        )

        return initial_params

    def initialize(self):
        pass

    def update_components(self, score, hyperparameters, component_dynamics, t):
        pass

    def fit_in_sample(self, params, y):
        gamma = np.zeros(self.harmonics)
        gamma_star = np.zeros(self.harmonics)

        km = params[0] / 1000
        kg = params[1] / 1000

        for i in range(self.harmonics):
            gamma[i] = params[i + 2] / 100
            gamma_star[i] = params[i + 2 + self.harmonics] / 100

        mu0 = params[self.harmonics * 2 + 2] / 100
        delta_0 = params[self.harmonics * 2 + 3] / 100
        delta_1 = params[self.harmonics * 2 + 4] / 100
        gamma_ = params[self.harmonics * 2 + 5] / 100
        xi = params[self.harmonics * 2 + 6] / 100
        zeta = params[self.harmonics * 2 + 7] / 100

        mu = mu0
        fit_in_sample = np.zeros(self.n)

        for t in range(self.n):
            phi = mu + np.sum(gamma)
            score = self.distribution.score(
                y[t],
                delta_0=delta_0,
                delta_1=delta_1,
                phi=phi,
                gamma=gamma_,
                xi=xi,
                zeta=zeta,
            )[0]
            fit_in_sample[t] = self.distribution.mean(
                delta_0=delta_0,
                delta_1=delta_1,
                phi=phi,
                gamma=gamma_,
                xi=xi,
                zeta=zeta,
            )
            mu += km * score

            for i in range(self.harmonics):
                lambda_i = (2 * np.pi * (i + 1)) / self.period
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

        km = params[0] / 1000
        kg = params[1] / 1000

        if np.abs(kg) > 2.0 or np.abs(km) > 2.0:
            print("d1")
            return np.inf

        for i in range(self.harmonics):
            gamma[i] = params[i + 2] / 100
            gamma_star[i] = params[i + 2 + self.harmonics] / 100

        mu0 = params[self.harmonics * 2 + 2]

        delta_0 = params[self.harmonics * 2 + 3] / 100
        delta_1 = params[self.harmonics * 2 + 4] / 100
        gamma_ = params[self.harmonics * 2 + 5] / 100
        xi = params[self.harmonics * 2 + 6] / 100
        zeta = params[self.harmonics * 2 + 7] / 100

        if np.exp(zeta) <= np.exp(gamma_):
            print("d2")
            return np.inf

        if np.abs(delta_0) > 2.0 or np.abs(delta_1) > 2.0:
            return np.inf

        mu_t = mu0

        sum_logpdf = 0.0

        for t, yt in enumerate(y):
            phi = mu_t + np.sum(gamma)

            if np.abs(phi) > 10:
                return np.inf

            # Add bounds checking for numerical stability

            try:
                logpdf = self.distribution.logpdf(
                    yt,
                    delta_0=delta_0,
                    delta_1=delta_1,
                    phi=phi,
                    gamma=gamma_,
                    xi=xi,
                    zeta=zeta,
                )
                score = self.distribution.score(
                    yt,
                    delta_0=delta_0,
                    delta_1=delta_1,
                    phi=phi,
                    gamma=gamma_,
                    xi=xi,
                    zeta=zeta,
                )[0]
                if np.isnan(logpdf) or np.isnan(score):
                    return np.inf
            except (ValueError, FloatingPointError):
                print("d3")
                return np.inf

            sum_logpdf += logpdf

            if (
                np.isinf(np.abs(phi))
                or np.isinf(np.abs(gamma_))
                or np.isinf(np.abs(xi))
                or np.isinf(np.abs(zeta))
            ):
                print("d4")
                return np.inf

            # Update level
            mu_t += km * score

            # Update seasonal components
            for i in range(self.harmonics):
                lambda_i = (2 * np.pi * (i + 1)) / self.period
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

        if np.isnan(sum_logpdf) or np.isinf(sum_logpdf):
            return np.inf

        regularization = 0.05 * np.sum(params**2)

        return -sum_logpdf + regularization  # Negative for minimization
