from GASModels.component import Component
from GASModels.distributions.distribution import Distribution
import numpy as np
from scipy.fft import fft
from scipy.optimize import curve_fit
from scipy import stats


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
        gamma = [0.2] * harmonics
        gamma_star = [0.2] * harmonics

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
        Estimate initial values that respect the bounds
        """
        n = len(y)
        y_positive = y[y > 0]

        if len(y_positive) == 0:
            y_positive = np.array([1e-6])

        # # 1. Learning rates - within bounds (1, 2000)
        # km = 50  # Will be 0.5 after scaling
        # kg = 20  # Will be 0.2 after scaling

        # # 2. Seasonal components - within bounds (-200, 200)
        # gamma_0 = np.zeros(self.harmonics) * 40  # Small values
        # gamma_star_0 = np.zeros(self.harmonics) * 40

        # # 3. Distribution parameters - CRITICAL: must respect bounds
        # mu0 = 10  # Will be 1.0 after scaling (within -1000, 1000)

        # # These must be within bounds after scaling:
        # delta_0 = 100  # Will be 0.1 after scaling (within -200, 200)
        # delta_1 = 100  # Will be 0.05 after scaling (within -100, 100)
        # gamma_ = 33  # Will be 1.0 after scaling (within 10, 500)
        # xi = 68  # Will be 1.0 after scaling (within 10, 500)
        # zeta = 66  # Will be 2.0 after scaling (within 20, 1000) AND zeta > gamma_

        # 1. Learning rates - within bounds (1, 2000)
        km = 20  # Will be 0.5 after scaling
        kg = 50  # Will be 0.2 after scaling

        # 2. Seasonal components - within bounds (-200, 200)
        gamma_0 = np.zeros(self.harmonics) * 40  # Small values
        gamma_star_0 = np.zeros(self.harmonics) * 40

        gamma_pi_0 = (np.zeros(self.harmonics)) * 5  # Small values
        gamma_star_pi_0 = (np.zeros(self.harmonics)) * 5

        # 3. Distribution parameters - CRITICAL: must respect bounds
        mu0 = 10  # Will be 1.0 after scaling (within -1000, 1000)

        # These must be within bounds after scaling:
        delta_0 = 50  # Will be 0.1 after scaling (within -200, 200)
        # delta_1 = 50  # Will be 0.05 after scaling (within -100, 100)
        gamma_ = 20  # Will be 1.0 after scaling (within 10, 500)
        xi = 200  # Will be 1.0 after scaling (within 10, 500)
        zeta = 200  # Will be 2.0 after scaling (within 20, 1000) AND zeta > gamma_

        # Verify bounds compliance
        test_params = np.concatenate(
            [
                [km, kg],
                gamma_0,
                gamma_star_0,
                [mu0, delta_0, gamma_, xi, zeta],
                gamma_pi_0,
                gamma_star_pi_0,
            ]
        )

        print(f"Initial parameter check:")
        print(f"km: {km} (bounds: 1-2000) - {'OK' if 1 <= km <= 2000 else 'VIOLATION'}")
        print(f"kg: {kg} (bounds: 1-2000) - {'OK' if 1 <= kg <= 2000 else 'VIOLATION'}")
        print(
            f"gamma_0: {gamma_0[0]} (bounds: -200-200) - {'OK' if all(-200 <= g <= 200 for g in gamma_0) else 'VIOLATION'}"
        )
        print(
            f"mu0: {mu0} (bounds: -1000-1000) - {'OK' if -1000 <= mu0 <= 1000 else 'VIOLATION'}"
        )
        # print(
        #     f"delta_0: {delta_0} (bounds: -200-200) - {'OK' if -200 <= delta_0 <= 200 else 'VIOLATION'}"
        # )
        # print(
        #     f"delta_1: {delta_1} (bounds: -100-100) - {'OK' if -100 <= delta_1 <= 100 else 'VIOLATION'}"
        # )
        print(
            f"gamma_: {gamma_} (bounds: 10-500) - {'OK' if 0 <= gamma_ <= 500 else 'VIOLATION'}"
        )
        print(f"xi: {xi} (bounds: 10-500) - {'OK' if 0 <= xi <= 500 else 'VIOLATION'}")
        print(
            f"zeta: {zeta} (bounds: 20-1000) - {'OK' if 0 <= zeta <= 1000 else 'VIOLATION'}"
        )
        print(
            f"zeta > gamma_: {zeta > gamma_} - {'OK' if zeta > gamma_ else 'VIOLATION'}"
        )

        # Construct parameter vector
        initial_params = np.concatenate(
            [
                [km, kg],
                gamma_0,
                gamma_star_0,
                [mu0, delta_0, gamma_, xi, zeta],
                gamma_pi_0,
                gamma_star_pi_0,
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
        # delta_1 = params[self.harmonics * 2 + 4] / 100
        # gamma_ = params[self.harmonics * 2 + 5] / 100
        # xi = params[self.harmonics * 2 + 6] / 100
        # zeta = params[self.harmonics * 2 + 7] / 100

        gamma_ = params[self.harmonics * 2 + 4] / 100
        xi = params[self.harmonics * 2 + 5] / 100
        zeta = params[self.harmonics * 2 + 6] / 100

        gamma_pi = np.zeros(self.harmonics)
        gamma_star_pi = np.zeros(self.harmonics)

        for i in range(self.harmonics):
            gamma_pi[i] = params[i + self.harmonics * 2 + 7] / 100
            gamma_star_pi[i] = params[i + self.harmonics * 3 + 7] / 100

        mu = mu0
        fit_in_sample = np.zeros(self.n)
        mean_gb2 = np.zeros(self.n)
        pi = np.zeros(self.n)

        for t in range(self.n):
            phi = mu + np.sum(gamma)

            score = self.distribution.score(
                y[t],
                t=t,
                delta_0=delta_0,
                # delta_1=delta_1,
                phi=phi,
                gamma=gamma_,
                xi=xi,
                zeta=zeta,
                gamma_pi=gamma_pi,
                gamma_star_pi=gamma_star_pi,
            )[0]
            mean_gb2[t], fit_in_sample[t] = self.distribution.mean(
                t=t,
                delta_0=delta_0,
                # delta_1=delta_1,
                phi=phi,
                gamma=gamma_,
                xi=xi,
                zeta=zeta,
                gamma_pi=gamma_pi,
                gamma_star_pi=gamma_star_pi,
            )
            mu += km * score
            # pi[t] = np.exp(delta_0 + delta_1 * phi) / (
            #     1 + np.exp(delta_0 + delta_1 * phi)
            # )

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

                seas = [
                    gamma_pi[i] * np.cos(2 * np.pi * (i + 1) * t / 365.25)
                    + gamma_star_pi[i] * np.sin(2 * np.pi * (i + 1) * t / 365.25)
                    for i in range(5)
                ]
                linear_comb = delta_0 + np.sum(seas)
                pi[t] = np.exp(linear_comb) / (1 + np.exp(linear_comb))

        return mean_gb2, fit_in_sample, pi

    def pit(self, params, y):
        gamma = np.zeros(self.harmonics)
        gamma_star = np.zeros(self.harmonics)

        km = params[0] / 1000
        kg = params[1] / 1000

        for i in range(self.harmonics):
            gamma[i] = params[i + 2] / 100
            gamma_star[i] = params[i + 2 + self.harmonics] / 100

        mu0 = params[self.harmonics * 2 + 2] / 100
        delta_0 = params[self.harmonics * 2 + 3] / 100
        # delta_1 = params[self.harmonics * 2 + 4] / 100
        # gamma_ = params[self.harmonics * 2 + 5] / 100
        # xi = params[self.harmonics * 2 + 6] / 100
        # zeta = params[self.harmonics * 2 + 7] / 100

        gamma_ = params[self.harmonics * 2 + 4] / 100
        xi = params[self.harmonics * 2 + 5] / 100
        zeta = params[self.harmonics * 2 + 6] / 100

        gamma_pi = np.zeros(self.harmonics)
        gamma_star_pi = np.zeros(self.harmonics)

        for i in range(self.harmonics):
            gamma_pi[i] = params[i + self.harmonics * 2 + 7] / 100
            gamma_star_pi[i] = params[i + self.harmonics * 3 + 7] / 100

        mu = mu0
        pit = np.zeros(self.n)

        for t in range(self.n):
            phi = mu + np.sum(gamma)

            score = self.distribution.score(
                y[t],
                t=t,
                delta_0=delta_0,
                # delta_1=delta_1,
                phi=phi,
                gamma=gamma_,
                xi=xi,
                zeta=zeta,
                gamma_pi=gamma_pi,
                gamma_star_pi=gamma_star_pi,
            )[0]
            pit[t] = self.distribution.cdf(
                y[t],
                t=t,
                delta_0=delta_0,
                # delta_1=delta_1,
                phi=phi,
                gamma=gamma_,
                xi=xi,
                zeta=zeta,
                gamma_pi=gamma_pi,
                gamma_star_pi=gamma_star_pi,
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

        return pit

    def get_quantile_residuals(self, pit):
        """
        Calculate quantile residuals from the fitted model.

        Quantile residuals are defined as the inverse standard normal CDF
        applied to the PIT values: r_t = Φ^{-1}(PIT_t)

        Parameters
        ----------
        params : array_like
            Model parameters
        y : array_like
            Observed time series data

        Returns
        -------
        quantile_residuals : ndarray
            Quantile residuals following standard normal distribution
            if the model is correctly specified
        """

        # Transform to quantile residuals using inverse standard normal CDF
        quantile_residuals = stats.norm.ppf(pit)

        return quantile_residuals

    def objective(self, params, y, bounds):

        gamma = np.zeros(self.harmonics)
        gamma_star = np.zeros(self.harmonics)

        km = params[0] / 1000
        kg = params[1] / 1000

        for i in range(self.harmonics):
            gamma[i] = params[i + 2] / 100
            gamma_star[i] = params[i + 2 + self.harmonics] / 100

        mu0 = params[self.harmonics * 2 + 2] / 100
        delta_0 = params[self.harmonics * 2 + 3] / 100
        # delta_1 = params[self.harmonics * 2 + 4] / 100
        # gamma_ = params[self.harmonics * 2 + 5] / 100
        # xi = params[self.harmonics * 2 + 6] / 100
        # zeta = params[self.harmonics * 2 + 7] / 100

        gamma_ = params[self.harmonics * 2 + 4] / 100
        xi = params[self.harmonics * 2 + 5] / 100
        zeta = params[self.harmonics * 2 + 6] / 100

        gamma_pi = np.zeros(self.harmonics)
        gamma_star_pi = np.zeros(self.harmonics)

        for i in range(self.harmonics):
            gamma_pi[i] = params[i + self.harmonics * 2 + 7] / 100
            gamma_star_pi[i] = params[i + self.harmonics * 3 + 7] / 100
        # Extensive parameter validation FIRST
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            return np.inf

        # # Check if parameters are within bounds (sanity check)
        for i, (param, (low, high)) in enumerate(zip(params, bounds)):
            if not (low <= param <= high):
                return np.inf

        if np.exp(zeta) <= 2 * np.exp(gamma_):
            return np.inf

        # Initialize state
        mu_t = mu0
        sum_logpdf = 0.0
        valid_count = 0

        # Time iteration with error handling
        for t, yt in enumerate(y):
            try:
                # Calculate phi with bounds
                seasonal_sum = np.sum(gamma)
                phi = mu_t + seasonal_sum

                # Calculate log-likelihood
                logpdf = self.distribution.logpdf(
                    yt,
                    t=t,
                    delta_0=delta_0,
                    # delta_1=delta_1,
                    phi=phi,
                    gamma=gamma_,
                    xi=xi,
                    zeta=zeta,
                    gamma_pi=gamma_pi,
                    gamma_star_pi=gamma_star_pi,
                )

                if np.isnan(logpdf) or np.isinf(logpdf):
                    continue  # Skip but don't fail

                sum_logpdf += logpdf
                valid_count += 1

                # Calculate score
                score_vec = self.distribution.score(
                    yt,
                    t=t,
                    delta_0=delta_0,
                    # delta_1=delta_1,
                    phi=phi,
                    gamma=gamma_,
                    xi=xi,
                    zeta=zeta,
                    gamma_pi=gamma_pi,
                    gamma_star_pi=gamma_star_pi,
                )
                score = score_vec[0]  # dL_dphi

            except (ValueError, FloatingPointError) as e:
                score = 0.0
                continue

            # Update level
            mu_t += km * score

            if np.isnan(mu_t) or np.isinf(mu_t) or np.abs(mu_t) > 10.0:
                mu_t = 0.0

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

        # Final validation
        if valid_count == 0:
            print("No valid observations")
            return np.inf

        if np.isnan(sum_logpdf) or np.isinf(sum_logpdf):
            print("Invalid final log-likelihood")
            return np.inf

        # Gentle regularization
        avg_loglik = sum_logpdf / valid_count
        regularization = 0.1 * np.sum(np.square(params / 100.0))
        return -avg_loglik + regularization
