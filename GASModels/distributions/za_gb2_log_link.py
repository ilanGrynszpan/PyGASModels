import numpy as np
import scipy.stats as stats
from scipy.special import digamma, polygamma, beta
from scipy.special import betainc, expit
from GASModels.distributions.distribution import Distribution


class ZAGB2LogLinkDistribution(Distribution):
    def __init__(self):
        super().__init__("ZAGB2LogLink", 6)  # Changed to 6 parameters

    def _get_parameters(self, **kwargs):
        """Extract and transform parameters from kwargs"""
        delta_0 = kwargs.get("delta_0")
        delta_1 = kwargs.get("delta_1")
        phi = kwargs.get("phi")
        gamma = kwargs.get("gamma")
        xi = kwargs.get("xi")
        zeta = kwargs.get("zeta")

        if any(param is None for param in [delta_0, delta_1, phi, gamma, xi, zeta]):
            raise ValueError("All ZA-GB2 parameters must be provided")

        return delta_0, delta_1, phi, gamma, xi, zeta

    def logpdf(self, y, **kwargs):
        delta_0, delta_1, phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        # Numerical stability for pi calculation
        linear_comb = delta_0 + delta_1 * phi
        # Use log-exp trick for numerical stability
        pi = np.exp(linear_comb) / (1 + np.exp(linear_comb))

        if y <= 0:
            return np.log(1 - pi)
        else:
            return (
                np.log(pi)
                - gamma
                + (np.exp(xi - gamma) - 1) * (np.log(y) - phi)
                - phi
                - np.log(beta(np.exp(xi), np.exp(zeta)))
                - (np.exp(xi) + np.exp(zeta))
                * np.log(1 + (y / np.exp(phi)) ** np.exp(-gamma))
            )

    def score(self, y, **kwargs):
        delta_0, delta_1, phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        # Numerical stability for pi calculation
        linear_comb = delta_0 + delta_1 * phi
        pi = np.exp(linear_comb) / (1 + np.exp(linear_comb))

        # Initialize scores for the 4 parameters we want (phi, gamma, xi, zeta)
        dL_dphi = 0.0
        dL_dgamma = 0.0
        dL_dxi = 0.0
        dL_dzeta = 0.0

        if y <= 0:
            dL_dphi = -delta_1 * pi
        else:
            z = (y / np.exp(phi)) ** np.exp(-gamma)
            dL_dphi = (
                delta_1 * (1 - pi)  # Zero-inflation component
                + (np.exp(xi) + np.exp(zeta)) * np.exp(-gamma) * z / (1 + z)
                - np.exp(xi - gamma)
            )

            dL_dgamma = (
                -1
                - np.exp(xi - gamma) * np.log(y / np.exp(phi))
                + (np.exp(xi) + np.exp(zeta))
                * np.exp(-gamma)
                * (z / (1 + z))
                * (np.log(y / np.exp(phi)))
            )

            dL_dxi = (
                (np.exp(xi - gamma) * np.log(y / np.exp(phi)))
                - np.exp(xi)
                * (digamma(np.exp(xi)) - digamma(np.exp(xi) + np.exp(zeta)))
                - np.log(1 + z) * np.exp(xi)
            )

            dL_dzeta = -np.exp(zeta) * (
                digamma(np.exp(zeta)) - digamma(np.exp(xi) + np.exp(zeta))
            ) - np.log(1 + z) * np.exp(zeta)

        return np.array([dL_dphi, dL_dgamma, dL_dxi, dL_dzeta])

    def fisher_information(self, **kwargs):
        """
        Fisher information matrix - placeholder for ZA-GB2
        This is complex for zero-inflated models
        """
        # Return identity matrix as placeholder
        return np.eye(4)

    def cdf(self, y, **kwargs):
        """
        CDF for Zero-Augmented GB2 distribution.

        F(y) = {
            0                   if y < 0
            1 - π               if y = 0
            (1 - π) + π * F_GB2(y) if y > 0
        }
        """
        delta_0, delta_1, phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        # π = logistic(delta_0 + delta_1 * exp(phi))
        linear_comb = delta_0 + delta_1 * phi
        pi = expit(linear_comb)  # numerically stable sigmoid

        if y <= 0:
            return 1 - pi
        else:
            # GB2 parameters
            sigma = np.exp(phi)
            p = np.exp(-gamma)
            a = np.exp(xi)
            b = np.exp(zeta)

            # GB2 CDF = I_{z/(1+z)}(a, b), with z = (y/σ)^p
            z = (y / sigma) ** p
            gb2_cdf = betainc(a, b, z / (1 + z))

            return (1 - pi) + pi * gb2_cdf

    def mean(self, **kwargs):
        """Return the mean of the zero-inflated distribution"""
        delta_0, delta_1, phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        if np.exp(zeta) <= np.exp(gamma):
            raise ValueError("Mean is undefined when zeta <= gamma")

        # Numerical stability for pi
        linear_comb = delta_0 + delta_1 * phi
        max_val = np.maximum(0, linear_comb)
        pi = np.exp(linear_comb - max_val) / (1 + np.exp(linear_comb - max_val))

        # GB2 component mean
        gb2_mean = (
            np.exp(phi)
            * beta(np.exp(xi) + np.exp(gamma), np.exp(zeta) - np.exp(gamma))
            / beta(np.exp(xi), np.exp(zeta))
        )

        # Zero-inflated mean: pi * GB2_mean + (1-pi)*0 = pi * GB2_mean
        expected_value = pi * gb2_mean

        return expected_value

    def variance(self, **kwargs):
        """Return the variance of the zero-inflated distribution"""
        delta_0, delta_1, phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        if np.exp(zeta) <= np.exp(gamma):
            raise ValueError("Variance is undefined when zeta <= gamma")

        # Numerical stability for pi
        linear_comb = delta_0 + delta_1 * phi
        max_val = np.maximum(0, linear_comb)
        pi = np.exp(linear_comb - max_val) / (1 + np.exp(linear_comb - max_val))

        # GB2 component moments
        ex = (
            np.exp(phi)
            * beta(np.exp(xi) + np.exp(gamma), np.exp(zeta) - np.exp(gamma))
            / beta(np.exp(xi), np.exp(zeta))
        )

        ex2 = (
            np.exp(phi) ** 2
            * beta(np.exp(xi) + 2 * np.exp(gamma), np.exp(zeta) - 2 * np.exp(gamma))
            / beta(np.exp(xi), np.exp(zeta))
        )

        # Zero-inflated variance: pi * E[X^2] - (pi * E[X])^2
        variance = pi * ex2 - (pi * ex) ** 2

        return variance
