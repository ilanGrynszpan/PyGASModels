import numpy as np
import scipy.stats as stats
from scipy.special import digamma, polygamma, beta
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
        linear_comb = delta_0 + delta_1 * np.exp(phi)
        # Use log-exp trick for numerical stability
        max_val = np.maximum(0, linear_comb)
        pi = np.exp(linear_comb - max_val) / (1 + np.exp(linear_comb - max_val))

        # Handle both scalar and array inputs
        y = np.asarray(y)
        result = np.zeros_like(y, dtype=float)

        # Zero observations
        zero_mask = y == 0
        result[zero_mask] = np.log(1 - pi)

        # Positive observations
        pos_mask = y > 0
        if np.any(pos_mask):
            y_pos = y[pos_mask]

            # Numerical safeguards
            y_safe = np.maximum(y_pos, 1e-10)
            exponent = np.exp(-gamma)
            z = (y_safe / np.exp(phi)) ** exponent

            # Use log1p for numerical stability
            log_pdf_pos = (
                np.log(pi)
                - gamma
                + (np.exp(xi - gamma) - 1) * (np.log(y_safe) - phi)
                - phi
                - np.log(beta(np.exp(xi), np.exp(zeta)))
                - (np.exp(xi) + np.exp(zeta)) * np.log1p(z)
            )

            result[pos_mask] = log_pdf_pos

        return result

    def score(self, y, **kwargs):
        delta_0, delta_1, phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        # Numerical stability for pi calculation
        linear_comb = delta_0 + delta_1 * np.exp(phi)
        max_val = np.maximum(0, linear_comb)
        pi = np.exp(linear_comb - max_val) / (1 + np.exp(linear_comb - max_val))

        y = np.asarray(y)

        # Initialize scores for the 4 parameters we want (phi, gamma, xi, zeta)
        dL_dphi = np.zeros_like(y)
        dL_dgamma = np.zeros_like(y)
        dL_dxi = np.zeros_like(y)
        dL_dzeta = np.zeros_like(y)

        # Zero observations
        zero_mask = y == 0
        if np.any(zero_mask):
            dL_dphi[zero_mask] = -delta_1 * pi * np.exp(phi)

        # Positive observations
        pos_mask = y > 0
        if np.any(pos_mask):
            y_pos = y[pos_mask]

            # Numerical safeguards
            y_safe = np.maximum(y_pos, 1e-10)
            exponent = np.exp(-gamma)
            z = (y_safe / np.exp(phi)) ** exponent
            z = np.clip(z, 1e-10, 1e10)

            z_ratio = z / (1 + z)
            log_y_phi = np.log(y_safe) - phi

            # Score for phi (includes zero-inflation effect)
            dL_dphi[pos_mask] = (
                delta_1 * (1 - pi) * np.exp(phi)  # Zero-inflation component
                + (np.exp(xi) + np.exp(zeta)) * np.exp(-gamma) * z_ratio
                - np.exp(xi - gamma)
            )

            dL_dgamma[pos_mask] = (
                -1
                + log_y_phi * np.exp(xi - gamma)
                - (np.exp(xi) + np.exp(zeta)) * np.exp(-gamma) * log_y_phi * z_ratio
            )

            dL_dxi[pos_mask] = (
                log_y_phi * np.exp(xi - gamma)
                - np.exp(xi)
                * (digamma(np.exp(xi)) - digamma(np.exp(xi) + np.exp(zeta)))
                - np.exp(xi) * np.log1p(z)
            )

            dL_dzeta[pos_mask] = -np.exp(zeta) * (
                digamma(np.exp(zeta)) - digamma(np.exp(xi) + np.exp(zeta))
            ) - np.exp(zeta) * np.log1p(z)

        # Return average score across observations (only 4 parameters)
        return np.array(
            [np.mean(dL_dphi), np.mean(dL_dgamma), np.mean(dL_dxi), np.mean(dL_dzeta)]
        )

    def fisher_information(self, **kwargs):
        """
        Fisher information matrix - placeholder for ZA-GB2
        This is complex for zero-inflated models
        """
        # Return identity matrix as placeholder
        return np.eye(4)

    def mean(self, **kwargs):
        """Return the mean of the zero-inflated distribution"""
        delta_0, delta_1, phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        if np.exp(zeta) <= np.exp(gamma):
            raise ValueError("Mean is undefined when zeta <= gamma")

        # Numerical stability for pi
        linear_comb = delta_0 + delta_1 * np.exp(phi)
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
        linear_comb = delta_0 + delta_1 * np.exp(phi)
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
