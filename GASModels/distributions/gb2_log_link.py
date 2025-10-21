import numpy as np
import scipy.stats as stats
from scipy.special import digamma, polygamma, beta
from GASModels.distributions.distribution import Distribution


class GB2LogLinkDistribution(Distribution):
    def __init__(self):
        super().__init__("GB2LogLink", 4)

    def _get_parameters(self, **kwargs):
        """Extract and transform parameters from kwargs"""
        phi = kwargs.get("phi")
        gamma = kwargs.get("gamma")
        xi = kwargs.get("xi")
        zeta = kwargs.get("zeta")

        if phi is None or gamma is None or xi is None or zeta is None:
            raise ValueError("All GB2 parameters must be provided")

        return phi, gamma, xi, zeta

    def logpdf(self, y, **kwargs):

        phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        log_pdf = (
            -np.log(gamma)
            + (xi / gamma - 1) * np.log(y / phi)
            - np.log(phi)
            - np.log(beta(xi, zeta))
            - (xi + zeta) * np.log(1 + (y / phi) ** (1 / gamma))
        )

        return log_pdf

    def score(self, y, **kwargs):

        phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        z = (y / np.exp(phi)) ** (np.exp(-gamma))

        dL_dphi = (np.exp(xi) + np.exp(zeta)) * np.exp(-gamma) * z / (1 + z) - np.exp(
            xi - gamma
        )
        dL_dgamma = (
            -1
            + np.log(y / np.exp(phi)) * np.exp(xi - gamma)
            - (np.exp(xi) + np.exp(zeta))
            * np.exp(-gamma)
            * np.log(y / np.exp(phi))
            * z
            / (1 + z)
        )
        dL_dxi = (
            np.log(y / np.exp(phi)) * np.exp(xi - gamma)
            - np.exp(xi) * (digamma(np.exp(xi)) - digamma(np.exp(xi) + np.exp(zeta)))
            - np.exp(xi) * np.log(1 + z)
        )
        dL_dzeta = -np.exp(zeta) * (
            digamma(np.exp(zeta)) - digamma(np.exp(xi) + np.exp(zeta))
        ) - np.exp(zeta) * np.log(1 + z)

        return np.array([dL_dphi, dL_dgamma, dL_dxi, dL_dzeta])

    def fisher_information(self, **kwargs):
        """
        Fisher information matrix for parameters [alpha, lambda_]
        """
        alpha, lambda_, shape, scale = self._get_parameters(**kwargs)

        # For Gamma(k, θ), the Fisher information in terms of (k, θ) is:
        # I(k,k) = polygamma(1, k)  (trigamma function)
        # I(k,θ) = 1/θ
        # I(θ,θ) = k/θ²

        trigamma_shape = polygamma(1, shape)
        I_kk = trigamma_shape
        I_ktheta = 1 / scale
        I_thetatheta = shape / (scale**2)

        # Transform Fisher information from (k, θ) space to (alpha, lambda_) space
        # Using the Jacobian of the transformation
        # J = [[dk/dalpha, dk/dlambda], [dθ/dalpha, dθ/dlambda]] = [[k, 0], [-θ, θ]]
        J = np.array([[shape, 0], [-scale, scale]])

        # Original Fisher information matrix in (k, θ) space
        I_original = np.array([[I_kk, I_ktheta], [I_ktheta, I_thetatheta]])

        # Transform: I_new = J^T @ I_original @ J
        I_transformed = J.T @ I_original @ J

        return I_transformed

    def mean(self, **kwargs):
        """Return the mean of the distribution: exp(lambda)"""
        phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        if np.exp(zeta) <= np.exp(gamma):
            raise ValueError("Mean is undefined when zeta <= gamma")

        expected_value = (
            np.exp(phi)
            * beta(np.exp(xi) + np.exp(gamma), np.exp(zeta) - np.exp(gamma))
            / beta(np.exp(xi), np.exp(zeta))
        )

        return expected_value

    def variance(self, **kwargs):
        """Return the variance of the distribution: shape * scale^2"""
        phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        if np.exp(zeta) <= np.exp(gamma):
            raise ValueError("Mean is undefined when zeta <= gamma")

        ex2 = (
            np.exp(phi) ** 2
            * beta(np.exp(xi) + 2 * np.exp(gamma), np.exp(zeta) - 2 * np.exp(gamma))
            / beta(np.exp(xi), np.exp(zeta))
        )
        ex = (
            np.exp(phi)
            * beta(np.exp(xi) + np.exp(gamma), np.exp(zeta) - np.exp(gamma))
            / beta(np.exp(xi), np.exp(zeta))
        )

        variance = ex2 - ex**2

        return variance
