import numpy as np
import scipy.stats as stats
from scipy.special import digamma, polygamma, beta
from scipy.special import betainc, expit
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
            -gamma
            + (np.exp(xi - gamma) - 1) * np.log(y / np.exp(phi))
            - phi
            - np.log(beta(np.exp(xi), np.exp(zeta)))
            - (np.exp(xi) + np.exp(zeta))
            * np.log(1 + (y / np.exp(phi)) ** (np.exp(-gamma)))
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

    def fisher_information(self, y, **kwargs):
        """
        Outer-product approximation of the Fisher information matrix:
        I_t ≈ s_t s_t'
        """

        # Get score vector
        s = self.score(y, **kwargs)

        # Outer product
        I = np.outer(s, s)

        return I

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

    def cdf(self, y, **kwargs):
        phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = exp(phi)
        z = (y / sigma) ** p
        u = z / (1 + z)
        F = betainc(a, b, u)

    def rvs(self, size=1, **kwargs):
        """
        Random sampling from GB2 distribution using Beta transformation.
        """

        phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)

        # Draw from Beta
        U = np.random.beta(a, b, size=size)

        # Avoid numerical explosion near 1
        eps = 1e-12
        U = np.clip(U, eps, 1 - eps)

        # Transform
        Y = sigma * (U / (1 - U)) ** (1 / p)

        if size == 1:
            return Y.item()

        return Y
