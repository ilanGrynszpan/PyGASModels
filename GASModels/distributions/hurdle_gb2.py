import numpy as np
from scipy.special import expit, beta, digamma, betainc
from GASModels.distributions.distribution import Distribution


class HurdleGB2LogLinkDistribution(Distribution):
    """
    Hurdle GB2:
    - Bernoulli for occurrence (dynamic pi_t)
    - GB2 for positive values
    """

    def __init__(self):
        super().__init__("HurdleGB2LogLink", 5)

    def logpdf(self, y, *, pi, phi, gamma, xi, zeta):
        """
        pi: P(y > 0)
        """
        if y <= 0:
            return np.log(1 - pi + 1e-12)

        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)

        z = (y / sigma) ** p

        log_gb2 = (
            np.log(p)
            - np.log(sigma)
            + (a * p - 1) * np.log(y / sigma)
            - np.log(beta(a, b))
            - (a + b) * np.log(1 + z)
        )

        return np.log(pi + 1e-12) + log_gb2

    def score_phi(self, y, *, phi, gamma, xi, zeta):
        """Score w.r.t phi (only for y>0)"""
        if y <= 0:
            return 0.0

        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)

        z = (y / sigma) ** p

        return p * ((a + b) * z / (1 + z) - a)

    def score_pi(self, y, pi):
        """Score for Bernoulli"""
        z = 1.0 if y > 0 else 0.0
        return z - pi

    def cdf(self, y, *, pi, phi, gamma, xi, zeta):
        if y <= 0:
            return 1 - pi

        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)

        z = (y / sigma) ** p
        u = z / (1 + z)
        u = np.clip(u, 1e-12, 1 - 1e-12)

        gb2_cdf = betainc(a, b, u)

        return (1 - pi) + pi * gb2_cdf
