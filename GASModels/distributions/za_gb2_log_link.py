import numpy as np
from scipy.special import digamma, beta, betainc, expit
from GASModels.distributions.distribution import Distribution


class ZAGB2LogLinkDistribution(Distribution):
    def __init__(self):
        super().__init__("ZAGB2LogLink", 6)

    def _get_parameters(self, **kwargs):
        delta_0 = kwargs.get("delta_0")
        delta_1 = kwargs.get("delta_1")
        phi = kwargs.get("phi")
        gamma = kwargs.get("gamma")
        xi = kwargs.get("xi")
        zeta = kwargs.get("zeta")

        if any(v is None for v in [delta_0, delta_1, phi, gamma, xi, zeta]):
            raise ValueError("All ZA-GB2 parameters must be provided")

        return delta_0, delta_1, phi, gamma, xi, zeta

    # =========================
    # LOGPDF
    # =========================
    def logpdf(self, y, **kwargs):
        delta_0, delta_1, phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        pi = expit(delta_0 + delta_1 * phi)

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

    # =========================
    # SCORE
    # =========================
    def score(self, y, **kwargs):
        delta_0, delta_1, phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        pi = expit(delta_0 + delta_1 * phi)

        a = np.exp(xi)
        b = np.exp(zeta)
        p = np.exp(-gamma)
        sigma = np.exp(phi)

        if y <= 0:
            dL_dphi = -delta_1 * pi
            return np.array([dL_dphi, 0.0, 0.0, 0.0])

        z = (y / sigma) ** p
        logy = np.log(y / sigma)

        # φ-score
        dL_dphi = delta_1 * (1 - pi) + p * ((a + b) * z / (1 + z) - a)

        # γ-score
        dL_dgamma = -1 - p * logy * (a - (a + b) * z / (1 + z))

        # ξ-score
        dL_dxi = a * p * logy - a * (digamma(a) - digamma(a + b)) - a * np.log(1 + z)

        # ζ-score
        dL_dzeta = -b * (digamma(b) - digamma(a + b)) - b * np.log(1 + z)

        return np.array([dL_dphi, dL_dgamma, dL_dxi, dL_dzeta])

    # =========================
    # CDF (for PIT!)
    # =========================
    def cdf(self, y, **kwargs):
        delta_0, delta_1, phi, gamma, xi, zeta = self._get_parameters(**kwargs)

        pi = expit(delta_0 + delta_1 * phi)

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
