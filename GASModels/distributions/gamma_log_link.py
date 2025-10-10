import numpy as np
import scipy.stats as stats
from scipy.special import digamma, polygamma
from GASModels.distributions.distribution import Distribution


class GammaLogLinkDistribution(Distribution):
    def __init__(self):
        super().__init__("GammaLogLink", 2)

    def _get_parameters(self, **kwargs):
        """Extract and transform parameters from kwargs"""
        alpha = kwargs.get("alpha")
        lambda_ = kwargs.get("lambda_")

        if alpha is None or lambda_ is None:
            raise ValueError("Both alpha and lambda_ parameters must be provided")

        # Transform parameters: (exp(alpha), exp(lambda - alpha))
        shape = np.exp(alpha)  # k = exp(alpha)
        scale = np.exp(lambda_ - alpha)  # theta = exp(lambda - alpha)

        return alpha, lambda_, shape, scale

    def logpdf(self, y, **kwargs):
        """
        Log probability density function for Gamma distribution
        with parameterization: shape = exp(alpha), scale = exp(lambda - alpha)

        Parameters:
        y: array-like, observed values
        alpha: shape parameter in log space
        lambda_: location parameter in log space (exp(lambda) is the mean)
        """
        alpha, lambda_, shape, scale = self._get_parameters(**kwargs)

        # Gamma logpdf: logpdf(y) = -log(Γ(k)) - k*log(θ) + (k-1)*log(y) - y/θ
        log_pdf = -stats.gamma.logpdf(y, a=shape, scale=scale)

        return log_pdf

    def score(self, y, **kwargs):
        """
        Score function (gradient of log-likelihood with respect to parameters)
        Returns gradient with respect to [alpha, lambda_]
        """
        alpha, lambda_, shape, scale = self._get_parameters(**kwargs)

        dL_dlambda = np.exp(alpha) * (y / np.exp(lambda_) - 1)
        dL_dalpha = np.exp(alpha) * (
            digamma(np.exp(alpha))
            - lambda_
            + alpha
            + 1
            + np.log(y)
            - y / np.exp(lambda_)
        )

        return np.array([dL_dlambda, dL_dalpha])

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
        alpha, lambda_, shape, scale = self._get_parameters(**kwargs)
        return np.exp(lambda_)

    def variance(self, **kwargs):
        """Return the variance of the distribution: shape * scale^2"""
        alpha, lambda_, shape, scale = self._get_parameters(**kwargs)
        return shape * (scale**2)
