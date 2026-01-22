import numpy as np


class DynamicsUtils:

    def __init__(self):
        pass

    def iterateGAS(self, distribution, hyperparameters, n, y, trend, seasonality):

        mu = hyperparameters["mu_0"]
        seasonal = hyperparameters["seasonal_0"]

        phi = mu + seasonality.get_gamma(seasonal)

        loglik = np.zeros(n)
        loglik[0] = distribution.logpdf(
            y[0], phi, hyperparameters["fixed_distribution_params"]
        )

        for t in range(2, n):

            score = distribution.scaled_score(
                y[t - 1], phi, hyperparameters["fixed_distribution_params"]
            )
            mu = trend.update(mu, score, hyperparameters["kappa_mu"])
            seasonal = seasonality.update(
                seasonal, score, hyperparameters["kappa_gamma"]
            )
            phi = mu + seasonality.get_gamma(seasonal)

            loglik[t] = distribution.logpdf(
                y[t], phi, hyperparameters["fixed_distribution_params"]
            )

        return -np.sum(loglik)
