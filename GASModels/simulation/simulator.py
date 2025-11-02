import numpy as np
import random
from scipy.stats import norm


class Simulator:
    def __init__(self):
        pass

    def simulate(self, n, harmonics, period, seed=None):
        random.seed(seed)
        mu = 10
        sigma = 1
        epsilon = norm.rvs(loc=0, scale=sigma, size=n)
        y = np.zeros(n)
        gamma = np.zeros(harmonics)
        gamma_star = np.zeros(harmonics)
        for i in range(harmonics):
            gamma[i] = random.uniform(0, 1)
            gamma_star[i] = random.uniform(0, 1)
        for t in range(n):
            seas = 0
            for i in range(harmonics):
                lambda_ = 2 * np.pi * (i + 1) / period
                seas += gamma[i] * np.cos(lambda_ * t) + gamma_star[i] * np.sin(
                    lambda_ * t
                )
            y[t] = mu + seas + epsilon[t]
        return y
