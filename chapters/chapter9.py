import numpy as np
from scipy.stats import norm
from .chapter7 import CI


def parametric_bootstrap(dist, n, statistic, iters, random_state=0):
    for i in range(iters):
        yield statistic(dist.rvs(n, random_state=random_state+i))


def parametric_bootstrap_variance(dist, n, statistic, iters, random_state=0):
    boots = list(parametric_bootstrap(dist, n, statistic, iters, random_state=random_state))
    vboot = np.var(boots, axis=0)
    return vboot
