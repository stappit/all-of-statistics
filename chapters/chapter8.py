import numpy as np
from scipy.stats import norm
from .chapter7 import CI


def bootstrap(observations, statistic, iters):
    n = len(observations)
    for _ in range(iters):
        yield statistic(np.random.choice(observations, n))


def bootstrap_variance(observations, statistic, iters):
    boots = list(bootstrap(observations, statistic, iters))
    vboot = np.var(boots)
    return vboot


def bootstrap_ci_normal(observations, statistic, iters, a):
    t = statistic(observations)
    vboot = bootstrap_variance(observations, statistic, iters)
    seboot = np.sqrt(vboot)
    z = np.abs(norm.ppf(a / 2))
    return CI(t - z*seboot, t + z*seboot)


def bootstrap_ci_pivot(observations, statistic, iters, a):
    theta = statistic(observations)
    boots = list(bootstrap(observations, statistic, iters))
    theta_l = np.percentile(boots, 100 - 100*a / 2)
    theta_r = np.percentile(boots, 100*a / 2)
    return CI(2*theta - theta_l, 2*theta - theta_r)


def bootstrap_ci_percentile(observations, statistic, iters, a):
    boots = list(bootstrap(observations, statistic, iters))
    theta_l = np.percentile(boots, 100*a / 2)
    theta_r = np.percentile(boots, 100 - 100*a / 2)
    return CI(theta_l, theta_r)
