import numpy as np
from scipy import stats


def beta_dist_from_mean_and_variance(mean, variance):
    """
    Returns a beta distribution with the specified mean and variance.
    I.e. implements the method of moments for the Beta distribution.
    """
    if mean <= 0 or mean >= 1:
        raise ValueError("Mean must be in the interval (0,1)")
    if variance >= mean * (1 - mean):
        raise ValueError(f"Variance too large: {variance} >= {mean*(1-mean)}")

    # For derivations of these formulas, see:
    # https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
    relative_variance = mean * (1 - mean) / variance - 1
    alpha = mean * relative_variance
    beta = (1 - mean) * relative_variance
    return stats.beta(a=alpha, b=beta)  # a,b can be recovered from dist.args


def gamma_dist_from_mean_and_variance(mean, variance):
    """Method of moments for gamma distribution."""
    alpha = mean**2 / variance
    scale = variance / mean
    return scipy.stats.gamma(a=alpha, scale=scale)


def lognorm_dist_from_mean_and_variance(mean, variance):
    """Method of moments for lognormal distribution.
    See https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    mean_squared = mean**2
    median = mean_squared / np.sqrt(variance + mean_squared)
    # median = mean / np.sqrt(1+ variance / mean_squared)
    sigma_squared = np.log(1 + variance / mean_squared)
    return stats.lognorm(s=np.sqrt(sigma_squared), scale=median)


def lognorm_dist_from_mean_and_median(mean, median):
    if mean <= median:
        raise ValueError(f"mean must be larger than median! {mean=}, {median=}")
    sigma = np.sqrt(2 * np.log(mean / median))
    return stats.lognorm(s=sigma, scale=median)


def normal_dist_from_mean_and_variance(mean, variance):
    """Method of moments for normal distribution."""
    return stats.norm(loc=mean, scale=np.sqrt(variance))


def truncnorm_dist_from_support(support_min, support_max, loc=0, scale=1):
    """Returns a truncatated normal distribution (as a frozen
    scipy.stats.truncnorm) with the specified support, location,
    and scale.

    This is a convenience function to convert the support of the desired
    distribution into the correct shape parameters for truncnorm(), which are,
    less intuitively, the endpoints of the support of the standardized
    distribution with loc=0 an scale=1.

    See scipy.stats.truncnorm documentation at
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    """
    a = (support_min - loc) / scale
    b = (support_max - loc) / scale
    return stats.truncnorm(a=a, b=b, loc=loc, scale=scale)
