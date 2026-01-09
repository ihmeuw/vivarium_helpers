import numpy as np
from scipy import stats


def beta_approx_median(a, b):
    return (a - 1 / 3) / (a + b - 2 / 3)


def normal_stdev_from_quantiles(quantiles, quantile_ranks):
    """
    Computes the standard deviation of a normal distribution that two quantiles
    with the specified ranks.
    """
    #     # If q = quantile, mu = mean, and sigma = std deviation, then
    #     # q = mu + q'*sigma, where q' is the standard normal quantile
    #     # and q is the transformed quantile, so sigma = (q-mu)/q'
    #     return (quantile - mean) / scipy.stats.norm().ppf(quantile_rank)
    # quantiles of the standard normal distribution with specified quantile_ranks
    stdnorm_quantiles = stats.norm.ppf(quantile_ranks)
    # Find sigma such that (upper - lower) = (q1 - q0)*sigma, where q1 and q2 are the standard normal
    # quantiles with the specified ranks
    stdev = (quantiles[1] - quantiles[0]) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    return stdev


def beta_dist_approx_from_mean_lower_upper(mean, lower, upper, quantile_ranks=(0.025, 0.975)):
    """
    Returns a scipy.stats Beta distribution with the specified mean and
    quantiles of ranks approximately equal to quantile_ranks.
    This is achieved by specifying that the variance of the Beta distribution
    is equal to the variance of a normal distribution with the same mean and
    the specified quantile. This will not work well if the variance of the required Beta
    distribution is too large (i.e. if the probability is concentrated near 0 and 1
    instead of near the mean), because there is no normal distribution with this behavior,
    and it also won't work well if lower and upper are not close to the actual quantiles
    for the specified ranks.
    """
    variance = normal_stdev_from_quantiles((lower, upper), quantile_ranks) ** 2
    return beta_dist_from_mean_var(mean, variance)


def normal_dist_approx_from_mean_lower_upper(
    mean, lower, upper, quantile_ranks=(0.025, 0.975)
):
    """Returns a frozen normal distribution with the specified mean, such that
    (lower, upper) are approximately equal to the quantiles with ranks
    (quantile_ranks[0], quantile_ranks[1]).
    """
    # quantiles of the standard normal distribution with specified quantile_ranks
    stdnorm_quantiles = stats.norm.ppf(quantile_ranks)
    # Find sigma such that (upper - lower) = (q1 - q0)*sigma, where q1 and q2 are the standard normal
    # quantiles with the specified ranks
    stdev = (upper - lower) / (stdnorm_quantiles[1] - stdnorm_quantiles[0])
    # Frozen normal distribution
    return stats.norm(loc=mean, scale=stdev)


def lognorm_dist_approx_from_median_lower_upper(
    median, lower, upper, quantile_ranks=(0.025, 0.975)
):
    """Returns a frozen lognormal distribution with the specified median, such that
    the values (lower, upper) are approximately equal to the quantiles with ranks
    (quantile_ranks[0], quantile_ranks[1]). More precisely, if q0 and q1 are
    the quantiles of the returned distribution with ranks quantile_ranks[0]
    and quantile_ranks[1], respectively, then q1/q0 = upper/lower. If the
    quantile ranks are symmetric about 0.5, lower and upper will coincide with
    q0 and q1 precisely when median^2 = lower*upper.
    """
    # Let Y ~ Norm(mu, sigma^2) and X = exp(Y), where mu = log(median)
    # so X ~ Lognorm(s=sigma, scale=exp(mu)) in scipy's notation.
    # We will determine sigma from the two specified quantiles lower and upper.

    # mean (and median) of the normal random variable Y = log(X)
    mu = np.log(median)
    # quantiles of the standard normal distribution corresponding to quantile_ranks
    stdnorm_quantiles = stats.norm.ppf(quantile_ranks)
    # quantiles of Y = log(X) corresponding to the quantiles (lower, upper) for X
    norm_quantiles = np.log([lower, upper])
    # standard deviation of Y = log(X) computed from the above quantiles for Y
    # and the corresponding standard normal quantiles
    sigma = (norm_quantiles[1] - norm_quantiles[0]) / (
        stdnorm_quantiles[1] - stdnorm_quantiles[0]
    )
    # Frozen lognormal distribution for X = exp(Y)
    # (s=sigma is the shape parameter; the scale parameter is exp(mu), which equals the median)
    return stats.lognorm(s=sigma, scale=median)


def lognorm_dist_approx_from_mean_lower_upper(
    mean, lower, upper, quantile_ranks=(0.025, 0.975)
):
    """Returns a frozen lognormal distribution with the specified mean, such that
    the values (lower, upper) are approximately equal to the quantiles with ranks
    (quantile_ranks[0], quantile_ranks[1])."""
    # Let Y ~ Norm(mu, sigma^2) and X = exp(Y), where mu = log(median)
    # so X ~ Lognorm(s=sigma, scale=exp(mu)) in scipy's notation.
    # Note that mean = exp(mu+sigma^2/2)

    # First we determine the unique sigma from the two specified quantiles lower and upper,
    # such that q1/q0 = upper/lower, where q0 and q1 are the quantiles of the specified ranks

    # quantiles of the standard normal distribution corresponding to quantile_ranks
    stdnorm_quantiles = stats.norm.ppf(quantile_ranks)
    # quantiles of Y = log(X) corresponding to the quantiles (lower, upper) for X
    norm_quantiles = np.log([lower, upper])
    # standard deviation of Y = log(X) computed from the above quantiles for Y
    # and the corresponding standard normal quantiles
    sigma = (norm_quantiles[1] - norm_quantiles[0]) / (
        stdnorm_quantiles[1] - stdnorm_quantiles[0]
    )
    # Solve for median = exp(mu) in terms of mean and sigma
    median = mean * np.exp(-(sigma**2) / 2)
    # Frozen lognormal distribution for X = exp(Y)
    # (s=sigma is the shape parameter; the scale parameter is exp(mu), which equals the median)
    return stats.lognorm(s=sigma, scale=median)
