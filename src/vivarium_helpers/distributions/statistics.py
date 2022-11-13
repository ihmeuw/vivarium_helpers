def mean_and_variance(distribution):
    """Return the mean and variance of a scipy.stats distribution."""
    return distribution.stats('mv')

def get_stats(moments='mv'):
    """Return a subset of the first four standardized moments of
    a scipy.stats distribution:
    Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’)
    Note that the kurtosis returned by scipy is the _excess_ kurtosis,
    i.e. the standardized fourth moment minus 3.
    """
    def stats(distribution):
        return distribution.stats(moments)
    return stats

def get_statistic_and_interval_probability(statistic, lower, upper):
    """Return the specified statistic of the distribution and the probability
    that a random variable lies in the interval [lower, upper].

    statistic: str
        The name of a method on a scipy.stats rv_continuous object that
        returns a single real number. Namely, one of 'mean', 'median',
        'var', 'std', or 'entropy'.
    lower: float
        The lower bound of the interval.
    upper: float
        The upper bound of the interval.

    returns: tuple of length 2
        Returns the tuple (statistic, probability), where statistic is the
        requested statistic from the distribution, and probability is
        P(lower < X < upper), where X~distribution.
    """
    def statistic_and_interval_probability(distribution):
        statistic = getattr(distribution, statistic)()
        prob = distribution.cdf(upper) - distribution.cdf(lower)
        return statistic, prob
    return statistic_and_interval_probability

def get_mean_and_interval_probability(lower, upper):
    return get_statistic_and_interval_probability('mean', lower, upper)

def get_median_and_interval_probability(lower, upper):
    return get_statistic_and_interval_probability('median', lower, upper)

def get_statistic_and_interval(statistic, desired_probability=0.95):
    """Return the specified statistic of the distribution and the central
    confidence interval with the desired probability (confidence),
    i.e., the interval with total area `desired_probability` and equal
    areas around the median.

    statistic: str
        The name of a method on a scipy.stats rv_continuous object that
        returns a single real number. Namely, one of 'mean', 'median',
        'var', 'std', or 'entropy'.
    desired_probability: float in [0,1]
        The desired probability that a random variable X~distribution
        falls in the central interval.

    returns: tuple of length 3
        Returns the tuple (statistic, q1, q2), where statistic is the requested
        statistic of the distribution, and (q1, q2) are the quantiles of ranks
        p1 and p2, where p1 = (1 - desired_probability)/2, and
        p2 = desired_probability + p1.
    """
    def statistic_and_interval(distribution):
        statistic = getattr(distribution, statistic)()
        interval = distribution.interval(desired_prob)
        return statistic, *interval
    return statistic_and_interval
