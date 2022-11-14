def get_mean_and_variance(distribution):
    """Return the mean and variance of a scipy.stats distribution."""
    return distribution.stats('mv')

def get_mean_and_std(distribution):
    """Return the mean and standard deviation of a scipy.stats distribution."""
    return distribution.mean(), distribution.std()

def get_mean_and_median(distribution):
    """Return the mean and median of a scipy.stats distribution."""
    return distribution.mean(), distribution.median()

def special_moments(moments='mv'):
    """Returns a function that takes as input a frozen scipy.stats.rv_continuous
    distribution and returns a subset of the first four commonly used moments:
    mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
    - The mean is the first raw moment.
    - The variance is the second central moment.
    - The skew is the third standardized moment.
    - The kurtosis returned by scipy is the excess kurtosis,
      i.e. the normalized fourth cumulant, which is equal to the
      fourth standardized moment minus 3.
    """
    def get_special_moments(distribution):
        return distribution.stats(moments)
    return stats

def statistic(statistic_name):
    """Returns a function of a frozen `scipy.stats.rv_continuous` object
    that returns the specified statistic of the distribution.

    statistic: str
        The name of a method on a scipy.stats rv_continuous object that
        returns a single real number. Namely, one of 'mean', 'median',
        'var', 'std', 'skew', 'kurtosis', or 'entropy'.
    """
    # Return tuples to make the return type compatible with outputs
    # returned by other functions in this module
    if statistic_name in ['skew', 'kurtosis']:
        get_statistic = lambda dist: (dist.stats(statistic_name[0]),)
    else:
        get_statistic = lambda dist: (getattr(dist, statistic_name)(),)
    # def distribution_statistic(distribution):
    #     if statistic in ['skew', 'kurtosis']:
    #         statistic = distribution.stats(statistic[0])
    #     else:
    #         statistic = getattr(distribution, statistic)()
    #     return statistic
    return get_statistic


def statistic_and_interval_probability(statistic_name, lower, upper):
    """Returns a function `statistic_and_interval_probability` that takes
    a frozen `scipy.stats.rv_continuous` object representing a probability
    distribution as input, and returns a length-2 tuple containing the
    specified statistic of the distribution and the probability that a
    random variable from the distribution lies in the interval [lower, upper].

    statistic: str
        The name of a method on a scipy.stats rv_continuous object that
        returns a single real number. Namely, one of 'mean', 'median',
        'var', 'std', 'skew', 'kurtosis', or 'entropy'.
    lower: float
        The lower bound of the interval.
    upper: float
        The upper bound of the interval.

    returns: function
        The returned function takes a
    returns: tuple of length 2
        Returns the tuple (statistic, probability), where statistic is the
        requested statistic from the distribution, and probability is
        P(lower < X < upper), where X~distribution.
    """
    get_statistic = statistic(statistic_name)
    def get_statistic_and_interval_probability(distribution):
        stat = get_statistic(distribution)
        prob = distribution.cdf(upper) - distribution.cdf(lower)
        return stat, prob
    return get_statistic_and_interval_probability

def mean_and_interval_probability(lower, upper):
    return statistic_and_interval_probability('mean', lower, upper)

def median_and_interval_probability(lower, upper):
    return statistic_and_interval_probability('median', lower, upper)

def statistic_and_central_interval(statistic_name, desired_probability=0.95):
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
        p1 and p2, where p1 = 0.5 - desired_probability/2, and
        p2 = 0.5 + desired_probability/2.
    """
    get_statistic = statistic(statistic_name)
    def get_statistic_and_central_interval(distribution):
        stat = get_statistic(distribution)
        interval = distribution.interval(desired_probability)
        return stat, *interval
    return get_statistic_and_central_interval

def quantiles(*quantile_ranks):
    def get_quantiles(distribution):
        return distribution.ppf(quantile_ranks)
    return get_quantiles

def quantile_ranks(*quantiles):
    def get_quantile_ranks(distribution):
        return distribution.cdf(quantiles)
    return get_quantile_ranks

def statistic_and_quantiles(statistic_name, *quantile_ranks):
    get_statistic = distribution_statistic(statistic_name)
    def get_statistic_and_quantiles(distribution):
        statistic = get_statistic(distribution)
        quantiles = distribution.ppf(quantile_ranks)
        return statistic, *quantiles
    return get_statistic_and_quantiles

def statistic_and_quantile_ranks(statistic_name, *quantiles):
    get_statistic = distribution_statistic(statistic_name)
    def get_statistic_and_quantile_ranks(distribution):
        statistic = get_statistic(distribution)
        quantile_ranks = distribution.cdf(quantiles)
        return statistic, *quantile_ranks
    return get_statistic_and_quantile_ranks

def moments(*orders):
    def get_moments(distribution):
        moments = [distribution.moment(order) for order in orders]
        return moments
    return get_moments

def central_interval(confidence):
    def get_central_interval(distribution):
        return distribution.interval(confidence)
    return get_central_interval

def interval_probability(lower, upper):
    def get_interval_probability(distribution):
        prob = distribution.cdf(upper) - distribution.cdf(lower)
        return prob
    return get_interval_probability

def combine(*functionals):
    def get_values(distribution):
        list_of_iterables = [functional(distribution) for functional in functionals]
        return [value for values in list_of_iterables for value in values]
    return get_values
