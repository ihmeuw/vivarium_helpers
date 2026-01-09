"""
Module defining functions that compute various descriptive statistics of
probability distributions represented by scipy.stats.rv_continuous_frozen
objects.
"""


def get_mean_and_variance(distribution):
    """Return the mean and variance of a scipy.stats distribution."""
    return distribution.stats("mv")


def get_mean_and_std(distribution):
    """Return the mean and standard deviation of a scipy.stats distribution."""
    return distribution.mean(), distribution.std()


def get_mean_and_median(distribution):
    """Return the mean and median of a scipy.stats distribution."""
    return distribution.mean(), distribution.median()


def get_support(distribution):
    """Return the support of a scipy.stats distribution."""
    return distribution.support()


def fisher_moments(moments="mv"):
    """Returns a function that takes as input a frozen scipy.stats.rv_continuous
    distribution and returns a subset of the first four commonly used moments
    computed by the scipy.stats.rv_continuous.stats() function.
    mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).
    Skew and kurtosis use Fisher's definitions as indicated in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.stats.html
    The moments are defined as follows:
    - The mean is the first raw moment.
    - The variance is the second central moment.
    - The skew is the third standardized moment.
    - The kurtosis returned by scipy is the excess kurtosis,
      i.e. the normalized fourth cumulant, which is equal to the
      fourth standardized moment minus 3.
    The moments parameter defaults to 'mv', the same as the stats() function.
    """

    def get_special_moments(distribution):
        return distribution.stats(moments)

    return get_special_moments


def raw_moments(*orders):
    """Returns a function that returns a distribution's raw moments
    of the specified orders.
    """
    print(orders)

    def get_raw_moments(distribution):
        moments = [distribution.moment(order) for order in orders]
        return moments

    return get_raw_moments


def statistic(statistic_name):
    """Returns a function of a scipy.stats.rv_continuous_frozen object that
    returns the specified descriptive statistic of the probability distribution
    it represents.

    statistic_name: str

        A descrpitive statistic (a single real number) that is either the name
        of a method of scipy.stats.rv_continuous_frozen that returns the
        statistic, or else can be easily computed from one of the standard
        methods. One of 'mean', 'median', 'var' or 'variance', 'std', 'skew',
        'kurtosis', 'entropy', 'support_min', or 'support_max'.
    """
    # Return tuples to make the return type compatible with outputs
    # returned by other functions in this module
    if statistic_name in ["variance", "skew", "kurtosis"]:
        get_statistic = lambda dist: (dist.stats(statistic_name[0]),)
    elif statistic_name == "support_min":
        get_statistic = lambda dist: (dist.support()[0],)
    elif statistic_name == "support_max":
        get_statistic = lambda dist: (dist.support()[1],)
    else:
        get_statistic = lambda dist: (getattr(dist, statistic_name)(),)
    # def distribution_statistic(distribution):
    #     if statistic in ['skew', 'kurtosis']:
    #         statistic = distribution.stats(statistic[0])
    #     else:
    #         statistic = getattr(distribution, statistic)()
    #     return statistic
    get_statistic.__name__ = f"get_{statistic_name}"
    return get_statistic


# Original, old version -- keep for now, until different versions are tested
def statistic_and_interval_probability1(statistic_name, lower, upper):
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


# Original, old version -- keep for now, until different versions are tested
def statistic_and_central_interval1(statistic_name, desired_probability):
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


# Original, old version -- keep for now, until different versions are tested
def statistic_and_quantiles1(statistic_name, *quantile_ranks):
    get_statistic = distribution_statistic(statistic_name)

    def get_statistic_and_quantiles(distribution):
        statistic = get_statistic(distribution)
        quantiles = distribution.ppf(quantile_ranks)
        return statistic, *quantiles

    return get_statistic_and_quantiles


# Original, old version -- keep for now, until different versions are tested
def statistic_and_quantile_ranks1(statistic_name, *quantiles):
    get_statistic = distribution_statistic(statistic_name)

    def get_statistic_and_quantile_ranks(distribution):
        statistic = get_statistic(distribution)
        quantile_ranks = distribution.cdf(quantiles)
        return statistic, *quantile_ranks

    return get_statistic_and_quantile_ranks


def central_interval(confidence):
    def get_central_interval(distribution):
        return distribution.interval(confidence)

    return get_central_interval


def interval_probability(lower, upper):
    def get_interval_probability(distribution):
        prob = distribution.cdf(upper) - distribution.cdf(lower)
        return prob

    return get_interval_probability


def concatenate(*functionals):
    def get_values(distribution):
        list_of_iterables = [functional(distribution) for functional in functionals]
        return [value for values in list_of_iterables for value in values]

    return get_values


def statistic_and_interval_probability(statistic_name, lower, upper):
    return concatenate(
        statistic(statistic_name),
        interval_probability(lower, upper),
    )


def mean_and_interval_probability(lower, upper):
    return statistic_and_interval_probability("mean", lower, upper)


def median_and_interval_probability(lower, upper):
    return statistic_and_interval_probability("median", lower, upper)


def statistic_and_central_interval(statistic_name, probability):
    return concatenate(statistic(statistic_name), central_interval(probability))


def statistic_and_quantiles(statistic_name, *quantile_ranks):
    return concatenate(statistic(statistic_name), quantiles(*quantile_ranks))


def statistic_and_quantile_ranks(statistic_name, *quantiles):
    return concatenate(statistic(statistic_name), quantile_ranks(*quantiles))
