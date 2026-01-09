import matplotlib.pyplot as plt
import numpy as np


def plot_rv_func(
    dist,
    funcname,
    lower=0.001,
    upper=0.999,
    numpoints=200,
    quantile_bounds=True,
    ax=None,
    **kwargs
):
    """Plot a built-in function of a scipy.stats.rv_continuous_frozen distribution dist.

    funcname can be one of: 'pdf', 'cdf', 'sf', 'logpdf', 'logcdf', 'logsf'.
    (lower, upper, numpoints) are the arguments (start, stop, num) of numpy.linspace.
    quantile_bounds is a boolean (default True) indicating whether lower and upper should be
    interpreted as quantile ranks (i.e. probabilities).
    If False, they are interpreted as values in the support of the distribution.
    ax is the matplotlib Axis object to plot on, defaults to matplotlib.pyplot.gca().
    kwargs is passed to ax.plot.
    """
    if ax is None:
        ax = plt.gca()

    # If lower and upper are interpreted as quantile ranks,
    # and f is NOT the quantile function (ppf) or inverse survival function,
    # find the correspondng quantiles for the x values.
    if quantile_bounds and funcname not in ["ppf", "isf"]:
        lower, upper = map(dist.ppf, [lower, upper])
    # If f is the quantile function (ppf) or inverse survival function,
    # and lower and upper are quantiles instead of quantile ranks,
    # compute the corresponding quantile ranks for our x values.
    elif funcname in ["ppf", "isf"] and not quantile_bounds:
        lower, upper = map(dist.cdf, [lower, upper])
    x = np.linspace(lower, upper, numpoints)

    # Set a few default keyword arguments for .plot() if not already set
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.8
    if all(key not in kwargs for key in ["linewidth", "lw"]):
        kwargs["linewidth"] = 2

    func = getattr(dist, funcname)
    ax.plot(x, func(x), **kwargs)
    return ax


def plot_pdf(
    dist, lower=0.001, upper=0.999, numpoints=200, quantile_bounds=True, ax=None, **kwargs
):
    return plot_rv_func(dist, "pdf", lower, upper, numpoints, quantile_bounds, ax, **kwargs)


def plot_cdf(
    dist, lower=0.001, upper=0.999, numpoints=200, quantile_bounds=True, ax=None, **kwargs
):
    return plot_rv_func(dist, "cdf", lower, upper, numpoints, quantile_bounds, ax, **kwargs)


def plot_sf(
    dist, lower=0.001, upper=0.999, numpoints=200, quantile_bounds=True, ax=None, **kwargs
):
    return plot_rv_func(dist, "sf", lower, upper, numpoints, quantile_bounds, ax, **kwargs)


def plot_logpdf(
    dist, lower=0.001, upper=0.999, numpoints=200, quantile_bounds=True, ax=None, **kwargs
):
    return plot_rv_func(
        dist, "logpdf", lower, upper, numpoints, quantile_bounds, ax, **kwargs
    )


def plot_logcdf(
    dist, lower=0.001, upper=0.999, numpoints=200, quantile_bounds=True, ax=None, **kwargs
):
    return plot_rv_func(
        dist, "logcdf", lower, upper, numpoints, quantile_bounds, ax, **kwargs
    )


def plot_logsf(
    dist, lower=0.001, upper=0.999, numpoints=200, quantile_bounds=True, ax=None, **kwargs
):
    return plot_rv_func(dist, "logsf", lower, upper, numpoints, quantile_bounds, ax, **kwargs)


def plot_ppf(
    dist, lower=0.001, upper=0.999, numpoints=200, quantile_bounds=True, ax=None, **kwargs
):
    return plot_rv_func(dist, "ppf", lower, upper, numpoints, quantile_bounds, ax, **kwargs)


def plot_isf(
    dist, lower=0.001, upper=0.999, numpoints=200, quantile_bounds=True, ax=None, **kwargs
):
    return plot_rv_func(dist, "isf", lower, upper, numpoints, quantile_bounds, ax, **kwargs)
