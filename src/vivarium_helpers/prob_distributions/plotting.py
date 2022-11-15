import numpy as np
import matplotlib.pyplot as plt

def plot_pdf(dist, lower=0.001, upper=0.999, numpoints=200, quantile_bounds=True, ax=None, **kwargs):
    """Plot the pdf of a scipy.stats distribution dist.

    (lower, upper, numpoints) are the arguments (start, stop, num) of numpy.linspace.
    quantile_bounds is a boolean (default True) indicating whether lower and upper should be
    interpreted as quantiles. If False, they are interpreted as x values in the
    support of the distribution.
    ax is the matplotlib Axis object to plot on, defaults to matplotlib.pyplot.gca().
    kwargs is passed to matplotlib.pyplot.plot.
    """
    if ax is None:
        ax = plt.gca()

    # If lower and upper are interpreted as quantiles, find the correspondng x values
    if quantile_bounds:
        lower, upper = map(dist.ppf, [lower, upper])
    x = np.linspace(lower, upper, numpoints)

    # Set a few default keyword arguments for .plot() if not already set
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.8
    if all(key not in kwargs for key in ['linewidth', 'lw']):
        kwargs['linewidth'] = 2

    ax.plot(x, dist.pdf(x), **kwargs)
    return ax
