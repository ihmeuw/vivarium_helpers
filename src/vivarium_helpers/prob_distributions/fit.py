import numpy as np
from scipy import stats
from scipy.optimize import minimize

from vivarium_helpers.prob_distributions import descriptive_stats


def l1_loss(x, y):
    x, y = map(np.asarray, [x, y])
    return np.abs(x - y).sum()


def l2_loss(x, y):
    x, y = map(np.asarray, [x, y])
    return ((x - y) ** 2).sum()


quadratic_loss = l2_loss


def weighted_l2_loss(x, y, weights):
    x, y, weights = map(np.asarray, [x, y, weights])
    return (weights * ((x - y) ** 2)).sum()


def l2_relative_error_loss(measured_val, true_val):
    """Compute the L2 norm of the relative errors between measured_val and
    true_val, where true_val is used for normalization in the denominator.
    true_val is replaced by the value max(|true_val|, 1e-8) before normalization
    in order to avoid division by 0.

    Idea taken from:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit

    My hypothesis (untested so far) is that this loss function may perform
    better than the un-normalized l2 loss in cases where the components of true_val are
    of drastically different magnitudes.
    """
    measured_val, true_val = map(np.asarray, [measured_val, true_val])
    return (((measured_val - true_val) / np.maximum(np.abs(true_val), 1e-8)) ** 2).sum()


# Code from Google AI
def log_loss(y_true, y_pred, epsilon=1e-15):
    """
    Calculates the log loss (binary cross-entropy) between true and predicted values.

    Args:
        y_true (array-like): True labels (0 or 1).
        y_pred (array-like): Predicted probabilities (values between 0 and 1).
        epsilon (float, optional): A small value to prevent log(0) errors. Defaults to 1e-15.

    Returns:
        float: The calculated log loss.
    """
    # Note: If one of y_true, y_pred is a pandas Series, NaN's will be
    # ignored, which is wrong, so we convert to arrays to fix this
    y_true, y_pred = map(np.asarray, (y_true, y_pred))
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def arglist(*args, **kwargs):
    return (*args, kwargs)


def _parse_arglist(arg_list):
    # Conform arg_list to the form (*args, kwargs) if
    # only a dict or only a sequence of values was passed
    if isinstance(arg_list, dict):
        arg_list = (arg_list,)
    elif len(arg_list) == 0 or not isinstance(arg_list[-1], dict):
        arg_list = (*arg_list, {})
    *args, kwargs = arg_list
    return args, kwargs


def fit(
    dist_family,
    data,
    descriptive_stats_func,
    initial_parameters,
    loss=quadratic_loss,
    **kwargs,
):
    # args stores extra fixed arguments to pass to distribution(),
    # as defined in documentation for `minimize`, except that
    # we use "arglists" to allow passing keyword arguments as well as
    # positional arguments
    fixed_arglist = kwargs.pop("args", ())
    fixed_args, fixed_kwargs = _parse_arglist(fixed_arglist)

    # # Extract positional and keyword parameters from parameter list
    pos_params, kwd_params = _parse_arglist(initial_parameters)
    print(fixed_kwargs, kwd_params)

    # Record # of positional arguments for convenience in objective_function
    num_positional = len(pos_params)
    # Save keys for keyword parameters to pass to distribution in objective_function.
    param_keys = kwd_params.keys()
    # Convert initial parameters into an array of values, to be compatible with minimize.
    initial_parameters = [*pos_params, *kwd_params.values()]

    print(initial_parameters)

    def dist_from_parameters(parameters):
        pos_params = parameters[:num_positional]
        kwd_params = dict(zip(param_keys, parameters[num_positional:]))
        return dist_family(*pos_params, *fixed_args, **kwd_params, **fixed_kwargs)

    def objective_function(parameters):
        dist = dist_from_parameters(parameters)
        computed_statistics = descriptive_stats_func(dist)
        return loss(computed_statistics, data)

    result = minimize(objective_function, initial_parameters, **kwargs)
    best_params = result.x
    return dist_from_parameters(best_params), result


def method_of_moments(
    dist_family,
    moments,
    initial_parameters,
    fisher=True,  # True: use Fisher moments from stats() False: use raw moments from moments().
    loss=l2_relative_error_loss,
    **kwargs,
):
    if isinstance(moments, dict):
        orders = moments.keys()
        moments = moments.values()
    else:
        orders = range(1, 1 + len(moments))

    if fisher:
        if len(moments) > 4:
            raise ValueError(f"More than 4 non-raw moments passed: {moments=}")
        order_to_abrv = dict(enumerate("mvsk", start=1))

        def convert_to_abrv(o):
            return order_to_abrv[o] if o in order_to_abrv else o

        orders = "".join(map(convert_to_abrv, orders))
        moment_func = descriptive_stats.fisher_moments(orders)
    else:
        moment_func = descriptive_stats.raw_moments(*orders)

    return fit(
        dist_family,
        moments,
        moment_func,
        initial_parameters,
        loss=loss,
        **kwargs,
    )
