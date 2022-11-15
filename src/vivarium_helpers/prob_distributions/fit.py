import numpy as np
from scipy import stats
from scipy.optimize import minimize

def l1_loss(x,y):
    x,y = map(np.asarray, [x,y])
    return np.abs(x-y).sum()

def l2_loss(x,y):
    x,y = map(np.asarray, [x,y])
    return ((x-y)**2).sum()

quadratic_loss = l2_loss

def weighted_l2_loss(x,y,weights):
    x,y,weights = map(np.asarray, [x,y,weights])
    return ((weights*(x-y))**2).sum()

def l2_relative_error_loss(x,y):
    """Compute the L2 norm of the relative errors between x and y,
    where x is the "measured" value and y is the "true" value
    appearing in the denominator for normalization.
    y is replaced by the value max(|y|, 1e-8) before normalization
    in order to avoid division by 0.
    Idea taken from:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit
    """
    x,y = map(np.asarray, [x,y])
    return (((x - y) / np.maximum(np.abs(y), 1e-8))**2).sum()

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

def method_of_moments(
    moments,
    distribution,
    initial_parameters,
    loss = quadratic_loss,
    **kwargs,
):
    mean, variance = moments
    # args stores extra fixed arguments to pass to distribution(),
    # as defined in documentation for `minimize`, except that
    # we use "arglists" to allow passing keyword arguments as well as
    # positional arguments
    fixed_arglist = kwargs.pop('args', ())
    fixed_args, fixed_kwargs = _parse_arglist(fixed_arglist)

    # # Extract positional and keyword parameters from parameter list
    # *pos_params, kwd_params = initial_parameters
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
        return distribution(
            *pos_params, *fixed_args, **kwd_params, **fixed_kwargs
        )

    def objective_function(parameters):
        dist = dist_from_parameters(parameters)
        mean_var = dist.stats()
        return loss(mean_var, [mean,variance])

    result = minimize(objective_function, initial_parameters, **kwargs)
    best_params = result.x
    return dist_from_parameters(best_params), result
