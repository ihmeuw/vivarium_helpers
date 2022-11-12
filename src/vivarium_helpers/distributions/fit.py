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

def arglist(*args, **kwargs):
    return [*args, kwargs]

def _parse_arglist(arg_list):
    # Conform arg_list to the form [*args, kwargs]
    if isinstance(arg_list, dict):
        arg_list = [arg_list]
    if len(arg_list) == 0 or not isinstance(arg_list[-1], dict):
        arg_list = [*arg_list, {}]
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
    # args stores extra (positional) arguments to pass to distribution(),
    # as defined in documentation for `minimize`
    if 'args' not in kwargs:
        print('No args')
        kwargs['args'] = ()

    fixed_args, fixed_kwargs = _parse_arglist(kwargs['args'])

    # Ensure that initial_parameters has the form [*args, kwargs]
    # if isinstance(initial_parameters, dict):
    #     initial_parameters = [initial_parameters]
    # print(initial_parameters)
    # if not isinstance(initial_parameters[-1], dict):
    #     initial_parameters = [*initial_parameters, {}]
    #
    # # Extract positional and keyword parameters from parameter list
    # *pos_params, kwd_params = initial_parameters
    pos_params, kwd_params = _parse_arglist(initial_parameters)
    print(fixed_kwargs, kwd_params)
    if not fixed_kwargs.keys().isdisjoint(kwd_params.keys()):
        raise ValueError(
            "Optimization keyword parameters overlap with fixed"
            f" keyword parameters: {fixed_kwargs.keys() & kwd_params.keys()}")
    # Record # of positional arguments for convenience in objective_function
    num_positional = len(pos_params)
    # Save keys for keyword parameters to pass to distribution in objective_function.
    param_keys = kwd_params.keys()
    # Convert initial parameters into a list of values, to be compatible with minimize.
    initial_parameters = [*pos_params, *kwd_params.values()]

    print(initial_parameters)
    def dist_from_parameters(parameters, *args):
        pos_params = parameters[:num_positional]
        kwd_params = dict(zip(param_keys, parameters[num_positional:]))
        assert (fixed_args, fixed_kwargs) == _parse_arglist(args), \
            f"{fixed_args=}, {fixed_kwargs=}, {args=}"
        kwd_args = {**kwd_params, **fixed_kwargs}
        return distribution(*pos_params, *fixed_args, **kwd_args)

    def objective_function(parameters, *args):
        dist = dist_from_parameters(parameters, *args)
        mv = dist.stats()
        return loss(mv, [mean,variance])

    result = minimize(objective_function, initial_parameters, **kwargs)
    best_params = result.x
    return dist_from_parameters(best_params, *kwargs['args']), result
