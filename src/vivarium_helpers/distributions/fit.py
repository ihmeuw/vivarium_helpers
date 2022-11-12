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

def method_of_moments(
    moments,
    distribution,
    initial_parameters,
    loss = quadratic_loss,
    **kwargs,
):
    mean, variance = moments
    if isinstance(initial_parameters, dict):
        initial_parameters = [initial_parameters]
    print(initial_parameters)
    if isinstance(initial_parameters[-1], dict):
        *pos_params, kwd_params = initial_parameters
        num_positional = len(pos_params)
        # Save keys for keyword parameters to pass to distribution in objective_function.
        param_keys = kwd_params.keys()
        # Convert initial parameters into a list of values, to be compatible with minimize.
        initial_parameters = [*pos_params, *kwd_params.values()]
#     if isinstance(initial_parameters[-1], dict):
#         num_positional = len(initial_parameters)-1
#         # Save keys for keyword parameters to pass to distribution in objective_function.
#         param_keys = initial_parameters[-1].keys()
#         # Convert initial parameters into a list of values, to be compatible with minimize.
#         initial_parameters = [*initial_parameters[:-1], *initial_parameters[-1].values()]
#     if len(initial_parameters) == 2 and isinstance(initial_parameters[1], dict):
#         num_positional = len(initial_parameters[0])
#         # Save keys for keyword parameters to pass to distribution in objective_function.
#         param_keys = initial_parameters[1].keys()
#         # Convert initial parameters into a list of values, to be compatible with minimize.
#         initial_parameters = [*initial_parameters[0], *initial_parameters[1].values()]

    print(initial_parameters)
    def dist_from_parameters(parameters):
        pos_params = parameters[:num_positional]
        kwd_params = dict(zip(param_keys, parameters[num_positional:]))
        return distribution(*pos_params, **kwd_params)

    def objective_function(parameters):
        dist = dist_from_parameters(parameters)
        mv = dist.stats()
        return loss(mv, [mean,variance])

    result = minimize(objective_function, initial_parameters, **kwargs)
    best_params = result.x
    return distribution(*best_params), result
