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
    mean,
    variance,
    distribution,
    initial_parameters,
    loss = quadratic_loss,
    **kwargs,
):
    # args stores extra (positional) arguments to pass to distribution(),
    # as defined in documentation for `minimize`
    if 'args' not in kwargs:
        print('No args')
        kwargs['args'] = ()

    def objective_function(parameters, *args):
        dist = distribution(*parameters, *args)
        mv = dist.stats()
        return loss(mv, [mean,variance])

    result = minimize(objective_function, initial_parameters, **kwargs)
    best_params = result.x
    return distribution(*best_params, *kwargs['args']), result
