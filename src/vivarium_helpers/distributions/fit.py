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
