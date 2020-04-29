import numpy as np
from math import erf

def gausspdf (x):
    res = (1/np.sqrt(2*np.pi))*np.exp(-(x**2)/2)
    return res

def gausscdf (x):
    x = 0.5*(1+erf(x/np.sqrt(2)))
    return x