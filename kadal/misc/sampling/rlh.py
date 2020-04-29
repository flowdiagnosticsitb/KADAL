"""
This function is adopted from: "Forrester, A., Sobester, A., & Keane, A. (2008). Engineering design
via surrogate modelling: a practical guide. John Wiley & Sons."

Generates a random latin hypercube within the [0,1]^k hypercube.

Inputs:
   n - desired number of points
   k - number of design variables(dimensions)
   Edges = if Edges = 1 the extreme bins will have their centres on the
   edges of the domain, otherwise the bins will be entirely contained
   within the domain (default setting)

Output:
   X - Latin Hypercube sampling plan of n points in k dimensions.
"""
import numpy as np

def rlh(k,n,**kwargs):
    edges = kwargs.get('Edges', 0)
    if edges != 0 and edges != 1:
        raise ValueError("Edges only accept 0 or 1")

    #Pre-allocate memory
    X = np.zeros(shape=[n,k])

    for i in range(0,k):
        X[:,i] = np.random.permutation(n)

    if edges == 1:
        X = (X)/(n-1)
    else:
        X = (X+0.5)/n
    return X