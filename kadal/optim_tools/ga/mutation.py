import numpy as np
from numpy.random import random_sample,randn
from copy import deepcopy

#Gaussian mutation ==> causing fitness value to be around maximum value
def gaussmut (variable,nvar,pmut,ub,lb):
    sdev = np.divide((ub-lb),4)
    mutChrom = variable
    for ii in range (0,nvar):
        if random_sample() < pmut:
            mutChrom[ii] = mutChrom[ii] + sdev[ii]*randn()
            if (mutChrom[ii]>ub[ii]) or (mutChrom[ii]<lb[ii]):
                mutChrom[ii] = lb[ii]+(ub[ii]-lb[ii])*random_sample()

    return mutChrom
