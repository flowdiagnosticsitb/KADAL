import numpy as np
from numpy.random import random_sample, randn
from copy import deepcopy


# Gaussian mutation ==> causing fitness value to be around maximum value
def gaussmut(variable, nvar, pmut, ub, lb):
    sdev = np.divide((ub - lb), 4)
    mutChrom = variable
    for ii in range(0, nvar):
        if random_sample() < pmut:
            mutChrom[ii] = mutChrom[ii] + sdev[ii] * randn()
            if (mutChrom[ii] > ub[ii]) or (mutChrom[ii] < lb[ii]):
                mutChrom[ii] = lb[ii] + (ub[ii] - lb[ii]) * random_sample()

    return mutChrom


def gaussmut_vec(feature, p_mut, ub, lb, rand_seed=0):
    """Vectorised Gaussian Mutation

    Based on mutation.gaussmut(). Looping structures have been
    vectorised for speed.

    Input feature can now be multiple samples at a time. Each row
    should contain the features for a single sample,

        i.e. feature.shape = [n_samp x n_dv]

    and the number of variables and samples are inferred from the array.
    N.B. No checks are made to ensure parent 2 is the same shape, or
    that upper and lower bounds are the correct length.

    The offspring are now returned in two separate arrays, with each
    row corresponding to the input rows in the parents.

    Args:
        feature (np.ndarray): [n_samp x n_dv] feature array. N.B. Must
            be 2D.
        p_mut (numeric): Probability of mutation.
        lb (np.ndarray): n_dv-len lower bounds.
        ub (np.ndarray): n_dv-len upper bounds.
        rand_seed (int, optional): The seed for np.random.default_rng().
            Defaults to 0.
    Returns:
        mut_feature (np.ndarray): [n_samp x n_dv] mutated feature array.
    """
    rng = np.random.default_rng(rand_seed)

    n_samp, n_dv = feature.shape
    mut_feature = deepcopy(feature)

    ub_lb = ub - lb
    sdev_i = (ub_lb) / 4  # sigma for a single feature (row)
    sdev = np.tile(sdev_i, (n_samp, 1))  # tile for as many rows as required.

    # Do mutation on features random features
    is_mut = rng.random(size=[n_samp, n_dv]) < p_mut  # mask for mutations
    mut_chrom = (feature[is_mut]
                 + sdev[is_mut] * rng.standard_normal(size=len(sdev[is_mut])))

    # Fix any exceeded upper or lower bounds
    # To correctly index the relevant dv bound, need to tile and mask
    lbs = np.tile(lb, (n_samp, 1))[is_mut]
    ubs = np.tile(ub, (n_samp, 1))[is_mut]
    ubs_lbs = np.tile(ub_lb, (n_samp, 1))[is_mut]

    # Get indices of exceeded bounds into a mask and randomise those valuess
    b_mask = np.where((mut_chrom < lbs) | (mut_chrom > ubs))
    mut_chrom[b_mask] = lbs[b_mask] + ubs_lbs[b_mask] * rng.random(len(b_mask[0]))

    # Copy mutated features into main array
    mut_feature[is_mut] = mut_chrom

    return mut_feature
