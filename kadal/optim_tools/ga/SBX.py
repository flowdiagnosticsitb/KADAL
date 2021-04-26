import numpy as np
from numpy.random import random_sample


# Simulated Binary Crossover (SBX)
def SBX(Parent_1, Parent_2, nvar, lb, ub):
    n = 17
    u = random_sample()

    beta_1 = np.zeros(shape=[nvar])
    beta_2 = np.zeros(shape=[nvar])
    beta_u = np.zeros(shape=[nvar])
    P_1 = np.zeros(shape=[nvar])
    P_2 = np.zeros(shape=[nvar])
    offspring = np.zeros(shape=[2, nvar])

    for i in range(0, nvar):
        div = (abs(Parent_2[i] - Parent_1[i]) + 1e-5)
        beta_1[i] = (Parent_1[i] + Parent_2[i] - 2 * lb[i]) / div
        beta_u[i] = (2 * ub[i] - Parent_1[i] - Parent_2[i]) / div
        if beta_1[i] <= 1:
            P_1[i] = 0.5 * beta_1[i] ** (n + 1)
            beta_1[i] = (2 * u * P_1[i]) ** (1 / (n + 1))
        else:
            P_1[i] = 0.5 * (2 - (1 / (beta_1[i] ** (n + 1))))
            beta_1[i] = (1 / (2 - 2 * u * P_1[i])) ** (1 / (n + 1))

        if beta_u[i] <= 1:
            P_2[i] = 0.5 * beta_u[i] ** (n + 1)
            beta_2[i] = (2 * u * P_2[i]) ** (1 / (n + 1))
        else:
            P_2[i] = 0.5 * (2 - (1 / (beta_u[i] ** (n + 1))))
            beta_2[i] = (1 / (2 - 2 * u * P_2[i])) ** (1 / (n + 1))

        offspring[0, i] = 0.5 * ((Parent_1[i] + Parent_2[i]) - beta_1[i] * abs(Parent_2[i] - Parent_1[i]))
        offspring[1, i] = 0.5 * ((Parent_1[i] + Parent_2[i]) + beta_2[i] * abs(Parent_2[i] - Parent_1[i]))

        if offspring[0, i] < lb[i] or offspring[0, i] > ub[i]:
            offspring[0, i] = lb[i] + (ub[i] - lb[i]) * random_sample()

        if offspring[1, i] < lb[i] or offspring[1, i] > ub[i]:
            offspring[1, i] = lb[i] + (ub[i] - lb[i]) * random_sample()

    offspring = np.vstack((offspring[0, :], offspring[1, :]))

    return offspring


def sbx_vec(parent_1, parent_2, lb, ub, rand_seed=0):
    """Vectorised Simulated Binary Crossover (SBX)

    Based on SBX.SBX(). Looping structures have been vectorised for
    speed.

    Input parents can now be multiple samples at a time. Each row
    should contain the features for a single sample,

        i.e. parent.shape = [n_samp x n_dv]

    and the number of variables and samples are inferred from parent 1.
    N.B. No checks are made to ensure parent 2 is the same shape, or
    that upper and lower bounds are the correct length.

    The offspring are now returned in two separate arrays, with each
    row corresponding to the input rows in the parents.

    Args:
        parent_1 (np.ndarray): [n_samp x n_dv] parent 1 feature array.
            N.B. Must be 2D.
        parent_2 (np.ndarray): [n_samp x n_dv] parent 2 feature array.
            N.B. Must be 2D.
        lb (np.ndarray): n_dv-len lower bounds.
        ub (np.ndarray): n_dv-len upper bounds.
        rand_seed (int, optional): The seed for np.random.default_rng().
            Defaults to 0.

    Returns:
        offspring_1 (np.ndarray): [n_samp x n_dv] child 1 feature array.
        offspring_2 (np.ndarray): [n_samp x n_dv] child 2 feature array.
    """

    n = 17
    u = np.random.default_rng(rand_seed).random()

    n_samp, n_dv = parent_1.shape

    # beta_1 = np.zeros([n_samp, nvar])
    beta_2 = np.zeros([n_samp, n_dv])
    # beta_u = np.zeros([n_samp, nvar])
    p_1 = np.zeros([n_samp, n_dv])
    p_2 = np.zeros([n_samp, n_dv])
    # offspring_1 = np.zeros([n_samp, nvar])
    # offspring_2 = np.zeros([n_samp, nvar])

    parent_abs_diff = np.abs(parent_2 - parent_1)
    parent_sum = parent_1 + parent_2

    div = parent_abs_diff + 1e-5
    beta_1 = (parent_sum - 2 * lb) / div
    beta_u = (2 * ub - parent_1 - parent_2) / div

    b1_mask = beta_1 <= 1
    p_1[b1_mask] = 0.5 * beta_1[b1_mask] ** (n + 1)
    beta_1[b1_mask] = (2 * u * p_1[b1_mask]) ** (1 / (n + 1))
    p_1[~b1_mask] = 0.5 * (2 - (1 / (beta_1[~b1_mask] ** (n + 1))))
    beta_1[~b1_mask] = (1 / (2 - 2 * u * p_1[~b1_mask])) ** (1 / (n + 1))

    bu_mask = beta_u <= 1
    p_2[bu_mask] = 0.5 * beta_u[bu_mask] ** (n + 1)
    beta_2[bu_mask] = (2 * u * p_2[bu_mask]) ** (1 / (n + 1))
    p_2[~bu_mask] = 0.5 * (2 - (1 / (beta_u[~bu_mask] ** (n + 1))))
    beta_2[~bu_mask] = (1 / (2 - 2 * u * p_2[~bu_mask])) ** (1 / (n + 1))

    offspring_1 = 0.5 * (parent_sum - beta_1 * parent_abs_diff)
    offspring_2 = 0.5 * (parent_sum + beta_2 * parent_abs_diff)

    # Fix exceeded upper and lower bounds
    ub_lb = ub - lb

    off_1_bounds = np.where((offspring_1 < lb) | (offspring_1 > ub))
    if len(off_1_bounds[0]) != 0:
        offspring_1[off_1_bounds] = lb + ub_lb * rng.random(len(off_1_bounds))

    off_2_bounds = np.where((offspring_2 < lb) | (offspring_2 > ub))
    if len(off_2_bounds[0]) != 0:
        offspring_2[off_2_bounds] = lb + ub_lb * rng.random(len(off_2_bounds))

    return offspring_1, offspring_2
