import numpy as np
import multiprocessing as mp
from kadal.optim_tools.ehvi.exi2d import exi2d


def ehvicalc(x,ypar,moboInfo,kriglist):
    """
    Wrapper for EHVI calculation.

    Args:
        x (nparray): Design Variables
        ypar (nparray): Current Pareto Front
        moboInfo (dict): Structure(Dictionary) containing necessary information for multiobjective Bayesian optimization.
        kriglist (list): List containing Kriging instances.

    Returns:
        HV (float): EHVI value
    """
    HV = EHVI(x, ypar, moboInfo, kriglist)

    return HV


def EHVI(x,ypar,moboInfo,kriglist):
    """
        ModelInfoKR{i} = Model Information of objective i
        ObjectiveInfoKR{i} = Objective Information of objective i

        Input :
            - x : Design variables
            - ypar: Current Pareto front
            - BayesMultiInfo: Structure(Dictionary) containing necessary information for multiobjective Bayesian optimization.
            - kriglist (list): List containing Kriging instances.
        """

    X = kriglist[0].KrigInfo["X"]
    nobj = len(kriglist)
    nsamp = np.size(X, 0)
    YO = np.zeros(shape=[nsamp, nobj])
    RefP = moboInfo["refpoint"]

    # prediction of each objective
    pred = np.zeros(shape=[nobj])
    SSqr = np.zeros(shape=[nobj])
    for ii in range(0, nobj):
        pred[ii], SSqr[ii] = kriglist[ii].predict(x, ["pred", "SSqr"])

    # Compute (negative of) hypervolume
    HV = -1 * exi2d(ypar, RefP, pred, SSqr)

    if HV == 0:  # give penalty to HV, to avoid error in CMA-ES when in an iteration produce all HV = 0
        HV = np.random.uniform(np.finfo("float").tiny, np.finfo("float").tiny * 100)

    return HV


def ehvicalc_vec(x, y_par, moboInfo, kriglist, pool=None):
    """Vectorised EHVI Function

    Vectorises above EHVI function as much as possible. Currently,
    still depends on the vectorisation of prediction.py prediction() and
    kadal.optim_tools.ehvi.exi2d.exi2d() for best performance boost.

    Args:
        x (np.ndarray): [n_pop, n_dv] Design variables for a population.
        y_par (np.ndarray): [n_par, n_obj] Current Pareto front.
        moboInfo (dict): Structure containing necessary information for
            multiobjective Bayesian optimization.
        kriglist ([]): n_obj-len list of objective Kriging instances.

    Returns:
        hv (np.ndarray/float): n_pop-len array of hypervolumes for each
            samp, if input x is 2D. If input x is a 1D input array,
            n_pop = 1 is assumed and all inputs are design variables;
            a single hv float is returned (legacy behaviour).
    """
    reshape = False
    if x.ndim == 1:
        reshape = True
        x = x.reshape(1, -1)

    n_pop = x.shape[0]
    # X = kriglist[0].KrigInfo["X"]
    n_obj = len(kriglist)
    ref_point = moboInfo["refpoint"]

    pred = np.zeros([n_pop, n_obj])
    SSqr = np.zeros([n_pop, n_obj])
    hv = np.zeros(n_pop)

    if moboInfo.get('n_cpu', 1) == 1 or pool is not None:
        # Prediction of each objective
        # Looks like prediction.py prediction is only set up for 1D arrays...
        for i in range(n_pop):
            for j in range(n_obj):
                pred[i, j], SSqr[i, j] = kriglist[j].predict(x[i, :],
                                                             ["pred", "SSqr"])

        # Compute (negative of) hypervolume
        # exi2d not easy to vectorise - just pass to mp.Pool.starmap
        for i in range(n_pop):
            hv[i] = -1 * exi2d(y_par, ref_point, pred[i, :], SSqr[i, :])

    else:
        def pool_hv():
            # Set up predict loop args
            p_args = [(x[i, :], ["pred", "SSqr"]) for i in range(n_pop)]
            # Run for all population for each Kriging
            for j in range(n_obj):
                res_p = pool.starmap(kriglist[j].predict, p_args)
                res_p = np.array(res_p).ravel().reshape(n_pop, 2)
                pred[:, j] = res_p[:, 0].copy()
                SSqr[:, j] = res_p[:, 1].copy()

            # Set up exi2d loop args
            hv_args = ((y_par, ref_point, pred[i, :], SSqr[i, :]) for i in range(n_pop))
            nonlocal hv
            hv[:] = pool.starmap(exi2d, hv_args)
            hv *= -1

        # Run each serial loop evaluation in parallel with mp.Pool
        # If a pool is provided, use it, else create a new one
        if pool is None:
            pool = mp.Pool(processes=moboInfo['n_cpu'])
            exit_pool = True
        else:
            exit_pool = False

        try:
            pool_hv()
        finally:
            # Close pool if created here
            if exit_pool:
                pool.close()
                pool.join()

    # give penalty to HV, to avoid error in CMA-ES when in an iteration produce all HV = 0
    z = hv == 0  # mask of values very near 0
    rng = np.random.default_rng()
    hv[z] = rng.uniform(np.finfo("float").tiny, np.finfo("float").tiny * 100,
                        size=np.count_nonzero(z))

    # If 1D input, expects float output (legacy behaviour)
    if reshape:
        hv = hv[0]

    return hv
