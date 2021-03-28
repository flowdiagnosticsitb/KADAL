import numpy as np
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

def ehvicalc_vec(x, y_par, moboInfo, kriglist):
    """Vectorised EHVI Function

    Vectorises above EHVI function as much as possible. Currently,
    still depends on the vectorsation of prediction.py prediction() and
    kadal.optim_tools.ehvi.exi2d.exi2d() for best performance boost.

    Args:
        x (np.ndarray): [n_pop, n_dv] Design variables for a population.
        y_par (np.ndarray): [n_par, n_obj] Current Pareto front.
        moboInfo (dict): Structure containing necessary information for
            multiobjective Bayesian optimization.
        kriglist ([]): n_obj-len list of objective Kriging instances.

    Returns:
        hv (np.ndarray): n_pop-len array of hypervolumes for each samp.
    """

    n_pop = x.shape[0]
    X = kriglist[0].KrigInfo["X"]
    n_obj = len(kriglist)
    ref_point = moboInfo["refpoint"]

    # prediction of each objective
    pred = np.zeros([n_pop, n_obj])
    SSqr = np.zeros([n_pop, n_obj])

    # Currently, looks like prediction.py predction is only set up for 1D arrays...
    for i in range(n_pop):
        for j in range(n_obj):
            pred[i, j], SSqr[i, j] = kriglist[j].predict(x[i, :], ["pred", "SSqr"])

    # Compute (negative of) hypervolume
    # Also need to investigate how to vectorise exi2d
    hv = np.zeros(n_pop)
    for i in range(n_pop):
        hv[i] = -1 * exi2d(y_par, ref_point, pred[i, :], SSqr[i, :])

    # give penalty to HV, to avoid error in CMA-ES when in an iteration produce all HV = 0
    z = hv == 0  # mask of values very near 0
    rng = np.random.default_rng()
    hv[z] = rng.uniform(np.finfo("float").tiny, np.finfo("float").tiny * 100,
                        size=np.count_nonzero(z))
    return hv