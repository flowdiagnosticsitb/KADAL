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