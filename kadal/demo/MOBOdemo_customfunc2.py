import sys
sys.path.insert(0, "..")
import numpy as np
from kadal.surrogate_models.kriging_model import Kriging
from kadal.surrogate_models.supports.initinfo import initkriginfo
from kadal.misc.sampling.samplingplan import sampling
from matplotlib import pyplot as plt
from kadal.optim_tools.MOBO import MOBO
from copy import deepcopy

def cust_func(x):
    x1 = x[:,0]
    x2 = x[:,1]
    f = np.zeros(np.shape(x))
    f[:,0] = 4*x1**2 + 4*x2**2
    f[:,1] = (x1-5)**2 + (x2-5)**2
    return f

def cheap_const(x):
    """This function should return either 1 or 0, 1 for feasible 0 for infeasible"""
    if np.ndim(x) == 1:
        x = np.array([x])
    x1 = x[:,0]
    x2 = x[:,1]
    g = (x1-5)**2 + x2**2 <= 25
    return g

def exp_const_eval(x):
    """This function should return the evaluation value of the constraint"""
    if np.ndim(x) == 1:
        x = np.array([x])
    x1 = x[:,0]
    x2 = x[:,1]
    g = (x1-8)**2 + (x2 + 3)**2
    return g

def construct_krig(X, y, g, lb, ub):
    # Define input for constraint Kriging
    KrigConstInfo = initkriginfo()
    KrigConstInfo['X'] = X
    KrigConstInfo['y'] = g.reshape(-1,1) # should be in shape (n,1)
    KrigConstInfo['problem'] = exp_const_eval
    KrigConstInfo["nrestart"] = 5
    KrigConstInfo["ub"] = ub
    KrigConstInfo["lb"] = lb
    KrigConstInfo["optimizer"] = "lbfgsb"
    KrigConstInfo['limittype'] = '>='  # value of the expensive constraints should be more than equal 7.7
    KrigConstInfo['limit'] = 7.7

    # Define input for first objective Kriging
    KrigInfo1 = initkriginfo()
    KrigInfo1["X"] = X
    KrigInfo1["y"] = y[:,0].reshape(-1,1)
    KrigInfo1["problem"] = cust_func
    KrigInfo1["nrestart"] = 5
    KrigInfo1["ub"] = ub
    KrigInfo1["lb"] = lb
    KrigInfo1["optimizer"] = "lbfgsb"

    # Define input for second objective Kriging
    KrigInfo2 = deepcopy(KrigInfo1)
    KrigInfo2['y'] = y[:,1].reshape(-1,1)

    # Run Kriging
    krigobj1 = Kriging(KrigInfo1, standardization=True, standtype='default', normy=False, trainvar=False)
    krigobj1.train(parallel=False)
    loocverr1, _ = krigobj1.loocvcalc()

    krigobj2 = Kriging(KrigInfo2, standardization=True, standtype='default', normy=False, trainvar=False)
    krigobj2.train(parallel=False)
    loocverr2, _ = krigobj2.loocvcalc()

    krigconst = Kriging(KrigConstInfo, standardization=True, standtype='default', normy=False, trainvar=False)
    krigconst.train(parallel=False)
    loocverrConst, _ = krigconst.loocvcalc()

    print('LOOCV 1: ', loocverr1)
    print('LOOCV 2: ', loocverr2)
    print('LOOCV Constraint: ', loocverrConst)

    # List of Kriging objects, objective and constraints should be separated
    kriglist = [krigobj1, krigobj2]
    expconstlist = [krigconst]

    return kriglist, expconstlist

def optimize(kriglist, expconstlist):
    moboInfo = dict()
    moboInfo['nup'] = 5
    moboInfo['acquifunc'] = "ehvi"
    moboInfo['acquifuncopt'] = "diff_evo"
    cheapconstlist = [cheap_const]
    mobo = MOBO(moboInfo, kriglist, autoupdate=True, multiupdate=5, expconst=expconstlist,
                chpconst = cheapconstlist)
    xupdate, yupdate, supdate, metricall = mobo.run(disp=True)
    return xupdate, yupdate, supdate, metricall


if __name__ == '__main__':
    nsample = 20
    nvar = 2
    lb = np.array([0, 0])
    ub = np.array([5, 3])
    sampoption = "halton"
    samplenorm, sample = sampling(sampoption, nvar, nsample, result="real", upbound=ub, lobound=lb)
    X = sample

    # Evaluate function
    y = cust_func(X)
    g = exp_const_eval(X)

    # Create Kriging
    kriglist, expconstlist = construct_krig(X, y, g, lb, ub)

    # Optimize
    xupdate, yupdate, supdate, metricall = optimize(kriglist, expconstlist)