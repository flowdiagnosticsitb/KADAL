import sys
sys.path.insert(0, "..")
import numpy as np
from kadal.surrogate_models.kriging_model import Kriging
from kadal.surrogate_models.supports.initinfo import initkriginfo
from kadal.misc.sampling.samplingplan import sampling
from matplotlib import pyplot as plt
from kadal.optim_tools.MOBO import MOBO
from copy import deepcopy

def generate_kriging():
    # Sampling
    nsample = 20
    nvar = 2
    nobj = 2
    lb = -1 * np.ones(shape=[nvar])
    ub = 1 * np.ones(shape=[nvar])
    sampoption = "halton"
    samplenorm, sample = sampling(sampoption, nvar, nsample, result="real", upbound=ub, lobound=lb)
    X = sample
    # Evaluate sample
    global y
    y = myproblem(X)

    # Initialize KrigInfo
    KrigInfo1 = initkriginfo("single")
    # Set KrigInfo
    KrigInfo1["X"] = X
    KrigInfo1["y"] = y[:,0].reshape(-1,1)
    KrigInfo1["problem"] = myproblem
    KrigInfo1["nrestart"] = 5
    KrigInfo1["ub"] = ub
    KrigInfo1["lb"] = lb
    KrigInfo1["optimizer"] = "lbfgsb"

    # Initialize KrigInfo
    KrigInfo2 = deepcopy(KrigInfo1)
    KrigInfo2['y'] = y[:,1].reshape(-1,1)

    # Run Kriging
    krigobj1 = Kriging(KrigInfo1, standardization=True, standtype='default', normy=False, trainvar=False)
    krigobj1.train(parallel=False)
    loocverr1, _ = krigobj1.loocvcalc()
    print("LOOCV error of Kriging model: ", loocverr1, "%")

    krigobj2 = Kriging(KrigInfo2, standardization=True, standtype='default', normy=False, trainvar=False)
    krigobj2.train(parallel=False)
    loocverr2, _ = krigobj2.loocvcalc()
    print("LOOCV error of Kriging model: ", loocverr2, "%")

    return krigobj1, krigobj2

def runopt(krigobj1, krigobj2):
    moboInfo = dict()
    moboInfo["nup"] = 3
    moboInfo["nrestart"] = 10
    moboInfo["acquifunc"] = "ehvi"
    moboInfo["acquifuncopt"] = "lbfgsb"

    Optim = MOBO(moboInfo,[krigobj1,krigobj2],autoupdate=True,multiupdate=5)
    xupdate,yupdate,supdate,metricall = Optim.run(disp=True)

    return xupdate,yupdate,metricall

def myproblem(x):
    r = 1
    a = 1 / (2 * r)
    m = np.size(x, 0)
    n = np.size(x, 1)
    f = np.zeros(shape=[m, 2])
    for i in range(0, m):
        f[i, 0] = (1 / (n ** a)) * np.sum(x[i, :] ** 2) ** a
        f[i, 1] = (1 / (n ** a)) * np.sum((1 - x[i, :]) ** 2) ** a
    if m == 1:
        f = f[0, :]
    return f

if __name__ == '__main__':
    krigobj1, krigobj2 = generate_kriging()
    xupdate,yupdate,metricall = runopt(krigobj1,krigobj2)

    print(metricall)
    plt.scatter(y[:,0],y[:,1])
    plt.scatter(yupdate[:, 0], yupdate[:, 1])
    plt.show()