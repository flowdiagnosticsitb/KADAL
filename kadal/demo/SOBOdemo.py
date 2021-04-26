import os
# Set a single thread per process for numpy with MKL/BLAS
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

import numpy as np

from kadal.optim_tools.SOBO import SOBO
from kadal.testcase.analyticalfcn.cases import evaluate
from kadal.surrogate_models.kriging_model import Kriging
from kadal.surrogate_models.supports.initinfo import initkriginfo
from kadal.misc.sampling.samplingplan import sampling


def generate_kriging(n_cpu):
    # Sampling
    nsample = 10
    nvar = 2
    lb = np.array([-5, -5])
    ub = np.array([5, 5])
    sampoption = "halton"
    samplenorm, sample = sampling(sampoption, nvar, nsample, result="real",
                                  upbound=ub, lobound=lb)
    X = sample
    # Evaluate sample
    # global y
    y = evaluate(X, "styblinski")

    # Initialize KrigInfo
    # global KrigInfo
    KrigInfo = initkriginfo()
    # Set KrigInfo
    KrigInfo["X"] = X
    KrigInfo["y"] = y
    KrigInfo["problem"] = "styblinski"
    KrigInfo["nrestart"] = 5
    KrigInfo["ub"] = ub
    KrigInfo["lb"] = lb
    KrigInfo["optimizer"] = "lbfgsb"

    # Run Kriging
    krigobj = Kriging(KrigInfo, standardization=True, standtype='default',
                      normy=False, trainvar=False)
    krigobj.train(n_cpu=n_cpu)
    loocverr, _ = krigobj.loocvcalc()
    print("LOOCV error of Kriging model: ", loocverr, "%")

    return krigobj


def runopt(krigobj, n_cpu):
    soboInfo = dict()
    soboInfo['nup'] = 35
    soboInfo['stalliteration'] = 40
    soboInfo['nrestart'] = 10
    soboInfo['acquifunc'] = 'EI'
    soboInfo['acquifuncopt'] = 'diff_evo'

    optim = SOBO(soboInfo, krigobj, autoupdate=True)
    xnext, ynext = optim.run(n_cpu)
    return xnext, ynext


if __name__ == '__main__':
    n_cpu = 12
    krigobj1 = generate_kriging(n_cpu)
    xnext, ynext = runopt(krigobj1, n_cpu)
