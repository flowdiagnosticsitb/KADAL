import sys
sys.path.insert(0, "..")
import numpy as np
from kadal.reliability_analysis.akmcs import AKMCS,mcpopgen
from kadal.testcase.RA.testcase import evaluate
from kadal.surrogate_models.kriging_model import Kriging
from kadal.surrogate_models.kpls_model import KPLS
from kadal.surrogate_models.supports.initinfo import initkriginfo
import time


def generate_krig(init_samp,n_krigsamp,nvar,problem):

    # Kriging Sample
    t1 = time.time()
    init_krigsamp = mcpopgen(type="lognormal",ndim=nvar,n_order=1,n_coeff=5, stddev=0.2, mean=1)
    ykrig = evaluate(init_krigsamp, type=problem)
    t2 = time.time()
    print("50 samp eval", t2 - t1)

    # Evaluate Kriging Sample and calculate PoF real
    init_samp_G = evaluate(init_samp, type=problem)
    total_samp = np.hstack((init_samp, init_samp_G)).transpose()
    positive_samp = total_samp[:, total_samp[nvar] >= 0]
    positive_samp = positive_samp.transpose()
    nsamp = np.size(init_samp, 0)
    npos = np.size(positive_samp, 0)
    Pfreal = 1 - npos / nsamp

    lb = np.floor(np.min(init_samp)) * np.ones(shape=[nvar])
    ub = np.ceil(np.max(init_samp)) * np.ones(shape=[nvar])

    # Set Kriging Info
    KrigInfo = initkriginfo("single")
    KrigInfo["X"] = init_krigsamp
    KrigInfo["y"] = ykrig
    KrigInfo["nvar"] = nvar
    KrigInfo["nsamp"] = n_krigsamp
    KrigInfo["nrestart"] = 5
    KrigInfo["ub"] = ub
    KrigInfo["lb"] = lb
    KrigInfo["nkernel"] = len(KrigInfo["kernel"])
    KrigInfo["n_princomp"] = 4
    KrigInfo["optimizer"] = "lbfgsb"

    #trainkrig
    t = time.time()
    krigobj = KPLS(KrigInfo, standardization=True, standtype='default', normy=False, trainvar=False)
    krigobj.train(parallel=False)
    loocverr, _ = krigobj.loocvcalc()
    elapsed = time.time() - t
    print("elapsed time for train Kriging model: ", elapsed, "s")
    print("LOOCV error of Kriging model: ", loocverr, "%")

    return krigobj,Pfreal


def run_akmcs(krigobj,init_samp,problem,filename):

    # Define AKMCS Information
    akmcsInfo = dict()
    akmcsInfo["init_samp"] = init_samp
    akmcsInfo["maxupdate"] = 70
    akmcsInfo["problem"] = problem

    # Run AKMCS
    t = time.time()
    akmcsobj = AKMCS(krigobj,akmcsInfo)
    akmcsobj.run(savedatato=filename)
    elapsed = time.time() - t
    print("elapsed time is : ", elapsed, "s")

if __name__ == '__main__':
    init_samp = mcpopgen(type="lognormal",ndim=40,n_order=6,n_coeff=1, stddev=0.2, mean=1)

    nvar = 40
    n_krigsamp = 50
    problem = 'hidimenra'
    filename = "akmcshidimen.csv"

    krigobj,Pfreal = generate_krig(init_samp,n_krigsamp,nvar,problem)
    run_akmcs(krigobj,init_samp,problem,filename)
    print(Pfreal)