import os
# Set a single thread per process for numpy with MKL/BLAS
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

import time
import numpy as np
import matplotlib.pyplot as plt

from kadal.reliability_analysis.akmcs import AKMCS, mcpopgen
from kadal.testcase.RA.testcase import evaluate
from kadal.surrogate_models.kriging_model import Kriging
from kadal.surrogate_models.supports.initinfo import initkriginfo


def generate_krig(init_samp, krigsamp, nvar, problem, n_cpu):
    # Monte Carlo Sampling
    t1 = time.time()
    init_krigsamp = krigsamp
    n_krigsamp = np.size(krigsamp, 0)
    ykrig = evaluate(init_krigsamp, type=problem)
    t2 = time.time()

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
    KrigInfo = initkriginfo(1)
    KrigInfo["X"] = init_krigsamp
    KrigInfo["y"] = ykrig
    KrigInfo["nvar"] = nvar
    KrigInfo["nsamp"] = n_krigsamp
    KrigInfo["nrestart"] = 5
    KrigInfo["ub"] = ub
    KrigInfo["lb"] = lb
    KrigInfo["nkernel"] = len(KrigInfo["kernel"])
    KrigInfo["optimizer"] = "lbfgsb"

    # trainkrig
    t = time.time()
    krigobj = Kriging(KrigInfo, standardization=True, standtype='default',
                      normy=False, trainvar=False)
    krigobj.train(n_cpu=n_cpu)
    loocverr, _ = krigobj.loocvcalc()
    elapsed = time.time() - t
    print("elapsed time for train Kriging model: ", elapsed, "s")
    print("LOOCV error of Kriging model: ", loocverr, "%")

    return krigobj, Pfreal


def run_akmcs(krigobj, init_samp, problem, filename, figloc, posdata, negdata):
    # Define AKMCS Information
    akmcsInfo = dict()
    akmcsInfo["init_samp"] = init_samp
    akmcsInfo["maxupdate"] = 100
    akmcsInfo["problem"] = problem

    # Run AKMCS
    t = time.time()
    akmcsobj = AKMCS(krigobj, akmcsInfo)
    akmcsobj.run(savedatato=filename, logging=False, saveimageto=figloc,
                 plotdatapos=posdata, plotdataneg=negdata)
    elapsed = time.time() - t
    print("elapsed time is : ", elapsed, "s")
    xtotal = akmcsobj.krigobj.KrigInfo['X']

    return xtotal


if __name__ == '__main__':
    nvar = 2
    n_krigsamp = 25
    problem = 'styblinski'
    filename = "akmcs2d.csv"
    figloc = 'akmcsupdate'
    n_cpu = 12

    init_samp = mcpopgen(type='normal', ndim=nvar, n_order=6, n_coeff=1,
                         stddev=1.5)
    ysamp = evaluate(init_samp, problem)
    krigsamp = mcpopgen(type='normal', ndim=nvar, n_order=1, n_coeff=2.5,
                        stddev=1.5)
    pos_MC = init_samp[ysamp.flatten() > 0, :]
    neg_MC = init_samp[ysamp.flatten() <= 0, :]

    krigobj, Pfreal = generate_krig(init_samp, krigsamp, nvar, problem, n_cpu)
    xtotal = run_akmcs(krigobj, init_samp, problem, filename, figloc,
                       pos_MC, neg_MC)
    print(Pfreal)

    updateX = xtotal[n_krigsamp:, :]

    np.savetxt('MCpop_styb.csv', init_samp, fmt='%10.5f', delimiter=',')
    np.savetxt('akmcsXtot_styb.csv', xtotal, fmt='%10.5f', delimiter=',')

    plt.figure(0, figsize=[10, 9])
    plt.scatter(pos_MC[:, 0], pos_MC[:, 1], c='yellow', label='Safe')
    plt.scatter(neg_MC[:, 0], neg_MC[:, 1], c='cyan', label='Fail')
    plt.scatter(krigsamp[:, 0], krigsamp[:, 1], c='red',
                label='Initial Population')
    plt.scatter(updateX[:, 0], updateX[:, 1], s=75, c='black', marker='x',
                label='Update')
    plt.xlabel('X1', fontsize=18)
    plt.ylabel('X2', fontsize=18)
    plt.tick_params(axis='both', which='both', labelsize=16)
    plt.legend(loc=1, prop={'size': 15})
    plt.show()
