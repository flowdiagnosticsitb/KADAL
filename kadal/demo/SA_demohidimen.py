import os
# Set a single thread per process for numpy with MKL/BLAS
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from kadal.testcase.RA.testcase import evaluate
from kadal.reliability_analysis.akmcs import mcpopgen
from kadal.surrogate_models.kriging_model import Kriging
from kadal.surrogate_models.kpls_model import KPLS
from kadal.surrogate_models.supports.initinfo import initkriginfo
from kadal.sensitivity_analysis.sobol_ind import SobolIndices as SobolI


def generate_krig(init_samp, n_krigsamp, nvar, problem, n_cpu):
    init_krigsamp = mcpopgen(type="lognormal",ndim=nvar,n_order=1,n_coeff=5, stddev=0.2, mean=1)
    print("Evaluating Kriging Sample")
    ykrig = evaluate(init_krigsamp, type=problem)
    print(np.count_nonzero(ykrig <= 0))

    lb = (np.min(init_samp, axis=0))
    ub = (np.max(init_samp, axis=0))

    Pfreal = None

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
    KrigInfo["n_princomp"] = 4
    KrigInfo["optimizer"] = "lbfgsb"

    #trainkrig
    drm = None
    t = time.time()
    krigobj = KPLS(KrigInfo, standardization=True, standtype='default', normy=False, trainvar=False)
    krigobj.train(n_cpu=n_cpu)
    loocverr, _ = krigobj.loocvcalc()
    elapsed = time.time() - t
    print("elapsed time to train Kriging model: ", elapsed, "s")
    print("LOOCV error of Kriging model: ", loocverr, "%")

    return krigobj,loocverr,drm

def pred(krigobj, init_samp, problem, drmmodel=None):

    nsamp = np.size(init_samp,axis=0)
    Gx = np.zeros(shape=[nsamp, 1])
    if nsamp < 10000:
        Gx = krigobj.predict(init_samp, ['pred'])
    else:
        run_times = int(np.ceil(nsamp / 10000))
        for i in range(run_times):
            start = i * 10000
            stop = (i + 1) * 10000
            if i != (run_times - 1):
                Gx[start:stop, :]=  krigobj.predict(init_samp[start:stop, :], ['pred'], drmmodel=drmmodel)
            else:
                Gx[start:, :] = krigobj.predict(init_samp[start:, :], ['pred'], drmmodel=drmmodel)

    init_samp_G = evaluate(init_samp, type=problem)

    subs = np.transpose((init_samp_G - Gx))
    subs1 = np.transpose((init_samp_G - Gx) / init_samp_G)
    RMSE = np.sqrt(np.sum(subs ** 2) / nsamp)
    RMSRE = np.sqrt(np.sum(subs1 ** 2) / nsamp)
    MAPE = 100 * np.sum(abs(subs1)) / nsamp
    print("RMSE = ", RMSE)
    print("MAPE = ", MAPE, "%")
    print("==============================")
    print("UQ")
    mean1 = np.mean(Gx)
    stdev1 = np.std(Gx)
    mean2 = np.mean(init_samp_G)
    stdev2 = np.std(init_samp_G)
    print("model\tmean\tstdev")
    print("real:\t",mean2,"\t",stdev2)
    print("pred:\t",mean1,"\t",stdev1)
    print("==============================")

def pred_sensitivity(krigobj,init_samp,nvar,second=False):
    lb = (np.min(init_samp, axis=0))
    ub = (np.max(init_samp, axis=0))
    lb = np.hstack((lb,lb))
    ub = np.hstack((ub,ub))
    testSA = SobolI(nvar, krigobj, None, ub, lb)
    result = testSA.analyze(True, True, second)
    print("PREDICTED SA")
    for key in result.keys():
        print(key+':')
        if type(result[key]) is not dict:
            print(result[key])
        else:
            for subkey in result[key].keys():
                print(subkey+':', result[key][subkey])

    return result

def real_sensi(krigobj,init_samp,nvar,second=False,prob='hidimenra'):
    lb = (np.min(init_samp, axis=0))
    ub = (np.max(init_samp, axis=0))
    lb = np.hstack((lb, lb))
    ub = np.hstack((ub, ub))
    testSA = SobolI(nvar, krigobj, prob, ub, lb)
    result = testSA.analyze(True, True, second)
    print("REAL SA")
    for key in result.keys():
        print(key + ':')
        if type(result[key]) is not dict:
            print(result[key])
        else:
            for subkey in result[key].keys():
                print(subkey + ':', result[key][subkey])

    return result

def plotFT (first_1,first_2,total_1,total_2,var1,vartot,width=0.4,label1='Predicted',label2='Real',title='',xlim=(0,0.1)):
    y_pos = np.arange(len(var1))
    plt.rcParams["font.weight"] = "bold"
    plt.figure(1, figsize=[15,7])
    ax1 = plt.subplot(1, 2, 1)
    ax1.barh(y_pos-width/2,first_1,width,label=label1)
    ax1.barh(y_pos+width/2,first_2,width,label=label2)
    ax1.grid(which='both',axis='both',linestyle='--')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(var1,fontsize=13,fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=13,width=3)
    ax1.invert_yaxis()
    ax1.set_xlim(xlim)
    ax1.set_title('1st Order '+title,fontsize=15,fontweight='bold')

    ax2 = plt.subplot(1,2,2,sharey=ax1)
    ax2.barh(y_pos-width/2,total_1,width,label=label1)
    ax2.barh(y_pos+width/2,total_2,width,label=label2)
    ax2.grid(which='both',axis='both',linestyle='--')
    ax2.set_title('Total Order '+title,fontsize=15,fontweight='bold')
    ax2.set_xlim(xlim)
    ax2.xaxis.set_tick_params(labelsize=13,width=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(vartot, fontsize=13, fontweight='bold')

    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.legend()
    plt.show();

if __name__ == '__main__':
    init_samp = mcpopgen(type="lognormal",ndim=40,n_order=6,n_coeff=1, stddev=0.2, mean=1)
    dic = dict()

    nvar = 40
    n_krigsamp = 50
    problem = 'hidimenra'
    n_cpu = 12

    # Create Kriging model
    t = time.time()
    krigobj, loocverr, drm= generate_krig(init_samp, n_krigsamp, nvar, problem,
                                          n_cpu)
    ktime = time.time() - t
    # Predict and UQ
    pred(krigobj,init_samp,problem,drmmodel=drm)
    # Sensitivity Analysis
    t1 = time.time()
    sa_pred = pred_sensitivity(krigobj, init_samp, nvar)
    SAtime = time.time()-t1
    print("time: ",SAtime," s")

    sa_real = real_sensi(None,init_samp, nvar)
    first_highest_index = np.argsort(-sa_real['first'])[:5]
    total_highest_index = np.argsort(-sa_real['total'])[:5]

    real_first = sa_real['first'][first_highest_index]
    pred_first = sa_pred['first'][first_highest_index]
    real_total = sa_real['total'][total_highest_index]
    pred_total = sa_pred['total'][total_highest_index]

    var1 = ['S'+str(i+1) for i in first_highest_index]
    var2 = ['S'+str(i+1) for i in total_highest_index]
    plotFT(real_first,pred_first,real_total,pred_total,var1,var2)


