import numpy as np
from kadal.surrogate_models.supports.errperf import errperf
from kadal.surrogate_models.supports.likelihood_func import likelihood
from kadal.surrogate_models.supports.prediction import prediction
from copy import deepcopy

def loocv2(KrigInfo,errtype="rmse"):
    # Take required info from KrigInfo

    nsamp = KrigInfo["nsamp"]
    F = KrigInfo["F"]
    y = KrigInfo["y"]
    LOOCVpred = np.zeros(shape=[len(y),1])

    for i in range(nsamp):
        KrigInfotemp = deepcopy(KrigInfo)
        xLOOCV = np.delete(KrigInfo['X_norm'],i,0)
        xpredLOOCV = KrigInfo['X_norm'][i,:]
        yLOOCV = np.delete(KrigInfo['y'],i,0)
        KrigInfotemp["F"] = np.delete(F,i,0)
        KrigInfotemp['y'] = yLOOCV[:]
        KrigInfotemp['X_norm'] = xLOOCV[:]
        KrigInfotemp['nsamp'] = nsamp-1
        if KrigInfotemp["kernel"] == ["iso_gaussian"]:
            KrigInfotemp = likelihood(KrigInfotemp['Theta'][0],KrigInfotemp,mode='all',trainvar=False)
        else:
            KrigInfotemp = likelihood(KrigInfotemp['Theta'],KrigInfotemp,mode='all',trainvar=False)
        KrigInfotemp['standardization'] = False
        KrigInfotemp['X'] = KrigInfotemp['X_norm']
        LOOCVpred[i,0] = prediction(xpredLOOCV, KrigInfotemp, predtypes=['pred'], drm=None)

    LOOCVerr = errperf(y, LOOCVpred, errtype)

    return LOOCVerr, LOOCVpred