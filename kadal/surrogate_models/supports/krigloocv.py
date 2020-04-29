import numpy as np
from kadal.surrogate_models.supports.errperf import errperf

def loocv(KrigInfo,errtype="rmse",num=None):

    #Take required info from KrigInfo
    nsamp = KrigInfo["nsamp"]
    if num == None:
        F = KrigInfo["F"]
        Psi = KrigInfo["Psi"]
        SigmaSqr = KrigInfo["SigmaSqr"]
        y = KrigInfo["y"]
    else:
        F = KrigInfo["F"][num]
        Psi = KrigInfo["Psi"][num]
        SigmaSqr = KrigInfo["SigmaSqr"][num]
        y = KrigInfo["y"][num]

    #Create Matrix B
    PsiF = np.hstack((SigmaSqr*Psi, F))
    sl = np.size(PsiF,1)-np.size(F.transpose(),1)
    Fones = np.hstack(( F.transpose(), np.ones(shape=[np.size(F.transpose(),0),sl]) ))
    B = np.linalg.inv( np.vstack((PsiF,Fones)) )

    LOOCVpred = np.zeros(shape=[nsamp])
    for i in range(0,nsamp):
        LOOCVpred[i] = 0
        for j in range(0,nsamp):
            LOOCVpred[i] = LOOCVpred[i] + np.dot((B[i,j]/B[i,i]),y[j,0])
        LOOCVpred[i] = -LOOCVpred[i] + y[i,0]

    LOOCVpred = LOOCVpred.reshape(-1,1)
    LOOCVerr = errperf(y,LOOCVpred,errtype)

    return (LOOCVerr,LOOCVpred)