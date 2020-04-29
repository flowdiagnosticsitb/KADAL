import numpy as np
import numpy.matlib
from kadal.misc.sampling.samplingplan import scale

def paregopre(yall,idx=False):
    """
    Pre-processing solutions for ParEGO

    Input:
      Yall - Original responses

    Output
       YN - Scalarized responses with random weights
    """
    NF = 11
    WGA = np.transpose([np.arange(0,1+1/(NF-1),1/(NF-1))])
    WGA = np.hstack((WGA,1-WGA))
    if idx is False:
        IDR = np.random.permutation(np.size(WGA,0))
        IDR = np.random.choice(IDR)
    else:
        IDR = idx
    WG = WGA[IDR,:]
    FFN,_ = scale(yall,0,range=np.vstack((np.min(yall,0),np.max(yall,0))))
    rho = 0.05

    YN = np.zeros(shape=[np.size(FFN,0),1])
    for IH in range(0,np.size(FFN,0)):
        YN[IH,:] = np.max((WG*(FFN[IH,:])),0) + rho*np.sum((WG*FFN[IH,:]),0)

    return YN
