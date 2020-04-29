import numpy as np

def hvolume2d(P,x):
    i = np.argsort(P[:, 0])
    S = P[i, :]
    h = 0
    if np.size(P,axis=0) != 0:
        k = len(S[:,0])
        for i in range(0,k):
            if i == 0:
                h = h + (x[0]-S[i,0])*(x[1]-S[i,1])
            else:
                h = h + (x[0]-S[i,0])*(S[i-1,1]-S[i,1])
    return h