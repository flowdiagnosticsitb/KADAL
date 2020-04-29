import numpy as np
import warnings
from copy import deepcopy

def paretopoint(B):
    """
    Let Fi(X), i=1...n, are objective functions
    for minimization.
    A point X* is said to be Pareto optimal one
    if there is no X such that Fi(X)<=Fi(X*) for
    all i=1...n, with at least one strict inequality.
    A=prtp(B),
    B - m x n input matrix: B=
    [F1(X1) F2(X1) ... Fn(X1);
     F1(X2) F2(X2) ... Fn(X2);
     .......................
     F1(Xm) F2(Xm) ... Fn(Xm)]
    A - an output matrix with rows which are Pareto
    points (rows) of input matrix B.
    [A,b]=prtp(B). b is a vector which contains serial
    numbers of matrix B Pareto points (rows).
    """
    A = []
    varargout = []
    sz1 = np.size(B,0)
    jj = 0
    kk = np.zeros(shape=[sz1])
    c = np.zeros(shape=[sz1,np.size(B,1)])
    bb = deepcopy(c)
    for k in range(0,sz1):
        j = 0
        ak = B[k,:]
        for i in range(0,sz1):
            if i !=k:
                bb[j,:] = ak-B[i,:]
                j = j + 1

        if np.any(np.transpose(bb[0:j,:]) < 0,axis=0).all():
            c[jj,:] = ak
            kk[jj] = k
            jj = jj + 1

    if jj:
        A = c[0:jj,:]
        varargout = kk[0:jj]
    else:
        warnings.warn("There are no Pareto points. The result is an empty matrix.")
    return (A,varargout)