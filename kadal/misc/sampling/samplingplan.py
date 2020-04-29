"""
Subroutines:
 - sampling : return normalized sample in range between 0 and 1
 - realval : return actual sample based on bound value

"""
import numpy as np
import numpy.matlib
from kadal.misc.sampling.haltonsampling import halton
from kadal.misc.sampling.rlh import rlh
from sobolsampling.sobol_new import sobol_points

def sampling(option,nvar,nsamp,result='normalized',upbound=None,lobound=None):

    if option.lower() == "halton":
        samplenorm = halton(nvar,nsamp)
    elif option.lower() == "sobolnew" or option.lower() == "sobol":
        samplenorm = sobol_points(nsamp+1, nvar)
        samplenorm = samplenorm[1:,:]
    elif option.lower() == "rlh":
        samplenorm = rlh(nvar, nsamp)
    else:
        raise NameError("sampling plan unavailable!")

    if result.lower() == "real" and lobound is not None and upbound is not None:
        checker = upbound - lobound;
        for numbers in checker:
            if numbers < 0:
                raise ValueError("Upper bound must bigger than lower bound!")
        sample = realval(lobound, upbound, samplenorm)
    elif result.lower() == "real" and (lobound is None or upbound is None):
        raise ValueError("lb and ub must have value")

    if result.lower() == "real":
        return samplenorm,sample
    else:
        print("real value returned as zero matrix")
        return samplenorm,np.zeros(np.shape(samplenorm))

def realval(lb,ub,samp):
    if len(ub) != len(lb):
        raise ValueError("Lower and upper bound have to be in the same dimension")
    if len(ub) != np.size(samp,axis=1):
        raise ValueError("sample and bound are not in the same dimension")
    ndim = len(ub)
    nsamp = np.size(samp,axis=0)
    realsamp = np.zeros(shape=[nsamp,ndim])
    for i in range(0, ndim):
        for j in range(0, nsamp):
            realsamp[j, i] = (samp[j, i] * (ub[i] - lb[i])) + lb[i]
    return realsamp

def standardize(X,y=None,type='default',norm_y=False,**kwargs):
    ranges = kwargs.get('range', np.array([None]))

    if type.lower()=='default':
        X_norm = np.empty(np.shape(X))
        y_norm = np.empty(np.shape(y))
        if ranges.any() == None:
            raise ValueError("Default normalization requires range value!")

        if norm_y == True:
            # Normalize to [0,1]
            for i in range(0, np.size(X, 1)):
                X_norm[:, i] = (X[:, i] - ranges[0, i]) / (ranges[1, i] - ranges[0, i])
            for jj in range(0, np.size(y, 1)):
                y_norm[:, jj] = (y[:, jj] - ranges[0, 1+i+jj]) / (ranges[1, 1+i+jj] - ranges[0, 1+i+jj])

            # Normalize to [-1,1]
            X_norm = (X_norm - 0.5) * 2
            y_norm = (y_norm - 0.5) * 2

            return X_norm,y_norm

        else:
            #Normalize to [0,1]
            for i in range(0,np.size(X,1)):
                X_norm[:,i] = (X[:,i]-ranges[0,i])/(ranges[1,i]-ranges[0,i])

            #Normalize to [-1,1]
            X_norm = (X_norm-0.5)*2

            return X_norm

    elif type.lower() == 'std':
        if norm_y == True:
            X_mean = np.mean(X, axis=0)
            X_std = X.std(axis=0, ddof=1)
            y_mean = np.mean(y, axis=0)
            y_std = y.std(axis=0, ddof=1)
            X_std[X_std == 0.] = 1.
            y_std[y_std == 0.] = 1.

            X_norm = (X - X_mean) / X_std
            y_norm = (y - y_mean) / y_std
            return X_norm, y_norm, X_mean, y_mean, X_std, y_std
        else:
            X_mean = np.mean(X, axis=0)
            X_std = X.std(axis=0, ddof=1)
            X_std[X_std == 0.] = 1.

            X_norm = (X - X_mean) / X_std
            return X_norm, X_mean, X_std

def scale(inst,strategy=0,**kwargs):
    """
    =========================================================================
    scale : scale data to [0 1] or N(0,1)
    -------------------------------------------------------------------------
    input
      inst    [m x n] : learning data
       strategy[1 x 1] : 0-sacle to [0 1]
                         1-scale to N(0,1)
       range   [2 x n] : first row  : the minimum of each column
                         second row : the maximum of each column
    -------------------------------------------------------------------------
    ouput
       scale_inst : scaling learning data
       range      : the same with input
    =========================================================================
    """
    range = kwargs.get('range',np.array([None]))
    n = np.size(inst[:,0],0)
    if strategy == 0:
        if range.any() == None:
            Max = np.matlib.repmat(np.max(inst,0),n,1)
            Min = np.matlib.repmat(np.min(inst,0),n,1)
            range = np.vstack((Min[0,:],Max[0,:]))
        else:
            Max = np.matlib.repmat(range[1,:],n,1)
            Min = np.matlib.repmat(range[0,:],n,1)
        M_m = Max-Min
        Temp = np.where(M_m[0,:] == 0)
        if Temp:
            M_m[:,Temp] = inst[:,Temp]
            if M_m[0,Temp] == 0:
                M_m[:,Temp] = 1
        scale_inst = (inst-Min)/M_m
    else:
        if range.any() == None:
            inst_c = np.mean(inst,0)
            inst_s = np.std(inst,0)
            range = np.vstack((inst_c,inst_s))
        else:
            inst_c = range[0,:]
            inst_s = range[1,:]
        scale_inst = inst - np.ones(shape=[n,1])*inst_c
        scale_inst = scale_inst*np.diag(1/inst_s)

    return (scale_inst, range)

if __name__ == '__main__':
    samplenorm,_ = sampling('sobol',3,10)
    print(samplenorm)
