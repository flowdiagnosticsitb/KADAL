import numpy as np; import numpy.matlib
from kadal.optim_tools.ehvi.exipsi import exipsi
from kadal.optim_tools.ehvi.gaussfcn import gausscdf,gausspdf
from kadal.optim_tools.ehvi.hvolume2d import hvolume2d
from copy import deepcopy

def exi2d(P,r,mu,s):
    i = np.argsort(P[:, 0])
    S = P[i, :]
    k = np.size(S,axis=0)

    c2 = np.sort(S[:,1])
    c1 = np.sort(S[:,0])
    c = np.zeros(shape=[k+1,k+1])

    for i in range (-1,k):
        if i == -1 :
            ii = 0
        else:
            ii =i+1
        for j in range (-1,k-ii):
            # For coordinate j determine hight fMax2
            if j == -1:
                fmax2 = r[1]
            else:
                fmax2 = c2[k-j-1]
            # For coordinate i determine the width of the staircase fMax1
            if i == -1:
                fmax1 = r[0]
            else:
                fmax1 = c1[k-i-1]
            # get cell coordinates
            if j == -1:
                cL1 = -1*np.inf
            else:
                cL1 = c1[j]
            if i == -1:
                cL2 = -1*np.inf
            else:
                cL2 = c2[i]
            if j == k-1 :
                cU1 = r[0]
            else:
                cU1 = c1[j+1]
            if i == k-1:
                cU2 = r[1]
            else:
                cU2 = c2[i+1]
            # SM = points that are dominated or equal to upper cell bound
            SM = np.asarray([[]])
            ss = 0
            for m in range(0,k):
                if cU1 <= S[m,0] and cU2 <= S[m,1]:
                    stemp = np.hstack((S[m,0],S[m,1]))
                    if ss != 0:
                        SM = np.vstack((stemp,SM))
                    else:
                        SM = np.asarray([deepcopy(stemp)])
                    ss = ss + 1

            f = np.hstack((fmax1,fmax2))
            if np.size(SM) == 0:
                sPlus = 0
            else:
                sPlus = hvolume2d(SM,f)
            # Marginal integration over the length of a cell
            Psi1 = exipsi(fmax1,cU1,mu[0],s[0]) - exipsi(fmax1,cL1,mu[0],s[0])
            # Marginal integration over the height of a cell
            Psi2 = exipsi(fmax2,cU2,mu[1],s[1]) - exipsi(fmax2,cL2,mu[1],s[1])
            # Cumulative Gaussian over length for correction constant
            GaussCDF1 = gausscdf((cU1-mu[0])/s[0]) - gausscdf((cL1-mu[0])/s[0])
            # Cumulative Gaussian over length for correction constant
            GaussCDF2 = gausscdf((cU2-mu[1])/s[1]) - gausscdf((cL2-mu[1])/s[1])
            c [i+1,j+1] = Psi1*Psi2 - sPlus*GaussCDF1*GaussCDF2

    treshold = 0
    for ii in range (0,k+1):
        for jj in range(0,k+1):
            if c[ii,jj] < treshold:
                c[ii,jj] = treshold
            else:
                pass
    res = np.sum(np.sum(c, axis=0))
    return res
