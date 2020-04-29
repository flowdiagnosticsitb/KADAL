import numpy as np
from numpy.linalg import solve as mldivide
from kadal.surrogate_models.supports.kernelfunc import calckernel


def likelihood (x, KrigInfo, mode='default', trainvar=True):
    """
    Calculates the negative of the concentrated ln-likelihood

    Args:
        x (nparray): vector of log(theta) parameters
        KrigInfo (dict): Dictionary contains Kriging Model
        mode (str): Mode of return. Defaults to 'default', only return NegLnLike
            available modes are 'default' and 'all'
        trainvar (bool): Train Kriging variance or not. Defaults to True

    Returns:
        NegLnLike (nparray): log-likelihood *-1 for minimising (default)
        KrigInfo (dict): Dictionary contains Kriging Model (optional)

    Copyright 2007 A I J Forrester
    This program is free software: you can redistribute it and/or modify  it
    under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or any
    later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
    General Public License for more details.

    You should have received a copy of the GNU General Public License and GNU
    Lesser General Public License along with this program. If not, see
    <http://www.gnu.org/licenses/>.
    """

    # Variables Initialization
    nvar = KrigInfo["nvar"]
    nsamp = KrigInfo["nsamp"]
    F = KrigInfo["F"]
    kernel = KrigInfo["kernel"]
    nkernel = KrigInfo["nkernel"]

    # Check Variables
    if KrigInfo["standardization"] is False:  # If standardization is not performed
        X = KrigInfo["X"]
        y = KrigInfo["y"]
    else:  # If standardization is performed
        X = KrigInfo["X_norm"]
        if "y_norm" in KrigInfo and KrigInfo["y_norm"][0] != 0:
            y = KrigInfo["y_norm"]
        else:
            y = KrigInfo["y"]

    if KrigInfo["type"].lower() == "kpls":
        plscoeff = KrigInfo["plscoeff"]
    else:
        pass

    if KrigInfo["kernel"] == ["iso_gaussian"]:
        x = np.array([x]*nvar)
    else:
        pass

    if type(x) is not np.ndarray:
        x = np.array([x])
    if KrigInfo["n_princomp"] is not False:
        nvar = KrigInfo["n_princomp"]

    # Set nugget and weighting function
    nugget,eps,wgkf = nuggetset(x,KrigInfo,nvar,nkernel,trainvar=trainvar)

    # Set theta and Kriging variance
    if trainvar is True:
        theta = 10**(x[0:nvar])
        SigmaSqr = 10 ** (x[nvar])
    else:
        theta = 10 ** (x[0:nvar])
        SigmaSqr = None

    # Set variables
    KrigInfo["Theta"] = x[0:nvar]
    KrigInfo["nugget"] = nugget
    KrigInfo["wgkf"] = wgkf
    n = np.ma.size(X,axis=0)

    #Pre-allocate memory
    Psi = np.zeros(shape=[n,n])
    PsiComp = np.zeros(shape=[n,n,nkernel])


    #Build upper half of correlation matrix
    if KrigInfo["type"].lower() == "kriging":
        for ii in range(0,nkernel):
            PsiComp[:,:,ii] = wgkf[ii]*calckernel(X,X,theta,nvar,type=kernel[ii])
        Psi = np.sum(PsiComp,2)

    elif KrigInfo["type"].lower() == "kpls":
        nvar = KrigInfo["nvar"]
        for ii in range(0,nkernel):
            PsiComp[:,:,ii] = wgkf[ii]*calckernel(X,X,theta,nvar,type=kernel[ii],plscoeff=plscoeff)
        Psi = np.sum(PsiComp,2)

    # Perform Main Calculation
    NegLnLike, BE, U, SigmaSqr = maincalc(Psi,eps,y,F,nsamp,n,SigmaSqr=SigmaSqr)

    KrigInfo["U"] = U
    KrigInfo["Psi"] = Psi
    KrigInfo["BE"] = BE
    KrigInfo["SigmaSqr"] = SigmaSqr
    KrigInfo["NegLnLike"] = NegLnLike

    if mode.lower() == "default":
        return NegLnLike
    elif mode.lower() == "all":
        return KrigInfo
    else:
        raise TypeError("Only have two modes, default and all, default return NegLnLike, all return KrigInfo")


def nuggetset(x,KrigInfo,nvar,nkernel,trainvar):
    if trainvar is True:
        if len(x) == nvar+1: # Nugget is not tunable, single kernel
            nugget = KrigInfo["nugget"]
            eps = 10. ** nugget
            wgkf = np.array([1])
        elif len(x) == nvar+2: # Nugget is tunable, single kernel
            nugget = x[nvar+1]
            eps = 10. ** nugget
            wgkf = np.array([1])
        elif len(x) == nvar+nkernel+1: # Nugget is not tunable, multiple kernels
            nugget = KrigInfo["nugget"]
            eps = 10. ** nugget
            weight = x[nvar+1:nvar+nkernel+1]
            wgkf = weight / np.sum(weight)
        elif len(x) == nvar+nkernel+2: # Nugget is tunable, multiple kernels
            nugget = x[nvar+1]
            eps = 10. ** nugget
            weight = x[nvar+2:nvar+nkernel+2]
            wgkf = weight / np.sum(weight)
        else:
            return ValueError("Nugget setting is not available")
    else:
        if len(x) == nvar:  # Nugget is not tunable, single kernel
            nugget = KrigInfo["nugget"]
            eps = 10. ** nugget
            wgkf = np.array([1])
        elif len(x) == nvar + 1:  # Nugget is tunable, single kernel
            nugget = x[nvar]
            eps = 10. ** nugget
            wgkf = np.array([1])
        elif len(x) == nvar + nkernel:  # Nugget is not tunable, multiple kernels
            nugget = KrigInfo["nugget"]
            eps = 10. ** nugget
            weight = x[nvar:nvar + nkernel]
            wgkf = weight / np.sum(weight)
        elif len(x) == nvar + nkernel + 1:
            nugget = x[nvar]
            eps = 10. ** nugget
            weight = x[nvar + 1:nvar + nkernel + 1]
            wgkf = weight / np.sum(weight)
        else:
            return ValueError("Nugget setting is not available")

    return (nugget,eps,wgkf)


def maincalc(Psi,eps,y,F,nsamp,n,SigmaSqr=None):
    # Add upper and lower halves and diagonal of ones plus
    # small number to reduce ill-conditioning
    Psi = Psi + (np.eye(n) * (eps))
    if np.any(np.linalg.eigvals(Psi) < 0):
        print("Not positive definite")

    Utemp = np.linalg.cholesky(Psi)  # Cholesky in Python Produce lower triangle
    U = np.transpose(Utemp)  # Cholesky in Matlab Produce Upper Triangle

    try:
        # Sum lns of diagonal to find ln(abs(det(Psi)))
        LnDetPsi = 2 * np.sum(np.log(abs(np.diag(U))))

        # Compute the coefficients of regression function
        temp11 = mldivide(np.transpose(U), y)  # just a temporary variable for debugging
        temp1 = (mldivide(U, temp11))  # just a temporary variable for debugging
        temp21 = mldivide(np.transpose(U), F)  # just a temporary variable for debugging
        temp2 = (mldivide(U, temp21))  # just a temporary variable for debugging
        tempmu = mldivide(np.dot(np.transpose(F), temp2),
                          np.dot(np.transpose(F), temp1))  # np.dot(np.transpose(F),temp1)/np.dot(np.transpose(F),temp2)
        BE = tempmu

        # Use back-substitution of Cholesky instead of inverse
        temp31 = mldivide(np.transpose(U), (y - np.dot(F, BE)))  # just a temporary variable for debugging
        temp3 = mldivide(U, temp31)  # just a temporary variable for debugging

        if SigmaSqr is not None:
            # Ln likelihood
            tempNegLnLike = -0.5 * LnDetPsi - nsamp / 2 * np.log(2 * np.pi) - nsamp / 2 * np.log(SigmaSqr) - np.dot(
                np.transpose(y - np.dot(F, BE)), temp3) / (2 * SigmaSqr)
            NegLnLike = -tempNegLnLike[0, 0]
        else:
            # Concentrated Ln-likelihood
            SigmaSqr = (np.dot(np.transpose(y - np.dot(F,BE)),(temp3)))/n
            tempNegLnLike    = -1*(-(n/2)*np.log(SigmaSqr) - 0.5*LnDetPsi)
            NegLnLike = tempNegLnLike[0, 0]

    except Exception as e:
        NegLnLike = 10000
        print(e)
        print("Matrix is ill-conditioned or an error occurred, penalty is used for NegLnLike value")
        print("Are you sure want to continue?")
        input("Press Enter to continue...")

    return (NegLnLike,BE,U,SigmaSqr)




