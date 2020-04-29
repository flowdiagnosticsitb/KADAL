import numpy as np
import numpy.matlib
from itertools import combinations
from copy import deepcopy
from scipy.special import factorial

def polytruncation(nix, nvar, q):
    """
    Generate polynomial indices for the trend function by using total-order
    truncation or hyperbolic truncation (the former is if q = 1 and the latter is if q < 1)

    inputs:
      nix - Maximum polynomial order
      nvar - number of variables
      q - hyperbolic truncation parameter

    output:
      idx - Indices of polynomial bases

    Author: Pramudita Satria Palar(pramsatriapalar@gmail.com, pramsp@ftmd.itb.ac.id)

    Generate index for polynomial chaos expansion
    """
    idx = []
    for i in range(nix, -1, -1):
        if i == nix:
            idx = MonCof(i, nvar)
        else:
            idx = np.vstack((idx, MonCof(i, nvar)))
    idx = np.flip(idx, 0)

    # Now truncate further (if q = 1, this equals to total-order expansion)
    if q < 1:
        try:
            pow = 1 / q
        except ZeroDivisionError:
            pow = float('Inf')
        idp = np.sum(idx**q, 1) ** pow
        idx = idx[idp <= (nix + 0.000001), :]

    return idx


def MonCof(d, n):
    """
    This function is used to generate monomial coefficient
    input : d = degree
            n = dimension
    """
    temp = np.arange(1, (d+n))
    c = np.array(list(combinations(temp, n-1)))
    c = c.astype(int)
    m = np.size(c, 0)
    t = np.ones(shape=[m, d+n-1], dtype=int)
    tempt1 = np.transpose(np.array([np.arange(0, m, dtype=int)]))
    tempt2 = np.matlib.repmat(tempt1, 1, n-1)
    tempt3 = tempt2 + (c-1) * m
    for ii in range(np.size(tempt3, 0)):
        for jj in range(np.size(tempt3, 1)):
            x1 = int(np.round(tempt3[ii, jj] % m))
            x2 = int(np.floor(tempt3[ii, jj] / m))
            t[x1, x2] = 0
    tempu = np.vstack((np.zeros(m), np.transpose(t)))
    u = np.vstack((tempu, np.zeros(m)))
    v = np.cumsum(u, 0)
    tempreshape1 = np.array([np.transpose(v)[np.transpose(u) == 0]])
    tempreshape = np.reshape(tempreshape1, (n+1, m), order='F')
    x = np.transpose(np.diff(tempreshape, 1, 0))
    return x


def compute_regression_mat(idx, X, bounds, polytype):
    """
    Create regression matrix using polynomial chaos expansion for Kriging

    Inputs
      idx - Indices of polynomial basis.
      X - Experimental design
      bounds - Bounds of the experimental design (note that in create_Kriging.m the X is already normalized to [-1,1])
      polytpe - Type of polynomial (a 1 x nvar vector). Use 1 for Legendre polynomial (uniform distribution), 2 = Hermite polynomial (normal distribution)

    Output
      F - Regression matrix

     Author: Pramudita Satria Palar(pramsatriapalar@gmail.com, pramsp@ftmd.itb.ac.id)
    """
    nsamp = np.size(X, 0)
    nvar = np.size(X, 1)
    F = np.ones(shape=[nsamp, np.size(idx, 0)])
    N = np.zeros(shape=[nsamp, nvar])

    for po in range(0, nvar):
        if polytype[po] == 1:
            tempres = 2 * ((np.transpose([X[:, po]]) - np.matlib.repmat(bounds[0, po], nsamp, 1)) /
                           (np.matlib.repmat(bounds[1, po], nsamp, 1) - np.matlib.repmat(bounds[0, po], nsamp, 1))) - 1
            N[:, po] = tempres[:, 0]
        elif polytype[po] == 2:
            tempres = (np.transpose([X[:, po]]) - np.matlib.repmat(bounds[0, po], nsamp, 1)) / \
                      (np.sqrt(2 * np.matlib.repmat(bounds[1, po], nsamp, 1) ** 2))
            N[:, po] = tempres[:,0]

    for ii in range(0, np.size(idx, 0)):
        h1 = np.ones(shape=[nsamp])
        ids = idx[ii, :].astype(int)
        for jo in range(0, np.size(idx, 1)):
            if polytype[jo] == 1:
                h1 = h1 * np.polyval(legendre(ids[jo], -1, 1), N[:, jo]) / np.sqrt(1 / (2 * ids[jo] + 1))
            elif polytype[jo] == 2:
                h1 = h1*((1/(2**(ids[jo]/2)))*hermite(ids[jo],x=N[:,jo]) / np.sqrt(factorial(ids[jo])))
        F[:, ii] = h1
    return F


def polymin (p1, p2):
    if type(p1) is int or isinstance(p1, float):
        p1 = np.array([[p1]])
    if type(p2) is int or isinstance(p2, float):
        p2 = np.array([[p2]])
    if p1.ndim == 1:
        p1 = np.array([p1])
    if p2.ndim == 1:
        p2 = np.array([p2])

    p1l = np.size(p1, 1)
    p2l = np.size(p2, 1)

    if p2l > p1l:
        term1 = np.hstack((np.zeros(shape=[1, np.size(p2, 1) - np.size(p1, 1)]), p1))
        term2 = p2
    elif p2l < p1l:
        term1 = p1
        term2 = np.hstack((np.zeros(shape=[1, np.size(p1, 1) - np.size(p2, 1)]), p2))
    elif p2l == p1l:
        term1 = np.hstack((np.zeros(shape=[1, np.size(p2, 1) - np.size(p1, 1)]), p1))
        term2 = np.hstack((np.zeros(shape=[1, np.size(p1, 1) - np.size(p2, 1)]), p2))

    p = (term1 - term2)[0,:]
    return p


def legendre(n,lb,ub):
    """
     legendre: Compute the legendre polynomials

    Inputs:
      n - Order of Legendre polynomials.
      lb - lower bounds for Legendre polynomials
      ub - Upper bounds for Legendre polynomials

    Output:
      Le - Coefficients of Legendre polynomials
    """
    if n < 0:
        raise ValueError("The order of legendre polynomial must be greater than or equal to 0")

    if isinstance(n, int) == False:
        n = int(n)

    # Call legendre recursive function
    L0 = np.array([1])
    L1 = np.array([1, -((ub + lb) / 2)])

    if n == 0:
        Le = L0
    elif n == 1:
        Le = L1
    else:
        # Perform Gram Schmidt orthogonalization
        for i in range(0, n-1):
            if i == 0:
                K = deepcopy(L0)
                L = deepcopy(L1)

            a1 = np.polyval(np.polyint(np.convolve(np.array([1,0]), np.convolve(L, L))), ub) - \
                 np.polyval(np.polyint(np.convolve(np.array([1,0]), np.convolve(L, L))), lb)
            a2 = np.polyval(np.polyint(np.convolve(L, L)), ub) - np.polyval(np.polyint(np.convolve(L, L)), lb)
            a = a1 / a2
            h1 = np.convolve(polymin(np.array([1, 0]), a), L)

            b1 = np.polyval(np.polyint(np.convolve(L, L)), ub) - np.polyval(np.polyint(np.convolve(L, L)), lb)
            b2 = np.polyval(np.polyint(np.convolve(K, K)), ub) - np.polyval(np.polyint(np.convolve(K, K)), lb)
            b = b1 / b2
            h2 = np.convolve(b, K)

            Le = polymin(h1, h2)

            K = deepcopy(L)
            L = deepcopy(Le)

        cons = 1 / (np.polyval(Le, 1))
        Le = Le * cons

    return Le.tolist()

def hermite(n,x=None):
    """
    HERMITE: compute the Hermite polynomials.

    h = hermite(n)
    h = hermite(n,x)

    Inputs:
       - n is the order of the Hermite polynomial (n>=0).
       - x is (optional) values to be evaluated on the resulting Hermite
         polynomial function.

     There are two possible outputs:
     1. If x is omitted then h is an array with (n+1) elements that contains
        coefficients of each Hermite polynomial term.
        E.g. calling h = hermite(3)
        will result h = [8 0 -12 0], i.e. 8x^3 - 12x

     2. If x is given, then h = Hn(x) and h is the same size of x.
        E.g., H2(x) = 4x^2 - 2
        calling h = hermite(2,[0 1 2])
        will result h = [-2 2 14]

     More information:
     - about the Hermite polynomial: http://mathworld.wolfram.com/HermitePolynomial.html% - some examples of this function:
     http://suinotes.wordpress.com/2010/05/26/hermite-polynomials-with-matlab/

     Authors: Avan Suinesiaputra (avan.sp@gmail.com)
              Fadillah Z Tala    (fadil.tala@gmail.com)
     rev.
     26/05/2010 - first creation.
                - bug fixed: error when hermite(0,x) is called (x isn't empty)
     24/09/2010 - bug fixed: the size of x does match with y in line 50.
                  (thanks to Shiguo Peng)
    """

    # check n
    if n<0:
        raise ValueError("The order of legendre polynomial must be greater than or equal to 0")

    if type(n) is not int:
        raise TypeError("The order of legendre polynomial must be an integer.")

    # call hermite recursive function
    h = hermite_rec(n)

    # evaluate the hermite polynomial function, given x
    if x is not None:
        y = h[-1] * np.ones(np.shape(x))
        p = 1
        for i in range (len(h)-1,0,-1):
            y = y + h[i-1] * x**p
            p = p+1
        h = np.reshape(y,np.size(x))

    return h

def hermite_rec(n):
    """
    This is the reccurence construction of a Hermite polynomial, i.e.:
    H0(x) = 1
    H1(x) = 2x
    H[n+1](x) = 2x Hn(x) - 2n H[n-1](x)
    """
    id = 1 #For probabilist (this is it!), 2 for physicst polynomial
    if 0 == n:
        h = 1
    elif 1 == n:
        h = np.hstack((id,0))
    else:
        h1 = np.zeros(shape=[1,n+1])
        h1[0,0:n] = id*hermite_rec(n-1)

        h2 = np.zeros(shape=[1,n+1])
        h2[0,2:n+1] = id*(n-1)*hermite_rec(n-2)

        h = (h1 - h2)[0,:]
    return h