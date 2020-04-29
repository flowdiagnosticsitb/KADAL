import numpy as np

def evaluate(X,type):

    if X.ndim == 1:
        X = np.array([X])
    nsample = np.size(X, 0)
    y = np.zeros(shape=[nsample, 1])
    if type.lower() == "branin":
        for ii in range(0, nsample):
            y[ii, 0] = branin(X[ii, :])
    elif type.lower() == "styblinski":
        for ii in range(0, nsample):
            y[ii, 0] = styb(X[ii, :])
    elif type.lower() == "sasena":
        y = sasena(X)
    elif type.lower() == "griewank":
        y = griewank(X)
    elif type.lower() == "schaffer":
        y = schaffer(X)
    elif type.lower() == "schaffer1":
        y = schaffer1(X)
    elif type.lower() == "fonseca":
        y = fonseca(X)
    elif type.lower() == "ishigami":
        y = ishigami(X)
    else:
        raise NameError("Test case unavailable!")
    return y

# 2D Cases
def styb(x):
    d = len(x)
    sum = 0
    for ii in range (0,d):
        xi = x[ii]
        new = xi**4 - 16*xi**2 + 5*xi
        sum = sum + new
    y = sum/2
    return y

def branin (x):
    a = 5.1/(4 * (np.pi)**2 )
    b = 5/ np.pi
    c = (1-(1/(8 * np.pi)))

    f = (x[1] - a*x[0]**2 + b*x[0] - 6)**2 + 10*(c* np.cos(x[0]) +1)
    return f

def sasena (x):
    """
    SASENA function (two variables).

    Input
      X - (nsamp x nvar) matrix of experimental design.

    Output
      Y - (nsamp x 1) vector of responses.
    """
    Y = np.zeros(shape=[np.size(x,0),1])
    for ii in range(0,np.size(x,0)):
        xtemp = x[ii,:]
        x1 = xtemp[0]
        x2 = xtemp[1]

        Y[ii,0] = 2 + 0.01*(x2 - x1**2)**2 + (1-x1)**2 + 2*(2-x2)**2 + 7 * np.sin(0.5*x1) * np.sin(0.7*x1*x2)
    return Y

def griewank (x):
    d = np.size(x,1)
    nn = np.size(x,0)
    Y = np.zeros(shape=[nn,1])

    for ii in range(0,nn):
        sum = 0
        prod = 1
        for jj in range(0,d):
            xi = x[ii,jj]
            sum = sum + xi**2 / 4000
            prod = prod * np.cos(xi/np.sqrt(jj+1))
        Y[ii,0] = sum - prod + 1

    return Y

#1D Cases
def case10(x):
    #bound [0,10]
    f = -x * np.sin(x)
    return f

#Multi-objective cases
def schaffer (x):
    """
    Generalized Schaffer problem
    Reference: "Emmerich, M. T., & Deutz, A. H. (2007, March). Test problems
    based on LamÃ© superspheres. In International Conference on Evolutionary
    Multi-Criterion Optimization (pp. 922-936). Springer, Berlin, Heidelberg."

    Inputs:
      X:  Vector of decision variables
      r:  Describes the shape of the Pareto front

    Output:
      fitness:    fitness function value

    Written by Kaifeng Yang, 20/1/2016
    """
    r = 1
    a = 1/(2*r)
    m = np.size(x,0)
    n = np.size(x,1)
    fitness = np.zeros(shape=[m,2])
    for i in range(0,m):
        fitness[i,0] = (1/(n**a))*np.sum(x[i,:]**2)**a
        fitness[i,1] = (1/(n**a))*np.sum((1-x[i,:])**2)**a
    if m == 1:
        fitness = fitness[0,:]
    return fitness

def fonseca (x):
    b = np.size(x[:,0])
    n = 2
    m = np.size(x, 0)
    one = np.ones(shape=[b,1])
    sum1 = np.zeros(shape=[b,1]);f1 = np.zeros(shape=[b,1])
    sum2 = np.zeros(shape=[b,1]);f2 = np.zeros(shape=[b,1])
    c  = np.ones(shape=[n])*(1/np.sqrt(n))
    for jj in range (0,b):
        sum1[jj,0] = -1*np.sum((x[jj,:]-(c))**2)
        sum2[jj,0] = -1*np.sum((x[jj,:]+(c))**2)
        f1[jj,0] = one[jj,0] - np.exp(sum1[jj,0])
        f2[jj,0] = one[jj,0] - np.exp(sum2[jj,0])
    f = np.hstack((f1,f2))
    if m == 1:
        f = f[0,:]
    return f

def schaffer1 (x):
    b = np.size(x[:,0])
    m = np.size(x, 0)
    f1 = np.zeros(shape=[b, 1])
    f2 = np.zeros(shape=[b, 1])
    for jj in range (0,b):
        f1[jj,0] = x[jj,0]**2
        f2[jj,0] = (x[jj,0]-2)**2
    f = np.hstack((f1, f2))
    if m == 1:
        f = f[0, :]
    return f

def ishigami(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    a = 7
    b = 0.1
    f = np.sin(x1) + a * (np.sin(x2))**2 + b * (np.sin(x1) * x3**4)
    return f.reshape(-1,1)
