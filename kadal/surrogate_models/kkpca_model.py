import numpy as np
from kadal.misc.sampling.samplingplan import sampling, standardize
from sklearn.decomposition import KernelPCA as drm
from scipy.optimize import minimize, fmin_cobyla
from kadal.surrogate_models.supports.trendfunction import polytruncation, compute_regression_mat
from kadal.surrogate_models.kriging_model import Kriging, kriginfocheck
from scipy.spatial.distance import cdist
from copy import deepcopy

class KKPCA(Kriging):
    """
    Create Kriging model based on the information from inputs and global variables.
    Inputs:
    KrigInfo (dict): Containing necessary information to create a Kriging model
    ub (int): log10 of hyperparam upper bound. Defaults to 5.
    lb (int): log10 of hyperparam lower bound. Defaults to -5.
    standardization (bool): Perform standardization to the samples. Defaults to False.
    standtype (str): Type of standardization. Defaults to "default".
        available options are "default" and "std"
    normy (bool): True or False, normalize y or not
    trainvar (bool): True or False, train Kriging variance or not

    Outputs:
     KrigInfo (dict): Trained kriging model

    Details of KrigInfo:
    REQUIRED PARAMETERS. These parameters need to be specified manually by
    the user. Otherwise, the process cannot continue.
        - KrigInfo['lb'] - Variables' lower bounds.
        - KrigInfo['ub'] - Variables' upper bounds.
        - KrigInfo['nvar'] - Number of variables.
        - KrigInfo['nsamp'] - Number of samples.
        - KrigInfo['X'] - Experimental design.
        - KrigInfo['y'] - Responses of the experimental design.

    EXTRA PARAMETERS. These parameters can be set by the user. If not
    specified, default values will be used (or computed for the experimetntal design and responses)
        - KrigInfo['problem'] - Function name to evaluate responses (No need to specify this if KrigInfo.X and KrigInfo.Y are specified).
        - KrigInfo['nugget'] - Nugget (noise factor). Default: 1e-6
        - KrigInfo['TrendOrder'] - Polynomial trend function order (note that this code uses polynomial chaos expansion). Default: 0 (ordinary Kriging).
        - KrigInfo['kernel'] - Kernel function. Available kernels are 'gaussian', 'exponential','matern32', 'matern52', and 'cubic'. Default: 'Gaussian'.
        - KrigInfo['nrestart'] - Number of restarts for hyperparameters optimization. Default: 1.
        - KrigInfo['LOOCVtype'] - Type of cross validation error. Default: 'rmse'.
        - KrigInfo['optimizer'] - Optimizer for hyperparameters optimization. Default: 'lbfgsb'.

    """

    def __init__(self, KrigInfo, ub=2, lb=-3, standardization=False, standtype="default", normy=True,
                 trainvar=True):
        """
        Initialize Kriging model

       Args:
            KrigInfo (dict): Dictionary that contains Kriging information.
            ub (int): log10 of hyperparam upper bound. Defaults to 5.
            lb (int): log10 of hyperparam lower bound. Defaults to -5.
            standardization (bool): Perform standardization to the samples. Defaults to False.
            standtype (str): Type of standardization. Defaults to "default".
                available options are "default" and "std"
            normy (bool): True or False, normalize y or not
            trainvar (bool): True or False, train Kriging variance or not. Defaults to True.
        """
        if "n_princomp" not in KrigInfo:
            KrigInfo["n_princomp"] = 1
            self.n_princomp = KrigInfo["n_princomp"]
        else:
            self.n_princomp = KrigInfo["n_princomp"]

        super().__init__(KrigInfo, ub, lb , standardization, standtype, normy, trainvar, inherit=True)

        if trainvar is True:
            self.nbhyp = self.n_princomp + 1
        else:
            self.nbhyp = self.n_princomp

        self.type = 'kriging'
        KrigInfo['type'] = self.type
        KrigInfo, scaling = kriginfocheck(KrigInfo, lb, ub, self.nbhyp)
        self.KrigInfo = KrigInfo
        self.scaling = scaling  # Scaling for CMA-ES Optimizer, otherwise, unused.
        self.sigmacmaes = (ub - lb) / 5  # Sigma for CMA-ES Optimizer, otherwise, unused.

        if self.standardization is True:
            self.standardize()
        else:
            pass

    def standardize(self):
        """
        Standardize Kriging samples and create regression matrix.

        Returns:
            None
        """
        Kriging.standardize(self)


    def train(self, parallel = False, disp=True, KPCAkernel="poly"):
        """
        Train Kriging model

        Args:
            parallel (bool): Parallel processing or not. Default to False.
            disp (bool): Display process or not. Default to True.

        Returns:
            None
        """
        # Create starting points
        if KPCAkernel.lower() == 'poly':
            upwstart = np.array([2.5,2.5,7])
            lowwstart = np.array([-1,-1,1])
            _, wstart = sampling('sobol', len(upwstart), 1,
                                   result="real", upbound=upwstart, lobound=lowwstart)
            # wstart = np.array([0,0,0])
        elif KPCAkernel.lower() == 'sigmoid':
            upwstart = np.array([2.5, 2.5])
            lowwstart = np.array([-2, -2])
            wstart = np.array([0, 0])
        elif KPCAkernel.lower() == 'rbf':
            upwstart = np.array([1.5])
            lowwstart = np.array([-0.5])
            wstart = np.array([0])
        elif KPCAkernel.lower() == 'gaussian':
            upwstart = np.array([1.5]*self.KrigInfo["nvar"])
            lowwstart = np.array([-0.5]*self.KrigInfo["nvar"])
            wstart = np.array([0]*self.KrigInfo["nvar"])
        elif KPCAkernel.lower() == 'linear':
            upwstart = np.array([2.5])
            lowwstart = np.array([-2])
            wstart = np.array([0])
        else:
            raise ValueError(KPCAkernel.lower() + " kernel option is not a valid kernel")

        # Define hyperparams bounds
        optimbound = np.transpose(np.vstack((lowwstart, upwstart)))

        # Run optimization
        print("Optimize Hyperparams")
        if self.standardization is True:
            original_X = self.KrigInfo['X_norm']
            self.KrigInfo['orig_X'] = original_X
        else:
            original_X = self.KrigInfo['X']
            self.KrigInfo['orig_X'] = original_X
        res = minimize(self.kpcaopt, wstart, method='L-BFGS-B', options={'maxfun':50, 'eps':1e-4},
                       bounds=optimbound, args=(KPCAkernel,original_X))
        wopt = res.x

        drm, loocverr = self.kpcaopt(wopt,KPCAkernel,original_X,out='all')
        self.KrigInfo['kpcaw'] = wopt
        return drm,loocverr

    def kpcaopt(self,w,KPCAkernel,orig_X,out='default'):
        # Calculate PLS coeff
        if KPCAkernel != "gaussian" and KPCAkernel != "precomputed":
            if KPCAkernel.lower() == 'poly' or KPCAkernel.lower() == 'polynomial':
                _drm = drm(self.n_princomp, kernel='poly', gamma=10**w[0], coef0=10**w[1], degree=np.round(w[2]))
            elif KPCAkernel.lower() == 'sigmoid':
                _drm = drm(self.n_princomp, kernel='sigmoid', gamma=10**w[0], coef0=10**w[1])
            elif KPCAkernel.lower() == 'rbf':
                _drm = drm(self.n_princomp, kernel='rbf', gamma=10**w[0])
            elif KPCAkernel.lower() == 'linear' or KPCAkernel.lower() == 'cosine':
                _drm = drm(self.n_princomp)

            self.KrigInfo["nvar"] = self.n_princomp
            if self.standardization is True:
                self.KrigInfo["X_norm"] = deepcopy(orig_X)
                _drm.fit(self.KrigInfo["X_norm"].copy())
                transformed = _drm.transform(self.KrigInfo["X_norm"].copy())
                self.KrigInfo["lb2"] = (np.min(transformed, axis=0)) # Create lowerbound for transformed X
                self.KrigInfo["ub2"] = (np.max(transformed, axis=0))  # Create upperbound for transformed X
                self.KrigInfo["X_norm"] = standardize(transformed, self.KrigInfo['y'],
                                                      type=self.standtype.lower(),
                                                      range=np.vstack((self.KrigInfo["lb2"], self.KrigInfo["ub2"])))
                self.KrigInfo['idx'] = polytruncation(self.KrigInfo["TrendOrder"], self.KrigInfo["nvar"], 1)
            else:
                self.KrigInfo["X"] = deepcopy(orig_X)
                _drm.fit(self.KrigInfo["X"].copy())
                transformed = _drm.transform(self.KrigInfo["X"].copy())
                self.KrigInfo["X"] = transformed
                self.KrigInfo['idx'] = polytruncation(self.KrigInfo["TrendOrder"], self.KrigInfo["nvar"], 1)

        else:
            n_features = np.size(orig_X,1)
            self.KrigInfo["nvar"] = self.n_princomp
            _drm = drm(self.n_princomp,kernel='precomputed')
            k_mat = customkernel(orig_X,orig_X,w,n_features,type='gaussian')
            if self.standardization is True:
                self.KrigInfo["X_norm"] = deepcopy(orig_X)
                transformed = _drm.fit_transform(k_mat)
                self.KrigInfo["lb2"] = (np.min(transformed, axis=0))  # Create lowerbound for transformed X
                self.KrigInfo["ub2"] = (np.max(transformed, axis=0))  # Create upperbound for transformed X
                self.KrigInfo["X_norm"] = standardize(transformed, self.KrigInfo['y'],
                                                      type=self.standtype.lower(),
                                                      range=np.vstack((self.KrigInfo["lb2"], self.KrigInfo["ub2"])))
                self.KrigInfo['idx'] = polytruncation(self.KrigInfo["TrendOrder"], self.KrigInfo["nvar"], 1)
            else:
                pass

        if out == 'default':
            self.KrigInfo["kernel"] = ["iso_gaussian"]
            Kriging.train(self, disp=False)
        else:
            self.KrigInfo["kernel"] = ["gaussian"]
            Kriging.train(self, disp=False, pre_theta=self.KrigInfo['Theta'])

        loocverr, _ = Kriging.loocvcalc(self, drm=_drm)

        if out == 'default':
            return loocverr
        elif out == 'all':
            return _drm, loocverr

def customkernel(XN,XM,w,nvar,type='gaussian'):
    if type == 'gaussian':
        K = gausskernel(XN,XM,w,nvar)
    elif type == 'poly':
        K = polykernel(XN,XM,w,nvar)
    else:
        raise NotImplementedError('other type of kernel is not supported')

    return K

def polykernel(XN,XM,w,nvar):
    g = 10**w[0]
    c = 10**w[1]
    d = np.round(w[2])
    K = (g*np.dot(XN, XM.T)+c)**d
    return K

def gausskernel(XN,XM,w,nvar):
    w = 10**w
    mdist = np.zeros((np.size(XN, 0), np.size(XM, 0), nvar))
    for ii in range(0, nvar):
        X1 = np.transpose(np.array([XN[:, ii]]))
        X2 = np.transpose(np.array([XM[:, ii]]))
        mdist[:, :, ii] = (cdist(X1, X2, 'euclidean') ** 2) / (w[ii] ** 2)
    Psi = np.exp(-0.5 * np.sum(mdist, 2))
    return Psi
