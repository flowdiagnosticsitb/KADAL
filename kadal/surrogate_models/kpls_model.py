from sklearn.cross_decomposition import PLSRegression as pls
from kadal.surrogate_models.kriging_model import Kriging,kriginfocheck
import cma
import logging

class KPLS(Kriging):
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

    def __init__(self, KrigInfo, ub=5, lb=-5, standardization=False, standtype="default", normy=True,
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

        self.type = 'kpls'
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

        # Calculate PLS coeff
        _pls = pls(self.n_princomp)
        if self.standardization is True:
            coeff_pls = _pls.fit(self.KrigInfo["X_norm"].copy(), self.KrigInfo['y'].copy()).x_rotations_
        else:
            coeff_pls = _pls.fit(self.KrigInfo["X"].copy(), self.KrigInfo['y'].copy()).x_rotations_
        self.KrigInfo["plscoeff"] = coeff_pls