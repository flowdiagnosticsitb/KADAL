import numpy as np
import multiprocessing as mp
from kadal.misc.sampling.samplingplan import sampling, standardize
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize, fmin_cobyla
from kadal.surrogate_models.supports.trendfunction import polytruncation, compute_regression_mat
from kadal.surrogate_models.supports.krigloocv import loocv
from kadal.surrogate_models.supports.krigloocv2 import loocv2
from kadal.surrogate_models.supports.likelihood_func import likelihood
from kadal.surrogate_models.supports.prediction import prediction
from kadal.optim_tools.ga.uncGA import uncGA
import cma
import logging


class Kriging:
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

    def __init__(self, KrigInfo, ub=5, lb=-5, standardization=False, standtype="default", normy=False,
                 trainvar=False, inherit=False):
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

        # Calculate KrigInfo['nsamp'] and KrigInfo['nvar']
        KrigInfo["nvar"] = np.size(KrigInfo['X'], 1)
        KrigInfo["nsamp"] = np.size(KrigInfo['X'], 0)

        if trainvar is True:
            self.nbhyp = KrigInfo["nvar"] + 1
        else:
            self.nbhyp = KrigInfo["nvar"]

        self.trainvar = trainvar
        self.normy = normy
        self.standardization = standardization
        self.standtype = standtype
        self.Y = KrigInfo["y"]
        self.X = KrigInfo["X"]
        self.lb = lb
        self.ub = ub

        if not inherit:
            self.type = 'kriging'
            KrigInfo['type'] = self.type
            KrigInfo, scaling = kriginfocheck(KrigInfo, lb, ub, self.nbhyp)
            self.KrigInfo = KrigInfo
            self.scaling = scaling  # Scaling for CMA-ES Optimizer, otherwise, unused.
            self.sigmacmaes = (ub-lb)/5  # Sigma for CMA-ES Optimizer, otherwise, unused.
            if self.standardization is True:
                self.standardize()
            else:
                pass
        else:
            pass

    def standardize(self):
        """
        Standardize Kriging samples and create regression matrix.

        Returns:
            None
        """
        # Create regression matrix
        self.KrigInfo['idx'] = polytruncation(self.KrigInfo["TrendOrder"], self.KrigInfo["nvar"], 1)

        # Standardize X and y
        if self.standardization is True:
            if self.standtype.lower() == "default":  # If standardization type is set to 'default'
                self.KrigInfo["normtype"] = "default"

                # Create normalization bound from -1 to 1
                bound = np.vstack((- np.ones(shape=[1, self.KrigInfo["nvar"]]),
                                   np.ones(shape=[1, self.KrigInfo["nvar"]])))

                # Normalize sample to -1 and 1
                if self.normy is True:  # If normalize y
                    self.KrigInfo["X_norm"], self.KrigInfo["y_norm"] = standardize(self.KrigInfo['X'], self.KrigInfo['y'],
                                                                                   type=self.standtype.lower(),
                                                                                   normy=True,
                                                                                   range=np.vstack(
                                                                                       (np.hstack((self.KrigInfo["lb"],
                                                                                                   np.min(self.Y))),
                                                                                        np.hstack((self.KrigInfo["ub"],
                                                                                                   np.max(self.Y))))))
                    self.KrigInfo["norm_y"] = True

                else:
                    self.KrigInfo["X_norm"] = standardize(self.KrigInfo['X'], self.KrigInfo['y'],
                                                          type=self.standtype.lower(),
                                                          range=np.vstack((self.KrigInfo["lb"], self.KrigInfo["ub"])))
                    self.KrigInfo["norm_y"] = False

            else:  # If standardization type is set to 'std'
                self.KrigInfo["normtype"] = "std"

                # create normalization with mean 0 and standard deviation 1
                if self.normy is True:
                    self.KrigInfo["X_norm"], self.KrigInfo["y_norm"], \
                    self.KrigInfo["X_mean"], self.KrigInfo["y_mean"], \
                    self.KrigInfo["X_std"], self.KrigInfo["y_std"] = standardize(
                        self.KrigInfo['X'], self.KrigInfo['y'], type=self.standtype.lower(), normy=True)

                    self.KrigInfo["norm_y"] = True
                else:
                    self.KrigInfo["X_norm"], self.KrigInfo["X_mean"], self.KrigInfo["X_std"] = \
                        standardize(self.KrigInfo['X'], self.KrigInfo['y'], type=self.standtype.lower())
                    self.KrigInfo["norm_y"] = False

                bound = np.vstack((np.min(self.KrigInfo["X_norm"], axis=0),
                                   np.max(self.KrigInfo["X_norm"], axis=0)))

            self.KrigInfo["standardization"] = True
            self.KrigInfo["F"] = compute_regression_mat(self.KrigInfo["idx"], self.KrigInfo["X_norm"], bound,
                                                        np.ones(shape=[self.KrigInfo["nvar"]]))
        else:
            self.KrigInfo["standardization"] = False
            self.KrigInfo["norm_y"] = False
            bound = np.vstack((np.min(self.KrigInfo["X"], axis=0),
                               np.max(self.KrigInfo["X"], axis=0)))
            self.KrigInfo["F"] = compute_regression_mat(self.KrigInfo["idx"], self.KrigInfo["X"], bound,
                                                        np.ones(shape=[self.KrigInfo["nvar"]]))

    def train(self, n_cpu=1, disp=True, pre_theta=None, pool=None):
        """
        Train Kriging model
        
        Args:
            n_cpu (bool): If > 1, uses parallel processing. Defaults
                to 1.
            disp (bool, optional): Print updates. Defaults to False.
            pre_theta #TODO: document this, if working.
            pool (int, optional): A multiprocessing.Pool instance.
                Will be passed to functions for use, if specified.
                Defaults to None.
        Returns:
            None
        """
        if disp:
            print("Begin train hyperparam.")

        # Isotropic gaussian kernel
        if self.KrigInfo["kernel"] == ["iso_gaussian"]:  # TODO: Should this be in a list?
            self.nbhyp = 1
        elif len(self.KrigInfo["kernel"]) != 1 and "iso_gaussian" in self.KrigInfo["kernel"]:
            raise NotImplementedError("Isotropic Gaussian kernel is not available for composite kernel")
        else:
            if len(self.KrigInfo["ubhyp"]) != self.nbhyp:
                self.nbhyp = len(self.KrigInfo["ubhyp"])

        # Create multiple starting points
        if self.KrigInfo['nrestart'] < 1:
            xhyp = self.nbhyp * [0]
        else:
            if self.nbhyp <= 40:
                _, xhyp = sampling('sobol', len(self.KrigInfo["ubhyp"]), self.KrigInfo['nrestart'],
                                   result="real", upbound=self.KrigInfo["ubhyp"], lobound=self.KrigInfo["lbhyp"])
            else:
                _, xhyp = sampling('sobolnew', len(self.KrigInfo["ubhyp"]), self.KrigInfo['nrestart'],
                                   result="real", upbound=self.KrigInfo["ubhyp"], lobound=self.KrigInfo["lbhyp"])

        # multiple starting from pre-trained theta:
        if pre_theta is not None:
            xhyp = np.random.rand(self.KrigInfo['nrestart']-1,len(self.KrigInfo["ubhyp"])) + pre_theta
            xhyp = np.vstack((pre_theta,xhyp))

        # Optimize hyperparam if number of hyperparameter is 1 using golden section method
        if self.nbhyp == 1:
            if self.KrigInfo["optimizer"] != "ga":
                res = minimize_scalar(likelihood, bounds=(self.lb, self.ub), method='golden', args=(self.KrigInfo,'default',
                                                                                                    self.trainvar) )
                if self.KrigInfo["kernel"] == ["iso_gaussian"]:
                    best_x = res.x
                else:
                    best_x = np.array([res.x])
                neglnlikecand = likelihood(best_x, self.KrigInfo, trainvar=self.trainvar)
            else:
                best_x, neglnlikecand,_ = uncGA(likelihood, lb=self.lb, ub=self.ub, npop=100, maxg=100,
                                              args=(self.KrigInfo,'default',self.trainvar))
            if disp:
                print(f"Best hyperparameter is {best_x}")
                print(f"With NegLnLikelihood of {neglnlikecand}")
        else:
            # Set Bounds and Constraints for Optimizer
            # Set Bounds for LBSGSB or SLSQP if one is used.
            if self.KrigInfo["optimizer"] == "lbfgsb" or self.KrigInfo["optimizer"] == "slsqp":
                optimbound = np.transpose(np.vstack((self.KrigInfo["lbhyp"], self.KrigInfo["ubhyp"])))
            # Set Constraints for Cobyla if used
            elif self.KrigInfo["optimizer"] == "cobyla":
                optimbound = []
                for i in range(len(self.KrigInfo["ubhyp"])):
                    # params aa and bb are not used, just to avoid error in Cobyla optimizer
                    optimbound.append(lambda x, Kriginfo, aa, bb, itemp=i: x[itemp] - self.KrigInfo["lbhyp"][itemp])
                    optimbound.append(lambda x, Kriginfo, aa, bb, itemp=i: self.KrigInfo["ubhyp"][itemp] - x[itemp])
            else:
                optimbound = None

            if disp:
                print(f"Training {self.KrigInfo['nrestart']} hyperparameter(s)")

            # Train hyperparams
            bestxcand, neglnlikecand = self.parallelopt(xhyp, n_cpu, optimbound,
                                                        disp=disp, pool=pool)

            # Search best hyperparams among the candidates
            I = np.argmin(neglnlikecand)
            best_x = bestxcand[I, :]

            if disp:
                print("Single Objective, train hyperparam, end.")
                print(f"Best hyperparameter is {best_x}")
                print(f"With NegLnLikelihood of {neglnlikecand[I]}")

            # Calculate Kriging model based on the best hyperparam.

        self.KrigInfo = likelihood(best_x,self.KrigInfo,mode='all',trainvar=self.trainvar)

    def loocvcalc(self, metrictype='mape',type='standard'):
        """
        Calculate Leave-one-out Cross Validation metric of Kriging model
        Args:
            metrictype (str) : Type of metric that want to be used. Defaults to MAPE
                available metrics are :
                    - 'e' : Error
                    - 'ae' : Absolute error
                    - 'mae' : Mean absolute error
                    - 'se' : Squared error
                    - 'mse' : Mean Squared error
                    - 'rmse' : Root Mean Squared error
                    - 're' : Relative error
                    - 'are' : Absolute relative error
                    - 'mare' : Mean absolute relative error
                    - 'sre' : Squared relative error
                    - 'msre' : Mean squared relative error
                    - 'rmsre' : Root mean squared relative error
                    - 'pe' : Percentage error
                    - 'ape' : Absolute percentage error
                    - 'mape' : Mean absolute percentage error
                    - 'spe' : Squared percentage error
                    - 'mspe' : Mean squared percentage error
                    - 'rmspe' : Root mean squared percentage error
                    - 'r2' : R2 criterion

        Returns:
            LOOCVerror : Value of chosen error metric.
            LOOCVpred : Prediction of LOOCV.

        """
        if type == 'fast':
            self.KrigInfo["LOOCVerror"],self.KrigInfo["LOOCVpred"] = loocv(self.KrigInfo, errtype=metrictype)
        elif type == 'standard':
            self.KrigInfo["LOOCVerror"], self.KrigInfo["LOOCVpred"] = loocv2(self.KrigInfo, errtype=metrictype)
        else:
            raise ValueError('"type" only accept fast or standard')
        return (self.KrigInfo["LOOCVerror"],self.KrigInfo["LOOCVpred"])

    def predict(self,x,predtypes=['pred'],drmmodel=None):
        """
        Prediction of Kriging model
        Args:
            x (nparray) : Prediction site (will be normalized to [-1,1])
            predtypes (list) :  Requested outputs at prediction site x.
                Valid predtypes are:
                    'pred' - for Kriging prediction.
                    'SSqr' - for Kriging prediction error.
                    'fpc' - Kriging trend function.
                    'lcb' -
                    'ebe' -
                    'EI' - for expected improvement.
                    'poi' -
                    'pof' -

        Returns:
            If only one output specified through predtypes, a single value
            or array is returned. Else a list of each output is returned.

        Raises:
            ValueError:
            KeyError:

        """
        result = prediction(x,self.KrigInfo,predtypes=predtypes, drm=drmmodel)
        return result

    def parallelopt(self,xhyp, n_cpu, optimbound, disp=True, pool=None):
        """
        Optimize hyperparameter using parallel processing

        Args:
            xhyp (nparray): Array of starting points.
            n_cpu (int): If > 1, uses parallel processing.
            loglvl (str): level of logging function.
            disp (bool, optional): Print progress to screen. Defaults
                to False.
            pool (mp.Pool, optional): A mp.Pool instance. If specified,
                it will be used for multiprocessing (overrides n_cpu).

         Returns:
             bestxcand (nparray): Array of best X candidates for each starting points.
             neglnlikecand (nparray): Array of corresponding Negative Ln-Likelihood value of best X candidates.
        """
        # Create array of solution candidate.
        bestxcand = np.zeros(shape=[self.KrigInfo['nrestart'], self.nbhyp])
        neglnlikecand = np.zeros(shape=[self.KrigInfo['nrestart']])

        # Try to identify number of core on machine fo multiprocessing
        # try:
        #     n_cpu = mp.cpu_count()
        #     skip_mp = False
        # except NotImplementedError:
        #     # No idea how many cores so just run sequentially
        #     skip_mp = True

        if n_cpu > 1 or pool is not None:
            # Calculate hyperparams in parallel
            print(f"Training in parallel on {n_cpu} cores.")

            hyperparam_inputs = []
            for ii in range(self.KrigInfo['nrestart']):
                xhyp_ii = xhyp[ii, :]
                hyperparam_inputs.append((self.KrigInfo, xhyp_ii, self.trainvar, self.KrigInfo['ubhyp'],
                                          self.KrigInfo['lbhyp'], self.sigmacmaes, self.scaling, optimbound))

            if pool is None:
                with mp.Pool(n_cpu) as pool:
                    results = pool.starmap(tune_hyperparameters,
                                           hyperparam_inputs)
            else:
                results = pool.starmap(tune_hyperparameters, hyperparam_inputs)

            # Collate results back into numpy arrays
            for i, (bestxcand_ii, neglnlikecand_ii) in enumerate(results):
                bestxcand[i] = bestxcand_ii
                neglnlikecand[i] = neglnlikecand_ii

        else:
            # Calculate hyperparams sequentially
            for ii in range(self.KrigInfo['nrestart']):

                if disp:
                    print(f'Training hyperparameter candidate no.{ii + 1}')

                xhyp_ii = xhyp[ii, :]
                p = (self.KrigInfo, xhyp_ii, self.trainvar,self.KrigInfo['ubhyp'], self.KrigInfo['lbhyp'],
                     self.sigmacmaes, self.scaling, optimbound)
                bestxcand_ii, neglnlikecand_ii = tune_hyperparameters(*p)
                bestxcand[ii, :] = bestxcand_ii
                neglnlikecand[ii] = neglnlikecand_ii

        return (bestxcand,neglnlikecand)


def tune_hyperparameters(KrigInfo, xhyp_ii, trainvar, ubhyp=None, lbhyp=None,
                         sigmacmaes=None, scaling=None, optimbound=None):
    """Estimate the best hyperparameters.

    Extracted hyperpamaeter tuning code into a function for
    parallelisation.

    Args:
        KrigInfo (dict): Dictionary that contains Kriging information.
        xhyp_ii (nparray): starting point number ii.
        ubhyp (nparray): upper bounds of hyperparams.
        lbhyp (nparray): lower bounds of hyperparams.
        sigmacmaes (float): initial sigma for cma-es.
        scaling (list): scaling for cma-es.
        optimbound: bounds for optimizer.

    Returns:
        bestxcand (np.array(float)): Best x candidate array
        neglnlikecand (float): Negative ln-likelihood candidate

    Raises:
        ValueError: If a required parameter for the chosen optimizer is
            missing.
    """
    if KrigInfo["optimizer"] == "cmaes":
        for p in (ubhyp, lbhyp, sigmacmaes, scaling):
            if p is None:
                raise ValueError(f'{p} must be set if optimizer is cmaes.')
        bestxcand, es = cma.fmin2(likelihood, xhyp_ii, sigmacmaes,
                                  {'bounds': [lbhyp.tolist(), ubhyp.tolist()],
                                   'scaling_of_variables': scaling,
                                   'verb_disp': 0, 'verbose': -9},
                                  args=(KrigInfo,'default',trainvar))
        neglnlikecand = es.result[1]

    elif KrigInfo["optimizer"] == "lbfgsb":
        if optimbound is None:
            raise ValueError('optimbound must be set if optimizer is lbfgsb.')
        res = minimize(likelihood, xhyp_ii, method='L-BFGS-B', options={'eps': 1e-03},
                       bounds=optimbound, args=(KrigInfo,'default',trainvar))
        bestxcand = res.x
        neglnlikecand = res.fun

    elif KrigInfo["optimizer"] == "slsqp":
        if optimbound is None:
            raise ValueError('optimbound must be set if optimizer is slsqp.')
        res = minimize(likelihood, xhyp_ii, method='SLSQP',
                       bounds=optimbound, args=(KrigInfo,'default',trainvar))
        bestxcand = res.x
        neglnlikecand = res.fun

    elif KrigInfo["optimizer"] == "cobyla":
        if optimbound is None:
            raise ValueError('optimbound must be set if optimizer is cobyla.')
        res = fmin_cobyla(likelihood, xhyp_ii, optimbound,
                          rhobeg=0.5, rhoend=1e-4, args=(KrigInfo,'default',trainvar))
        bestxcand = res
        neglnlikecand = likelihood(res, KrigInfo, trainvar=trainvar)

    else:
        msg = (f"{KrigInfo['optimizer']} in KrigInfo['Optimizer'] is not "
               f"recognised.")
        raise KeyError(msg)
    return bestxcand, neglnlikecand


def kriginfocheck(KrigInfo, lb, ub, nbhyp):
    """
    Function to check the Kriging information and set Kriging Information to default value if
    required parameters are not supplied.

    Args:
        KrigInfo: Dictionary that contains Kriging information.
        ub (int): log10 of hyperparam upper bound. Defaults to 5.
        lb (int): log10 of hyperparam lower bound. Defaults to -5.
        nbhyp (int): number of hyperparameter
        loglvl (str): level of logging function

    Returns:
        KrigInfo: Checked/Modified Kriging Information

    """
    eps = np.finfo(float).eps

    # Check if number of restart is specified. If not set to 1
    if 'nrestart' not in KrigInfo:
        KrigInfo['nrestart'] = 1
    elif KrigInfo['nrestart'] < 1:
        raise ValueError('Minimum value of KrigInfo["nrestart"] is 1')

    # Check if optimizer option is specified. If not set to lbfgsb
    if 'optimizer' not in KrigInfo:
        KrigInfo['optimizer'] = 'lbfgsb'
        print('The optimizer is not specified, set to lbfgsb')
    else:
        availoptmzr = ["lbfgsb", "cmaes", "cobyla", "slsqp","ga"]  # check if the specified optimizer is available
        if KrigInfo['optimizer'].lower() not in availoptmzr:
            raise ValueError(KrigInfo["optimizer"], " is not a valid optimizer.")

    # Check if Trend order is specified. If not set to zero
    if 'TrendOrder' not in KrigInfo:
        KrigInfo['TrendOrder'] = 0
    elif KrigInfo['TrendOrder'] < 0:
        raise ValueError("The order of the polynomial trend should be a positive value.")
    else:
        pass

    # Check if nugget settings are specified. If not, set to -6 and 'fixed'
    if "nugget" not in KrigInfo:
        KrigInfo["nugget"] = -6
        KrigInfo["nuggetparam"] = "fixed"
        print("Nugget is not defined, set nugget to 1e-06")
    elif type(KrigInfo["nugget"]) is not list:
        KrigInfo["nuggetparam"] = "fixed"
        nnugget = 1
    elif len(KrigInfo["nugget"]) == 2:
        KrigInfo["nuggetparam"] = "tuned"
        nnugget = len(KrigInfo["nugget"]);
        if KrigInfo["nuggetparam"][0] > KrigInfo["nuggetparam"][1]:
            raise TypeError("The lower bound of the nugget should be lower than the upper bound.")

    # Check Kernel Settings
    if "kernel" not in KrigInfo:
        KrigInfo["kernel"] = ["gaussian"]
        nkernel = 1
        print("Kernel is not defined, set kernel to gaussian")
    elif type(KrigInfo["kernel"]) is not list:
        nkernel = 1
        KrigInfo["kernel"] = [KrigInfo["kernel"]]
    else:
        nkernel = len(KrigInfo["kernel"])
    KrigInfo["nkernel"] = nkernel

    # Check if n_princomp appears
    if 'n_princomp' in KrigInfo and KrigInfo['type'] == 'kriging':
        KrigInfo['n_princomp'] = False
    elif 'n_princomp' not in KrigInfo and KrigInfo['type'] == 'kriging':
        KrigInfo['n_princomp'] = False
    elif 'n_princomp' in KrigInfo and KrigInfo['type'] == 'kpls':
        pass
    else:
        print("Check kriging_model.py line 485!")

    if 'limit' in KrigInfo:
        if 'limittype' not in KrigInfo:
            KrigInfo['limittype'] = '>='
            print("KrigInfo['limittype'] is set to greater than equal'>='.")
        else:
            pass

    # Check overall hyperparam
    lbhyp = lb * np.ones(shape=[nbhyp])
    ubhyp = ub * np.ones(shape=[nbhyp])
    if nnugget == 1 and nkernel == 1:  # Fixed nugget, one kernel function
        pass
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
    elif nnugget > 1 and nkernel == 1:  # Tunable nugget, one kernel function
        lbhyp = np.hstack((lbhyp, KrigInfo["nugget"][0]))
        ubhyp = np.hstack((ubhyp, KrigInfo["nugget"][1]))
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
        scaling[-1] = (KrigInfo["nugget"][1] - KrigInfo["nugget"][0]) / (ub - lb)
    elif nnugget == 1 and nkernel > 1:  # Fixed nugget, multiple kernel functions
        lbhyp = np.hstack((lbhyp, np.zeros(shape=[nkernel]) + eps))
        ubhyp = np.hstack((ubhyp, np.ones(shape=[nkernel])))
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
        scaling[-nkernel:] = 1 / (ub - lb)
    elif nnugget > 1 and nkernel > 1:  # Tunable nugget, multiple kernel functions
        lbhyp = np.hstack((lbhyp, KrigInfo["nugget"][0], np.zeros(shape=[nkernel]) + eps))
        ubhyp = np.hstack((ubhyp, KrigInfo["nugget"][1], np.ones(shape=[nkernel])))
        nbhyp = len(lbhyp)
        scaling = np.ones(nbhyp)
        scaling[-nkernel - 1] = (KrigInfo["nugget"][1] - KrigInfo["nugget"][0]) / (ub - lb)
        scaling[-nkernel:] = 1 / (ub - lb)
    KrigInfo["lbhyp"] = lbhyp
    KrigInfo["ubhyp"] = ubhyp

    return KrigInfo,scaling
