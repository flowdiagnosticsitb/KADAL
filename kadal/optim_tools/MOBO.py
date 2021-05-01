import numpy as np
from copy import deepcopy
import scipy.io as sio
import multiprocessing as mp
from kadal.testcase.analyticalfcn.cases import evaluate
from kadal.surrogate_models.kriging_model import Kriging
from kadal.optim_tools import searchpareto
from kadal.optim_tools.parego import paregopre
from kadal.optim_tools.acquifunc_opt import run_single_opt,run_multi_opt
from kadal.surrogate_models.supports.trendfunction import compute_regression_mat
from kadal.surrogate_models.supports.likelihood_func import likelihood
import time


class MOBO:
    """
    Perform multi-objective Bayesian Optimization

    Args:
        moboInfo (dict): Dictionary containing necessary information
            for multi-objective Bayesian optimization.
        kriglist ([kriging_model.Kriging]): List of objective Kriging
            instances.
        autoupdate (bool, optional): Automatically continue evaluations
            of the objective functions. Defaults to True.
        multiupdate (int, optional): Number of suggested samples
            returned for each iteration. Defaults to 0.
        savedata (bool, optional): Save data for each iteration.
            Defaults to True.
        expconst ([kriging_model.Kriging], optional): Kriging instances
            for constraints. Defaults to None.
        chpconst ([func], optional): Constraint functions. Defaults to
            None. Expected output of the constraint functions is 1 if
            satisfied and 0 if not. The constraint functions MUST have
            an input of x (the decision variable to be evaluated).

    Returns:
        xupdate (np.ndarray): Array of design variables updates.
        yupdate (np.ndarray): Array of objectives updates
        metricall (np.ndarray): Array of metric values of the updates.
    """

    def __init__(self, moboInfo, kriglist, autoupdate=True, multiupdate=0,
                 savedata=True, expconst=None, chpconst=None):
        """
        Initialize MOBO class

        Args:
            moboInfo (dict): Dictionary containing necessary information
                for multi-objective Bayesian optimization.
            kriglist ([kriging_model.Kriging]): List of objective
                Kriging instances.
            autoupdate (bool, optional): Automatically continue
                evaluations of the objective functions. Defaults to
                True.
            multiupdate (int, optional): Number of suggested samples
                returned for each iteration. Defaults to 0.
            savedata (bool, optional): Save data for each iteration.
                Defaults to True.
            expconst ([kriging_model.Kriging], optional): Kriging
                instances for constraints. Defaults to None.
            chpconst ([func], optional): Constraint functions. Defaults
                to None. Expected output of the constraint functions is
                1 if satisfied and 0 if not. The constraint functions
                MUST have an input of x (the decision variable to be
                evaluated).
        """
        n_krig = len(kriglist)
        self.moboInfo = moboinfocheck(moboInfo, autoupdate, n_krig)
        self.kriglist = kriglist
        self.krignum = n_krig
        self.autoupdate = autoupdate
        self.multiupdate = multiupdate
        self.savedata = savedata
        self.krigconstlist = expconst
        self.cheapconstlist = chpconst

    def run(self, disp=True, infeasible=None, n_cpu=1):
        """
        Run multi objective unconstrained Bayesian optimization.

        Args:
            disp (bool, optional): Print progress. Defaults to True.
            infeasible (np.ndarray, optional): Indices of infeasible
                samples to delete. Defaults to None.
            n_cpu (int, optional): The number of processors to use in a
                multiprocessing.Pool. Default 1 will not run a pool.

        Returns:
            xalltemp (np.ndarray): [n_kb, n_dv] array of update design
                variable values.
            yalltemp (np.ndarray): [n_kb, n_obj] array of update
                objective values.
            salltemp (np.ndarray): [n_kb, n_obj] array of update
                objective uncertainty values.
            metricall (np.ndarray): [n_kb, 1] array of update metric
                values.
        """

        print(f'Running with n_cpu: {n_cpu} for supported functions.')
        if n_cpu == 1:
            return self._run(disp=disp, pool=None)
        else:
            with mp.Pool(processes=n_cpu) as pool:
                return self._run(disp=disp, pool=pool)

    def _run(self,disp=True,infeasible=None, pool=None):
        """
        Run multi objective unconstrained Bayesian optimization.

        Args:
            disp (bool, optional): Print progress. Defaults to True.
            infeasible (np.ndarray, optional): Indices of infeasible
                samples to delete. Defaults to None.
            pool (int, optional): A multiprocessing.Pool instance.
                Will be passed to functions for use, if specified.
                Defaults to None.

        Returns:
            xalltemp (np.ndarray): [n_kb, n_dv] array of update design
                variable values.
            yalltemp (np.ndarray): [n_kb, n_obj] array of update
                objective values.
            salltemp (np.ndarray): [n_kb, n_obj] array of update
                objective uncertainty values.
            metricall (np.ndarray): [n_kb, 1] array of update metric
                values.
        """
        self.nup = 0  # Number of current iteration
        self.Xall = self.kriglist[0].KrigInfo['X']
        n_samp = self.kriglist[0].KrigInfo["nsamp"]
        n_krig = len(self.kriglist)
        self.yall = np.zeros([n_samp, n_krig])
        for ii in range(n_krig):
            self.yall[:, ii] = self.kriglist[ii].KrigInfo["y"][:, 0]

        if infeasible is not None:
            self.yall = np.delete(self.yall.copy(), infeasible, 0)
            self.Xall = np.delete(self.Xall.copy(), infeasible, 0)

        self.ypar, _ = searchpareto.paretopoint(self.yall)

        print("Begin multi-objective Bayesian optimization process.")
        if self.autoupdate and disp:
            print(f"Update no.: {self.nup + 1}, F-count: {n_samp}, "
                  f"Maximum no. updates: {self.moboInfo['nup'] + 1}")

        # If the optimizer is ParEGO, create a scalarized Kriging
        if self.moboInfo['acquifunc'].lower() == 'parego':
            self.KrigScalarizedInfo = deepcopy(self.kriglist[0].KrigInfo)
            self.KrigScalarizedInfo['y'] = paregopre(self.yall)
            self.scalkrig = Kriging(self.KrigScalarizedInfo,
                                    standardization=True,
                                    standtype='default',
                                    normy=False,
                                    trainvar=False)
            self.scalkrig.train(disp=False, pool=pool)


        # Perform update on design space
        if self.moboInfo['acquifunc'].lower().startswith('ehvi'):
            self.ehviupdate(disp, pool=pool)
        elif self.moboInfo['acquifunc'].lower() == 'parego':
            self.paregoupdate(disp, pool=pool)
        else:
            raise ValueError(f"{self.moboInfo['acquifunc']} is not a valid "
                             f"acquisition function.")

        # Finish optimization and return values
        if disp:
            print("Optimization finished, now creating the final outputs.")

        xupdate = self.Xall[(-self.moboInfo['nup'] * self.multiupdate):, :]
        yupdate = self.yall[(-self.moboInfo['nup'] * self.multiupdate):, :]
        supdate = self.spredall[(-self.moboInfo['nup'] * self.multiupdate):, :]
        metricall = self.metricall

        return xupdate, yupdate, supdate, metricall


    def ehviupdate(self, disp=True, pool=None):
        """
        Update MOBO using EHVI algorithm.

        Args:
            disp (bool, optional): Print progress. Defaults to True.
            pool (int, optional): A multiprocessing.Pool instance.
                Will be passed to functions for use, if specified.
                Defaults to None.

        Returns:
             None
        """
        self.spredall = deepcopy(self.yall)
        self.spredall[:] = 0
        while self.nup < self.moboInfo['nup']:
            # Iteratively update the reference point for hypervolume computation
            # if EHVI is used as the acquisition function
            if self.moboInfo['refpointtype'].lower() == 'dynamic':
                rp = (np.max(self.yall, 0)
                      + (np.max(self.yall, 0) - np.min(self.yall, 0)) * 2)
                self.moboInfo['refpoint'] = rp

            # Perform update(s)
            if self.multiupdate < 1:
                raise ValueError("Number of multiple update must be > 1")
            else:
                res = self.simultpredehvi(disp=disp, pool=pool)
                xnext, yprednext, sprednext, metricnext = res

            if self.nup == 0:
                self.metricall = metricnext.reshape(-1, 1)
            else:
                self.metricall = np.vstack((self.metricall, metricnext))

            # Break Loop if auto is false
            if self.autoupdate is False:
                self.Xall = np.vstack((self.Xall, xnext))
                self.yall = np.vstack((self.yall, yprednext))
                self.spredall = np.vstack((self.spredall, sprednext))
                break

            # Evaluate and enrich experimental design
            self.enrich(xnext, pool=pool)

            # Update number of iterations
            self.nup += 1

            # Show optimization progress
            if disp:
                print(f"Update no.: {self.nup+1}, F-count: {np.size(self.Xall, 0)}, "
                      f"Maximum no. updates: {self.moboInfo['nup']+1}")


    def paregoupdate(self, disp=True, pool=None):
        """
        Update MOBO using ParEGO algorithm.

        Args:
            disp (bool, optional): Print progress. Defaults to True.
            pool (int, optional): A multiprocessing.Pool instance.
                Will be passed to functions for use, if specified.
                Defaults to None.

        Returns:
             None
        """
        while self.nup < self.moboInfo['nup']:
            # Perform update(s)
            if self.multiupdate < 0:
                raise ValueError("Number of multiple update must be greater or "
                                 "equal to 0")
            elif self.multiupdate in (0, 1):
                x_n, met_n = run_single_opt(self.scalkrig, self.moboInfo,
                                            krigconstlist=self.krigconstlist,
                                            cheapconstlist=self.cheapconstlist,
                                            pool=pool)
                xnext = x_n
                metricnext = met_n
                yprednext = np.zeros(shape=[2])
                for ii, krigobj in enumerate(self.kriglist):
                    yprednext[ii] = krigobj.predict(xnext, ['pred'])
            else:
                xnext, yprednext, metricnext = self.simultpredparego(pool=pool)

            if self.nup == 0:
                self.metricall = metricnext.reshape(-1, 1)
            else:
                self.metricall = np.vstack((self.metricall, metricnext))

            # Break Loop if auto is false
            if not self.autoupdate:
                self.Xall = np.vstack((self.Xall, xnext))
                self.yall = np.vstack((self.yall, yprednext))
                break

            # Evaluate and enrich experimental design
            self.enrich(xnext, pool=pool)

            # Update number of iterations
            self.nup += 1

            # Show optimization progress
            if disp:
                print(f"Update no.: {self.nup+1}, "
                      f"F-count: {np.size(self.Xall, 0)}, "
                      f"Maximum no. updates: {self.moboInfo['nup']+1}")


    def simultpredehvi(self, disp=False, pool=None):
        """
        Perform multi updates on EHVI MOBO using Kriging believer method.

        Args:
            disp (bool, optional): Print progress. Defaults to True.
            pool (int, optional): A multiprocessing.Pool instance.
                Will be passed to functions for use, if specified.
                Defaults to None.

        Returns:
            xalltemp (np.ndarray): [n_kb, n_dv] array of update design
                variable values.
            yalltemp (np.ndarray): [n_kb, n_obj] array of update
                objective values.
            salltemp (np.ndarray): [n_kb, n_obj] array of update
                objective uncertainty values.
            metricall (np.ndarray): [n_kb, 1] array of update metric
                values.
        """
        n_krig = len(self.kriglist)
        n_dv = self.kriglist[0].KrigInfo["nvar"]

        krigtemp = [deepcopy(obj) for obj in self.kriglist]
        yprednext = np.zeros([n_krig])
        sprednext = np.zeros([n_krig])

        xalltemp = np.empty([self.multiupdate, n_dv])
        yalltemp = np.empty([self.multiupdate, n_krig])
        salltemp = np.empty([self.multiupdate, n_krig])
        metricall = np.empty([self.multiupdate, 1])

        ypartemp = self.ypar
        yall = self.yall

        for ii in range(self.multiupdate):
            t1 = time.time()
            if disp:
                print(f"update number {ii+1}")

            xnext, metrictemp = run_multi_opt(krigtemp, self.moboInfo,
                                              ypartemp,
                                              krigconstlist=self.krigconstlist,
                                              cheapconstlist=self.cheapconstlist,
                                              pool=pool)

            bound = np.vstack((-np.ones([1, n_dv]), np.ones([1, n_dv])))

            for jj, krig in enumerate(krigtemp):
                yprednext[jj], sprednext[jj] = krig.predict(xnext, ['pred', 's'])
                krig.KrigInfo['X'] = np.vstack((krig.KrigInfo['X'], xnext))
                krig.KrigInfo['y'] = np.vstack((krig.KrigInfo['y'], yprednext[jj]))
                krig.standardize()
                krig.KrigInfo["F"] = compute_regression_mat(krig.KrigInfo["idx"],
                                                            krig.KrigInfo["X_norm"],
                                                            bound,
                                                            np.ones([n_dv]))
                krig.KrigInfo = likelihood(krig.KrigInfo['Theta'],
                                           krig.KrigInfo,
                                           mode='all',
                                           trainvar=krig.trainvar)

            xalltemp[ii, :] = xnext[:]
            yalltemp[ii, :] = yprednext[:]
            salltemp[ii, :] = sprednext[:]
            metricall[ii, :] = metrictemp

            yall = np.vstack((yall, yprednext))
            ypartemp, _ = searchpareto.paretopoint(yall)

            if disp:
                print(f"time: {time.time() - t1:.2f} s")

        return xalltemp, yalltemp, salltemp, metricall


    def simultpredparego(self, pool=None):
        """
        Perform multi updates on ParEGO MOBO by varying the weighting function.

        Args:
            pool (int, optional): A multiprocessing.Pool instance.
                Will be passed to functions for use, if specified.
                Defaults to None.

        Returns:
             xalltemp (nparray) : Array of design variables updates.
             yalltemp (nparray) : Array of objectives value updates.
             metricall (nparray) : Array of metric of the updates.
        """
        idxs = np.random.choice(11, self.multiupdate)
        scalinfotemp = deepcopy(self.KrigScalarizedInfo)
        xalltemp = self.Xall[:,:]
        yalltemp = self.yall[:,:]
        yprednext = np.zeros(shape=[len(self.kriglist)])

        for ii,idx in enumerate(idxs):
            print(f"update number {ii + 1}")
            scalinfotemp['X'] = xalltemp
            scalinfotemp['y'] = paregopre(yalltemp,idx)
            krigtemp = Kriging(scalinfotemp, standardization=True,
                               standtype='default', normy=False, trainvar=False)
            krigtemp.train(disp=False, pool=pool)
            x_n, met_n = run_single_opt(krigtemp, self.moboInfo,
                                        krigconstlist=self.krigconstlist,
                                        cheapconstlist=self.cheapconstlist,
                                        pool=pool)
            xnext = x_n
            metricnext = met_n
            for jj, krigobj in enumerate(self.kriglist):
                yprednext[jj] = krigobj.predict(xnext, ['pred'])
            if ii == 0:
                xallnext = deepcopy(xnext)
                yallnext = deepcopy(yprednext)
                metricall = deepcopy(metricnext)
            else:
                xallnext = np.vstack((xallnext, xnext))
                yallnext = np.vstack((yallnext, yprednext))
                metricall = np.vstack((metricall, metricnext))

        yalltemp = np.vstack((yalltemp,yprednext))
        xalltemp = np.vstack((xalltemp,xnext))

        return xallnext, yallnext, metricall


    def enrich(self, xnext, pool=None):
        """
        Evaluate and enrich experimental design.

        Args:
            xnext: Next design variable(s) to be evaluated.
            pool (int, optional): A multiprocessing.Pool instance.
                Will be passed to functions for use, if specified.
                Defaults to None.

        Returns:
            None
        """
        # Evaluate new sample
        obj_krig_problem = self.kriglist[0].KrigInfo['problem']
        if isinstance(obj_krig_problem, str):
            if np.ndim(xnext) == 1:
                ynext = evaluate(xnext, obj_krig_problem)
            else:
                ynext = np.zeros(shape=[np.size(xnext, 0), len(self.kriglist)])
                for ii in range(np.size(xnext,0)):
                    ynext[ii,:] = evaluate(xnext[ii,:],
                                           obj_krig_problem)
        elif callable(obj_krig_problem):
            ynext = obj_krig_problem(xnext)
        else:
            raise ValueError('KrigInfo["problem"] is not a string nor a '
                             'callable function!')

        if self.krigconstlist is not None:
            for idx, constobj in enumerate(self.krigconstlist):
                con_krig_problem = constobj.KrigInfo['problem']
                if isinstance(con_krig_problem, str):
                    ynext_const = evaluate(xnext, con_krig_problem)
                elif callable(con_krig_problem):
                    ynext_const = con_krig_problem(xnext).reshape(-1, 1)
                else:
                    raise ValueError('KrigConstInfo["problem"] is not a string '
                                     'nor a callable function!')
                constobj.KrigInfo['X'] = np.vstack((constobj.KrigInfo['X'], xnext))
                constobj.KrigInfo['y'] = np.vstack((constobj.KrigInfo['y'], ynext_const))
                constobj.standardize()
                constobj.train(disp=False, pool=pool)

        # Treatment for failed solutions, Reference : "Forrester, A. I., SÃ³bester, A., & Keane, A. J. (2006). Optimization with missing data.
        # Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 462(2067), 935-945."
        if np.isnan(ynext).any():
            for jj in range(len(self.kriglist)):
                SSqr, y_hat = self.kriglist[jj].predict(xnext, ['SSqr','pred'])
                ynext[0,jj] = y_hat + SSqr

        # Enrich experimental design
        self.yall = np.vstack((self.yall, ynext))
        self.Xall = np.vstack((self.Xall, xnext))
        self.ypar,I = searchpareto.paretopoint(self.yall)  # Recompute non-dominated solutions

        if self.moboInfo['acquifunc'].lower().startswith('ehvi'):
            for index, krigobj in enumerate(self.kriglist):
                krigobj.KrigInfo['X'] = self.Xall
                krigobj.KrigInfo['y'] = self.yall[:,index].reshape(-1,1)
                krigobj.standardize()
                krigobj.train(disp=False, pool=pool)
        elif self.moboInfo['acquifunc'] == 'parego':
            self.KrigScalarizedInfo['X'] = self.Xall
            self.KrigScalarizedInfo['y'] = paregopre(self.yall)
            self.scalkrig = Kriging(self.KrigScalarizedInfo,
                                    standardization=True, standtype='default',
                                    normy=False, trainvar=False)
            self.scalkrig.train(disp=False, pool=pool)
            for index, krigobj in enumerate(self.kriglist):
                krigobj.KrigInfo['X'] = self.Xall
                krigobj.KrigInfo['y'] = self.yall[:,index].reshape(-1,1)
                krigobj.standardize()
                krigobj.train(disp=False, pool=pool)
        else:
            raise NotImplementedError(self.moboInfo["acquifunc"],
                                      " is not a valid acquisition function.")

        # Save data
        if self.savedata:
            I = I.astype(int)
            Xbest = self.Xall[I,:]
            sio.savemat(self.moboInfo["filename"],
                        {"xbest": Xbest, "ybest": self.ypar})


def moboinfocheck(moboInfo, autoupdate, n_krig):
    """
    Function to check the MOBO information and set MOBO Information to default value if
    required parameters are not supplied.

    Args:
        moboInfo (dict): Structure containing necessary information
            for multi-objective Bayesian optimization.
        autoupdate (bool): True or False, depends on your decision
            to evaluate your function automatically or not.
        n_krig (int): The number of objective functions.

     Returns:
         moboInfo (dict): Checked/Modified MOBO Information
    """
    # Check necessary parameters
    if "nup" not in moboInfo:
        if autoupdate is True:
            raise ValueError("Number of updates for Bayesian optimization, moboInfo['nup'], is not specified")
        else:
            moboInfo["nup"] = 1
            print("Number of updates for Bayesian optimization has been set to 1")
    else:
        if autoupdate == True:
            pass
        else:
            moboInfo["nup"] = 1
            print("Manual mode is active, number of updates for Bayesian optimization is forced to 1")

    # Set default values
    if "acquifunc" not in moboInfo:
        moboInfo["acquifunc"] = "EHVI"
        print("The acquisition function is not specified, set to EHVI")
    else:
        availacqfun = ["ehvi", "ehvi_vec", "ehvi_kmac3d", "parego"]
        if moboInfo["acquifunc"].lower() not in availacqfun:
            raise ValueError(moboInfo["acquifunc"], " is not a valid acquisition function.")
        else:
            pass

    # Set necessary params for multiobjective acquisition function
    if moboInfo["acquifunc"].lower() in ("ehvi", "ehvi_vec", "ehvi_kmac3d"):
        if "refpoint" not in moboInfo:
            moboInfo["refpointtype"] = 'dynamic'
        else:
            moboInfo["refpointtype"] = 'static'
            if len(moboInfo["refpoint"]) != n_krig:
                msg = (f'refpoint {moboInfo["refpoint"]} should have the same '
                       f'dimension as the number of Kriging models ({n_krig})!')
                raise ValueError(msg)
        
        if 'refpointtype' in moboInfo:
            refpointavail = ['dynamic','static']
            if moboInfo["refpointtype"].lower() not in refpointavail:
                raise ValueError(moboInfo["refpointtype"],' is not valid type')

        if n_krig == 2 and moboInfo["acquifunc"].lower() not in ("ehvi", "ehvi_vec"):
            msg = 'For 2-objective optimization, set "acquifunc" to "ehvi" or "ehvi_vec"'
            raise NotImplementedError(msg)
        elif n_krig == 3 and moboInfo["acquifunc"].lower() != 'ehvi_kmac3d':
            msg = 'For 3-objective optimization, set "acquifunc" to "ehvi_kmac3d"'
            raise NotImplementedError(msg)

    elif moboInfo["acquifunc"].lower() == "parego":
        moboInfo["krignum"] = 1
        if "paregoacquifunc" not in moboInfo:
            moboInfo["paregoacquifunc"] = "EI"

    # If moboInfo['acquifuncopt'] (optimizer for the acquisition function) is not specified set to 'sampling+cmaes'
    if "acquifuncopt" not in moboInfo:
        moboInfo["acquifuncopt"] = "lbfgsb"
        print("The acquisition function optimizer is not specified, set to L-BFGS-B.")
    else:
        availableacqoptimizer = ['lbfgsb', 'cobyla', 'cmaes', 'ga', 'diff_evo', 'fcmaes']
        if moboInfo["acquifuncopt"].lower() not in availableacqoptimizer:
            raise ValueError(moboInfo["acquifuncopt"], " is not a valid acquisition function optimizer.")
        else:
            pass

    if "nrestart" not in moboInfo:
        moboInfo["nrestart"] = 1
        print(
            "The number of restart for acquisition function optimization is not specified, setting BayesInfo.nrestart to 1.")
    else:
        if moboInfo["nrestart"] < 1:
            raise ValueError("BayesInfo['nrestart'] should be at least one")
        print("The number of restart for acquisition function optimization is specified to ",
              moboInfo["nrestart"], " by user")

    if "filename" not in moboInfo:
        moboInfo["filename"] = "temporarydata.mat"
        print("The file name for saving the results is not specified, set the name to temporarydata.mat")
    else:
        print("The file name for saving the results is not specified, set the name to ", moboInfo["filename"])

    if "ehvisampling" not in moboInfo:
        moboInfo['ehvisampling'] = 'default'
        print("EHVI sampling is not specified, set sampling to default")
    else:
        availsamp = ['default','efficient']
        if moboInfo['ehvisampling'].lower() not in availsamp:
            raise ValueError(moboInfo['ehvisampling'], ' is not a valid option')
        else:
            print("EHVI sampling is set to ", moboInfo['ehvisampling'])

    if "n_cpu" not in moboInfo:
        moboInfo['n_cpu'] = 1
        print("n_cpu not specified, set to 1")
    else:
        if moboInfo["n_cpu"] < 1:
            raise ValueError("BayesInfo['n_cpu'] should be at least one")
        print(f"n_cpu is set to {moboInfo['n_cpu']} by user")

    return moboInfo
