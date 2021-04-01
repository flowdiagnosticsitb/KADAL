import numpy as np
from copy import deepcopy
import scipy.io as sio
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
        moboInfo (dict): Dictionary containing necessary information for multi-objective Bayesian optimization.
        kriglist (list): List of Kriging object.
        autoupdate (bool): True or False, depends on your decision to evaluate your function automatically or not.
        multiupdate (int): Number of suggested samples returned for each iteration.
        expconst (list): List of constraints Kriging object.
        chpconst (list): List of cheap constraint function.

    Returns:
        xupdate (nparray): Array of design variables updates.
        yupdate (nparray): Array of objectives updates
        metricall (nparray): Array of metric values of the updates.
    """

    def __init__(self, moboInfo, kriglist, autoupdate=True, multiupdate=0, savedata=True, expconst=None, chpconst=None):
        """
        Initialize MOBO class

        Args:
            moboInfo (dict): Dictionary containing necessary information for multi-objective Bayesian optimization.
            kriglist (list): List of Kriging object.
            autoupdate (bool): True or False, depends on your decision to evaluate your function automatically or not.
            multiupdate (int): Number of suggested samples returned for each iteration.
            savedata (bool): Save data for each iteration or not. Defaults to True.
            expconst (list): List of constraints Kriging object.
            chpconst (list): List of cheap constraint function.

        """
        self.moboInfo = moboinfocheck(moboInfo, autoupdate)
        self.kriglist = kriglist
        self.krignum = len(self.kriglist)
        self.autoupdate = autoupdate
        self.multiupdate = multiupdate
        self.savedata = savedata
        self.krigconstlist = expconst
        self.cheapconstlist = chpconst

    def run(self,disp=True,infeasible=None):
        """
        Run multi objective unconstrained Bayesian optimization.

        Args:
            disp (bool): Display process or not. Defaults to True.
            infeasible (np.ndarray): Indices of infeasible to samples
                to delete. Defaults to None.

        Returns:
            xupdate (nparray): Array of design variables updates.
            yupdate (nparray): Array of objectives updates
            metricall (nparray): Array of metric values of the updates.

        """

        self.nup = 0  # Number of current iteration
        self.Xall = self.kriglist[0].KrigInfo['X']
        self.yall = np.zeros(shape=[np.size(self.kriglist[0].KrigInfo["y"],axis=0),len(self.kriglist)])
        for ii in range(np.size(self.yall,axis=1)):
            self.yall[:,ii] = self.kriglist[ii].KrigInfo["y"][:,0]

        if infeasible is not None:
            self.yall = np.delete(self.yall.copy(),infeasible,0)
            self.Xall = np.delete(self.Xall.copy(), infeasible, 0)
        else:
            pass

        self.ypar,_ = searchpareto.paretopoint(self.yall)

        print("Begin multi-objective Bayesian optimization process.")
        if self.autoupdate and disp:
            print(f"Update no.: {self.nup+1}, F-count: {np.size(self.Xall,0)}, "
                  f"Maximum no. updates: {self.moboInfo['nup']+1}")
        else:
            pass

        # If the optimizer is ParEGO, create a scalarized Kriging
        if self.moboInfo['acquifunc'].lower() == 'parego':
            self.KrigScalarizedInfo = deepcopy(self.kriglist[0].KrigInfo)
            self.KrigScalarizedInfo['y'] = paregopre(self.yall)
            self.scalkrig = Kriging(self.KrigScalarizedInfo,standardization=True,standtype='default',normy=False,
                                    trainvar=False)
            self.scalkrig.train(disp=False)
        else:
            pass

        # Perform update on design space
        if self.moboInfo['acquifunc'].lower() in ('ehvi', 'ehvi_vec'):
            self.ehviupdate(disp)
        elif self.moboInfo['acquifunc'].lower() == 'parego':
            self.paregoupdate(disp)
        else:
            raise ValueError(self.moboInfo["acquifunc"], " is not a valid acquisition function.")

        # Finish optimization and return values
        if disp:
            print("Optimization finished, now creating the final outputs.")

        if self.multiupdate == 0 or self.multiupdate == 1:
            xupdate = self.Xall[-self.moboInfo['nup']:, :]
            yupdate = self.yall[-self.moboInfo['nup']:, :]
            supdate = self.spredall[-self.moboInfo['nup']:, :]
        else:
            xupdate = self.Xall[(-self.moboInfo['nup']*self.multiupdate):, :]
            yupdate = self.yall[(-self.moboInfo['nup']*self.multiupdate):, :]
            supdate = self.spredall[(-self.moboInfo['nup']*self.multiupdate):, :]
        metricall = self.metricall

        return xupdate,yupdate,supdate,metricall


    def ehviupdate(self, disp):
        """
        Update MOBO using EHVI algorithm.

        Args:
            disp (bool): Display process or not.

        Returns:
             None
        """
        self.spredall = deepcopy(self.yall)
        self.spredall[:] = 0
        while self.nup < self.moboInfo['nup']:
            # Iteratively update the reference point for hypervolume computation if EHVI is used as the acquisition function
            if self.moboInfo['refpointtype'].lower() == 'dynamic':
                self.moboInfo['refpoint'] = np.max(self.yall,0)+(np.max(self.yall,0)-np.min(self.yall,0))*2

            # Perform update(s)
            if self.multiupdate < 0:
                raise ValueError("Number of multiple update must be greater or equal to 0")
            elif self.multiupdate == 0 or self.multiupdate == 1:
                xnext, metricnext = run_multi_opt(self.kriglist, self.moboInfo, self.ypar, self.krigconstlist,
                                                  self.cheapconstlist)
                yprednext = np.zeros(shape=[2])
                sprednext = np.zeros(shape=[2])
                for ii,krigobj in enumerate(self.kriglist):
                    yprednext[ii], sprednext[ii] = krigobj.predict(xnext,['pred','s'])
            else:
                xnext, yprednext, sprednext, metricnext = self.simultpredehvi(disp)

            if self.nup == 0:
                self.metricall = metricnext
            else:
                self.metricall = np.vstack((self.metricall,metricnext))

            # Break Loop if auto is false
            if self.autoupdate is False:
                self.Xall = np.vstack((self.Xall, xnext))
                self.yall = np.vstack((self.yall, yprednext))
                self.spredall = np.vstack((self.spredall, sprednext))
                break
            else:
                pass

            # Evaluate and enrich experimental design
            self.enrich(xnext)

            # Update number of iterations
            self.nup += 1

            # Show optimization progress
            if disp:
                print(f"Update no.: {self.nup+1}, F-count: {np.size(self.Xall, 0)}, "
                      f"Maximum no. updates: {self.moboInfo['nup']+1}")


    def paregoupdate(self, disp):
        """
        Update MOBO using ParEGO algorithm.

        Args:
            disp (bool): Display process or not.

        Returns:
             None
        """
        while self.nup < self.moboInfo['nup']:
            # Perform update(s)
            if self.multiupdate < 0:
                raise ValueError("Number of multiple update must be greater or equal to 0")
            elif self.multiupdate == 0 or self.multiupdate == 1:
                xnext, metricnext = run_single_opt(self.scalkrig,self.moboInfo,self.krigconstlist,self.cheapconstlist)
                yprednext = np.zeros(shape=[2])
                for ii, krigobj in enumerate(self.kriglist):
                    yprednext[ii] = krigobj.predict(xnext, ['pred'])
            else:
                xnext, yprednext, metricnext = self.simultpredparego()

            if self.nup == 0:
                self.metricall = metricnext
            else:
                self.metricall = np.vstack((self.metricall,metricnext))

            # Break Loop if auto is false
            if self.autoupdate is False:
                self.Xall = np.vstack((self.Xall, xnext))
                self.yall = np.vstack((self.yall, yprednext))
                break
            else:
                pass

            # Evaluate and enrich experimental design
            self.enrich(xnext)

            # Update number of iterations
            self.nup += 1

            # Show optimization progress
            if disp:
                print(f"Update no.: {self.nup+1}, F-count: {np.size(self.Xall, 0)}, "
                      f"Maximum no. updates: {self.moboInfo['nup']+1}")


    def simultpredehvi(self,disp=False):
        """
        Perform multi updates on EHVI MOBO using Kriging believer method.

        Returns:
             xalltemp (nparray) : Array of design variables updates.
             yalltemp (nparray) : Array of objectives value updates.
             metricall (nparray) : Array of metric of the updates.
        """

        krigtemp = [0]*len(self.kriglist)
        for index,obj in enumerate(self.kriglist):
            krigtemp[index] = deepcopy(obj)
        yprednext = np.zeros(shape=[len(krigtemp)])
        sprednext = np.zeros(shape=[len(krigtemp)])
        ypartemp = self.ypar
        yall = self.yall

        for ii in range(self.multiupdate):
            t1 = time.time()
            if disp:
                print(f"update number {ii+1}")
            else:
                pass

            xnext, metrictemp = run_multi_opt(krigtemp, self.moboInfo, ypartemp, self.krigconstlist,
                                              self.cheapconstlist)
            bound = np.vstack((- np.ones(shape=[1, krigtemp[0].KrigInfo["nvar"]]),
                               np.ones(shape=[1, krigtemp[0].KrigInfo["nvar"]])))

            for jj in range(len(krigtemp)):
                yprednext[jj], sprednext[jj] = krigtemp[jj].predict(xnext,['pred','s'])
                krigtemp[jj].KrigInfo['X'] = np.vstack((krigtemp[jj].KrigInfo['X'], xnext))
                krigtemp[jj].KrigInfo['y'] = np.vstack((krigtemp[jj].KrigInfo['y'], yprednext[jj]))
                krigtemp[jj].standardize()
                krigtemp[jj].KrigInfo["F"] = compute_regression_mat(krigtemp[jj].KrigInfo["idx"],
                                                                    krigtemp[jj].KrigInfo["X_norm"], bound,
                                                                    np.ones(shape=[krigtemp[jj].KrigInfo["nvar"]]))
                krigtemp[jj].KrigInfo = likelihood(krigtemp[jj].KrigInfo['Theta'], krigtemp[jj].KrigInfo, mode='all',
                                                   trainvar=krigtemp[jj].trainvar)

            if ii == 0:
                xalltemp = deepcopy(xnext)
                yalltemp = deepcopy(yprednext)
                salltemp = deepcopy(sprednext)
                metricall = deepcopy(metrictemp)
            else:
                xalltemp = np.vstack((xalltemp,xnext))
                yalltemp = np.vstack((yalltemp,yprednext))
                salltemp = np.vstack((salltemp,sprednext))
                metricall = np.vstack((metricall,metrictemp))

            yall = np.vstack((yall,yprednext))
            ypartemp,_ = searchpareto.paretopoint(yall)

            if disp:
                print("time: ",time.time()-t1," s")

        return [xalltemp,yalltemp,salltemp,metricall]


    def simultpredparego(self):
        """
        Perform multi updates on ParEGO MOBO by varying the weighting function.

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
            krigtemp = Kriging(scalinfotemp, standardization=True, standtype='default', normy=False,
                               trainvar=False)
            krigtemp.train(disp=False)
            xnext, metricnext = run_single_opt(krigtemp,self.moboInfo,self.krigconstlist,self.cheapconstlist)
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


    def enrich(self,xnext):
        """
        Evaluate and enrich experimental design.

        Args:
            xnext: Next design variable(s) to be evaluated.

        Returns:
            None
        """
        # Evaluate new sample
        if type(self.kriglist[0].KrigInfo['problem']) == str:
            if np.ndim(xnext) == 1:
                ynext = evaluate(xnext, self.kriglist[0].KrigInfo['problem'])
            else:
                ynext = np.zeros(shape=[np.size(xnext, 0), len(self.kriglist)])
                for ii in range(np.size(xnext,0)):
                    ynext[ii,:] = evaluate(xnext[ii,:],  self.kriglist[0].KrigInfo['problem'])
        elif callable(self.kriglist[0].KrigInfo['problem']):
            ynext = self.kriglist[0].KrigInfo['problem'](xnext)
        else:
            raise ValueError('KrigInfo["problem"] is not a string nor a callable function!')

        if self.krigconstlist is not None:
            for idx, constobj in enumerate(self.krigconstlist):
                if type(constobj.KrigInfo['problem']) == str:
                    ynext_const = evaluate(xnext, constobj.KrigInfo['problem'])
                elif callable(constobj.KrigInfo['problem']):
                    ynext_const = constobj.KrigInfo['problem'](xnext).reshape(-1,1)
                else:
                    raise ValueError('KrigConstInfo["problem"] is not a string nor a callable function!')
                constobj.KrigInfo['X'] = np.vstack((constobj.KrigInfo['X'], xnext))
                constobj.KrigInfo['y'] = np.vstack((constobj.KrigInfo['y'], ynext_const))
                constobj.standardize()
                constobj.train(disp=False)
        else:
            pass

        # Treatment for failed solutions, Reference : "Forrester, A. I., SÃ³bester, A., & Keane, A. J. (2006). Optimization with missing data.
        # Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 462(2067), 935-945."
        if np.isnan(ynext).any() is True:
            for jj in range(len(self.kriglist)):
                SSqr, y_hat = self.kriglist[jj].predict(xnext, ['SSqr','pred'])
                ynext[0,jj] = y_hat + SSqr

        # Enrich experimental design
        self.yall = np.vstack((self.yall, ynext))
        self.Xall = np.vstack((self.Xall, xnext))
        self.ypar,I = searchpareto.paretopoint(self.yall)  # Recompute non-dominated solutions

        if self.moboInfo['acquifunc'] == 'ehvi':
            for index, krigobj in enumerate(self.kriglist):
                krigobj.KrigInfo['X'] = self.Xall
                krigobj.KrigInfo['y'] = self.yall[:,index].reshape(-1,1)
                krigobj.standardize()
                krigobj.train(disp=False)
        elif self.moboInfo['acquifunc'] == 'parego':
            self.KrigScalarizedInfo['X'] = self.Xall
            self.KrigScalarizedInfo['y'] = paregopre(self.yall)
            self.scalkrig = Kriging(self.KrigScalarizedInfo, standardization=True, standtype='default', normy=False,
                               trainvar=False)
            self.scalkrig.train(disp=False)
            for index, krigobj in enumerate(self.kriglist):
                krigobj.KrigInfo['X'] = self.Xall
                krigobj.KrigInfo['y'] = self.yall[:,index].reshape(-1,1)
                krigobj.standardize()
                krigobj.train(disp=False)
        else:
            raise ValueError(self.moboInfo["acquifunc"], " is not a valid acquisition function.")

        # Save data
        if self.savedata:
            I = I.astype(int)
            Xbest = self.Xall[I,:]
            sio.savemat(self.moboInfo["filename"],{"xbest":Xbest,"ybest":self.ypar})


def moboinfocheck(moboInfo, autoupdate):
    """
    Function to check the MOBO information and set MOBO Information to default value if
    required parameters are not supplied.

    Args:
         moboInfo (dict): Structure containing necessary information for multi-objective Bayesian optimization.
         autoupdate (bool): True or False, depends on your decision to evaluate your function automatically or not.

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
        availacqfun = ["ehvi", "ehvi_vec", "parego"]
        if moboInfo["acquifunc"].lower() not in availacqfun:
            raise ValueError(moboInfo["acquifunc"], " is not a valid acquisition function.")
        else:
            pass

    # Set necessary params for multiobjective acquisition function
    if moboInfo["acquifunc"].lower() in ("ehvi", "ehvi_vec"):
        if "refpoint" not in moboInfo:
            moboInfo["refpointtype"] = 'dynamic'
        else:
            moboInfo["refpointtype"] = 'static'
        
        if 'refpointtype' in moboInfo:
            refpointavail = ['dynamic','static']
            if moboInfo["refpointtype"].lower() not in refpointavail:
                raise ValueError(moboInfo["refpointtype"],' is not valid type')

    elif moboInfo["acquifunc"].lower() == "parego":
        moboInfo["krignum"] = 1
        if "paregoacquifunc" not in moboInfo:
            moboInfo["paregoacquifunc"] = "EI"

    # If moboInfo['acquifuncopt'] (optimizer for the acquisition function) is not specified set to 'sampling+cmaes'
    if "acquifuncopt" not in moboInfo:
        moboInfo["acquifuncopt"] = "lbfgsb"
        print("The acquisition function optimizer is not specified, set to L-BFGS-B.")
    else:
        availableacqoptimizer = ['lbfgsb', 'cobyla', 'cmaes', 'ga', 'diff_evo']
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
