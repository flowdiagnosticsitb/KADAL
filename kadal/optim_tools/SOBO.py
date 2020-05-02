import numpy as np
from copy import deepcopy
from kadal.testcase.analyticalfcn.cases import evaluate
from kadal.optim_tools.acquifunc_opt import run_single_opt

class SOBO:
    """
    Perform single-objective Bayesian Optimization

    Args:
        soboInfo (dict): Dictionary containing necessary information for single-objective Bayesian optimization.
        krigobj (object): List of Kriging object.
        autoupdate (bool): True or False, depends on your decision to evaluate your function automatically or not.
        expconst (list): List of constraints Kriging object.
        chpconst (list): List of cheap constraint function.

    Returns:
        xupdate (nparray): Matrix of updated samples after optimization.
        yupdate (nparray): Response matrix of updated sample solutions after optimization.
    """

    def __init__(self, soboInfo, krigobj, autoupdate=True, expconst=None, chpconst=None):
        """
        Initialize SOBO class

        Args:
            soboInfo (dict): Dictionary containing necessary information for single-objective Bayesian optimization.
            krigobj (object): List of Kriging object.
            autoupdate (bool): True or False, depends on your decision to evaluate your function automatically or not.
            expconst (list): List of constraints Kriging object.
            chpconst (list): List of cheap constraint function.
        """
        self.soboInfo = soboInfocheck(soboInfo,autoupdate)
        self.krigobj = krigobj
        self.autoupdate = autoupdate
        self.krigconstlist = expconst
        self.cheapconstlist = chpconst

    def run(self, disp=True):
        """
        Run multi objective unconstrained Bayesian optimization.

        Args:
            disp (bool): Display process or not. Defaults to True

        Returns:
            xupdate (nparray): Array of design variables updates.
            yupdate (nparray): Array of objectives updates
        """
        self.nup = 0  # Number of current iteration
        self.Xall = self.krigobj.KrigInfo['X']
        self.yall = self.krigobj.KrigInfo['y']
        self.yhist = np.array([np.min(self.yall)])
        self.istall = 0

        print("Begin single-objective Bayesian optimization process.")
        while self.nup < self.soboInfo['nup']:

            if self.autoupdate and disp:
                print(f"Update no.: {self.nup + 1}, F-count: {np.size(self.Xall, 0)}, "
                      f"Best f(x): {self.yhist[self.nup]}, Stall counter: {self.istall}")
            else:
                pass

            # Find next suggested point
            self.xnext, self.metricnext = run_single_opt(self.krigobj,self.soboInfo,self.krigconstlist,self.cheapconstlist)

            # Break Loop if autoupdate is False
            if self.autoupdate is False:
                break
            else:
                pass

            # Evaluate response for next decision variable
            if type(self.krigobj.KrigInfo['problem']) == str:
                self.ynext = evaluate(self.xnext,self.krigobj.KrigInfo['problem'])
            elif callable(self.krigobj.KrigInfo['problem']):
                self.ynext = self.krigobj.KrigInfo['problem'](self.xnext)

            # Treatment for failed solutions, Reference : "Forrester, A. I., SÃ³bester, A., & Keane, A. J. (2006). Optimization with missing data.
            # Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 462(2067), 935-945."
            if np.isnan(self.ynext).any() is True:
                SSqr, y_hat = self.krigobj.predict(self.xnext, ['SSqr', 'pred'])
                self.ynext = y_hat + SSqr

            # Enrich experimental design
            self.krigobj.KrigInfo['X'] = np.vstack((self.krigobj.KrigInfo['X'], self.xnext))
            self.krigobj.KrigInfo['y'] = np.vstack((self.krigobj.KrigInfo['y'], self.ynext))

            # Re-train Kriging model
            self.krigobj.standardize()
            self.krigobj.train(disp=False)

            if self.nup == 0:
                self.xupdate = deepcopy(self.xnext)
                self.yupdate = deepcopy(self.ynext)
            else:
                self.xupdate = np.vstack((self.xupdate,self.xnext))
                self.yupdate = np.vstack((self.yupdate,self.ynext))

            self.nup += 1
            self.yhist = np.vstack((self.yhist, np.min(self.krigobj.KrigInfo['y'])))

            # Check stall iteration
            if self.yhist[self.nup,0] == self.yhist[self.nup-1,0]:
                self.istall += 1
                if self.istall == self.soboInfo['stalliteration']:
                    break
                else:
                    pass
            else:
                self.istall = 0

        print("Optimization finished, now creating the final outputs.")
        y_opt = np.min(self.krigobj.KrigInfo['y'])
        min_pos = np.argmin(self.krigobj.KrigInfo['y'])
        x_opt = self.krigobj.KrigInfo['y'][min_pos,:]
        if self.autoupdate:
            return x_opt,y_opt
        else:
            return self.xnext,self.ynext


def soboInfocheck(soboInfo, autoupdate):
    """
    Function to check the SOBO information and set SOBO Information to default value if
    required parameters are not supplied.

    Args:
         soboInfo (dict): Structure containing necessary information for single-objective Bayesian optimization.
         autoupdate (bool): True or False, depends on your decision to evaluate your function automatically or not.

     Returns:
         soboInfo (dict): Checked/Modified MOBO Information
    """
    if "nup" not in soboInfo:
        if autoupdate is True:
            raise ValueError("Number of updates for Bayesian optimization, soboInfo['nup'], is not specified")
        else:
            soboInfo["nup"] = 1
            print("Number of updates for Bayesian optimization has been set to 1")
    else:
        if autoupdate == True:
            pass
        else:
            soboInfo["nup"] = 1
            print("Manual mode is active, number of updates for Bayesian optimization is forced to 1")

    # Set default values
    if "stalliteration" not in soboInfo:
        if autoupdate is True:
            soboInfo["stalliteration"] = int(np.floor(soboInfo["nup"]/2))
            print("The number of stall iteration is not specified, set to nup/2.")
        else:
            soboInfo["stalliteration"] = 1
            print("Number of stall iteration for Bayesian optimization has been set to 1")
    else:
        if autoupdate is True:
            print("The number of stall iteration is specified to ", soboInfo["stalliteration"]," by user")
        else:
            soboInfo["stalliteration"] = 1
            print("Number of stall iteration for Bayesian optimization has been set to 1")

    if "acquifunc" not in soboInfo:
        soboInfo["acquifunc"] = "EI"
        print("The acquisition function is not specified, set to EI")
    else:
        availacqfun = ["ei","pred","lcb","poi","ebe"]
        if soboInfo["acquifunc"].lower() not in availacqfun:
            raise ValueError(soboInfo["acquifunc"], " is not a valid acquisition function.")
        else:
            if soboInfo["acquifunc"].lower() == "lcb":
                if "sigmalcb" not in soboInfo:
                    soboInfo["sigmalcb"] = 3
                    print("The sigma for lower confidence bound is not specified, set to 3.")
                else:
                    print("The sigma for lower confidence bound is specified to ", soboInfo["sigmalcb"], " by user")

    # If soboInfo['acquifuncopt'] (optimizer for the acquisition function) is not specified set to 'sampling+cmaes'
    if "acquifuncopt" not in soboInfo:
        soboInfo["acquifuncopt"] = "lbfgsb"
        print("The acquisition function optimizer is not specified, set to L-BFGS-B.")
    else:
        availableacqoptimizer = ['lbfgsb', 'cobyla', 'cmaes']
        if soboInfo["acquifuncopt"].lower() not in availableacqoptimizer:
            raise ValueError(soboInfo["acquifuncopt"], " is not a valid acquisition function optimizer.")
        else:
            pass

    if "nrestart" not in soboInfo:
        soboInfo["nrestart"] = 1
        print(
            "The number of restart for acquisition function optimization is not specified, setting soboInfo.nrestart to 1.")
    else:
        if soboInfo["nrestart"] < 1:
            raise ValueError("soboInfo['nrestart'] should be at least one")
        print("The number of restart for acquisition function optimization is specified to ",
              soboInfo["nrestart"], " by user")

    if "filename" not in soboInfo:
        soboInfo["filename"] = "temporarydata.mat"
        print("The file name for saving the results is not specified, set the name to temporarydata.mat")
    else:
        print("The file name for saving the results is not specified, set the name to ", soboInfo["filename"])

    return soboInfo