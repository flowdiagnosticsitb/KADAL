import numpy as np
from kadal.misc.sampling.samplingplan import realval
from kadal.testcase.RA.testcase import evaluate
from comet_ml import Experiment
import matplotlib.pyplot as plt
import time


class AKMCS:
    """
    Create AK-MCS model (Active Kriging - Monte Carlo Simulation) for reliability analysis

    Args:
        krigobj (object): Kriging object for AKMCS analysis
        akmcsInfo (dict): Dictionary that contains AKMCS model information.
            detail akmcsInfo:
                - akmcsInfo["init_samp"] (nparray): Initial Monte-Carlo population
                - akmcsInfo["maxupdate"] (int): Maximum number of update. Defaults to 120
                - akmcsInfo["problem"] (str): Type of case

    Returns:
        updatedX (nparray): updated samples.
        minUiter (nparray): minimum U value for each iteration
    """

    def __init__(self, krigobj, akmcsInfo):
        """
        Initialize akmcs

        Args:
            krigobj (object): Kriging object for AKMCS analysis
            akmcsInfo (dict): Dictionary that contains AKMCS model information.
                detail akmcsInfo:
                    - akmcsInfo["init_samp"] (nparray): Initial Monte-Carlo population
                    - akmcsInfo["maxupdate"] (int): Maximum number of update. Defaults to 120
                    - akmcsInfo["problem"] (str): Type of case
        """
        akmcsInfo = akmcsInfocheck(akmcsInfo)
        self.krigobj = krigobj
        self.akmcsInfo = akmcsInfo
        self.init_samp = akmcsInfo['init_samp']
        self.maxupdate = akmcsInfo['maxupdate']
        self.nsamp = np.size(self.init_samp, axis=0)
        self.Gx = np.zeros(shape=[self.nsamp,1])
        self.sigmaG = np.zeros(shape=[1,self.nsamp])
        self.stop_criteria = 100 #assign large number
        self.logging = None

    def run(self, autoupdate=True, disp=True, savedatato=None, logging=False, saveimageto=None, plotdatapos=None,
            plotdataneg=None, loggingAPIkey=None, logname=None, logworkspace=None):
        """
        Run AKMCS analysis

        Args:
            autoupdate (bool): Perform automatic update on design space or not. Default to True.
            disp (bool): Display progress or not. Default to True.
            savedatato (str): Filename to save update data. e.g.: 'filename.csv'

        Return:
             None
        """
        #logging
        if logging:
            if loggingAPIkey is None or logname is None or logworkspace is None:
                raise ValueError('Logging is turned on, APIkey, project and workspace must be specified.')
            self.logging = Experiment(api_key=loggingAPIkey,
                                    project_name=logname, workspace=logworkspace)
            if savedatato is not None:
                self.logging.set_name(savedatato)
            else:
                pass

        else:
            pass
        # Calculate Gx and SigmaG
        # Split init_samp to avoid memory error
        krig_initsamp = self.krigobj.KrigInfo['X']
        t1 = time.time()
        if self.nsamp < 10000:
            self.Gx,self.sigmaG = self.krigobj.predict(self.init_samp, ['pred','s'])
        else:
            run_times = int(np.ceil(self.nsamp/10000))
            for i in range(run_times):
                start = i * 10000
                stop = (i+1) * 10000
                if i != (run_times - 1):
                    self.Gx[start:stop, :], self.sigmaG[:,start:stop] = \
                        self.krigobj.predict(self.init_samp[start:stop, :], ['pred','s'])
                else:
                    self.Gx[start:, :], self.sigmaG[:,start:] = \
                        self.krigobj.predict(self.init_samp[start:, :], ['pred','s'])
        t2 = time.time()

        # Calculate probability of failure
        self.Pf = self.pfcalc()

        # Calculate learning function U
        self.lfucalc()
        self.stopcrit()
        self.updateX = np.array([self.xnew])
        self.minUiter = np.array([self.minU])
        if disp:
            print(f"Done iter no: 0, Pf: {self.Pf}, minU: {self.minU}")

        # Update samples automatically
        while autoupdate:
            labeladded = False
            for i in range(self.maxupdate):
                # Evaluate new samples and append into Kriging object information
                t = time.time()
                ynew = evaluate(self.xnew, type=self.akmcsInfo['problem'])
                self.krigobj.KrigInfo['y'] = np.vstack((self.krigobj.KrigInfo['y'],ynew))
                self.krigobj.KrigInfo['X'] = np.vstack((self.krigobj.KrigInfo['X'], self.xnew))
                self.krigobj.KrigInfo['nsamp'] += 1

                # standardize model and train updated kriging model
                t3 = time.time()
                self.krigobj.standardize()
                self.krigobj.train(disp=False)
                t4 = time.time()

                # Calculate Gx and SigmaG
                # Split init_samp to avoid memory error
                if self.nsamp < 10000:
                    self.Gx, self.sigmaG = self.krigobj.predict(self.init_samp, ['pred', 's'])
                else:
                    run_times = int(np.ceil(self.nsamp / 10000))
                    for ii in range(run_times):
                        start = ii * 10000
                        stop = (ii + 1) * 10000
                        if ii != (run_times - 1):
                            self.Gx[start:stop, :], self.sigmaG[:, start:stop] = \
                                self.krigobj.predict(self.init_samp[start:stop, :], ['pred', 's'])
                        else:
                            self.Gx[start:, :], self.sigmaG[:, start:] = \
                                self.krigobj.predict(self.init_samp[start:, :], ['pred', 's'])

                t5 = time.time()
                # Calculate Pf, COV and LFU
                self.Pf = self.pfcalc()
                self.cov = self.covpf()
                self.lfucalc()
                self.stopcrit()
                t6 = time.time()

                # Update variables
                self.updateX = np.vstack((self.updateX,self.xnew))
                self.minUiter = np.vstack((self.minUiter,self.minU))
                elapsed = time.time() - t
                if disp:
                    print(f"iter no: {i+1}, Pf: {self.Pf}, stopcrit: {self.stop_criteria}, time(s): {elapsed}, "
                          f"ynew: {ynew}")

                if logging:
                    self.logging.log_parameter('krigtype',self.krigobj.KrigInfo['type'])
                    outdict = {"Prob_fail":self.Pf,
                            "stopcrit":self.stop_criteria,
                            "time(s)":elapsed
                    }
                    self.logging.log_metrics(outdict,step=i+1)

                if savedatato is not None:
                    temparray = np.array([i,self.Pf,self.stop_criteria,elapsed])
                    if i == 0:
                        totaldata = temparray[:]
                    else:
                        totaldata = np.vstack((totaldata,temparray))
                    filename =  savedatato
                    np.savetxt(filename, totaldata, delimiter=',', header='iter,Pf,stopcrit,time(s)')
                else:
                    pass

                if saveimageto is not None:
                    imagefile = saveimageto + str(i) + ".PNG"
                    title = "Pf = " + str(self.Pf)
                    plt.figure(0, figsize=[10, 9])
                    if not labeladded:
                        plt.scatter(plotdatapos[:, 0], plotdatapos[:, 1], c='yellow', label='Feasible')
                        plt.scatter(plotdataneg[:, 0], plotdataneg[:, 1], c='cyan', label='Infeasible')
                        plt.scatter(krig_initsamp[:, 0], krig_initsamp[:, 1], c='red', label='Initial Kriging Population')
                        plt.scatter(self.updateX[:, 0], self.updateX[:, 1], s=75, c='black', marker='x', label='Update')
                        labeladded = True
                    else:
                        plt.scatter(plotdatapos[:, 0], plotdatapos[:, 1], c='yellow')
                        plt.scatter(plotdataneg[:, 0], plotdataneg[:, 1], c='cyan')
                        plt.scatter(krig_initsamp[:, 0], krig_initsamp[:, 1], c='red')
                        plt.scatter(self.updateX[:, 0], self.updateX[:, 1], s=75, c='black', marker='x')
                    plt.xlabel('X1', fontsize=18)
                    plt.ylabel('X2', fontsize=18)
                    plt.tick_params(axis='both', which='both', labelsize=16)
                    plt.legend(loc=1, prop={'size': 15})
                    plt.title(title,fontdict={'fontsize':20})
                    plt.savefig(imagefile, format='png')
                else:
                    pass

                # Break condition
                if self.stop_criteria <= 0.05 and i >= 15:
                    break
                else:
                    pass

            print(f"COV: {self.cov}")
            if self.cov <= 0.05:
                break
            else:
                pass
            break  # temporary break for debugging, delete/comment this line later

    def pfcalc(self):
        nGless = len([i for i in self.Gx if i <= 0])
        nsamp = np.size(self.init_samp, axis=0)
        Pf = nGless / nsamp
        return Pf

    def covpf(self):
        nmc = np.size(self.init_samp, axis=0)
        if self.Pf == 0:
            cov = 1000
        else:
            cov = np.sqrt((1 - self.Pf) / (self.Pf * nmc))
        return cov

    def lfucalc(self):
        self.U = abs(self.Gx) / self.sigmaG.reshape(-1,1)
        self.minU = np.min(self.U)
        minUloc = np.argmin(self.U)
        self.xnew = self.init_samp[minUloc,:]

    def stopcrit(self):
        nsamp = np.size(self.init_samp, axis=0)
        temp1 = self.Gx - 1.96 * self.sigmaG.reshape(-1,1)
        temp2 = self.Gx + 1.96 * self.sigmaG.reshape(-1,1)
        pfp = len([i for i in temp1 if i <= 0])/nsamp
        pfn = len([i for i in temp2 if i <= 0])/nsamp
        pf0 = len([i for i in self.Gx if i <= 0])/nsamp
        if pf0 == 0:
            self.stop_criteria = 100
        else:
            self.stop_criteria = (pfp-pfn)/pf0


def mcpopgen(lb=None,ub=None,n_order=6,n_coeff=1,type="random",ndim=2,stddev=1,mean=0,rand_seed=None):
    if rand_seed is not None:
        np.random.seed(rand_seed)
    nmc = int(n_coeff*10**n_order)
    if type.lower() == "normal" or type.lower() == "gaussian":
        pop = stddev*np.random.randn(nmc,ndim)+mean
    elif type.lower() == "lognormal":
        var = stddev**2
        sigma = np.sqrt(np.log(var/(mean**2)+1))
        mu = np.log((mean**2)/np.sqrt(var + mean**2))
        pop = np.exp(sigma*np.random.randn(nmc,ndim)+mu)
    elif type.lower() == 'gumbel':
        beta = (stddev/np.pi) * np.sqrt(6)
        pop = np.random.gumbel(mean,beta,(nmc,ndim))
    elif type.lower()== "random":
        if lb.any() == None or ub.any() == None:
            raise ValueError("type 'random' is selected, please input lower bound and upper bound value")
        else:
            pop = realval(lb, ub, np.random.rand(nmc,len(lb)))
    else:
        raise ValueError("Monte Carlo sampling type not supported")
    return pop


def akmcsInfocheck(akmcsInfo):
    """
    Function to check the AKMCS information and set AKMCS Information to default value if
    required parameters are not supplied.

    Args:
        akmcsInfo: Dictionary that contains AKMCS information.

    Returns:
        akmcsInfo: Checked/Modified AKMCS Information
    """
    if "init_samp" not in akmcsInfo:
        raise ValueError('akmcsInfo["init_samp"] must be defined')
    else:
        pass

    if "maxupdate" not in akmcsInfo:
        akmcsInfo["maxupdate"] = 120
        print("Maximum update is set to ", akmcsInfo["maxupdate"])
    else:
        print("Maximum update is set to ", akmcsInfo["maxupdate"], " by user.")

    if "problem" not in akmcsInfo:
        raise ValueError('akmcsInfo["problem"] must be defined')
    else:
        pass

    return akmcsInfo
