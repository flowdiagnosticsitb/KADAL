import numpy as np
from kadal.misc.sampling.samplingplan import sampling
from kadal.testcase.RA.testcase import evaluate
from copy import deepcopy

class SobolIndices:
    """
    Create Sobol Indices for sensitivity analysis
    """

    def __init__(self, nvar, krigobj=None, problem=None, ub=None, lb=None, nMC=2e5):
        """
        Initialize model

        Args:
            nvar (int):  Number of variables.
            krigobj (object): Kriging object, default to None.
            problem (str/callable/None): problem type, default to None.
        """
        self.nvar = nvar
        self.krigobj = krigobj
        self.problem = problem
        self.n = int(nMC)
        if ub is not None and lb is not None:
            _,init_mat = sampling("sobolnew", self.nvar*2, self.n, result="real", upbound=ub, lobound=lb)
        else:
            init_mat,_ = sampling("sobolnew", self.nvar*2, self.n)
        self.A = init_mat[:, :self.nvar]
        self.B = init_mat[:, self.nvar:]
        del init_mat
        self.ya = None
        self.yb = None
        self.fo_2 = None
        self.denom = None

    def analyze(self,first=True,total=False,second=False):
        """
        Run sensitivity analysis
        Args:
            first (bool): calculate first order or not
            total (bool): calculate total order or not
        Return:
             indices (dictionary): dictionary containing the sobol indices
        """
        if self.krigobj is not None:
            nsamp = np.size(self.A, axis=0)
            self.ya = np.zeros(shape=[nsamp, 1])
            if nsamp <= 10000:
                self.ya = self.krigobj.predict(self.A, ['pred'])
            else:
                run_times = int(np.ceil(nsamp / 10000))
                for i in range(run_times):
                    start = i * 10000
                    stop = (i + 1) * 10000
                    if i != (run_times - 1):
                        self.ya[start:stop, :] = self.krigobj.predict(self.A[start:stop, :], ['pred'])
                    else:
                        self.ya[start:, :] = self.krigobj.predict(self.A[start:, :], ['pred'])

            self.yb = np.zeros(shape=[nsamp, 1])
            if nsamp <= 10000:
                self.yb = self.krigobj.predict(self.B, ['pred'])
            else:
                run_times = int(np.ceil(nsamp / 10000))
                for i in range(run_times):
                    start = i * 10000
                    stop = (i + 1) * 10000
                    if i != (run_times - 1):
                        self.yb[start:stop, :] = self.krigobj.predict(self.B[start:stop, :], ['pred'])
                    else:
                        self.yb[start:, :] = self.krigobj.predict(self.B[start:, :], ['pred'])

        elif self.krigobj is None and self.problem is not None:
            if not callable(self.problem):
                self.ya = evaluate(self.A,self.problem)
                self.yb = evaluate(self.B,self.problem)
            else:
                self.ya = self.problem(self.A)
                self.yb = self.problem(self.B)
        else:
            raise ValueError("Either krigobj or problem must be not None")

        self.fo_2 = (np.sum(self.ya)/self.n)**2
        self.denom = (np.sum(self.ya**2)/self.n) - self.fo_2

        indices = dict()
        if first is True or total is True:
            indices["first"],indices["total"] = self.calc_ft_order(first,total)

        if second is True:
            indices["second"] = self.calc_second_order(indices["first"])
        elif second is True and first is False:
            indices["first"],indices["total"] = self.calc_ft_order(True,True)
            indices["second"] = self.calc_second_order(indices["first"])
        else:
            pass

        return indices

    def calc_ft_order(self,first=True,total=False):
        """
        Calculate first and total order Sobol Indices

        Return:
            s1 (numpy array): 1st order sobol indices.
        """
        s1 = np.zeros(self.nvar)
        st = np.zeros(self.nvar)

        for ii in range(self.nvar):
            C_i = deepcopy(self.B)
            C_i[:,ii] = self.A[:,ii]

            # Use Kriging to predict Monte-Carlo
            if self.krigobj is not None:
                nsamp = np.size(C_i, axis=0)
                yci = np.zeros(shape=[nsamp, 1])
                if nsamp <= 10000:
                    yci = self.krigobj.predict(C_i, ['pred'])
                else:
                    run_times = int(np.ceil(nsamp / 10000))
                    for i in range(run_times):
                        start = i * 10000
                        stop = (i + 1) * 10000
                        if i != (run_times - 1):
                            yci[start:stop, :] = self.krigobj.predict(C_i[start:stop, :], ['pred'])
                        else:
                            yci[start:, :] = self.krigobj.predict(C_i[start:, :], ['pred'])

            elif self.krigobj is None and self.problem is not None:
                if not callable(self.problem):
                    yci = evaluate(C_i, self.problem)
                else:
                    yci = self.problem(C_i)

            if first:
                s1[ii] = ((1/self.n)*np.sum(self.ya*yci) - self.fo_2)/self.denom
            if total:
                st[ii] = 1 - (((1 / self.n) * np.sum(self.yb * yci) - self.fo_2) / self.denom)

        return [s1,st]

    def calc_second_order(self,s1):
        """
        Calculate second order indices
        :return:
        """
        s2 = dict()

        for ii in range(self.nvar-1):
            for jj in range(ii+1, self.nvar):
                C_ij = deepcopy(self.B)
                C_ij[:, ii] = self.A[:, ii]
                C_ij[:, jj] = self.A[:, jj]

                # Use Kriging to predict Monte-Carlo
                if self.krigobj is not None:
                    nsamp = np.size(C_ij, axis=0)
                    yci = np.zeros(shape=[nsamp, 1])
                    if nsamp <= 10000:
                        yci = self.krigobj.predict(C_ij, ['pred'])
                    else:
                        run_times = int(np.ceil(nsamp / 10000))
                        for i in range(run_times):
                            start = i * 10000
                            stop = (i + 1) * 10000
                            if i != (run_times - 1):
                                yci[start:stop, :] = self.krigobj.predict(C_ij[start:stop, :], ['pred'])
                            else:
                                yci[start:, :] = self.krigobj.predict(C_ij[start:, :], ['pred'])

                elif self.krigobj is None and self.problem is not None:
                    if not callable(self.problem):
                        yci = evaluate(C_ij, self.problem)
                    else:
                        yci = self.problem(C_ij)

                vij =  ((1/self.n)*np.sum(self.ya*yci) - self.fo_2)
                key = "x"+str(ii+1)+"-x"+str(jj+1)
                s2[key] = (vij/self.denom) - s1[ii] - s1[jj]

        return s2


if __name__ == '__main__':
    ub = np.array([np.pi, np.pi, np.pi]*2)
    lb = -ub
    testSA = SobolIndices(40,None,"hidimenra",ub,lb)
    result = testSA.analyze(True,True,False)
    print(result)