import sys

sys.path.insert(0, "..")
import numpy as np
from surrogate_models.kriging_model import Kriging
from surrogate_models.supports.initinfo import initkriginfo
from copy import deepcopy
from optim_tools import searchpareto
from misc.sampling.samplingplan import sampling, standardize
from optim_tools.acquifunc_opt import multiconstfun
from optim_tools.MOBO import MOBO
import time
from matplotlib import pyplot as plt
from misc.constfunc import sweepdiffcheck
from misc.constfunc import constraints_check
import pandas as pd


class Problem:
    def __init__(self, X, y, cldat, area_2, limit):
        self.X = X
        self.y = y
        self.cldat = cldat
        self.area2 = area_2
        self.ypar = None
        self.ub = limit[0, :]
        self.lb = limit[1, :]

    def createkrig(self):
        # define variables
        lb = self.lb
        ub = self.ub
        loocverrCD, loocverrNOISE, loocverrCL = [None, None, None]

        # Set Const Kriging
        KrigConstInfo = initkriginfo("single")
        KrigConstInfo["X"] = self.X
        KrigConstInfo["y"] = self.cldat.reshape(-1, 1)
        KrigConstInfo["nrestart"] = 5
        KrigConstInfo["ub"] = ub
        KrigConstInfo["lb"] = lb
        KrigConstInfo["optimizer"] = "lbfgsb"
        KrigConstInfo["limittype"] = ">="
        KrigConstInfo["limit"] = 0.15

        # KrigAreaInfo = initkriginfo("single")
        # KrigAreaInfo["X"] =  np.delete(self.X, [3,5,10], axis=1)
        # KrigAreaInfo["y"] = self.area2.reshape(-1,1)
        # KrigAreaInfo["nrestart"] = 5
        # KrigAreaInfo["ub"] = np.max(KrigAreaInfo["X"], axis=0)
        # KrigAreaInfo["lb"] = np.min(KrigAreaInfo["X"], axis=0)
        # KrigAreaInfo["optimizer"] = "cobyla"

        # Set Kriging Info
        KrigMultiInfo1 = initkriginfo("single")
        KrigMultiInfo1["X"] = self.X
        KrigMultiInfo1["y"] = self.y[:, 0].reshape(-1, 1)
        KrigMultiInfo1["nrestart"] = 7
        KrigMultiInfo1["ub"] = ub
        KrigMultiInfo1["lb"] = lb
        KrigMultiInfo1["optimizer"] = "lbfgsb"

        KrigMultiInfo2 = deepcopy(KrigMultiInfo1)
        KrigMultiInfo2["y"] = self.y[:, 1].reshape(-1, 1)

        self.krigobj1 = Kriging(
            KrigMultiInfo1,
            standardization=True,
            standtype="default",
            normy=False,
            trainvar=False,
        )
        # self.krigobj1.train(parallel=False)
        # loocverrCD, _ = self.krigobj1.loocvcalc(metrictype='r2')

        self.krigobj2 = Kriging(
            KrigMultiInfo2,
            standardization=True,
            standtype="default",
            normy=False,
            trainvar=False,
        )
        # self.krigobj2.train(parallel=False)
        # loocverrNOISE, _ = self.krigobj2.loocvcalc(metrictype='r2')

        self.krigconst = Kriging(
            KrigConstInfo,
            standardization=True,
            standtype="default",
            normy=False,
            trainvar=False,
        )
        # self.krigconst.train(parallel=False)
        # loocverrCL, _ = self.krigconst.loocvcalc(metrictype='r2')

        # print('LOOCV CD (r2): ', loocverrCD)
        # print('LOOCV Noise (r2): ', loocverrNOISE)
        # print('LOOCV CL (r2): ', loocverrCL)

        # # Create Kriging for Area (uncomment if needed)
        # self.krigarea = Kriging(KrigAreaInfo, standardization=True, standtype='default', normy=False, trainvar=False)
        # self.krigarea.train(parallel=False)
        # loocverrAREA, _ = self.krigarea.loocvcalc()

        self.kriglist = [self.krigobj1, self.krigobj2]
        self.expconst = [self.krigconst]

        return loocverrCD, loocverrNOISE, loocverrCL

    def calcdist(self, Xpoints):
        infeasiblesamp = np.where(self.cldat <= 0.15)[0]
        self.Xall = self.kriglist[0].KrigInfo["X"]
        self.yall = np.zeros(
            shape=[np.size(self.kriglist[0].KrigInfo["y"], axis=0), len(self.kriglist)]
        )
        for ii in range(np.size(self.yall, axis=1)):
            self.yall[:, ii] = self.kriglist[ii].KrigInfo["y"][:, 0]

        if infeasiblesamp is not None:
            self.yall = np.delete(self.yall.copy(), infeasiblesamp, 0)
            self.Xall = np.delete(self.Xall.copy(), infeasiblesamp, 0)
        else:
            pass

        self.ypar, _ = searchpareto.paretopoint(self.yall)
        for jj in range(np.size(self.ypar, axis=0)):
            current_X = self.Xall[
                np.where(
                    (self.yall[:, 0] == self.ypar[jj, 0])
                    & (self.yall[:, 1] == self.ypar[jj, 1])
                )
            ]
            norm_curX = (
                standardize(current_X, range=np.vstack((self.lb, self.ub))) / 2 + 0.5
            )
            norm_new = (
                standardize(Xpoints, range=np.vstack((self.lb, self.ub))) / 2
                + 0.5
                - norm_curX
            ) * 100
            fileloc = "../innout/tim/Out/dist_nextpoints8_AT_" + str(jj + 1) + ".csv"
            np.savetxt(
                fileloc,
                np.vstack((current_X, norm_new)),
                delimiter=",",
                header="x,z,le_sweep_1,dihedral_1,chord_1,tc_1,proj_span_1,chord_2,le_sweep_2,dihedral_2,tc_2",
                fmt="%s",
            )
        return self.ypar

    def geomconst(self, vars):
        # constraint 'geomconst' should have input of the design variables
        vars = np.array(vars)
        proj_area_1, area_1, proj_area_2, area_2 = constraints_check.calc_areas(
            vars[6], vars[4], vars[3], vars[7], vars[9], total_proj_area=0.00165529
        )
        s1_min = 0.3 * 0.00165529
        s1_max = 0.9 * 0.00165529
        s1_satisfied = constraints_check.min_max_satisfied(
            proj_area_1, min_val=s1_min, max_val=s1_max, disp=False
        )
        tip_angle = constraints_check.triangular_tip_angle(vars[8], vars[7], area_2)
        tip_satisfied = constraints_check.min_max_satisfied(tip_angle, 7, disp=False)
        stat = s1_satisfied & tip_satisfied
        return stat


if __name__ == "__main__":
    CDloocv = []
    Nloocv = []
    CLloocv = []
    lim = np.loadtxt("../innout/tim/In/const.csv", delimiter=",")
    for ii in range(6, 7):
        print("--" * 35)
        print("loop no.", ii + 1)
        print("--" * 35)
        filepath = "../innout/tim/In/opt_data17_AT_mod.csv"
        df = pd.read_csv(filepath, sep=",", index_col="code")
        data = df.values
        X = data[:, 0:11].astype(float)
        y = data[:, 14:16].astype(float)
        cldat = data[:, 13].astype(float)
        area_2 = data[:, 11].astype(float)

        df1 = pd.read_csv("../innout/tim/Out/nextpoints18_AT.csv", sep=",")
        data1 = df1.values
        Xupdate = data1[:, 0:11].astype(float)
        yupdate = data1[:, 14:16].astype(float)
        cldatnsga = data1[:, 13].astype(float)
        area_2nsga = data1[:, 11].astype(float)
        supdate = data1[:, 17:19].astype(float)

    errtestcd = np.array([0.01, 0.009, 0.0085, 0.01, 0.007])
    errtestnoise = np.array([0.5, 0.9, 0.85, 1, 0.7])
    plt.scatter(
        y[cldat > 0.15, 0],
        y[cldat > 0.15, 1],
        c="#1f77b4",
        label="initial feasible samples",
    )
    plt.scatter(
        y[cldat <= 0.15, 0],
        y[cldat <= 0.15, 1],
        marker="x",
        c="k",
        label="initial infeasible samples",
    )
    plt.scatter(
        yupdate[:, 0], yupdate[:, 1], c="#ff7f0e", label="predicted next samples"
    )
    plt.errorbar(
        yupdate[:, 0], yupdate[:, 1], yerr=supdate[:, 1], fmt="o", color="orange"
    )
    plt.errorbar(
        yupdate[:, 0], yupdate[:, 1], xerr=supdate[:, 0], fmt="o", color="orange"
    )
    plt.ylabel("dB(A)")
    plt.xlabel("CD")
    plt.legend()
    plt.show()