"""Sample 3-Objective Problem

Author: Tim Jim, Tohoku University

Scroll down and check the n_cpu that you want to use.

Then run using:
    python mobo_3d.py test3d

Or:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python mobo_3d.py test3d

"""
import os
# Set a single thread per process for numpy with MKL/BLAS
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
import sys
import time
import argparse
import pathlib
import shutil
import pickle
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from kadal.surrogate_models.kriging_model import Kriging
from kadal.surrogate_models.supports.initinfo import initkriginfo
from kadal.optim_tools.MOBO import MOBO

import constraints as cons

class Problem:

    def __init__(self, X, y, dv_min, dv_max, g=None, x_labels=None,
                 y_labels=None, g_labels=None, h=None):
        """Setup problem.

        Dimensions extracted from:
            n_samp, n_dv = X.shape
            n_obj = y.shape[1]

        Args:
            X (np.ndarray): [n_samp x n_dv] design vars input array.
            y (np.ndarray): [n_samp x n_obj] objectives input array.
            dv_min (np.ndarray/[]): n_dv-len array of minimum design
                var values.
            dv_max (np.ndarray/[]): n_dv-len array of maximum design
                var values.
            g (np.ndarray): [n_samp x n_con] expensive constraints
                input array. Defaults to None.
            x_labels ([str/int], optional): n_dv-len labels for design
                vars. Default None will result in int labels in
                range(n_dv).
            y_labels ([str/int], optional): n_obj-len labels for
                objectives. Default None will result in int labels in
                range(n_obj).
            g_labels ([str/int], optional): n_con-len labels for
                constraints. Default None will result in int labels in
                range(n_con).
            h ([func]): List of cheap constraint functions. The
                constraint functions must return 1 if the constraint
                is satisfied and 0 if not. The constraint functions are
                passed an array of design variables, x.
        """
        n_samp, n_dv = X.shape
        n_obj = y.shape[1]
        n_con = 0 if g is None else g.shape[1]

        # Check dimensions
        if y.shape[0] != n_samp or (g is not None and g.shape[0] != n_samp):
            raise ValueError(f'Ensure X, y, and g have the same shape[0]!')
        if x_labels is not None and len(x_labels) != n_dv:
            msg = (f'Variable inputs do not match:\n'
                   f'len(x_labels) {x_labels}, n_dv from X: {n_dv}')
            raise ValueError(msg)
        if y_labels is not None and len(y_labels) != y.shape[1]:
            msg = (f'Objective inputs do not match:\n'
                   f'len(y_labels) {y_labels}, n_obj from y: {n_obj}')
            raise ValueError(msg)

        self.n_samp = n_samp
        self.n_obj = n_obj
        self.n_con = n_con

        self.X = X
        self.y = y
        self.g = g
        self.x_labels = list(range(n_dv)) if x_labels is None else x_labels
        self.y_labels = list(range(n_obj)) if y_labels is None else y_labels
        self.g_labels = list(range(n_con)) if g_labels is None else g_labels

        self.h = h  # Cheap constraint functions

        self.ub = np.array(dv_max)  # Convert to np.ndarray, in case [] input
        self.lb = np.array(dv_min)

        # Settings applied to Kriging models
        self.obj_krig_map = None
        self.con_krig_map = None

        self.obj_krig = []  # Generated objective Kriging models appended here
        self.con_krig = []  # Generated expensive constraint Kriging models
        self.obj_loocve = []
        self.con_loocve = []
        self.obj_time = []
        self.con_time = []
        self.total_train_time = None

        # Results from updates
        self.xupdate = []
        self.yupdate = []
        self.supdate = []
        self.metricall = []
        self.total_update_time = None

    def create_krig(self, obj_krig_map=None, con_krig_map=None, n_cpu=1):
        """Initialise and train Kriging models.

        Default settings for the objective or constraint Krigings can
        be overidden by setting the obj_krig_map or con_krig_map dict.
        The dictionary should take the form:

            e.g.
            map = {'default': {'nrestart': 5,
                               'optimizer': 'lbfgsb',
                               },
                   'CD': {'optimizer': 'cobyla',
                          },
                   'CL': {'nrestart': 10,
                          'limittype': '>=',
                          'limit': 0.15},
                          },
                   }

        -where the dict key is used to identify the objective or
        constraint by label (int, if no explicit x_label and y_label set
        previously). The subdict key-value pairs are set in each
        surrogate_models.supports.initinfo.initkriginfo('single'). The
        'default' dict is applied first and can be overridden by the
        following dictionaries.

        Args:
            obj_krig_map (dict(dict()), optional): Map specific settings
                onto objective Kriging models via the labels.
            con_krig_map (dict(dict()), optional): Map specific settings
                onto constraint Kriging models via the labels.
            n_cpu (int, optional): If > 1, uses parallel processing.
                Defaults to 1.
        """
        def apply_krig_map(krig_info, map, label):
            """Helper func. Apply 'default' dict, then labeled dict"""
            if 'default' in map:
                for k, v in map['default'].items():
                    print(f"Setting {label} Kriging defaults '{k}': {v}")
                    krig_info[k] = v
            if label in map:
                for k, v in map[label].items():
                    print(f"Setting {label} Kriging '{k}': {v}")
                    krig_info[k] = v

        # Set up Kriging for each objective
        obj_infos = []
        for i in range(self.n_obj):
            krig_multi_info = initkriginfo()
            krig_multi_info["X"] = self.X
            krig_multi_info["y"] = self.y[:, i].reshape(-1, 1)
            krig_multi_info["ub"] = self.ub
            krig_multi_info["lb"] = self.lb

            label = self.y_labels[i]
            if obj_krig_map is not None:
                apply_krig_map(krig_multi_info, obj_krig_map, label)
            obj_infos.append(krig_multi_info)

        # Set up Kriging for each constraint
        con_infos = []
        for i in range(self.n_con):
            krig_multi_info = initkriginfo()
            krig_multi_info["X"] = self.X
            krig_multi_info["y"] = self.g[:, i].reshape(-1, 1)
            krig_multi_info["ub"] = self.ub
            krig_multi_info["lb"] = self.lb

            label = self.g_labels[i]
            if con_krig_map is not None:
                apply_krig_map(krig_multi_info, con_krig_map, label)
            con_infos.append(krig_multi_info)

        # Train Kriging models
        start_total_train = time.time()
        for i, krig_info in enumerate(obj_infos):
            krig_obj = Kriging(krig_info, standardization=True,
                               standtype='default', normy=False,
                               trainvar=False)
            start_train = time.time()
            krig_obj.train(n_cpu=n_cpu)
            t = time.time() - start_train
            print(f'{self.y_labels[i]} training time: {t:.2f} seconds')
            loocve, _ = krig_obj.loocvcalc()
            print(f'Objective {self.y_labels[i]} LOOCVE: {loocve}')
            self.obj_krig.append(krig_obj)
            self.obj_loocve.append(loocve)
            self.obj_time.append(t)

        for i, krig_info in enumerate(con_infos):
            krig_con = Kriging(krig_info, standardization=True,
                               standtype='default', normy=False,
                               trainvar=False)
            start_train = time.time()
            t = time.time() - start_train
            print(f'{self.y_labels[i]} training time: {t:.2f} seconds')
            krig_con.train(n_cpu=n_cpu)
            loocve, _ = krig_con.loocvcalc()
            print(f'Constraint {self.g_labels[i]} LOOCVE: {loocve}')
            self.con_krig.append(krig_con)
            self.con_loocve.append(loocve)
            self.con_time.append(t)

        elapsed = time.time() - start_total_train
        print(f'Total training time: {elapsed:.2f} seconds')

        # Save data for summary
        self.total_train_time = elapsed
        self.obj_krig_map = obj_krig_map
        self.con_krig_map = con_krig_map

        # # Create Kriging for Area (uncomment if needed)
        # self.krigarea = Kriging(KrigAreaInfo, standardization=True, standtype='default', normy=False, trainvar=False)
        # self.krigarea.train(parallel=False)
        # loocverrAREA, _ = self.krigarea.loocvcalc()

    def save_state(self, out_pkl):
        print(f'Saving problem to: {out_pkl}')
        with open(out_pkl, 'wb') as f:
            pickle.dump(self, f)

    def update_sample(self, mobo_info, n_kb=5, n_cpu=1):
        # infeasiblesamp = np.where(self.cldat <= 0.15)[0]
        mobo = MOBO(mobo_info, self.obj_krig, autoupdate=False,
                    multiupdate=n_kb, savedata=False,
                    expconst=self.con_krig, chpconst=self.h)
        start_update = time.time()
        xupdate, yupdate, supdate, metricall = mobo.run(disp=True,
                                                        infeasible=None,
                                                        n_cpu=n_cpu)
        elapsed = time.time() - start_update
        print(f'Total update time: {elapsed:.2f} seconds')
        self.total_update_time = elapsed
        self.xupdate.append(xupdate)
        self.yupdate.append(yupdate)
        self.supdate.append(supdate)
        self.metricall.append(metricall)
        return xupdate, yupdate, supdate, metricall

    def summary(self, out=None, elapsed=None):
        elapsed = f'{elapsed/60:.2f}' if elapsed is not None else 'None'
        sum_text = [f'Total training time: '
                    f'{self.total_train_time/60:.2f} minutes',
                    f'Total update time: '
                    f'{self.total_update_time/60:.2f} minutes',
                    f'Elapsed time: {elapsed} minutes']

        sum_text.append('\nObjective Kriging Settings')
        if self.obj_krig_map is None:
            sum_text.append('None')
        else:
            for krig, settings in self.obj_krig_map.items():
                for setting, val in settings.items():
                    sum_text.append(f'{krig} - {setting}: {val}')

        sum_text.append('\nObjectives:\tLOOCVE,\ttime')
        for i in range(len(self.obj_krig)):
            sum_text.append(f'{self.y_labels[i]}:\t{self.obj_loocve[i]:.4f},\t'
                            f'{self.obj_time[i]:.2f} s')

        sum_text.append('\nConstraint Kriging Settings')
        if self.con_krig_map is None:
            sum_text.append('None')
        else:
            for krig, settings in self.con_krig_map.items():
                for setting, val in settings.items():
                    sum_text.append(f'{krig} - {setting}: {val}')

        sum_text.append('\nConstraints:\tLOOCVE,\ttime')
        n_con = len(self.con_krig)
        if n_con == 0:
            sum_text.append('None')
        for i in range(n_con):
            sum_text.append(f'{self.g_labels[i]}:\t{self.con_loocve[i]:.4f},\t'
                            f'{self.con_time[i]:.2f} s')

        sum_str = '\n'.join(sum_text)
        print(sum_str)
        if out is not None:
            print(f'Writing optimisation summary to: {out}')
            with open(out, 'w') as f_out:
                f_out.write(sum_str)
                f_out.write('\n')

    def plot_updates(self, out_dir=None, i_update=-1, title=None):
        # plot_groups = [[0, 1], [0, 2], [1, 2]]
        n_obj = len(self.obj_krig)
        plot_groups = itertools.combinations(range(n_obj), 2)
        y = self.y
        yupdate = self.yupdate[i_update]
        supdate = self.supdate[i_update]

        for p_group in plot_groups:
            a, b = p_group
            fig, axes = plt.subplots()
            if title is not None:
                axes.set_title(title)
            axes.scatter(y[:, a], y[:, b], c='#1f77b4', label='initial samples')
            axes.scatter(yupdate[:, a], yupdate[:, b], c='#ff7f0e', label='predicted next samples')
            axes.errorbar(yupdate[:, a], yupdate[:, b], xerr=supdate[:, a], yerr=supdate[:, b], fmt='o', color='orange')
            # axes.errorbar(yupdate[:, a], yupdate[:, b], yerr=supdate[:, b], fmt='o', color='orange')

            x_label = self.y_labels[a]
            y_label = self.y_labels[b]
            axes.set_xlabel(x_label)
            axes.set_ylabel(y_label)
            axes.legend()
            if out_dir is not None:
                out_dir = pathlib.Path(out_dir)
                fig.savefig(out_dir / f'{y_label}_vs_{x_label}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out_name', help="Opt case written to dir f'opt_{out_name}'")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing opt dir")
    parser.add_argument('--reload', action='store_true', help="Load previously save class")
    args = parser.parse_args()
    name = args.out_name
    overwrite = args.overwrite
    reload = args.reload

    opt = f'opt_{name}'
    src_opt_data = 'opt_data.csv'
    opt_data_in = f'{opt}/opt_data.csv'
    next_out = f'{opt}/next_samples.csv'
    next_out_xlsx = f'{opt}/next_samples.xlsx'
    summary_out = f'{opt}/summary.txt'
    out_pkl = f'{opt}/model.pkl'  # Save and reload trained Krig models

    # Check we're not overwriting an existing file
    if reload:
        if not pathlib.Path(opt_data_in).exists():
            raise ValueError(f'{opt_data_in} from previous run is missing.')
    else:
        if pathlib.Path(opt_data_in).exists() and not overwrite:
            msg = (f"{opt_data_in} already exists. Check out_name or use "
                   "'--reload' or '--overwrite' flags.")
            raise ValueError(msg)
        # Copy opt_data.csv into new opt directory
        p_opt = pathlib.Path(opt)
        p_opt.mkdir(exist_ok=True)
        shutil.copy(src_opt_data, opt_data_in)

    s_ref = 268.56 / 2  # Half aircraft

    geom_params = ['x', 'z', 'chord_0', 'tc_0', 'twist_0',
                   'le_sweep_1', 'dihedral_1', 'chord_1',
                   'tc_1', 'twist_1', 'proj_span_1',
                   'le_sweep_2', 'dihedral_2', 'chord_2',
                   'tc_2', 'twist_2', 'proj_span_2',
                   'le_sweep_3', 'dihedral_3', 'chord_3',
                   'tc_3', 'twist_3']
    objectives = ['CD_total', 'CM_x_abs', 'SELa_000.0']

    df = pd.read_csv(opt_data_in, sep=',', index_col='name')

    X = df[geom_params].to_numpy()  # [n_samp, n_dv] Design vars
    y = df[objectives].to_numpy()  # [n_samp, n_obj] Objective vals
    h = [cons.g_0, cons.g_1, cons.g_2, cons.g_3,
         cons.g_4, cons.g_5, cons.g_6, cons.g_7]  # Cheap constraints
    de_cons = [cons.nlc_0, cons.nlc_1, cons.nlc_2, cons.nlc_3,
               cons.nlc_4, cons.nlc_5, cons.nlc_6]  # Cheap constraints passed to DE

    dv_min = cons.dv_min
    dv_max = cons.dv_max

    krig_default = {'nrestart': 10,
                    'optimizer': 'lbfgsb',
                    }
    obj_krig_map = {'default': krig_default,
                    'CD_total': {},
                    'CM_x_abs': {},
                    'SELa_000.0': {},
                    }
    
    n_kb = 8  # number of Kriging Believer sampes 
    n_cpu = 40  # number of CPUs to use
    update_info = {'nup': 1,
                   'nrestart': 3,  # number of solver restarts per KB sample
                   # 'acquifunc': 'ehvi_vec',
                   'acquifunc': 'ehvi_kmac3d',
                   # 'acquifuncopt': 'ga',
                   # 'ga_kwargs': {'disp': True,
                   #               'n_pop': 500},
                   'acquifuncopt': 'diff_evo',
                   'de_kwargs': {'disp': False,
                                 # 'init': 'random',  # Overwritten if using ENDS
                                 'strategy': 'best2bin',
                                 'constraints': de_cons,
                                 'popsize': 15,
                                 'maxiter': 500,
                                 'tol': 1e-3,
                                 'mutation': (0.5, 1),
                                 'recombination': 0.85,
                                 'polish': False},
                   'n_cpu': n_cpu,
                   'ehvisampling': 'default',  # 'default' / 'efficient'
                   'refpoint': np.array([0.0500, 0.18, 100]),
                   }

    # output headers and cycle column
    header = ','.join(['Cycle'] + geom_params + objectives + ['metric'] + [f's_{o}' for o in objectives])
    cycle = np.array([opt] * n_kb).reshape(-1, 1)

    # Run optimisation
    t_opt = time.time()
    # If specified, reload existing problem
    if reload and pathlib.Path(out_pkl).exists():
        print(f'Loading problem from: {out_pkl}')
        with open(out_pkl, 'rb') as f:
            optim = pickle.load(f)
    else:
        optim = Problem(X, y, dv_min, dv_max, x_labels=geom_params, y_labels=objectives, h=h)
        optim.create_krig(obj_krig_map=obj_krig_map, con_krig_map=None, n_cpu=n_cpu)
        optim.save_state(out_pkl)
    xupdate, yupdate, supdate, metricall = optim.update_sample(update_info, n_kb, n_cpu=n_cpu)
    elapsed = time.time() - t_opt

    print(f'Total optimisation time: {elapsed/60:.2f} mins')

    totalupdate = np.hstack((cycle, xupdate, yupdate, metricall, supdate))
    np.savetxt(next_out, totalupdate, delimiter=",", header=header, comments="", fmt="%s")

    optim.plot_updates(out_dir=pathlib.Path(opt), title=opt)

    optim.summary(out=summary_out, elapsed=elapsed)
    optim.save_state(out_pkl)
