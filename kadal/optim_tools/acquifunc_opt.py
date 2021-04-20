import time
import numpy as np
from kadal.misc.sampling.samplingplan import realval
from scipy.optimize import minimize, fmin_cobyla, differential_evolution
from kadal.optim_tools.ehvi.EHVIcomputation import ehvicalc, ehvicalc_vec, ehvicalc_kmac3d
from kadal.optim_tools.ga.uncGA import uncGA, uncGA_vec, uncGA2
import cma


def print_res(r_t, metric, x, success=None, msg=None, n_eval=None,
              n_gen=None, i_restart=None, n_restart=None):
    """Helper function to print iteration loop updates"""
    restart = f'Restart {i_restart + 1} of {n_restart} done, ' if i_restart else ''
    evals = f'{n_eval} func evals, ' if n_eval else ''
    gens = f'{n_gen} generations, ' if n_gen else ''
    msg = f' - {msg}' if msg is not None else ''
    loop_time = int(time.time() - r_t)
    conv = f'\nConverged: {success}{msg}' if success is not None else ''
    print(f'metric: {metric}, x: {x}')
    print(f'{restart}'
          f'{evals}{gens}{loop_time} s'
          f'{conv}')


def run_single_opt(krigobj, soboInfo, krigconstlist=None, cheapconstlist=None,
                   pool=None):
    """
    Optimize the single-objective acquisition function to find the next
    sampling point.

    The available optimizers for the acquisition functions are:
        'cmaes', 'lbfgsb', 'diff_evo', 'cobyla'

    Note that this function runs for both unconstrained and constrained
    single-objective Bayesian optimization.

    Args:
        krigobj (kriging_model.Kriging): Objective Kriging instance.
        soboInfo (dict): A structure containing necessary information for
            Bayesian optimization.
        krigconstlist ([kriging_model.Kriging], optional): Kriging
            instances for constraints. Defaults to None.
        cheapconstlist ([func], optional): Constraint functions.
            Defaults to None. Expected output of the constraint
            functions is 1 if satisfied and 0 if not.
            The constraint functions MUST have an input of x (the
            decision variable to be evaluated).
        pool (mp.Pool, optional): An existing mp.Pool instance can be
            specified and passed to solvers/acquisition functions for
            multiprocessing, if supported. Default is None.
    Returns:
        xnext (np.ndarray): n_dv-len array of suggested next sampling
            point as discovered by the optimization of the acquisition
            function.
        fnext (np.ndarray): n_obj-len array of  optimized acquisition
            function fitness metrics.
    """
    acquifuncopt = soboInfo["acquifuncopt"]
    acquifunc = soboInfo["acquifunc"]

    if acquifunc.lower() == 'parego':
        acquifunc = soboInfo['paregoacquifunc']
    else:
        msg = "Only soboInfo['acquifunc'] = 'parego' is currently handled"
        raise NotImplementedError(msg)

    n_restart = soboInfo["nrestart"]
    n_var = krigobj.KrigInfo["nvar"]

    low_bound = krigobj.KrigInfo["lb"]
    up_bound = krigobj.KrigInfo["ub"]
    if acquifuncopt.lower() == 'cmaes':
        Xrand = realval(low_bound, up_bound,
                        np.random.rand(n_restart, n_var))
        xnextcand = np.zeros(shape=[n_restart, n_var])
        fnextcand = np.zeros(shape=[n_restart])
        sigmacmaes = 1  # np.mean((KrigNewMultiInfo["ub"] - KrigNewMultiInfo["lb"]) / 6)
        for im in range(0, n_restart):
            if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
                xnextcand[im, :], es = cma.fmin2(krigobj.predict, Xrand[im, :], sigmacmaes,
                                                 {'verb_disp': 0,'verbose': -9},
                                                 args=(acquifunc))
                fnextcand[im] = es.result[1]
            else:  # For constrained problem
                xnextcand[im, :], es = cma.fmin2(singleconstfun, Xrand[im, :], sigmacmaes,
                                                 {'verb_disp': 0, 'verbose': -9},
                                                 args=(krigobj, acquifunc, krigconstlist,cheapconstlist))
                fnextcand[im] = es.result[1]
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == 'lbfgsb':
        Xrand = realval(low_bound, up_bound,
                        np.random.rand(n_restart, n_var))
        xnextcand = np.zeros(shape=[n_restart, n_var])
        fnextcand = np.zeros(shape=[n_restart])
        lbfgsbbound = np.hstack((low_bound.reshape(-1, 1), up_bound.reshape(-1, 1)))
        for im in range(0, n_restart):
            if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
                res = minimize(krigobj.predict,Xrand[im,:] ,method='L-BFGS-B', bounds=lbfgsbbound, args=(acquifunc))
                xnextcand[im,:] = res.x
                fnextcand[im] = res.fun
            else:  # For constrained problem (on progress)
                res = minimize(singleconstfun,Xrand[im,:], method='L-BFGS-B', bounds=lbfgsbbound,
                               args=(krigobj, acquifunc, krigconstlist,cheapconstlist))
                xnextcand[im, :] = res.x
                fnextcand[im] = res.fun
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == 'diff_evo':
        if 'de_kwargs' in soboInfo:
            de_kwargs = soboInfo['de_kwargs']
        else:
            de_kwargs = {}

        de_kwargs['init'] = de_kwargs.get('init', 'latinhypercube')

        optimbound = list(zip(low_bound, up_bound))
        de_args = (singleconstfun, optimbound)
        if 'constraints' in de_kwargs:
            cheapconstlist = None  # DE handles cheap constraint functions directly
        args = (krigobj, acquifunc, krigconstlist, cheapconstlist, None, 'inf')
        de_kwargs['args'] = args

        if pool is not None:
            workers = pool.map
            # If MP, set n_cpu to 1 to stop pool in EHVI - pass in existing pool
            soboInfo['n_cpu'] = 1
        else:
            workers = 1  # Default DE flag

        xnextcand = np.zeros(shape=[n_restart, n_var])
        fnextcand = np.zeros(shape=[n_restart])

        for im in range(n_restart):
            r_t = time.time()

            res = differential_evolution(*de_args, **de_kwargs, workers=workers)

            xnextcand[im, :] = res.x
            fnextcand[im] = res.fun
            print_res(r_t, res.fun, res.x, success=res.success, msg=res.message,
                      n_eval=res.nfev, n_gen=res.nit, i_restart=im,
                      n_restart=n_restart)

        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == 'cobyla':
        Xrand = realval(low_bound, up_bound,
                        np.random.rand(n_restart, n_var))
        xnextcand = np.zeros(shape=[n_restart, n_var])
        fnextcand = np.zeros(shape=[n_restart])
        optimbound = []
        for i in range(len(up_bound)):
            optimbound.append(lambda x, krigobj, aa, bb, cc, itemp=i: x[itemp] - krigobj.KrigInfo["lb"][itemp])
            optimbound.append(lambda x, krigobj, aa, bb, cc, itemp=i: krigobj.KrigInfo["ub"][itemp] - x[itemp])
        for im in range(0, n_restart):
            if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
                res = fmin_cobyla(krigobj.predict, Xrand[im,:], optimbound,
                                  rhobeg=0.5, rhoend=1e-4, args=(acquifunc))
                xnextcand[im, :] = res
                fnextcand[im] = krigobj.predict(res, acquifunc)
            else:
                res = fmin_cobyla(singleconstfun, Xrand[im, :], optimbound,
                                  rhobeg=0.5, rhoend=1e-4, args=(krigobj, acquifunc, krigconstlist,cheapconstlist))
                xnextcand[im, :] = res
                fnextcand[im] = singleconstfun(res, krigobj, acquifunc, krigconstlist,cheapconstlist)
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    return (xnext,fnext)


def run_multi_opt(kriglist, moboInfo, ypar, krigconstlist=None,
                  cheapconstlist=None, pool=None):
    """
    Optimize the multi-objective acquisition function to find the next
    sampling point.

    The available optimizers for the acquisition functions are:
        'ga', 'diff_evo', 'cmaes', 'lbfgsb', 'cobyla'

    Note that this function runs for both unconstrained and constrained
    multi-objective Bayesian optimization.

    Args:
        kriglist ([kriging_model.Kriging]): n_obj-len list of objective
            Kriging instances.
        moboInfo (dict): A structure containing necessary information
            for Bayesian optimization.
        ypar (np.ndarray): [n_par, n_obj] Current Pareto front
            solutions.
        krigconstlist ([kriging_model.Kriging], optional): Kriging
            instances for constraints. Defaults to None.
        cheapconstlist ([func], optional): Constraint functions.
            Defaults to None. Expected output of the constraint
            functions is 1 if satisfied and 0 if not.
            The constraint functions MUST have an input of x (the
            decision variable to be evaluated).
        pool (mp.Pool, optional): An existing mp.Pool instance can be
            specified and passed to solvers/acquisition functions for
            multiprocessing, if supported. Default is None.

    Returns:
        xnext (np.ndarray): n_dv-len array of suggested next sampling
            point as discovered by the optimization of the acquisition
            function.
        fnext (np.ndarray): n_obj-len array of  optimized acquisition
            function fitness metrics.
    """
    acquifuncopt = moboInfo["acquifuncopt"]
    acquifunc = moboInfo["acquifunc"]

    if acquifunc.lower() == 'ehvi':
        acqufunhandle = ehvicalc
    elif acquifunc.lower() == 'ehvi_vec':
        acqufunhandle = ehvicalc_vec
    elif acquifunc.lower() == 'ehvi_kmac3d':
        acqufunhandle = ehvicalc_kmac3d
    else:
        raise ValueError(f"Acquisition function handle {acquifunc} is not available")

    n_restart = moboInfo["nrestart"]
    n_var = kriglist[0].KrigInfo["nvar"]
    low_bound = kriglist[0].KrigInfo["lb"]
    up_bound = kriglist[0].KrigInfo["ub"]

    if acquifuncopt.lower() == 'cmaes':
        Xrand = realval(low_bound, up_bound,
                        np.random.rand(n_restart, n_var))
        xnextcand = np.zeros(shape=[n_restart, n_var])
        fnextcand = np.zeros(shape=[n_restart])
        sigmacmaes = 1  # np.mean((KrigNewMultiInfo["ub"] - KrigNewMultiInfo["lb"]) / 6)
        for im in range(0, n_restart):
            if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
                xnextcand[im, :], es = cma.fmin2(acqufunhandle, Xrand[im, :], sigmacmaes,
                                                 {'verb_disp': 0, 'verbose': -9},
                                                 args=(ypar, moboInfo, kriglist))
                fnextcand[im] = es.result[1]
            else:  # For constrained problem
                xnextcand[im, :], es = cma.fmin2(multiconstfun, Xrand[im, :], sigmacmaes,
                                                 {'verb_disp': 0, 'verbose': -9},
                                                 args=(ypar, kriglist, moboInfo, krigconstlist, cheapconstlist))
                fnextcand[im] = es.result[1]
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == 'ga':
        # Load SciPy DE settings from de_kwargs dict
        if 'ga_kwargs' in moboInfo:
            ga_kwargs = moboInfo['ga_kwargs']
        else:
            ga_kwargs = {}

        # TODO Redefine 'ehvisampling' to pass in n_pop and std_scale too - maybe dict?
        if moboInfo['ehvisampling'] == 'efficient':
            init_seed = efficientsamp(kriglist, ypar)
        else:
            init_seed = None

        func = multiconstfun
        args = (ypar, kriglist, moboInfo, krigconstlist, cheapconstlist)

        xnextcand = np.zeros(shape=[n_restart, n_var])
        fnextcand = np.zeros(shape=[n_restart])

        for im in range(n_restart):
            r_t = time.time()
            x, metric, _ = uncGA2(func, lb=low_bound, ub=up_bound, args=args,
                                  **ga_kwargs, initialization=init_seed, pool=pool)

            xnextcand[im, :] = x
            fnextcand[im] = metric
            # TODO: Fill more outputs - maybe have a general results class like SciPY results object
            print_res(r_t, metric, x, n_gen=_[-1, 0], i_restart=im, n_restart=n_restart)

        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == 'lbfgsb':
        Xrand = realval(low_bound, up_bound,
                        np.random.rand(n_restart, n_var))
        xnextcand = np.zeros(shape=[n_restart, n_var])
        fnextcand = np.zeros(shape=[n_restart])
        lbfgsbbound = np.hstack((low_bound.reshape(-1, 1), up_bound.reshape(-1, 1)))
        for im in range(0, n_restart):
            if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
                res = minimize(acqufunhandle,Xrand[im,:],method='L-BFGS-B',bounds=lbfgsbbound,args=(ypar,moboInfo,
                                                                                                    kriglist))
                xnextcand[im,:] = res.x
                fnextcand[im] = res.fun
            else:  # For constrained problem
                res = minimize(multiconstfun,Xrand[im,:],method='L-BFGS-B',bounds=lbfgsbbound,
                               args=(ypar, kriglist, moboInfo, krigconstlist, cheapconstlist))
                xnextcand[im, :] = res.x
                fnextcand[im] = res.fun
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == 'diff_evo':
        # Load SciPy DE settings from de_kwargs dict
        if 'de_kwargs' in moboInfo:
            de_kwargs = moboInfo['de_kwargs']
        else:
            de_kwargs = {}

        # Set ENDS or DE sample init, if specified. Else, default to 'latinhypercube'
        if moboInfo['ehvisampling'] == 'efficient':
            n_pop_factor = de_kwargs.get('popsize', 10)
            n_var = n_var
            n_pop = n_var * n_pop_factor
            de_kwargs['init'] = efficientsamp(kriglist, ypar, n_pop=n_pop)
        else:
            de_kwargs['init'] = de_kwargs.get('init', 'latinhypercube')

        optimbound = list(zip(low_bound, up_bound))
        de_args = (multiconstfun, optimbound)
        if 'constraints' in de_kwargs:
            cheapconstlist = None  # DE handles cheap constraint functions directly
        # Can't pass pool to acqufunc (child process)
        args = (ypar, kriglist, moboInfo, krigconstlist, cheapconstlist, None, 'inf')
        de_kwargs['args'] = args

        if pool is not None:
            workers = pool.map
            # If MP, set n_cpu to 1 to stop pool in EHVI - pass in existing pool
            moboInfo['n_cpu'] = 1
        else:
            workers = 1  # Default DE flag

        xnextcand = np.zeros(shape=[n_restart, n_var])
        fnextcand = np.zeros(shape=[n_restart])

        for im in range(n_restart):
            r_t = time.time()

            res = differential_evolution(*de_args, **de_kwargs, workers=workers)

            xnextcand[im, :] = res.x
            fnextcand[im] = res.fun
            print_res(r_t, res.fun, res.x, success=res.success, msg=res.message,
                      n_eval=res.nfev, n_gen=res.nit, i_restart=im,
                      n_restart=n_restart)

        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == 'cobyla':
        Xrand = realval(low_bound, up_bound,
                        np.random.rand(n_restart, n_var))
        xnextcand = np.zeros(shape=[n_restart, n_var])
        fnextcand = np.zeros(shape=[n_restart])
        optimbound = []
        for i in range(len(up_bound)):
            optimbound.append(lambda x, cc, kriglist, dd, aa, bb, itemp=i: x[itemp] - kriglist[0].KrigInfo["lb"][itemp])
            optimbound.append(lambda x, cc, kriglist, dd, aa, bb, itemp=i: kriglist[0].KrigInfo["ub"][itemp] - x[itemp])
        for im in range(0, n_restart):
            if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
                res = fmin_cobyla(acqufunhandle, Xrand[im,:], optimbound,
                                  rhobeg=0.5, rhoend=1e-4, args=(ypar, moboInfo, kriglist))
                xnextcand[im, :] = res
                fnextcand[im] = acqufunhandle(res, ypar, moboInfo, kriglist)
            else:
                res = fmin_cobyla(multiconstfun, Xrand[im, :], optimbound,
                                  rhobeg=0.5, rhoend=1e-4, args=(ypar, kriglist, moboInfo, krigconstlist, cheapconstlist))
                xnextcand[im, :] = res
                fnextcand[im] = multiconstfun(res,ypar, kriglist, moboInfo, krigconstlist, cheapconstlist)
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    return xnext,fnext


def singleconstfun(x, krigobj, acquifunc, krigconstlist=None,
                   cheapconstlist=None, pool=None, mode='tiny',
                   __warned=set()):
    """
    Calculate the single objective acquisition function value

    Args:
        x (np.ndarray): Decision variable to be evaluated. If
            a 2D array is given, specify in the form [n_pop, n_dv].
        ypar (np.ndarray): [n_par, n_obj] Array containing the current
            non-dominated solutions in the objective space.
        krigobj (kriging_model.Kriging): Objective Kriging instance.
        krigconstlist ([kriging_model.Kriging]): List of constraint
            Kriging objects. Defaults to None.
        cheapconstlist ([func]]): List of constraints functions.
            Defaults to None. Expected output of a constraint function
            is 1 if the constraint is satisfied and 0 if not.
            Functions MUST have an input of x (the decision variable
            to be evaluated). If it can accept 2D [n_pop, n_dv] arrays,
            vectorised numpy evaluations will be performed.
        pool (mp.Pool, optional): An existing mp.Pool instance.
            #TODO: Add pool methods for predict here
        mode (str/None, optional): ['tiny'/'inf'] Passed to
            replace_zero_hv() if not None.

    Returns:
        fx (np.ndarray/float): The acquisition function value(s). If an
            [n_pop, n_dv]-shape input x array is used, fx is returned
            as an n_pop-len array. If input x is a 1D feature array,
            fx will be returned as a single float.
    """
    # Calculate unconstrained acquisition function
    metric = krigobj.predict(x, acquifunc)

    # Check if 1D feature array -> temporarily convert to 2D array.
    reshape = False
    if x.ndim == 1:
        x = np.array([x]).reshape(1, -1)
        reshape = True

    n_pop = x.shape[0]

    if krigconstlist is None:
        pof = 1
    else:
        # Change to list if the type is not list
        if not isinstance(krigconstlist, list):
            krigconstlist = list(krigconstlist)

        n_krig_con = len(krigconstlist)
        pof = np.zeros([n_pop, n_krig_con])
        for i in range(n_pop):
            for ii in range(n_krig_con):
                # predict can only handle 1D inputs at the moment
                pof[i, ii] = krigconstlist[ii].predict(x[i, :], 'PoF')
        pof = np.prod(pof, axis=1)  # n_pop-len PoF array

    if cheapconstlist is None:
        coeff = 1
    else:
        # Change to list if the type is not list
        if not isinstance(cheapconstlist, list):
            cheapconstlist = list(cheapconstlist)

        coeff = np.zeros([n_pop, len(cheapconstlist)])

        if __warned:
            # Eval pop members one at a time
            for i in range(n_pop):
                for jj in range(len(cheapconstlist)):
                    coeff[i, jj] = cheapconstlist[jj](x[i, :])
        try:
            # See if cheap constraints can handle [n_pop, n_dv] arrays
            for jj in range(len(cheapconstlist)):
                coeff[:, jj] = cheapconstlist[jj](x)

        except Exception as e:
            print(f'{e}\n N.B. Cheap constraints coded to handle input '
                  f'x.shape = [n_samp, n_dv] will run faster! Doing '
                  f'sequential run.')
            __warned.add(True)  # So we only print warning once.
            for i in range(n_pop):
                for jj in range(len(cheapconstlist)):
                    coeff[i, jj] = cheapconstlist[jj](x[i, :])

        coeff = np.prod(coeff, axis=1)  # n_pop-len coefficient array

    fx = pof * coeff * metric

    # Give a penalty to zero metric for some solvers
    if mode is not None:
        replace_zero(metric, mode)

    # If input was 1D feature array return single float
    if reshape:
        fx = fx[0]
    return fx


def multiconstfun(x, ypar, kriglist, moboInfo, krigconstlist=None,
                  cheapconstlist=None, pool=None, mode='tiny',
                  __warned=set()):
    """Calculate the multiobjective acquisition function value.

    Args:
        x (np.ndarray): Decision variable to be evaluated. If
            a 2D array is given, specify in the form [n_pop, n_dv].
        ypar (np.ndarray): [n_par, n_obj] Array containing the current
            non-dominated solutions in the objective space.
        kriglist ([kriging_model.Kriging]): List of objective Kriging
            instances.
        moboInfo (dict): A structure containing necessary information
            for Bayesian optimization.
        krigconstlist ([kriging_model.Kriging], optional): List of
            constraint Kriging objects. Defaults to None.
        cheapconstlist ([func]], optional): List of constraints
            functions. Defaults to None.
            Expected output of a constraint function is 1 if the
            constraint is satisfied and 0 if not.
            Functions MUST have an input of x (the decision variable
            to be evaluated). If it can accept 2D [n_pop, n_dv] arrays,
            vectorised numpy evaluations will be performed.
        pool (mp.Pool, optional): An existing mp.Pool instance.
        mode (str/None, optional): ['tiny'/'inf'] Passed to
            replace_zero_hv() if not None.

    Returns:
        fx (np.ndarray/float): The acquisition function value(s). If an
            [n_pop, n_dv]-shape input x array is used, fx is returned
            as an n_pop-len array. If input x is a 1D feature array,
            fx will be returned as a single float.
    """
    # TODO: Remove this logic and pass in the acquifunhandle for deep loop functions!
    acquifunc = moboInfo['acquifunc']
    if acquifunc.lower() == 'ehvi':
        acqufunhandle = ehvicalc
    elif acquifunc.lower() == 'ehvi_vec':
        acqufunhandle = ehvicalc_vec
    elif acquifunc.lower() == 'ehvi_kmac3d':
        acqufunhandle = ehvicalc_kmac3d
    else:
        msg = "Only moboInfo['acquifunc'] = 'ehvi' is currently handled"
        raise NotImplementedError(msg)

    # Check if 1D feature array -> temporarily convert to 2D array.
    reshape = False
    if x.ndim == 1:
        x = np.array([x]).reshape(1, -1)
        reshape = True

    n_pop = x.shape[0]

    # Calculate penalty due to expensive constraints
    if krigconstlist is None:
        pof = 1
    else:
        # Change to list if the type is not list
        if not isinstance(krigconstlist, list):
            krigconstlist = list(krigconstlist)

        n_krig_con = len(krigconstlist)
        pof = np.zeros([n_pop, n_krig_con])
        for i in range(n_pop):
            for ii in range(n_krig_con):
                # predict can only handle 1D inputs at the moment
                pof[i, ii] = krigconstlist[ii].predict(x[i, :], 'PoF')
        pof = np.prod(pof, axis=1)  # n_pop-len PoF array

    # Calculate penalty due to cheap constraints
    if cheapconstlist is None:
        coeff = 1
    else:
        # Change to list if the type is not list
        if not isinstance(cheapconstlist, list):
            cheapconstlist = list(cheapconstlist)

        coeff = np.zeros([n_pop, len(cheapconstlist)])

        if __warned:
            # Eval pop members one at a time
            for i in range(n_pop):
                for jj in range(len(cheapconstlist)):
                    coeff[i, jj] = cheapconstlist[jj](x[i, :])
        try:
            # See if cheap constraints can handle [n_pop, n_dv] arrays
            for jj in range(len(cheapconstlist)):
                coeff[:, jj] = cheapconstlist[jj](x)

        except Exception as e:
            print(f'{e}\n N.B. Cheap constraints coded to handle input '
                  f'x.shape = [n_samp, n_dv] will run faster! Doing '
                  f'sequential run.')
            __warned.add(True)  # So we only print warning once.
            for i in range(n_pop):
                for jj in range(len(cheapconstlist)):
                    coeff[i, jj] = cheapconstlist[jj](x[i, :])

        coeff = np.prod(coeff, axis=1)  # n_pop-len coefficient array

    if acquifunc.lower() in ('ehvi_vec', 'ehvi_kmac3d'):
        metric = acqufunhandle(x, ypar, moboInfo, kriglist, pool=pool)
    else:
        metric = acqufunhandle(x, ypar, moboInfo, kriglist)

    fx = pof * coeff * metric

    # Give a penalty to zero metric for some solvers
    if mode is not None:
        replace_zero(metric, mode)

    # If input was 1D feature array return single float
    if reshape:
        fx = fx[0]
    return fx


def replace_zero(hv, mode):
    """Helper function to modify zero values in an array.

    Args:
        hv (np.ndarray): An array of values. Modified in-place.
        mode (str): ['tiny'/'inf'] The replacement method. If 'tiny'
            is specified, zeros are replaced with a positive value
            near np.finfo("float").tiny. If 'inf', then zeros are
            replaced with np.inf.
    """
    z = hv == 0  # mask of values = 0
    if mode == 'tiny':
        # give penalty to HV, to avoid error in CMA-ES when all HV = 0
        rng = np.random.default_rng()
        hv[z] = rng.uniform(np.finfo("float").tiny, np.finfo("float").tiny * 100,
                            size=np.count_nonzero(z))
    elif mode == 'inf':
        # Scipy differential evolution does not like tiny values near zero
        # https://github.com/scipy/scipy/issues/13784
        hv[z] = np.inf
    else:
        raise ValueError("mode flag must be set to 'tiny' or 'inf'.")


def efficientsamp(kriglist, y_par, n_pop=300, std_scale=0.2):
    """Effective Non-Dominated Sampling

    Generate an n_pop-sized population of samples using a normal
    distribution around the current n-objective Pareto solutions.

    Args:
        kriglist ([KrigInfo]): A list containing Kriging KrigInfo
            from which to get the current decision variables 'X_norm',
            and bounds 'ub' and 'lb'.
        y_par (np.ndarray): [n_par, n_obj] Current Pareto front.
        n_pop (int, optional): The number of samples to generate.
            Defaults to 300.
        std_scale (numeric, optional) The standard deviation scaling
            factor. Defaults to 0.2.

    Returns:
        init_seed (np.ndarray): [n_pop, n_dv] Array of samples.
    """
    # Stack objective values into same shape as y_par
    objs = np.hstack([k.KrigInfo["y"] for k in kriglist])
    mask = (objs[:, None] == y_par).all(-1).any(-1)  # mask of matching rows
    # assert (objs[mask] == y_par).all()
    idx = np.where(mask)[0]  # indices of matching rows
    initialization = kriglist[0].KrigInfo["X_norm"][idx, :] / 2 + 0.5

    n_var = len(kriglist[0].KrigInfo["ub"])
    std_s = np.std(kriglist[0].KrigInfo['X_norm'], 0) * std_scale
    if initialization.ndim == 1:
        samplenorm = np.random.normal(initialization, std_s, (n_pop, n_var))
    else:
        n_init = np.size(initialization, 0)
        n_batch = int(n_pop / n_init)
        samplenorm = np.zeros((n_pop, n_var))
        for ij in range(n_init - 1):
            dist = np.random.normal(initialization[ij, :], std_s, (n_batch, n_var))
            samplenorm[ij * n_batch:(ij + 1) * n_batch, :] = dist
        n_remain = np.size(samplenorm[(ij + 1) * n_batch:, :], 0)
        dist = np.random.normal(initialization[(ij + 1), :], std_s, (n_remain, n_var))
        samplenorm[(ij + 1) * n_batch:, :] = dist
        samplenorm[samplenorm < 0] = 0
        samplenorm[samplenorm > 1] = 1

    init_seed = realval(kriglist[0].KrigInfo["lb"],
                        kriglist[0].KrigInfo["ub"],
                        samplenorm)
    return init_seed
