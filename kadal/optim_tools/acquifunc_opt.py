import numpy as np
import multiprocessing as mp
from kadal.misc.sampling.samplingplan import realval
from scipy.optimize import minimize, fmin_cobyla, differential_evolution
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from kadal.optim_tools.ehvi.EHVIcomputation import ehvicalc, ehvicalc_vec
from kadal.optim_tools.ga.uncGA import uncGA, uncGA_vec, uncGA2
import cma


def run_single_opt(krigobj, soboInfo, krigconstlist=None, cheapconstlist=None):
    """
   Run the optimization of multi-objective acquisition function to find the next sampling point.

   Args:
     krigobj (object): Kriging object.
     soboInfo (dict): A structure containing necessary information for Bayesian optimization.
     krigconstlist (list): List of Kriging object for constraints. Defaults to None.
     cheapconstlist (list): List of constraints function. Defaults to None.
            Expected output of the constraint functions is 1 if the constraint is satisfied and 0 if not.
            The constraint functions MUST have an input of x (the decision variable to be evaluated)

   Returns:
     xnext (nparray): Suggested next sampling point as discovered by the optimization of the acquisition function
     fnext (nparray): Optimized acquisition function

   The available optimizers for the acquisition function are 'cmaes', 'lbfgsb', 'cobyla'.
   Note that this function runs for both unconstrained and constrained single-objective Bayesian optimization.
   """
    acquifuncopt = soboInfo["acquifuncopt"]
    acquifunc = soboInfo["acquifunc"]

    if acquifunc.lower() == 'parego':
        acquifunc = soboInfo['paregoacquifunc']
    else:
        pass

    if acquifuncopt.lower() == 'cmaes':
        Xrand = realval(krigobj.KrigInfo["lb"], krigobj.KrigInfo["ub"],
                        np.random.rand(soboInfo["nrestart"], krigobj.KrigInfo["nvar"]))
        xnextcand = np.zeros(shape=[soboInfo["nrestart"], krigobj.KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[soboInfo["nrestart"]])
        sigmacmaes = 1  # np.mean((KrigNewMultiInfo["ub"] - KrigNewMultiInfo["lb"]) / 6)
        for im in range(0, soboInfo["nrestart"]):
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
        Xrand = realval(krigobj.KrigInfo["lb"], krigobj.KrigInfo["ub"],
                        np.random.rand(soboInfo["nrestart"], krigobj.KrigInfo["nvar"]))
        xnextcand = np.zeros(shape=[soboInfo["nrestart"], krigobj.KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[soboInfo["nrestart"]])
        lbfgsbbound = np.hstack((krigobj.KrigInfo["lb"].reshape(-1, 1), krigobj.KrigInfo["ub"].reshape(-1, 1)))
        for im in range(0,soboInfo["nrestart"]):
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
        xnextcand = np.zeros(shape=[soboInfo["nrestart"], krigobj.KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[soboInfo["nrestart"]])
        optimbound = np.hstack((krigobj.KrigInfo["lb"].reshape(-1, 1), krigobj.KrigInfo["ub"].reshape(-1, 1)))
        for im in range(0, soboInfo["nrestart"]):
            if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
                res = differential_evolution(krigobj.predict, optimbound, args=(acquifunc,))
                xnextcand[im, :] = res.x
                fnextcand[im] = res.fun
            else:
                res = differential_evolution(singleconstfun, optimbound,
                                             args=(krigobj, acquifunc, krigconstlist,cheapconstlist))
                xnextcand[im, :] = res.x
                fnextcand[im] = res.fun
        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == 'cobyla':
        Xrand = realval(krigobj.KrigInfo["lb"], krigobj.KrigInfo["ub"],
                        np.random.rand(soboInfo["nrestart"], krigobj.KrigInfo["nvar"]))
        xnextcand = np.zeros(shape=[soboInfo["nrestart"], krigobj.KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[soboInfo["nrestart"]])
        optimbound = []
        for i in range(len(krigobj.KrigInfo["ub"])):
            optimbound.append(lambda x, krigobj, aa, bb, cc, itemp=i: x[itemp] - krigobj.KrigInfo["lb"][itemp])
            optimbound.append(lambda x, krigobj, aa, bb, cc, itemp=i: krigobj.KrigInfo["ub"][itemp] - x[itemp])
        for im in range(0, soboInfo["nrestart"]):
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


def run_multi_opt(kriglist, moboInfo, ypar, krigconstlist=None, cheapconstlist=None):
    """
    Run the optimization of multi-objective acquisition function to find the next sampling point.

    Args:
      kriglist (list): A list containing Kriging instances.
      moboInfo (dict): A structure containing necessary information for Bayesian optimization.
      ypar (nparray): Array contains the current non-dominated solutions.
      krigconstlist (list): List of Kriging object for constraints. Defaults to None.
      cheapconstlist (list): List of constraints function. Defaults to None.
            Expected output of the constraint functions is 1 if the constraint is satisfied and 0 if not.
            The constraint functions MUST have an input of x (the decision variable to be evaluated)

    Returns:
      xnext (nparray): Suggested next sampling point as discovered by the optimization of the acquisition function
      fnext (nparray): Optimized acquisition function

    The available optimizers for the acquisition function are 'cmaes', 'lbfgsb', 'cobyla'.
    Note that this function runs for both unconstrained and constrained single-objective Bayesian optimization.
    """
    acquifuncopt = moboInfo["acquifuncopt"]
    acquifunc = moboInfo["acquifunc"]

    if acquifunc.lower() == 'ehvi':
        acqufunhandle = ehvicalc
    elif acquifunc.lower() == 'ehvi_vec':
        acqufunhandle = ehvicalc_vec
    else:
        raise ValueError("Acquisition function handle is not available")

    if acquifuncopt.lower() == 'cmaes':
        Xrand = realval(kriglist[0].KrigInfo["lb"], kriglist[0].KrigInfo["ub"],
                        np.random.rand(moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]))
        xnextcand = np.zeros(shape=[moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[moboInfo["nrestart"]])
        sigmacmaes = 1  # np.mean((KrigNewMultiInfo["ub"] - KrigNewMultiInfo["lb"]) / 6)
        for im in range(0, moboInfo["nrestart"]):
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
        if moboInfo['nrestart'] != 1:
            print(f"'nrestart' ignored for 'acquifuncopt': '{acquifuncopt}'")

        templst = []
        if moboInfo['ehvisampling'] == 'efficient':
            for ij in range(np.size(ypar, 0)):
                idx = np.where((kriglist[0].KrigInfo["y"] == ypar[ij, 0]) &
                               (kriglist[1].KrigInfo["y"] == ypar[ij, 1]))[0][0]
                templst.append(idx)

            init_seed = kriglist[0].KrigInfo["X_norm"][templst, :] / 2 + 0.5
        else:
            init_seed = None

        if krigconstlist is None and cheapconstlist is None:
            print('####### GA1')
            # xnext, fnext, _ = uncGA(acqufunhandle, lb=kriglist[0].KrigInfo["lb"], ub=kriglist[0].KrigInfo["ub"],
            #                         args=(ypar, moboInfo, kriglist), initialization=init_seed)
            func = acqufunhandle
            args = (ypar, moboInfo, kriglist)
            # xnext, fnext, _ = uncGA_vec(acqufunhandle,
            #                             lb=kriglist[0].KrigInfo["lb"],
            #                             ub=kriglist[0].KrigInfo["ub"],
            #                             args=(ypar, kriglist, moboInfo,
            #                                   krigconstlist, cheapconstlist),
            #                             initialization=init_seed, pool=pool)
        else:
            print('####### GA2')
            # xnext, fnext, _ = uncGA(multiconstfun, lb=kriglist[0].KrigInfo["lb"], ub=kriglist[0].KrigInfo["ub"],
            #                         args=(ypar, kriglist, moboInfo, krigconstlist, cheapconstlist),
            #                         initialization=init_seed)
            func = multiconstfun
            args = (ypar, kriglist, moboInfo, krigconstlist, cheapconstlist)

        # Start pool before GA so no time wasting creating new pools
        if moboInfo.get('n_cpu', 1) == 1:
            pool = None
        else:
            pool = mp.Pool(processes=moboInfo['n_cpu'])

        try:
            xnext, fnext, _ = uncGA2(func,
                                        lb=kriglist[0].KrigInfo["lb"],
                                        ub=kriglist[0].KrigInfo["ub"],
                                        args=args,
                                        initialization=init_seed, pool=pool)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    elif acquifuncopt.lower() == 'lbfgsb':
        Xrand = realval(kriglist[0].KrigInfo["lb"], kriglist[0].KrigInfo["ub"],
                        np.random.rand(moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]))
        xnextcand = np.zeros(shape=[moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[moboInfo["nrestart"]])
        lbfgsbbound = np.hstack((kriglist[0].KrigInfo["lb"].reshape(-1, 1), kriglist[0].KrigInfo["ub"].reshape(-1, 1)))
        for im in range(0,moboInfo["nrestart"]):
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
        if moboInfo['ehvisampling'] == 'efficient':
            init_seed = efficientsamp(kriglist, ypar, npop=300)
        else:
            init_seed = 'latinhypercube'

        optimbound = np.hstack((kriglist[0].KrigInfo["lb"].reshape(-1, 1), kriglist[0].KrigInfo["ub"].reshape(-1, 1)))
        if krigconstlist is None and cheapconstlist is None:  # For unconstrained problem
            func = acqufunhandle
            args = (ypar, moboInfo, kriglist)
            # res = differential_evolution(acqufunhandle, optimbound, init=init_seed, args=(ypar,moboInfo,kriglist))

        else:
            func = multiconstfun
            args = (ypar, kriglist, moboInfo, krigconstlist, cheapconstlist)
            # res = differential_evolution(multiconstfun, optimbound, init=init_seed, args=(ypar, kriglist, moboInfo, krigconstlist, cheapconstlist))

        if '_n_cpu' in moboInfo:
            # Set from previous KB cycle
            pass
        elif moboInfo.get('n_cpu', 1) == 1:
            moboInfo['_n_cpu'] = 1
        else:
            # Set n_cpu to 1 to stop pool in EHVI - pool set up by scipy for DE
            moboInfo['n_cpu'] = 1
            moboInfo['_n_cpu'] = moboInfo['n_cpu']

        if 'de_kwargs' in moboInfo:
            kwargs = moboInfo['de_kwargs']
        else:
            kwargs = {}

        xnextcand = np.zeros(shape=[moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[moboInfo["nrestart"]])

        # res = differential_evolution(func, optimbound, init=init_seed,
        #                              args=args, workers=moboInfo['_n_cpu'],
        #                              **kwargs)
        # xnext = res.x
        # fnext = res.fun

        # Manual convergence manger...
        import time
        for im in range(moboInfo["nrestart"]):
            print(f'Restart {im + 1} of {moboInfo["nrestart"]}')
            with DifferentialEvolutionSolver(func, optimbound, init=init_seed,
                                             args=args, workers=moboInfo['_n_cpu'],
                                             **kwargs) as de_solver:
                f_old = 0
                i = 0
                fit_err = np.nan
                tol = kwargs.get('tol', 1e-2)
                while i < 50 or fit_err >= tol:
                    t_start = time.time()
                    xnext, fnext = de_solver.next()
                    xnextcand[im,:] = xnext
                    fnextcand[im] = fnext
                    # seems like de iss replacing tiny number with zero, breaking convergence
                    if fnext == 0:
                        # Set a tiny number so at least we don't get nans in fit_err
                        # If a nan, the first non-nan number causes lage jump over tol
                        fnext = np.random.uniform(np.finfo("float").tiny, np.finfo("float").tiny * 100)

                    fit_err = 100 * np.abs(fnext - f_old) / fnext
                    f_old = fnext

                    # print(f'{i} fit_err: {fit_err}, xnext: {xnext}, fnext: {fnext}, time: {int(time.time()-t_start)} s')
                    i += 1
                    if i > 200:
                        print('Hit max generations.')
                        break

        I = np.argmin(fnextcand)
        xnext = xnextcand[I, :]
        fnext = fnextcand[I]

    elif acquifuncopt.lower() == 'cobyla':
        Xrand = realval(kriglist[0].KrigInfo["lb"], kriglist[0].KrigInfo["ub"],
                        np.random.rand(moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]))
        xnextcand = np.zeros(shape=[moboInfo["nrestart"], kriglist[0].KrigInfo["nvar"]])
        fnextcand = np.zeros(shape=[moboInfo["nrestart"]])
        optimbound = []
        for i in range(len(kriglist[0].KrigInfo["ub"])):
            optimbound.append(lambda x, cc, kriglist, dd, aa, bb, itemp=i: x[itemp] - kriglist[0].KrigInfo["lb"][itemp])
            optimbound.append(lambda x, cc, kriglist, dd, aa, bb, itemp=i: kriglist[0].KrigInfo["ub"][itemp] - x[itemp])
        for im in range(0, moboInfo["nrestart"]):
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


def singleconstfun(x, krigobj, acquifunc, krigconstlist=None, cheapconstlist=None):
    """
    Calculate the single objective acquisition function value

    Args:
        x (nparray): Decision variable to be evaluated.
        krigobj (object): The kriging object.
        acquifunc (str): Acquisition function metric.
        krigconstlist (list): List of Kriging object for constraints. Defaults to None.
        cheapconstlist (list): List of constraints function. Defaults to None.
            Expected output of the constraint functions is 1 if the constraint is satisfied and 0 if not.
            The constraint functions MUST have an input of x (the decision variable to be evaluated)

    Returns:
        fx (float): The acquisition function value.
    """
    # Calculate unconstrained acquisition function
    acquifuncval = krigobj.predict(x, acquifunc)

    if krigconstlist is not None:

        # Change to list if the type is not list
        if type(krigconstlist) is not list:
            krigconstlist = [krigconstlist]
        else:
            pass

        nkrigcon = len(krigconstlist)

        pof = np.zeros(shape=[nkrigcon])
        for ii in range(nkrigcon):
            pof[ii] = krigconstlist[ii].predict(x, 'PoF')
        pof = np.prod(pof)

    else:
        pof = 1

    if cheapconstlist is None:
        pass

    else:
        # Change to list if the type is not list
        if type(cheapconstlist) is not list:
            cheapconstlist = [cheapconstlist]
        else:
            pass

        coeff = np.zeros(shape=[len(cheapconstlist)])
        for jj in range(len(cheapconstlist)):
            coeff[jj] = cheapconstlist[jj](x)
        coeff = np.prod(coeff)

    fx = pof*coeff*acquifuncval

    return fx


def multiconstfun(x, ypar, kriglist, moboInfo, krigconstlist=None,
                  cheapconstlist=None, pool=None):
    """
    Calculate the multi objective acquisition function value

    Args:
        x (np.ndarray): [n_pop, n_dv] Decision variable to be evaluated.
        ypar (np.ndarray): Array contains the current non-dominated solutions.
        kriglist (list): List of Kriging object.
        moboInfo (dict): A structure containing necessary information for Bayesian optimization.
        krigconstlist ([Kriging]]): List of Kriging object for constraints. Defaults to None.
        cheapconstlist ([func]]): List of constraints function. Defaults to None.
            Expected output of the constraint functions is 1 if the constraint is satisfied and 0 if not.
            The constraint functions MUST have an input of x (the decision variable to be evaluated)

    Returns:
        fx (np.ndarray/float): The acquisition function value(s). If an
            [n_pop, n_dv] shape input x array is used, fx returned as an
            n_pop-len array. If input x is a 1D feature array, fx will
            be returned as a single float.
    """
    acquifunc = moboInfo['acquifunc']
    if acquifunc.lower() == 'ehvi':
        acqufunhandle = ehvicalc
    elif acquifunc.lower() == 'ehvi_vec':
        acqufunhandle = ehvicalc_vec
    else:
        msg = "Only moboInfo['acquifunc'] = 'ehvi' is currently handled"
        raise NotImplementedError(msg)

    if x.ndim == 1:
        n_pop = 1
        # n_dv = len(x)
    else:
        n_pop, n_dv = x.shape

    if krigconstlist is None:
        pof = 1

    else:
        # Change to list if the type is not list
        if isinstance(krigconstlist, list):
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
        try:
            # See if cheap constraints can handle [n_pop, n_dv] arrays
            for jj in range(len(cheapconstlist)):
                coeff[:, jj] = cheapconstlist[jj](x)

        except Exception as e:
            print(f'{e}\n N.B. Cheap constraints coded to handle input '
                  f'x.shape = [n_samp, n_dv] will run faster! Trying '
                  f'sequential run.')

            for i in range(n_pop):
                for jj in range(len(cheapconstlist)):
                    coeff[i, jj] = cheapconstlist[jj](x[i, :])

        coeff = np.prod(coeff, axis=1)  # n_pop-len coefficient array

    if acquifunc.lower() == 'ehvi_vec':
        metric = acqufunhandle(x, ypar, moboInfo, kriglist, pool=pool)
    else:
        metric = acqufunhandle(x, ypar, moboInfo, kriglist)

    fx = pof * coeff * metric

    return fx


def efficientsamp(kriglist, ypar, npop=300):
    nvar = len(kriglist[0].KrigInfo["ub"])
    templst = []
    for ij in range(np.size(ypar, 0)):
        idx = np.where((kriglist[0].KrigInfo["y"] == ypar[ij, 0]) &
                       (kriglist[1].KrigInfo["y"] == ypar[ij, 1]))[0][0]
        templst.append(idx)

    initialization = kriglist[0].KrigInfo["X_norm"][templst, :] / 2 + 0.5

    if initialization.ndim == 1:
        samplenorm = np.random.normal(initialization, np.std(kriglist[0].KrigInfo['X_norm'], 0) * 0.2, (npop, nvar))
    else:
        n_init = np.size(initialization, 0)
        nbatch = int(npop / n_init)
        samplenorm = np.zeros((npop, nvar))
        for ij in range(n_init - 1):
            samplenorm[ij * nbatch:(ij + 1) * nbatch, :] = np.random.normal(initialization[ij, :],
                                                                            np.std(kriglist[0].KrigInfo['X_norm'],
                                                                                   0) * 0.2,
                                                                            (nbatch, nvar))
        samplenorm[(ij + 1) * nbatch:, :] = np.random.normal(initialization[(ij + 1), :],
                                                             np.std(kriglist[0].KrigInfo['X_norm'], 0) * 0.2,
                                                             (np.size(samplenorm[(ij + 1) * nbatch:, :], 0), nvar))
        samplenorm[samplenorm < 0] = 0
        samplenorm[samplenorm > 1] = 1

    init_seed =realval(kriglist[0].KrigInfo["lb"], kriglist[0].KrigInfo["ub"], samplenorm)
    return init_seed