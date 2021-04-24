import numpy as np
from kadal.optim_tools.ehvi.exi2d import exi2d
try:
    from kadal.extern.HDYE_3D_Update import kmac
except ImportError as e:
    import pathlib
    f_path = pathlib.Path(__file__).parent
    lib_path = (f_path / '../../extern/HDYE_3D_Update').resolve()
    msg = (f"{e}\n\nKMAC library has probably not been compiled for this "
           f"python version yet. Go to:\n{lib_path}\nand run 'cmake . && make'."
           f" You may need to check the paths in 'CMakeLists.txt' first.")
    raise ImportError(msg)


def ehvicalc(x,ypar,moboInfo,kriglist):
    """
    Wrapper for EHVI calculation.

    Args:
        x (nparray): Design Variables
        ypar (nparray): Current Pareto Front
        moboInfo (dict): Structure(Dictionary) containing necessary information for multiobjective Bayesian optimization.
        kriglist (list): List containing Kriging instances.

    Returns:
        HV (float): EHVI value
    """
    HV = EHVI(x, ypar, moboInfo, kriglist)

    return HV


def EHVI(x,ypar,moboInfo,kriglist):
    """
        ModelInfoKR{i} = Model Information of objective i
        ObjectiveInfoKR{i} = Objective Information of objective i

        Input :
            - x : Design variables
            - ypar: Current Pareto front
            - BayesMultiInfo: Structure(Dictionary) containing necessary information for multiobjective Bayesian optimization.
            - kriglist (list): List containing Kriging instances.
        """
    X = kriglist[0].KrigInfo["X"]
    nobj = len(kriglist)
    nsamp = np.size(X, 0)
    YO = np.zeros(shape=[nsamp, nobj])
    RefP = moboInfo["refpoint"]

    # prediction of each objective
    pred = np.zeros(shape=[nobj])
    SSqr = np.zeros(shape=[nobj])
    for ii in range(0, nobj):
        pred[ii], SSqr[ii] = kriglist[ii].predict(x, ["pred", "SSqr"])

    # Compute (negative of) hypervolume
    HV = -1 * exi2d(ypar, RefP, pred, SSqr)

    if HV == 0:  # give penalty to HV, to avoid error in CMA-ES when in an iteration produce all HV = 0
        HV = np.random.uniform(np.finfo("float").tiny, np.finfo("float").tiny * 100)

    return HV


def pool_predict(pool, x, kriglist):
    """Helper function for multiprocessing Kriging.predict().

    N.B. Might want to inline this later once everything else
    is optimised.

    Args:
        pool (mp.Pool): A multiprocessing pool instance.
        x (np.ndarray): [n_pop, n_dv] Design variables for a population.
        kriglist ([kriging_model.Kriging]): n_obj-len list of objective
            Kriging instances.

    Returns:
        pred (np.ndarray): [n_pop, n_obj] predicted mean vectors.
        SSqr (np.ndarray): [n_pop, n_obj] mean vector standard
            deviations.
    """
    n_pop = x.shape[0]
    n_obj = len(kriglist)
    pred = np.zeros([n_pop, n_obj])
    SSqr = np.zeros([n_pop, n_obj])

    # Set up predict loop args
    p_args = [(x[i, :], ["pred", "SSqr"]) for i in range(n_pop)]
    # Run for all population for each Kriging
    for j in range(n_obj):
        res_p = pool.starmap(kriglist[j].predict, p_args)
        res_p = np.array(res_p).ravel().reshape(n_pop, 2)
        pred[:, j] = res_p[:, 0].copy()
        SSqr[:, j] = res_p[:, 1].copy()

    return pred, SSqr


def pool_hv_exi2d(pool, x, kriglist, y_par, ref_point):
    """Helper function for multiprocessing exi2d.

    Args:
        pool (mp.Pool): A multiprocessing pool instance.
        x (np.ndarray): [n_pop, n_dv] Design variables for a population.
        kriglist ([kriging_model.Kriging]): n_obj-len list of objective
            Kriging instances.
        y_par (np.ndarray): [n_par, n_obj] Current Pareto front.
        ref_point  (np.ndarray): n_obj-len array indicating the
            reference point location.

    Returns:
        hv (np.ndarray): n_pop-len array of hypervolumes for each
            population member.
    """
    n_pop = x.shape[0]
    hv = np.zeros(n_pop)

    pred, SSqr = pool_predict(pool, x, kriglist)

    # Set up exi2d loop args and pass to pool
    hv_args = ((y_par, ref_point, pred[i, :], SSqr[i, :])
               for i in range(n_pop))
    hv[:] = pool.starmap(exi2d, hv_args)
    hv *= -1
    return hv


def ehvicalc_vec(x, y_par, moboInfo, kriglist, pool=None):
    """Vectorised EHVI Function

    Vectorises above EHVI function as much as possible. Currently,
    still depends on the vectorisation of prediction.py prediction() and
    kadal.optim_tools.ehvi.exi2d.exi2d() for best performance boost.

    Args:
        x (np.ndarray): [n_pop, n_dv] Design variables for a population.
        y_par (np.ndarray): [n_par, n_obj] Current Pareto front.
        moboInfo (dict): Structure containing necessary information for
            multi-objective Bayesian optimization.
        kriglist ([kriging_model.Kriging]): n_obj-len list of objective
            Kriging instances.
        pool (mp.Pool, optional): An existing mp.Pool instance can be
            specified to reduce the overhead of starting a new mp.Pool
            with every iteration.

    Returns:
        hv (np.ndarray/float): n_pop-len array of hypervolumes for each
            samp, if input x is 2D. If input x is a 1D input array,
            n_pop = 1 is assumed and all inputs are design variables;
            a single hv float is returned (legacy behaviour).
    """
    reshape = False
    if x.ndim == 1:
        reshape = True
        x = x.reshape(1, -1)

    n_pop = x.shape[0]
    n_obj = len(kriglist)
    ref_point = moboInfo["refpoint"]

    if pool is None:
        pred = np.zeros([n_pop, n_obj])
        SSqr = np.zeros([n_pop, n_obj])
        hv = np.zeros(n_pop)

        # Prediction of each objective
        # Looks like prediction.py prediction is only set up for 1D arrays...
        for i in range(n_pop):
            for j in range(n_obj):
                pred[i, j], SSqr[i, j] = kriglist[j].predict(x[i, :],
                                                             ["pred", "SSqr"])

        # Compute (negative of) hypervolume
        for i in range(n_pop):
            hv[i] = -1 * exi2d(y_par, ref_point, pred[i, :], SSqr[i, :])

    else:
        # Parallelise serial evaluations with mp.Pool
        hv = pool_hv_exi2d(pool, x, kriglist, y_par, ref_point)

    # If 1D input, expects float output (legacy behaviour)
    if reshape:
        hv = hv[0]

    return hv


def ehvicalc_kmac3d(x, y_par, moboInfo, kriglist, pool=None):
    """Calculate 3D EHVI using Leiden Uni's KMAC c++ code.

    Uses the multi 'sliceupdate' mode.

    Args:
        x (np.ndarray): [n_pop, n_dv] Design variables for a population.
        y_par (np.ndarray): [n_par, n_obj] Current Pareto front.
        moboInfo (dict): Structure containing necessary information for
            multi-objective Bayesian optimization.
        kriglist ([kriging_model.Kriging]): n_obj-len list of objective
            Kriging instances.
        pool (mp.Pool, optional): An existing mp.Pool instance can be
            specified to reduce the overhead of starting a new mp.Pool
            with every iteration.

    Returns:
        hv (np.ndarray/float): n_pop-len array of hypervolumes for each
            samp, if input x is 2D. If input x is a 1D input array,
            n_pop = 1 is assumed and all inputs are design variables;
            a single hv float is returned (legacy behaviour).
        """
    reshape = False
    if x.ndim == 1:
        reshape = True
        x = x.reshape(1, -1)

    n_pop = x.shape[0]
    n_obj = len(kriglist)
    ref_point = moboInfo["refpoint"]

    hv = np.zeros(n_pop)

    if pool is None:
        pred = np.zeros([n_pop, n_obj])
        SSqr = np.zeros([n_pop, n_obj])
        # Prediction of each objective
        for i in range(n_pop):
            for j in range(n_obj):
                pred[i, j], SSqr[i, j] = kriglist[j].predict(x[i, :], ["pred", "SSqr"])

    else:
        pred, SSqr = pool_predict(pool, x, kriglist)

    # Invert inputs as KMAC expects maximisation - can do multiple samples at once
    # Compute (negative of) hypervolume using KMAC C++ code from Leiden Uni
    hv[:] = -kmac.ehvi3d_sliceupdate_multi(-y_par, -ref_point, -pred, SSqr)

    # If 1D input, expects float output (legacy behaviour)
    if reshape:
        hv = hv[0]

    return hv
