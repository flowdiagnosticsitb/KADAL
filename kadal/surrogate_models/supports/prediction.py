import numpy as np
from numpy.linalg import solve as mldivide
from kadal.surrogate_models.supports.kernelfunc import calckernel
from kadal.misc.sampling.samplingplan import standardize
from kadal.surrogate_models.supports.trendfunction import compute_regression_mat
from scipy.special import erf
from scipy.spatial.distance import cdist


def get_val(KrigInfo, key, num=None):
    """Helper function to get values from KrigInfo dictionary.

    Multiobjective Kriging values are not mapped to single values and
    need to be extracted using the num parameter.

    If running multiobjective Kriging, values are stored in a
    list/dictionary under each key in the Kriging model dictionary,
    where the index/key is the objective function number 'num'.

    Args:
        KrigInfo (dict): The Kriging model dictionary.
        key (str): The variable key of desired value.
        num (int, optional): The objective function number. Defaults
            to None.

    Returns:
        Value extracted from dictionary.

    Raises:
        KeyError: If key does not exist in KrigInfo or if specified
            key also requires a multiobjective function number and
            num is not specified or nonexistent.
        IndexError: If a num is specified that is outside the range of
            the list for the desired key variable.
    """
    try:
        val = KrigInfo[key]
    except KeyError:
        msg = f"'Key {key}' was not found in Kriging model dictionary."
        raise KeyError(msg)

    if num is not None:
        try:
            val = val[num]
        except KeyError:
            msg = (
                f"Objective function number '{num}' was not found for key "
                f"'{key}' in Kriging model dictionary."
            )
            raise KeyError(msg)
        except IndexError:
            msg = (
                f"Objective function number '{num}' was not found for key "
                f"'{key}' in Kriging model dictionary."
            )
            raise IndexError(msg)

    # I'm not sure which one you're using!
    if isinstance(val, dict):
        keys = ", ".join([str(v) for v in val.keys()])
        msg = f"{key} has multiobjective function keys: {keys}. Specify 'num'."
        raise ValueError(msg)

    if isinstance(val, list):
        keys = ", ".join([str(v) for v in range(len(val))])
        msg = f"{key} has multiobjective function keys: {keys}. Specify 'num'."
        raise ValueError(msg)
    return val


def prediction(x, KrigInfo, predtypes, num=None, drm=None, **kwargs):
    """Predict response using a given Kriging model.

    Calculates expected improvement (for optimization), SSqr, Kriging
    prediction, and prediction from regression function

    Information: This function is a modification from "Forrester, A.,
    Sobester, A., & Keane, A. (2008). Engineering design via surrogate
    modelling:  a practical guide. John Wiley & Sons."

    Variables used from Kriging model dictionary KrigInfo:
        Xnorm - (nsamp x nvar) matrix of normalized experimental design.
        Y - (nsamp x 1) vector of responses.
        PHI - (nsamp x nind) matrix of regression function.
        idx - (nind x nvar) matrix consisting of polynomial index for
            regression function.
        kernel - Type of kernel function.
        wgkf - (1 x nkrnl) vector of weights for kernel functions.
        U - Choleski factorisation of correlation matrix.
        xparam - Hyperparameters of the Kriging model.
        BE - Coefficients of regression function
        SigmaSqr - SigmaSqr (Kriging variance) of the Kriging model

    N.B. Output will be a list if several predtypes are specified.

    Author: Pramudita Satria Palar(pramsatriapalar@gmail.com, pramsp@ftmd.itb.ac.id)

    Args:
        x (list/np.array): Prediction site (will be normalized to [-1,1])
        KrigInfo (dict): A structure containing necessary information of
            a constructed Kriging model.
        predtypes (str/[str]): Requested outputs at prediction site x.
            Valid predtypes are:
            'pred' - for Kriging prediction.
            'SSqr' - for Kriging prediction error.
            'fpc' - Kriging trend function.
            'lcb' -
            'ebe' -
            'EI' - for expected improvement.
            'poi' -
            'pof' -
        num (int, optional): Objective Function number. Defaults to None.

    Returns:
        If only one output specified through predtypes, a single value
        or array is returned. Else a list of each output is returned.

    Raises:
        ValueError:
        KeyError:
    """
    nvar = KrigInfo["nvar"]
    if KrigInfo["n_princomp"] is not False:
        nvar = KrigInfo["n_princomp"]
    kernel = KrigInfo["kernel"]
    nkernel = KrigInfo["nkernel"]
    # p = 2  # from reference  # Apparently unused?

    # If vector, turn into 1D array
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # # Is this extra check necessary?
    # if KrigInfo['multiobj'] is True and 'num' in KrigInfo:
    #     wgkf = KrigInfo['wgkf'][num]
    #     idx = KrigInfo['idx'][num]

    if KrigInfo["standardization"] is False:
        X = KrigInfo["X"]
        y = get_val(KrigInfo, "y", num)
    else:
        X = KrigInfo["X_norm"]
        if "y_norm" in KrigInfo:
            y = get_val(KrigInfo, "y_norm", num)
        else:
            y = get_val(KrigInfo, "y", num)

    theta = 10 ** get_val(KrigInfo, "Theta", num)
    U = get_val(KrigInfo, "U", num)
    PHI = get_val(KrigInfo, "F", num)
    BE = get_val(KrigInfo, "BE", num)
    wgkf = get_val(KrigInfo, "wgkf", num)
    idx = get_val(KrigInfo, "idx", num)
    SigmaSqr = get_val(KrigInfo, "SigmaSqr", num)

    if KrigInfo["type"].lower() == "kpls":
        plscoeff = get_val(KrigInfo, "plscoeff", num)

    if KrigInfo["standardization"] is True:
        if KrigInfo["normtype"] == "default":
            x = standardize(
                x, 0, type="default", range=np.vstack((KrigInfo["lb"], KrigInfo["ub"]))
            )
        elif KrigInfo["normtype"] == "std":
            x = (x - KrigInfo["X_mean"]) / KrigInfo["X_std"]
        else:
            msg = (
                f"Kriging model dictionary 'normtype' value: "
                f"'{KrigInfo['normtype']}' is not recognised."
            )
            raise ValueError(msg)

    if drm is not None:
        if drm.kernel != "precomputed":
            x = drm.transform(x.copy())
            if KrigInfo["standardization"] is True:
                x = standardize(
                    x,
                    0,
                    type="default",
                    range=np.vstack((KrigInfo["lb2"], KrigInfo["ub2"])),
                )
        else:
            feat = np.size(x, 1)
            k_mat = customkernel(
                x, KrigInfo["orig_X"], KrigInfo["kpcaw"], feat, type="gaussian"
            )
            x = drm.transform(k_mat)
            if KrigInfo["standardization"] is True:
                x = standardize(
                    x,
                    0,
                    type="default",
                    range=np.vstack((KrigInfo["lb2"], KrigInfo["ub2"])),
                )

    # Calculate number of sample points
    n = np.ma.size(X, axis=0)
    npred = np.size(x, axis=0)

    # Construct regression matrix for prediction
    bound = np.vstack(
        (-np.ones(shape=[1, KrigInfo["nvar"]]), np.ones(shape=[1, KrigInfo["nvar"]]))
    )
    PC = compute_regression_mat(idx, x, bound, np.ones(shape=[KrigInfo["nvar"]]))
    fpc = np.dot(PC, BE)

    PsiComp = np.zeros(shape=[n, npred, nkernel])

    # Fill psi vector
    if KrigInfo["type"].lower() == "kriging":
        for ii in range(0, nkernel):
            psi_i = calckernel(X, x, theta, nvar, type=kernel[ii])
            PsiComp[:, :, ii] = wgkf[ii] * psi_i
        psi = np.sum(PsiComp, 2)

    elif KrigInfo["type"].lower() == "kpls":
        for ii in range(0, nkernel):
            psi_i = calckernel(
                X, x, theta, KrigInfo["nvar"], type=kernel[ii], plscoeff=plscoeff
            )
            PsiComp[:, :, ii] = wgkf[ii] * psi_i
        psi = np.sum(PsiComp, 2)

    else:
        msg = (
            f"Kriging model dictionary 'type' value: '{KrigInfo['type']}'"
            f"is not recognised."
        )
        raise ValueError(msg)

    # Calculate prediction
    f = fpc + np.dot(
        np.transpose(psi), mldivide(U, mldivide(np.transpose(U), (y - np.dot(PHI, BE))))
    )

    if num == None:
        if KrigInfo["norm_y"] == True:
            f = stdtoreal(f, KrigInfo)
    else:
        if KrigInfo["norm_y"] == True:
            f = stdtoreal(f, KrigInfo, num=num)

    # Compute sigma-squared error
    dummy1 = mldivide(U, mldivide(np.transpose(U), psi))
    dummy2 = mldivide(U, mldivide(np.transpose(U), PHI))
    term1 = 1 - np.sum(np.transpose(psi) * np.transpose(dummy1), 1)
    ux = (np.dot(np.transpose(PHI), dummy1)) - np.transpose(PC)
    term2 = ux * (mldivide(np.dot(np.transpose(PHI), dummy2), ux))
    SSqr = np.dot(SigmaSqr, (term1 + term2))
    s = abs(SSqr) ** 0.5

    # Switch prediction type
    if isinstance(predtypes, str):
        predtypes = [predtypes]

    outputs = []  # Collect requested outputs into a list
    for pred in predtypes:
        if pred.lower() == "pred":
            output = f
        elif pred.lower() == "ssqr":
            output = SSqr.T
        elif pred.lower() == "s":
            output = s.T
        elif pred.lower() == "fpc":
            output = fpc
        elif pred.lower() == "lcb":
            output = f - np.dot(KrigInfo["sigmalcb"], SSqr)
        elif pred.lower() == "ebe":
            output = -SSqr
        elif pred.lower() == "ei":
            yBest = np.min(y)
            if SSqr.all() == 0:
                ExpImp = 0
            else:
                EITermOne = (yBest - f) * (
                    0.5 + 0.5 * erf((1 / np.sqrt(2)) * (yBest - f) / np.transpose(s))
                )
                EITermTwo = (
                    np.transpose(s)
                    / np.sqrt(2 * np.pi)
                    * np.exp(-0.5 * (yBest - f) ** 2 / np.transpose(SSqr))
                )

                # Give penalty for CMA-ES optimizer, if both term produce 0.
                # Else in certain conditions, it may leads to error in CMA-ES.
                realmin = np.finfo(float).tiny
                if not EITermOne.any() and not EITermTwo.any():
                    tiny_number = np.random.uniform(realmin, realmin * 100)
                    ExpImp = np.array([[tiny_number]])
                else:
                    ExpImp = EITermOne + EITermTwo + realmin
            output = -ExpImp
        elif pred.lower() == "poi":
            ProbImp = 0.5 + 0.5 * erf(
                1 / np.sqrt(2) * (np.min(y) - f) / np.transpose(s)
            )
            output = -ProbImp
        elif pred.lower() == "pof":
            if KrigInfo["limittype"] == ">" or KrigInfo["limittype"] == ">=":
                ProbFeas = 0.5 + 0.5 * erf(
                    1 / np.sqrt(2) * ((f - KrigInfo["limit"]) / np.transpose(s))
                )
            elif KrigInfo["limittype"] == "<" or KrigInfo["limittype"] == "<=":
                ProbFeas = 0.5 + 0.5 * erf(
                    1 / np.sqrt(2) * ((KrigInfo["limit"] - f) / np.transpose(s))
                )
            else:
                raise ValueError("Limit Type is not available yet!")
            output = ProbFeas
        else:
            msg = f"Specified prediction type: '{pred}' is not recognised."
            raise NotImplementedError(msg)
        outputs.append(output)

    # If only one output specified, try to return as single value or array.
    if len(outputs) == 1:
        try:
            return outputs[0].item()
        except ValueError:
            return outputs[0]
    else:
        return outputs


def stdtoreal(f, KrigInfo, num=None):
    if KrigInfo["normtype"] == "default":
        if num == None:
            ymax = np.max(KrigInfo["y"])
            ymin = np.min(KrigInfo["y"])
        else:
            ymax = np.max(KrigInfo["y"][num])
            ymin = np.min(KrigInfo["y"][num])
        f = f / 2 + 0.5
        f = f * (ymax - ymin) + ymin
    elif KrigInfo["normtype"] == "std":
        f = KrigInfo["y_mean"] + KrigInfo["y_std"] * f

    return f


def customkernel(XN, XM, w, nvar, type="gaussian"):
    if type == "gaussian":
        K = gausskernel(XN, XM, w, nvar)
    elif type == "poly":
        K = polykernel(XN, XM, w, nvar)
    else:
        raise NotImplementedError("other type of kernel is not supported")

    return K


def polykernel(XN, XM, w, nvar):
    g = 10 ** w[0]
    c = 10 ** w[1]
    d = w[2]
    K = (g * np.dot(XN, XM.T) + c) ** d
    return K


def gausskernel(XN, XM, w, nvar):
    w = 10 ** w
    mdist = np.zeros((np.size(XN, 0), np.size(XM, 0), nvar))
    for ii in range(0, nvar):
        X1 = np.transpose(np.array([XN[:, ii]]))
        X2 = np.transpose(np.array([XM[:, ii]]))
        mdist[:, :, ii] = (cdist(X1, X2, "euclidean") ** 2) / (w[ii] ** 2)
    Psi = np.exp(-0.5 * np.sum(mdist, 2))
    return Psi
