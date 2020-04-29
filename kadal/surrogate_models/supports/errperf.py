import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def errperf (T,P,type='rmse'):
    """
    Calculate error performance metric of the predicted value

    Args:
        T (nparray): True value
        P (nparray): Predicted value

    Return:
        V : Error metric
    """

    #Check the dimension between T and P
    if np.shape(T) != np.shape(P):
        raise Exception("The size of T and P is different")
    elif np.ndim(T) == 1:
        T = T.reshape(-1,1)
        P = P.reshape(-1,1)
    else:
        a,b = np.shape(T)
        if a == 1:
            T = T.reshape(-1, 1)
            P = P.reshape(-1, 1)
        else:
            pass

    #covert type to lowercase
    if not isinstance(type,str):
        raise TypeError("Type should be a string")
    type = type.lower()

    #Compute metric

    if type == 'e': #Errors
        V = T-P
    elif type == 'ae': #Absolute Errors
        Ve = errperf(T,P,'e')
        V = abs(Ve)
    elif type == 'mae': #Mean absolute error
        Vae = errperf(T,P,'ae')
        V = np.mean(Vae)
    elif type == 'se': #Squared error
        Ve = errperf(T,P,'e')
        V = Ve**2
    elif type == 'mse': #Mean SE
        Vse = errperf(T,P,'se')
        V = np.mean(Vse)
    elif type == 'rmse': #Root mean SE
        Vmse = errperf(T,P,'mse')
        V = np.sqrt(Vmse)
    elif type == 're': #Relative error
        num_zero = np.count_nonzero(T == 0)
        if num_zero > 0:
            idx1, idx2 = np.where(T == 0)
            Ve = errperf(T, P, 'e')
            V = Ve / T
            for i, item in enumerate(idx1):
                V[item,idx2[i]] = T[i]-P[i]
        else:
            Ve = errperf(T,P,'e')
            V = Ve/T
    elif type == 'are': #Absolute relative error
        Vre = errperf(T,P,'re')
        V = abs(Vre)
    elif type == 'mare': #Mean ARE
        Vare = errperf(T,P,'are')
        V = np.mean(Vare)
    elif type == 'sre': #Squared RE
        Vre = errperf(T,P,'re')
        V = Vre**2
    elif type == 'msre': #Mean SRE
        Vsre = errperf(T,P,'sre')
        V = np.mean(Vsre)
    elif type == 'rmsre': #Root mean squared RE
        Vmsre = errperf(T,P,'msre')
        V = np.sqrt(Vmsre)
    elif type == 'pe': #Percentage Error
        Vre = errperf(T,P,'re')
        V = Vre*100
    elif type == 'ape': #Absolute PE
        Vpe = errperf(T,P,'pe')
        V = abs(Vpe)
    elif type == 'mape': #Mean APE
        Vape = errperf(T,P,'ape')
        V = np.mean(Vape)
    elif type == 'spe': #Squared PE
        Vpe = errperf(T,P,'pe')
        V = Vpe**2
    elif type == 'mspe': #Mean SPE
        Vspe = errperf(T,P,'spe')
        V = np.mean(Vspe)
    elif type == 'rmspe': #Root mean squared PE
        Vmspe = errperf(T,P,'mspe')
        V = np.sqrt(Vmspe)
    elif type == 'r2': #R2 criterion
        SStot = np.sum((T-np.mean(T))**2)
        SSres = np.sum((T-P)**2)
        V = 1 - (SSres/SStot)
    else:
        raise Exception("Type is invalid")

    return V