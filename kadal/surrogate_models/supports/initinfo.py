class KrigingModel:
    """A container class for a Kriging model.
    
    A class implementation of init_kriginfo for future work.
    
    Date: 24/05/2019
    Author: Tim Jim
    """
    def __init__(self, objectives=1):
        """Initialise a new Kriging model.
        
        Args:
            objectives (int, optional): Number of objective functions.
            Defaults to 1.
    
        Raises:
            TypeError: If objectives is not an integer.
            ValueError: If objectives < 1.
        """
        self.X = None
        self.nvar = None
        self.problem = 'branin'
        self.nsamp = None
        self.nrestart = 5
        self.kernel = 'gaussian'
        self.nugget = -6
        self.standardization = True

        if not isinstance(objectives, int):
            raise TypeError('The objectives parameter must be an integer.')

        elif objectives == 1:
            self.krignum = None
            self.multiobj = False

        elif objectives > 1:
            self.krignum = 1
            self.multiobj = True

        else:
            raise ValueError("The number of objectives must be 1 or greater.")

        multikeys = ['y', 'lb', 'ub', 'Theta', 'U', 'Psi', 'BE', 'y_mean',
                     'y_std', 'SigmaSqr', 'idx', 'F', 'wgkf', 'plscoeff']
        for key in multikeys:
            setattr(self, key, [None, ] * objectives)


def init_kriginfo(objectives=1):
    """Initialise the Kriging model dictionary structure.
    
    Initialize the values of KrigInfo, the value will be set to default value.
    You can change the value of KrigInfo outside this function.
    
    If objectives is set to 1, a single objective Kriging will be initialised.
    Otherwise, if objectives > 1, a multiobjective Kriging model will be
    initialised.
    
    For multiobjective Kriging, some values are initialised with lists with
    the length of objectives, a Kriging model is built for each objective.

    Default values:
        KrigInfo['X'] = None
        KrigInfo['y'] = [None] * objectives
        KrigInfo['nvar'] = None
        KrigInfo['problem'] = 'branin'
        KrigInfo['nsamp']= None
        KrigInfo['nrestart'] = 5
        KrigInfo['ub']= [None] * objectives
        KrigInfo['lb']= [None] * objectives
        KringInfo'[Theta'] = [None] * objectives
        KringInfo'[U'] = [None] * objectives
        KringInfo'[Psi'] = [None] * objectives
        KringInfo'[BE'] = [None] * objectives
        KringInfo'[y_mean'] = [None] * objectives
        KringInf'[y_std'] = [None] * objectives
        KringInfo'[SigmaSqr'] = [None] * objectives
        KringInfo'[idx'] = [None] * objectives
        KringInfo'[F'] = [None] * objectives
        KringInfo'[wgkf'] = [None] * objectives
        KringInfo'[plscoeff'] = [None] * objectives
        KrigInfo['kernel'] = 'gaussian'
        KrigInfo['nugget'] = -6

    Args:
        objectives (int, optional): Number of objective functions.
            Defaults to 1.

    Returns:
        KrigInfo - A structure containing information of the constructed Kriging of the objectives function.

    Raises:
        TypeError: If objectives is not an integer.
        ValueError: If objectives < 1.
    """
    KrigInfo = dict()
    KrigInfo['X'] = None
    KrigInfo['nvar'] = None
    KrigInfo['problem'] = 'branin'
    KrigInfo['nsamp'] = None
    KrigInfo['nrestart'] = 5
    KrigInfo['kernel'] = 'gaussian'
    KrigInfo['nugget'] = -6
    KrigInfo['standardization'] = True

    if not isinstance(objectives, int):
        raise TypeError('The objectives parameter must be an integer.')

    elif objectives == 1:
        KrigInfo['krignum'] = None
        KrigInfo['multiobj'] = False

    elif objectives > 1:
        KrigInfo['krignum'] = 1
        KrigInfo['multiobj'] = True

    else:
        raise ValueError("The number of objectives must be 1 or greater.")

    multikeys = ['y', 'lb', 'ub', 'Theta', 'U', 'Psi', 'BE', 'y_mean', 'y_std',
                 'SigmaSqr', 'idx', 'F', 'wgkf', 'plscoeff',"NegLnLike"]
    for key in multikeys:
        KrigInfo[key] = [0] * objectives

    return KrigInfo


def initkriginfo(type, objective=1):
    """Wrapper for init_kriginfo.

    Preserves old API behaviour.

    Args:
        type (str): 'single' or 'multi'. Now unused.
        objective (int, optional): Number of objective functions.
    """
    # print('DeprecationWarning: Consider changing initkriginfo calls to init_kriginfo')
    return init_kriginfo(objectives=objective)


def copymultiKrigInfo(KrigMultiInfo, num):
    """
    Function for copying multi-objective KrigInfo into single KrigInfo

    Inputs:
        KrigMultiInfo - Multi-objective KrigInfo
        num - Index of objective

    Output:
        KrigNewInfo - A structure containing information of the constructed Kriging of the objective function
                      taken from KrigMultiInfo.
    """

    KrigNewInfo = dict()
    KrigNewInfo['X'] = KrigMultiInfo['X']
    KrigNewInfo['y'] = KrigMultiInfo['y']
    KrigNewInfo['nvar'] = KrigMultiInfo['nvar']
    KrigNewInfo['problem'] = KrigMultiInfo['problem']
    KrigNewInfo['nsamp'] = KrigMultiInfo['nsamp']
    KrigNewInfo['nrestart'] = KrigMultiInfo['nrestart']
    KrigNewInfo['ub'] = KrigMultiInfo['ub']
    KrigNewInfo['lb'] = KrigMultiInfo['lb']
    KrigNewInfo['kernel'] = KrigMultiInfo['kernel']
    KrigNewInfo['nugget'] = KrigMultiInfo['nugget']
    KrigNewInfo['optimizer'] = KrigMultiInfo['optimizer']
    keys = ['Theta', 'U', 'Psi', 'BE', 'y_mean', 'y_std', 'SigmaSqr', 'idx', 'F', 'wgkf', 'plscoeff',"NegLnLike"]
    for key in keys:
        KrigNewInfo[key] = KrigMultiInfo[key][num]

    return KrigNewInfo