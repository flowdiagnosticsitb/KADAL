from copy import deepcopy
import numpy as np
from numpy.random import random_sample
from kadal.misc.sampling import haltonsampling
from sobolsampling.sobol_new import sobol_points
from kadal.optim_tools.ga import SBX, mutation


def uncGA (fitnessfcn, lb, ub, opt='min',disp=False,npop=300,maxg=200,args=None,initialization=None):
    # only required for multi-objective fcn
    if isinstance(ub, int) or isinstance(ub, float):
        nvar = 1
        ub = np.array([ub])
        lb = np.array([lb])
    else:
        nvar = len(ub)
    pmut = 0.1     #mutation probability
    pcross = 0.95   #crossover probability
    history  = np.zeros(shape=[maxg,2]) #ask Kemas for explanation


    #Initialize population
    # samplenorm = haltonsampling.halton(nvar, npop)
    if initialization is None:
        samplenorm = sobol_points(npop, nvar)
    else:
        if initialization.ndim == 1:
            samplenorm = np.random.normal(initialization, np.std(args[1][0].KrigInfo['X_norm'],0)*0.2, (npop,nvar))
        else:
            n_init = np.size(initialization,0)
            nbatch = int(npop/n_init)
            samplenorm = np.zeros((npop,nvar))
            for ij in range(n_init-1):
                samplenorm[ij*nbatch:(ij+1)*nbatch, :] = np.random.normal(initialization[ij,:],
                                                                          np.std(args[1][0].KrigInfo['X_norm'],0)*0.2,
                                                                          (nbatch,nvar))
            print('+++++Check this size! in uncGA.py', np.size(samplenorm[(ij + 1) * nbatch:, :], 0))
            breakpoint()
            samplenorm[(ij+1)*nbatch:, :] = np.random.normal(initialization[(ij+1), :],
                                                             np.std(args[1][0].KrigInfo['X_norm'],0)*0.2,
                                                             (np.size(samplenorm[(ij+1)*nbatch:, :],0), nvar))
            samplenorm[samplenorm < 0] = 0
            samplenorm[samplenorm > 1] = 1

    population = np.zeros(shape=[npop,nvar+1])
    for i in range(0, npop):
        for j in range(0, nvar):
            population[i, j] = (samplenorm[i, j] * (ub[j] - lb[j])) + lb[j]
            # for i in range (0,npop):
            #     for j in range (0,nvar):
            #         population[i,j] = lb[j] + (ub[j]-lb[j])*random_sample()
        if args == None:
            temp= fitnessfcn(population[i, 0:nvar])
        else:
            temp= fitnessfcn(population[i,0:nvar],*args)
        population[i,nvar] = deepcopy(temp)

    #Evolution loop
    generation = 1
    oldFitness = 0
    while generation <= maxg:
        # for generation 1:1
        tempopulation = deepcopy(population)

        #Tournament Selection
        matingpool = np.zeros(shape=[npop,nvar])
        for kk in range (0,npop):
            ip1 = int(np.ceil(npop*random_sample())) #random number 1
            ip2 = int(np.ceil(npop*random_sample())) #random number 2
            while ip1 >= npop or ip2 >=npop:
                ip1 = int(np.ceil(npop * random_sample()))
                ip2 = int(np.ceil(npop * random_sample()))
            if ip2 == ip1: #In case random number 1 = random number 2
                while ip2 == ip1 or ip2>=npop:
                    ip2 = int(np.ceil(npop*random_sample()))

            lst  = np.arange(0,nvar)
            Ft1  = population[ip1,lst]
            Ft2  = population[ip2,lst]
            Fit1 = population[ip1,nvar]
            Fit2 = population[ip2,nvar]

            #Switch case, in Python we use if and elif instead of switch-case
            if opt == "max":
                if Fit1>Fit2:
                    matingpool [kk,:] = Ft1
                else :
                    matingpool [kk,:] = Ft2
            elif opt == "min":
                if Fit1<Fit2:
                    matingpool [kk,:] = Ft1
                else :
                    matingpool [kk,:] = Ft2
            else:
                pass


        #Crossover with tournament seelection
        child = np.zeros(shape=[2,nvar])
        lst = np.arange(0, nvar)
        for jj in range (0,npop,2):
            idx1 = int(np.ceil(npop*random_sample()))
            idx2 = int(np.ceil(npop*random_sample()))
            while idx1 >= npop or idx2 >= npop or idx1==idx2:
                idx1 = int(np.ceil(npop * random_sample()))
                idx2 = int(np.ceil(npop * random_sample()))
            if (random_sample() < pcross):
                child = SBX.SBX(matingpool[idx1, :], matingpool[idx2, :], nvar, lb, ub)
                tempopulation[jj,0:nvar] = child [0,:]
                tempopulation[jj+1,0:nvar] = child [1,:]
            else:
                tempopulation[jj, 0:nvar] = matingpool [idx1,:]
                tempopulation[jj + 1, 0:nvar] = matingpool [idx2,:]
            if args == None:
                tempopulation[jj, nvar]= fitnessfcn(tempopulation[jj, lst])
                tempopulation[jj + 1, nvar]= fitnessfcn(tempopulation[jj + 1, lst])
            else:
                tempopulation[jj,nvar]= fitnessfcn(tempopulation[jj,lst],*args)
                tempopulation[jj+1,nvar]= fitnessfcn(tempopulation[jj+1,lst],*args)

        #Combined Population for Elitism
        compopulation = np.vstack((population,tempopulation))

        #Sort Population based on their fitness value
        if opt == 'max':
            i = np.argsort(compopulation[:,nvar]) [::-1]
            compopulation = compopulation[i,:]
        elif opt == 'min':
            i = np.argsort(compopulation[:, nvar])
            compopulation = compopulation[i,:]

        #Record Optimum Solution
        bestFitness = compopulation[0,nvar]
        bestx   = compopulation[0,0:nvar]

        #Mutation
        for kk in range (1,(2*npop)):
            compopulation[kk,0:nvar] = mutation.gaussmut(compopulation[kk, 0:nvar], nvar, pmut, ub, lb)
            if args == None:
                compopulation[kk, nvar]= fitnessfcn(compopulation[kk, 0:nvar])
            else:
                compopulation[kk,nvar]= fitnessfcn (compopulation[kk,0:nvar],*args)

        history[generation-1,0]=generation
        history[generation-1,1]=bestFitness

        fiterr =  100*(abs(bestFitness-oldFitness))/bestFitness
        if disp:
            print("Done, generation ", generation, " | Best X = ", bestx, " | Fitness Error (%)= ", fiterr)
        generation = generation+1
        if fiterr <= 10**(-2) and generation >= 50:
            break

        oldFitness = bestFitness
        #Next Population
        for i in range (0,npop):
            population[i,:] = compopulation[i,:]


    #Show Best Fitness and Design Variables
    # print("Best Fitness = ",bestFitness)
    # for i in range (0,nvar):
    #     print("X",i+1," = ",bestx[i])

    return (bestx,bestFitness,history)


def uncGA_vec(fit_func, lb, ub, min_fit=True, disp=False, n_pop=300,
              max_gen=200, p_mut=0.1, p_cross=0.95, rand_seed=0, args=None,
              kwargs=None, initialization=None, min_gen=50, min_fit_err=1e-2,
              pool=None):
    """Vectorised uncGA

    Args:
        fit_func (func): Function used to evalutate the fitness of an
            individual.
        lb (np.ndarray): n_dv-len array of lower bounds.
        ub (np.ndarray): n_dv-len array of upper bounds.
        min_fit (bool, optional): Minimise or maximise fitness.
            Defaults to True (minimise fitness).
        disp (bool, optional): Print progress. Defaults to False.
        n_pop (int, optional): Population number. Defaults to 300.
        max_gen (int, optional): Max generations. Defaults to 200.
        p_mut (numeric, optional): Mutation probability. Between 0
            and 1. Defaults to 0.1.
        p_cross (numeric, optional): Crossover probability. Between 0
            and 1. Defaults to 0.95.
        rand_seed (int, optional): The seed for np.random.default_rng().
            Defaults to 0.
        args ([]/(), optional): Additional arguments passed to the
            specified fit_func.
        kwargs ([]/(), optional): Additional keyword arguments passed
            to the specified fit_func.
        initialization:
        min_gen (numeric, optional): The minimim number of generations,
            if after which the percentage change in fitness error is
            below the min_fit_err value, the optimisation stops.
            Defaults to 50.
        min_fit_err (numeric, optional): The percentage minimim fitness
            error. Defaults to 1e-2.

    Returns:
        (best_x, best_fitness, history): A tuple of results, where:
            best_x (np.ndarray): n_dv-len array of optimised design
                variables.
            best_fitness (float): The optimised fitness value
            history (np.ndarray): [max_gen, 2] A history of the fitness
                for each generation.
    """
    if args is None:
        args = []
    # only required for multi-objective function
    if np.isscalar(ub):
        n_dv = 1
        ub = np.array([ub])
        lb = np.array([lb])
    else:
        n_dv = len(ub)

    history = np.zeros([max_gen, 2])

    # Initialize population
    if initialization is None:
        # samplenorm = haltonsampling.halton(nvar, n_pop)
        sample_norm = sobol_points(n_pop, n_dv)
    elif initialization.ndim == 1:
        x_norm = args[1][0].KrigInfo['X_norm']
        sample_norm = np.random.normal(initialization,
                                       np.std(x_norm, 0) * 0.2,
                                       (n_pop, n_dv))
    else:
        n_init = np.size(initialization, 0)
        nbatch = int(n_pop / n_init)
        sample_norm = np.zeros((n_pop, n_dv))
        x_norm = args[1][0].KrigInfo['X_norm']

        for ij in range(n_init - 1):
            sample_norm[ij * nbatch:(ij + 1) * nbatch, :] = np.random.normal(initialization[ij, :],
                                                                             np.std(x_norm, 0) * 0.2,
                                                                             (nbatch, n_dv))
        print('+++++Check this size! in uncGA.py', np.size(sample_norm[(ij + 1) * nbatch:, :], 0))
        breakpoint()
        sample_norm[(ij + 1) * nbatch:, :] = np.random.normal(initialization[(ij + 1), :],
                                                              np.std(x_norm, 0) * 0.2,
                                                              (np.size(sample_norm[(ij+1)*nbatch:, :], 0), n_dv))
        sample_norm[sample_norm < 0] = 0
        sample_norm[sample_norm > 1] = 1

    population = np.zeros(shape=[n_pop, n_dv + 1])
    # population[:, :n_dv] = lb + (ub - lb) * random_sample(size=n_dv)
    population[:, :n_dv] = (sample_norm * (ub - lb)) + lb
    population[:, n_dv] = fit_func(population[:, :n_dv], *args, pool=pool)

    # Evolution loop
    generation = 1
    old_fitness = 0
    rng = np.random.default_rng(rand_seed)
    while generation <= max_gen:
        # for generation 1:1
        temp_pop = population.copy()

        # Tournament Selection
        # Unroll loop by preallocating valid list of indices!
        # Set up pairs of random indices in semi-open interval [0, n_pop)
        ip_1s = rng.integers(1, n_pop, size=n_pop)  # n_pop-len random index
        ip_2s = rng.integers(1, n_pop, size=n_pop)  # n_pop-len random index

        # For each random index pair ip_1 and ip_2, check not the same
        ip_equal = ip_1s == ip_2s
        while ip_equal.any():
            # Find the index in ip_2 and replace with a new random index
            i = np.where(ip_equal)[0]
            new_ips = rng.integers(1, n_pop, size=len(i))
            ip_2s[i] = new_ips
            ip_equal = ip_1s == ip_2s

        # Order the population according to preallcated 'random' indices
        feat_1s = population[ip_1s, :n_dv]
        feat_2s = population[ip_2s, :n_dv]
        fit_1s = population[ip_1s, n_dv]
        fit_2s = population[ip_2s, n_dv]

        if min_fit:
            fit_mask = fit_1s < fit_2s
        else:
            fit_mask = fit_1s > fit_2s

        mating_pool = np.zeros([n_pop, n_dv])
        mating_pool[np.where(fit_mask)[0], :] = feat_1s[fit_mask]
        mating_pool[np.where(~fit_mask)[0], :] = feat_2s[~fit_mask]

        # Crossover with tournament selection
        # Stepping through pairs, so only need array length of half population
        idx_1s = rng.integers(1, n_pop, size=int(np.ceil(n_pop / 2)))
        idx_2s = rng.integers(1, n_pop, size=int(np.ceil(n_pop / 2)))

        # For each random index pair ip_1 and ip_2, check not the same
        ip_equal = idx_1s == idx_2s
        while ip_equal.any():
            # Find the index in ip_2 and replace with a new random index
            i = np.where(ip_equal)[0]
            new_ips = rng.integers(1, n_pop, size=len(i))
            idx_2s[i] = new_ips
            ip_equal = idx_1s == idx_2s

        # Run for pairs of samples less than probability of crossover
        p_rand = rng.random(size=int(np.floor(n_pop / 2)))  # random floats for crossover
        cross_mask = p_rand < p_cross
        ix = np.where(cross_mask)[0]
        parent_1s = mating_pool[idx_1s[ix], :]
        parent_2s = mating_pool[idx_2s[ix], :]
        child_1s, child_2s = SBX.sbx_vec(parent_1s, parent_2s, lb, ub,
                                         rand_seed=rand_seed)

        temp_pop[ix, :n_dv] = child_1s
        temp_pop[ix + 1, :n_dv] = child_2s

        # For others, just copy design vars from mating pool
        ix = np.where(~cross_mask)[0]
        temp_pop[ix, :n_dv] = mating_pool[idx_1s[ix], :]
        temp_pop[ix + 1, :n_dv] = mating_pool[idx_2s[ix], :]

        # Eval fitness of population
        temp_pop[:, n_dv] = fit_func(temp_pop[:, :n_dv], *args, pool=pool)

        # Combined Population for Elitism
        total_pop = np.vstack((population, temp_pop))

        # Sort Population based on their fitness value
        if min_fit:
            i = np.argsort(total_pop[:, n_dv])
        else:
            i = np.argsort(total_pop[:, n_dv])[::-1]

        total_pop = total_pop[i, :]

        # Record Optimum Solution
        best_fitness = total_pop[0, n_dv]
        best_x = total_pop[0, :n_dv]

        # Mutation
        total_pop[1:, :n_dv] = mutation.gaussmut_vec(total_pop[1:, :n_dv].copy(),
                                                     p_mut, ub, lb, rand_seed)
        total_pop[1:, n_dv] = fit_func(total_pop[1:, :n_dv].copy(), *args, pool=pool)

        history[generation - 1, 0] = generation
        history[generation - 1, 1] = best_fitness

        fit_err = 100 * np.abs(best_fitness - old_fitness) / best_fitness

        if disp:
            print("GA: generation ", generation, " | Best X = ", best_x,
                  " | Fitness Error (%)= ", fit_err)

        generation = generation + 1
        if fit_err <= min_fit_err and generation >= min_gen:
            break

        # Next Population
        population[:] = total_pop[:n_pop, :].copy()
        old_fitness = best_fitness

    return (best_x, best_fitness, history)


def uncGA2(fit_func, lb, ub, min_fit=True, disp=False, n_pop=300, maxg=200,
           pmut=0.1, pcross=0.95, rand_seed=0, args=None, initialization=None,
           pool=None):
    """

    Args:
        fit_func:
        lb:
        ub:
        min_fit (bool, optional): Minimise or maximise fitness.
            Defaults to True (minimise fitness).
        disp:
        n_pop (int, optional): Population number. Defaults to 300.
        maxg (int, optional): Max generations. Defaults to 200.
        pmut (numeric, optional): Mutation probability. Between 0 and 1.
            Defaults to 0.1.
        pcross (numeric, optional): Crossover probability.  Between 0 and 1.
            Defaults to 0.95.
        rand_seed (int, optional): The seed for np.random.default_rng().
            Defaults to 0.
        args:
        initialization:

    Returns:

    """
    # only required for multi-objective fcn
    if np.isscalar(ub):
        n_dv = 1
        ub = np.array([ub])
        lb = np.array([lb])
    else:
        n_dv = len(ub)

    history = np.zeros([maxg, 2])
    print('init pop')
    # Initialize population
    if initialization is None:
        # samplenorm = haltonsampling.halton(nvar, n_pop)
        sample_norm = sobol_points(n_pop, n_dv)
    else:
        x_norm = args[1][0].KrigInfo['X_norm']
        if initialization.ndim == 1:
            sample_norm = np.random.normal(initialization,
                                           np.std(x_norm, 0) * 0.2,
                                           (n_pop, n_dv))
        else:
            n_init = np.size(initialization, 0)
            nbatch = int(n_pop / n_init)
            sample_norm = np.zeros((n_pop, n_dv))
            # TODO: Is there an off by one error here with ij?
            for ij in range(n_init - 1):
                sample_norm[ij * nbatch:(ij + 1) * nbatch, :] = np.random.normal(initialization[ij, :],
                                                                                 np.std(x_norm, 0) * 0.2,
                                                                                 (nbatch, n_dv))
            sample_norm[(ij + 1) * nbatch:, :] = np.random.normal(initialization[(ij + 1), :],
                                                                  np.std(x_norm, 0) * 0.2,
                                                                  (
                                                                  np.size(sample_norm[(ij + 1) * nbatch:, :], 0), n_dv))
            sample_norm[sample_norm < 0] = 0
            sample_norm[sample_norm > 1] = 1

    population = np.zeros(shape=[n_pop, n_dv + 1])
    population = np.zeros(shape=[n_pop, n_dv + 1])
    # population[:, :n_dv] = lb + (ub - lb) * random_sample(size=n_dv)
    population[:, :n_dv] = (sample_norm * (ub - lb)) + lb
    population[:, n_dv] = fit_func(population[:, :n_dv], *args, pool=pool)

    print('before loop')
    # Evolution loop
    generation = 1
    old_fitness = 0
    rng = np.random.default_rng(rand_seed)
    while generation <= maxg:
        # for generation 1:1
        temp_pop = deepcopy(population)

        # Tournament Selection
        mating_pool = np.zeros([n_pop, n_dv])

        for kk in range(0, n_pop):
            ip1 = int(np.ceil(n_pop * random_sample()))  # random number 1
            ip2 = int(np.ceil(n_pop * random_sample()))  # random number 2
            while ip1 >= n_pop or ip2 >= n_pop:
                ip1 = int(np.ceil(n_pop * random_sample()))
                ip2 = int(np.ceil(n_pop * random_sample()))
            # In case random number 1 = random number 2
            if ip2 == ip1:
                while ip2 == ip1 or ip2 >= n_pop:
                    ip2 = int(np.ceil(n_pop * random_sample()))

            lst = np.arange(0, n_dv)
            Ft1 = population[ip1, lst]
            Ft2 = population[ip2, lst]
            fit_1 = population[ip1, n_dv]
            fit_2 = population[ip2, n_dv]

            # Switch case, in Python we use if and elif instead of switch-case
            if not min_fit:
                if fit_1 > fit_2:
                    mating_pool[kk, :] = Ft1
                else:
                    mating_pool[kk, :] = Ft2
            else:
                if fit_1 < fit_2:
                    mating_pool[kk, :] = Ft1
                else:
                    mating_pool[kk, :] = Ft2

        # Crossover with tournament seelection
        child = np.zeros(shape=[2, n_dv])
        lst = np.arange(0, n_dv)
        for jj in range(0, n_pop, 2):
            idx1 = int(np.ceil(n_pop * random_sample()))
            idx2 = int(np.ceil(n_pop * random_sample()))
            while idx1 >= n_pop or idx2 >= n_pop or idx1 == idx2:
                idx1 = int(np.ceil(n_pop * random_sample()))
                idx2 = int(np.ceil(n_pop * random_sample()))
            if (random_sample() < pcross):
                child = SBX.SBX(mating_pool[idx1, :], mating_pool[idx2, :], n_dv, lb, ub)
                temp_pop[jj, 0:n_dv] = child[0, :]
                temp_pop[jj + 1, 0:n_dv] = child[1, :]
            else:
                temp_pop[jj, 0:n_dv] = mating_pool[idx1, :]
                temp_pop[jj + 1, 0:n_dv] = mating_pool[idx2, :]
            if args == None:
                temp_pop[jj, n_dv] = fit_func(temp_pop[jj, lst], pool=pool)
                temp_pop[jj + 1, n_dv] = fit_func(temp_pop[jj + 1, lst], pool=pool)
            else:
                temp_pop[jj, n_dv] = fit_func(temp_pop[jj, lst], *args, pool=pool)
                temp_pop[jj + 1, n_dv] = fit_func(temp_pop[jj + 1, lst], *args, pool=pool)

        # Combined Population for Elitism
        total_pop = np.vstack((population, temp_pop))

        # Sort Population based on their fitness value
        if not min_fit:
            i = np.argsort(total_pop[:, n_dv])[::-1]
            total_pop = total_pop[i, :]
        else:
            i = np.argsort(total_pop[:, n_dv])
            total_pop = total_pop[i, :]

        # Record Optimum Solution
        best_fitness = total_pop[0, n_dv]
        best_x = total_pop[0, 0:n_dv]

        # Mutation
        for kk in range(1, (2 * n_pop)):
            total_pop[kk, 0:n_dv] = mutation.gaussmut(total_pop[kk, 0:n_dv], n_dv, pmut, ub, lb)
            if args == None:
                total_pop[kk, n_dv] = fit_func(total_pop[kk, 0:n_dv], pool=pool)
            else:
                total_pop[kk, n_dv] = fit_func(total_pop[kk, 0:n_dv], *args, pool=pool)

        history[generation - 1, 0] = generation
        history[generation - 1, 1] = best_fitness

        fiterr = 100 * (abs(best_fitness - old_fitness)) / best_fitness
        if disp:
            print("Done, generation ", generation, " | Best X = ", best_x, " | Fitness Error (%)= ", fiterr)
        generation = generation + 1
        if fiterr <= 10 ** (-2) and generation >= 50:
            break

        old_fitness = best_fitness
        # Next Population
        for i in range(0, n_pop):
            population[i, :] = total_pop[i, :]

    # Show Best Fitness and Design Variables
    # print("Best Fitness = ",bestFitness)
    # for i in range (0,nvar):
    #     print("X",i+1," = ",bestx[i])

    return (best_x, best_fitness, history)