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
            samplenorm = np.random.normal(initialization,.08,(npop,nvar))
        else:
            n_init = np.size(initialization,0)
            nbatch = int(npop/n_init)
            samplenorm = np.zeros((npop,nvar))
            for ij in range(n_init-1):
                samplenorm[ij*nbatch:(ij+1)*nbatch, :] = np.random.normal(initialization[ij,:],.075,(nbatch,nvar))
            samplenorm[(ij+1)*nbatch:, :] = np.random.normal(initialization[(ij+1), :], .075,
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