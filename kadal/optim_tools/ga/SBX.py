import numpy as np
from copy import deepcopy
from numpy.random import random_sample

#Simulated Binary Crossover (SBX)
def SBX (Parent_1,Parent_2,nvar,lb,ub):

    n = 17
    u = random_sample()

    beta_1 = np.zeros(shape=[nvar])
    beta_2 = np.zeros(shape=[nvar])
    beta_u = np.zeros(shape=[nvar])
    P_1 = np.zeros(shape=[nvar])
    P_2 = np.zeros(shape=[nvar])
    offspring = np.zeros(shape=[2,nvar])

    for i in range (0,nvar):
        div = (abs(Parent_2[i]-Parent_1[i])+1e-5)
        beta_1 [i] = (Parent_1[i] + Parent_2[i] - 2*lb[i])/ div
        beta_u [i] = (2*ub[i] - Parent_1[i] - Parent_2[i])/ div
        if beta_1[i] <= 1:
            P_1[i] = 0.5*beta_1[i]**(n+1)
            beta_1[i] = (2*u*P_1[i])**(1/(n+1))
        else:
            P_1[i] = 0.5*(2-(1/(beta_1[i]**(n+1))))
            beta_1[i] = (1/(2-2*u*P_1[i]))**(1/(n+1))

        if beta_u[i] <= 1:
            P_2[i] = 0.5*beta_u[i]**(n+1)
            beta_2[i] = (2*u*P_2[i])**(1/(n+1))
        else:
            P_2[i] = 0.5*(2-(1/(beta_u[i]**(n+1))))
            beta_2[i] = (1/(2-2*u*P_2[i]))**(1/(n+1))

        offspring[0,i] = 0.5* ((Parent_1[i]+Parent_2[i])-beta_1[i]*abs(Parent_2[i]-Parent_1[i]))
        offspring[1,i] = 0.5* ((Parent_1[i]+Parent_2[i])+beta_2[i]*abs(Parent_2[i]-Parent_1[i]))

        if offspring[0,i] < lb[i] or offspring[0,i]>ub[i]:
            offspring[0,i] = lb[i]+(ub[i]-lb[i])*random_sample()

        if offspring[1,i] < lb[i] or offspring[1,i]>ub[i]:
            offspring[1,i] = lb[i]+(ub[i]-lb[i])*random_sample()

    offspring = np.vstack((offspring[0,:],offspring[1,:]))

    return offspring