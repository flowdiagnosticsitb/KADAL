#Code for creating halton sampling in low n-dimensions
# references : - https://gist.github.com/tupui/cea0a91cc127ea3890ac0f002f887bae
#              - https://www.w3resource.com/python-exercises/list/python-data-type-list-exercise-34.php

import numpy as np

def primes (n):
    #Defining prime numbers for base using sieve of erasthostenes
    not_prime = []
    prime = []
    for i in range(2, n+1):
        if i not in not_prime:
            prime.append(i)
            for j in range(i*i, n+1, i):
                not_prime.append(j)
    return prime

def vandercorput(n_sample,base=2):
    #generate sample using van der corput sequence per dimension
    sequence=[]
    for i in range(0,n_sample):
        f=1. ;   r=0.
        while i > 0:
            i, remainder = divmod(i, base)
            f = f/base
            r = r+f*remainder
        sequence.append(r)
    return sequence

def halton (dimension,n_sample):
    # halton sequence general form of van der corput sequence in n-dimensions
    big_number = 1000       # just an input for base, as long as dim <= len(base) the program won't error
    base = primes(big_number)[:dimension]
    #print("base = ",base)             # for debugging
    sample = [vandercorput(n_sample + 1, dim) for dim in base] # looping van der corput for each dimension
    sample = np.stack(sample, axis= -1)[1:]     #arrange the array
    sample[1:n_sample,:] = sample[0:n_sample - 1,:]
    sample[0,:] = 0
    return sample


