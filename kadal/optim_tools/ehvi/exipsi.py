import numpy as np
from kadal.optim_tools.ehvi.gaussfcn import gausscdf,gausspdf

def exipsi(a,b,m,s):
    x = s*gausspdf((b-m)/s) + (a-m)*gausscdf((b-m)/s)
    return x