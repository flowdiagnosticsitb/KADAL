<img src="https://github.com/flowdiagnosticsitb/KADAL/blob/master/images/kadal.jpg" width="700">

Kriging for Analysis, Design optimization, And expLoration (KADAL) is a Python program developed by [Flow Diagnostics Research Group](https://flowdiagnostics.ftmd.itb.ac.id "Our Lab's Homepage") from Institut Teknologi Bandung (ITB) that contains collections of Bayesian Optimization tools including various surrogate modeling methods, sampling techniques, and optimization methods.
Currently, the program is under development and not yet available on Pypi. The coverage of the program are still limited to:

* Kriging
  * Ordinary Kriging
  * Regression Kriging
  * Polynomial Kriging
  * Composite Kernel Kriging
  * Kriging with Partial Least Square
* Bayesian Optimization
  * Unconstrained Single-Objective Bayesian Optimization (Expected Improvement)
  * Unconstrained Multi-Objective Bayesian Optimization (ParEGO, EHVI)
* Reliability Analysis
  * AK-MCS
* Sensitivity Analysis
  * Sobol indices
  
## Required packages
MySVR has the following dependencies:

* `numpy`
* `scipy`
* `matplotlib`
* `sobolsampling`
* `scikit-learn`
* `cma`
* `comet-ml`

KADAL has been tested on Python 3.6.1

# Quick Examples
The demo codes are available in the main folder. 
* KrigDemo.py is a demo script for creating surrogate model.
* MOBOdemo.py is a demo script for performing unconstrained multi-objective optimization for Schaffer test function.
* SOBOdemo.py is a demo script for performing unconstrained single objective optimization for Branin test function.

# Contact
The original program was written by Pramudita Satria Palar, Kemas Zakaria, Ghifari Adam Faza, and maintained by Aerodynamics Research Group ITB. 

e-mail: pramsp@ftmd.itb.ac.id

Special thanks to:
- Timothy Jim (Tohoku University)
- Potsawat Boonjaipetch (Tohoku University)
