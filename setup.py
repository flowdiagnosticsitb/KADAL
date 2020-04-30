'''
Author: Ghifari Adam Faza <ghifariadamf@gmail.com>
This package is distributed under BSD-3 license.
'''

from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
LONG_DESCRIPTION= """
Kriging for Analysis, Design optimization, And expLoration (KADAL) is Flow Diagnostics Lab in-house 
Python code that contains collections of Bayesian Optimization tools including various surrogate modeling methods, 
sampling techniques, and optimization methods. Some surrogate models that included in our program are Ordinary Kriging, 
Regression Kriging, Polynomial Kriging, Composite Kernel Kriging, and Kriging with Partial Least Square. 
In Bayesian optimization module, we have Single Objective Bayesian Optimization (SOBO) 
algorithm using Expected Improvement (EI) and Multi-Objective Bayesian Optimization (MOBO) using 
Pareto efficient global optimization (ParEGO) and Expected Hypervolume Improvement (EHVI) algorithm. 
We also have uncertainty quantification (UQ) module, global sensitivity analysis (GSA) module based on Sobol Indices, 
and reliability analysis module based on Active Kriging – Monte Carlo Simulation (AK-MCS). 
Our code is under active development and we aim to incorporate more sophisticated methods.
"""


metadata = dict(
    name='kadal',
    version='1.0.2',
    description='Kriging for Analysis, Design optimization, And expLoration.',
    long_description=LONG_DESCRIPTION,
    author='Pramudita S. Palar, Ghifari Adam F, Kemas Zakaria, Timothy Jim',
    author_email='ghifariadamf@gmail.com',
    license='BSD 3-Clause',
    packages=[
        'kadal',
        'kadal.surrogate_models',
        'kadal.surrogate_models.supports',
        'kadal.reliability_analysis',
        'kadal.sensitivity_analysis',
        'kadal.testcase',
        'kadal.testcase.RA',
        'kadal.testcase.analyticalfcn',
        'kadal.misc.sampling',
        'kadal.optim_tools',
        'kadal.optim_tools.ga',
        'kadal.optim_tools.ehvi'
    ],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'comet-ml',
        'matplotlib',
        'sobolsampling'
    ],
    python_requires='>=3.6.*',
    zip_safe=False,
    include_package_data=True,
    url = 'https://fazaghifari.github.io/portfolio/kadal/', # use the URL to the github repo
    download_url = 'https://fazaghifari.github.io/portfolio/kadal/',
)

setup(**metadata)
