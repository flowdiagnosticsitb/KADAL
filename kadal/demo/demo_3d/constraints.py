"""The constraint functions for optimsiation.

The same constraint functions are used for initial sample generation.

In this file, there are two versions of the same constraints:
    g_0 - g_7: constraint functions in the format used by KADAL as part
        of the multiconstfunc. These can be written to accept 2D inputs
        to take advantage of numpy vectorisation, if the selected
        optimiser can evaluate several samples at once. Should return 1
        if satisfied, else 0.

    n_0 - n_7: the same functions but rewritten using scipy.optimize
        NonlinearConstraint functions. This is currently only used if
        the selected optimiser is SciPy's 'diff_evo', as it can use
        the constraint functions directly. Other optimisers use the
        constraint functions as a penalty factor on the probability of
        feasibility.
        N.B. diff_evo cannot take advantage of multi-sample evaluation,
        so the functions here do not need to be written in a vectorised
        form.

Last updated: 30/04/2021
"""
__author__ = "Tim Jim"
__version__ = '1.0.0'

import argparse
import numpy as np
import pandas as pd
import scipy.optimize as so

# Explicitly set parameters and limits
lhs_params = {'x': [10, 30],
              'z': [-0.4, 1.8],
              'chord_0': [10, 30],
              'tc_0': [0.025, 0.10],
              'twist_0': [-8, 8],
              'le_sweep_1': [-78, 78],
              'dihedral_1': [-30, 30],
              'chord_1': [5, 25],
              'tc_1': [0.02, 0.10],
              'twist_1': [-6, 6],
              'proj_span_1': [2, 10],
              'le_sweep_2': [-78, 78],
              'dihedral_2': [-45, 45],
              'chord_2': [3, 20],
              'tc_2': [0.02, 0.10],
              'twist_2': [-6, 6],
              'proj_span_2': [1, 8],
              'le_sweep_3': [-78, 78],
              'dihedral_3': [-85, 85],
              'chord_3': [1, 15],
              'tc_3': [0.02, 0.10],
              'twist_3': [-6, 6],
              }

limits = pd.DataFrame(data=lhs_params, index=['dv_min', 'dv_max'])
params = lhs_params.keys()
dv_min = limits.loc['dv_min'].to_numpy()
dv_max = limits.loc['dv_max'].to_numpy()


# Geometric constraints
S_TOTAL = 268.56 / 2  # For half aircraft! ==> 268.56 * 1.2 / 2 for B planform
S1_MIN = 0.3 * S_TOTAL
S2_MIN = 0.1 * S_TOTAL
S3_MIN = 0.01 * S_TOTAL
SPAN3_MIN = 0.3
SPAN3_MAX = 3
X_AFT_MAX = 42


# Set up constraint functions
# Area constraints
def g_0(x):
    """S_1 > S_1_min"""
    reshape = False
    if x.ndim == 1:
        reshape = True
        x = x.reshape(1, -1)
    S1 = (x[:, 2] + x[:, 7]) / 2 * x[:, 10]
    if reshape:
        S1 = S1.reshape(-1)
    return (S1_MIN - S1 <= 0).astype(int)


def g_1(x):
    """S_2 > S_2_min"""
    reshape = False
    if x.ndim == 1:
        reshape = True
        x = x.reshape(1, -1)
    S2 = (x[:, 7] + x[:, 13]) / 2 * x[:, 16]
    if reshape:
        S2 = S2.reshape(-1)
    return (S2_MIN - S2 <= 0).astype(int)


def g_2(x):
    """S_3 > S_3_min"""
    reshape = False
    if x.ndim == 1:
        reshape = True
        x = x.reshape(1, -1)
    S1 = (x[:, 2] + x[:, 7]) / 2 * x[:, 10]
    S2 = (x[:, 7] + x[:, 13]) / 2 * x[:, 16]
    S3 = S_TOTAL - (S1 + S2)
    if reshape:
        S3 = S3.reshape(-1)
    return (S3_MIN - S3 <= 0).astype(int)


# Use span instead of proj_span due to extended dihedral limits on tips
def g_3(x):
    """span_3_min < span_3"""
    reshape = False
    if x.ndim == 1:
        reshape = True
        x = x.reshape(1, -1)
    S1 = (x[:, 2] + x[:, 7]) / 2 * x[:, 10]
    S2 = (x[:, 7] + x[:, 13]) / 2 * x[:, 16]
    S3 = S_TOTAL - (S1 + S2)
    proj_span_3 = 2 * S3 / (x[:, 13] + x[:, 19])
    span_3 = proj_span_3 / np.cos(np.radians(x[:, 18]))
    if reshape:
        span_3 = span_3.reshape(-1)
    return (SPAN3_MIN - span_3 <= 0).astype(int)


def g_4(x):
    """span_3 < span_3_max"""
    reshape = False
    if x.ndim == 1:
        reshape = True
        x = x.reshape(1, -1)
    S1 = (x[:, 2] + x[:, 7]) / 2 * x[:, 10]
    S2 = (x[:, 7] + x[:, 13]) / 2 * x[:, 16]
    S3 = S_TOTAL - (S1 + S2)
    proj_span_3 = 2 * S3 / (x[:, 13] + x[:, 19])
    span_3 = proj_span_3 / np.cos(np.radians(x[:, 18]))
    if reshape:
        span_3 = span_3.reshape(-1)
    return (span_3 - SPAN3_MAX <= 0).astype(int)


# Max rearward location at centreline
def g_5(x):
    """Max aft wing root location: x_offset + chord_0"""
    reshape = False
    if x.ndim == 1:
        reshape = True
        x = x.reshape(1, -1)
    x_aft = x[:, 0] + x[:, 2]
    if reshape:
        x_aft = x_aft.reshape(-1)
    return (x_aft - X_AFT_MAX <= 0).astype(int)


# Don't want aerofoil LE/TE or max t/c point protruding through the fuselage
# Restrict max/min diamond aerofoil LE/TE at centreline due to twist
# Assume linear between know stations from OpenVSP model and PW cut
# Assume alpha/twist at 50% chord and c_o & max thickness doesn't vary with alpha
def g_6(x):
    """Min z_0.5c, z_LE, or z_TE of wing at centreline"""
    reshape = False
    if x.ndim == 1:
        reshape = True
        x = x.reshape(1, -1)
    # z_min(LE/TE) = z - abs(c_0/2 * tan(twist_0))
    z_le_te = x[:, 1] - np.abs(x[:, 2] / 2 * np.tan(np.radians(x[:, 4])))

    # z_min_50% chord = z - 0.5 * t/c * c
    z_50 = x[:, 1] - 0.5 * x[:, 3] * x[:, 2]

    # Create array to store, for each sample:
    # max/min z, corresponding x-location, x_a, x_b, z_min_a, z_min_b
    dat = np.zeros([x.shape[0], 6])

    # Check if diamond centre is lower than LE/TE
    # If 50% chord is z_min, check that x location
    # if z_50 < z_le_te:
    #     z_i = z_50
    #     x_i = x[0] + 0.5 * x[2]  # x_LE + 0.5 * c_0 = x_50% chord
    i = z_50 < z_le_te
    dat[i, 0] = z_50[i]
    dat[i, 1] = x[i, 0] + 0.5 * x[i, 2]  # x_LE + 0.5 * c_0 = x_50% chord

    # # Else, check the x location of the LE or TE
    # else:
    #     # For twist_0 > 0 check TE limit, else check LE limit
    #     if x[4] > 0:
    #         x_i = x[0] + x[2]  # x_LE + c_0 = x_TE
    #     else:
    #         x_i = x[0]
    #     z_i = z_le_te
    dat[~i, 0] = z_le_te[~i]
    j = x[~i, 4] > 0
    ii = np.where(~i)[0]  # Masking a mask... need to get indices of not i
    dat[ii[j], 1] = x[ii[j], 0] + x[ii[j], 2]  # x_LE + c_0 = x_TE
    dat[ii[~j], 1] = x[ii[~j], 0]

    # # Max/min z changes depending on x location
    # # Model intermediate x as linear change in z between fuselage stations
    # # Fuselage section 2 - 3
    # if 8.84 <= x_i < 16.64:
    #     x_a = 8.84
    #     x_b = 16.64
    #     z_min_a = 0.66040 - 2.16 / 2
    #     z_min_b = 0.66560 - 2.5 / 2
    # # Fuselage section 3 - 4
    # elif 16.64 <= x_i < 36.4:
    #     x_a = 16.64
    #     x_b = 36.4
    #     z_min_a = z_min_b = 0.66560 - 2.5 / 2
    # # Fuselage section 4 - x_42
    # elif 36.4 <= x_i <= 42:
    #     x_a = 36.4
    #     x_b = 42
    #     z_min_a = 0.66560 - 2.5 / 2
    #     z_min_b = -0.46594752  # From PW cut at x = 42
    # else:
    #     # Handled by constraint g_5(x)
    #     # print(f'x_i: {x_i} is outside of the modelled range')
    #     return 0  # Return 0 for failed in KADAL '# 1
    # z_min = (z_min_b - z_min_a) / (x_b - x_a) * (x_i - x_a) + z_min_a

    # Fuselage section 2 - 3
    i = np.logical_and((8.84 <= dat[:, 1]), (dat[:, 1] < 16.64))
    dat[i, 2:4] = 8.84, 16.64  # x_a, x_b
    dat[i, 4:6] = 0.66040 - 2.16 / 2, 0.66560 - 2.5 / 2  # z_min_a, z_min_b
    # Fuselage section 3 - 4
    i = np.logical_and((16.64 <= dat[:, 1]), (dat[:, 1] < 36.4))
    dat[i, 2:4] = 16.64, 36.4  # x_a, x_b
    dat[i, 4:6] = 0.66560 - 2.5 / 2, 0.66560 - 2.5 / 2  # z_min_a, z_min_b
    # Fuselage section 4 - x_42
    i = np.logical_and((36.4 <= dat[:, 1]), (dat[:, 1] < 42))
    dat[i, 2:4] = 36.4, 42  # x_a, x_b
    dat[i, 4:6] = 0.66560 - 2.5 / 2, -0.46594752  # z_min_a, z_min_b
    z_min = (dat[:, 5] - dat[:, 4]) / (dat[:, 3] - dat[:, 2]) * (dat[:, 1] - dat[:, 2]) + dat[:, 4]

    if reshape:
        z_min = z_min.reshape(-1)
    return (z_min - dat[:, 0] <= 0).astype(int)


def g_7(x):
    """Max z_0.5c, z_LE, or z_TE of wing at centreline"""
    reshape = False
    if x.ndim == 1:
        reshape = True
        x = x.reshape(1, -1)

    # z_max(LE/TE) = z + abs(c_0/2 * tan(twist_0))
    z_le_te = x[:, 1] + np.abs(x[:, 2] / 2 * np.tan(np.radians(x[:, 4])))

    # z_max_50% chord = z + 0.5 * t/c * c
    z_50 = x[:, 1] + 0.5 * x[:, 3] * x[:, 2]

    # Create array to store, for each sample:
    # max/min z, corresponding x-location, x_a, x_b, z_max_a, z_max_b
    dat = np.zeros([x.shape[0], 6])

    # Check if diamond centre is higher than LE/TE
    # If 50% chord is z_max, check that x location
    # if z_50 > z_le_te:
    #     z_i = z_50
    #     x_i = x[0] + 0.5 * x[2]  # x_LE + 0.5 * c_0 = x_50% chord
    i = z_50 > z_le_te
    dat[i, 0] = z_50[i]
    dat[i, 1] = x[i, 0] + 0.5 * x[i, 2]  # x_LE + 0.5 * c_0 = x_50% chord

    # # Else, check the x location of the LE or TE
    # else:
    #     # For twist_0 < 0 check TE limit, else check LE limit
    #     if x[4] < 0:
    #         x_i = x[0] + x[2]  # x_LE + c_0 = x_TE
    #     else:
    #         x_i = x[0]
    #     z_i = z_le_te
    dat[~i, 0] = z_le_te[~i]
    j = x[~i, 4] < 0
    ii = np.where(~i)[0]  # Masking a mask... need to get indices of not i
    dat[ii[j], 1] = x[ii[j], 0] + x[ii[j], 2]  # x_LE + c_0 = x_TE
    dat[ii[~j], 1] = x[ii[~j], 0]

    # # Max/min z changes depending on x location
    # # Model intermediate x as linear change in z between fuselage stations
    # # Fuselage section 2 - 3
    # if 8.84 <= x_i < 16.64:
    #     x_a = 8.84
    #     x_b = 16.64
    #     z_max_a = 0.66040 + 2.16 / 2
    #     z_max_b = 0.66560 + 2.5 / 2
    # # Fuselage section 3 - 4
    # elif 16.64 <= x_i < 36.4:
    #     x_a = 16.64
    #     x_b = 36.4
    #     z_max_a = z_max_b = 0.66560 + 2.5 / 2
    # # Fuselage section 4 - x_42
    # elif 36.4 <= x_i <= 42:
    #     x_a = 36.4
    #     x_b = 42
    #     z_max_a = 0.66560 + 2.5 / 2
    #     z_max_b = 1.7876549  # From PW cut at x = 42
    # else:
    #     # Handled by constraint g_5(x)
    #     # print(f'x_i: {x_i} is outside of the modelled range')
    #     return 1
    # z_max = (z_max_b - z_max_a) / (x_b - x_a) * (x_i - x_a) + z_max_a

    # Fuselage section 2 - 3
    i = np.logical_and((8.84 <= dat[:, 1]), (dat[:, 1] < 16.64))
    dat[i, 2:4] = 8.84, 16.64  # x_a, x_b
    dat[i, 4:6] = 0.66040 + 2.16 / 2, 0.66560 + 2.5 / 2  # z_min_a, z_min_b
    # Fuselage section 3 - 4
    i = np.logical_and((16.64 <= dat[:, 1]), (dat[:, 1] < 36.4))
    dat[i, 2:4] = 16.64, 36.4  # x_a, x_b
    dat[i, 4:6] = 0.66560 + 2.5 / 2, 0.66560 + 2.5 / 2  # z_min_a, z_min_b
    # Fuselage section 4 - x_42
    i = np.logical_and((36.4 <= dat[:, 1]), (dat[:, 1] < 42))
    dat[i, 2:4] = 36.4, 42  # x_a, x_b
    dat[i, 4:6] = 0.66560 + 2.5 / 2, 1.7876549  # z_min_a, z_min_b
    z_max = (dat[:, 5] - dat[:, 4]) / (dat[:, 3] - dat[:, 2]) * (dat[:, 1] - dat[:, 2]) + dat[:, 4]

    if reshape:
        z_max = z_max.reshape(-1)
    return (dat[:, 0] - z_max <= 0).astype(int)


# scipy optimize non-linear constraints version for differential evolution
def n_0(x):
    """S_1 > S_1_min"""
    S1 = (x[2] + x[7]) / 2 * x[10]
    return S1


def n_1(x):
    """S_2 > S_2_min"""
    S2 = (x[7] + x[13]) / 2 * x[16]
    return S2


def n_2(x):
    """S_3 > S_3_min"""
    S1 = (x[2] + x[7]) / 2 * x[10]
    S2 = (x[7] + x[13]) / 2 * x[16]
    S3 = S_TOTAL - (S1 + S2)
    return S3


nlc_0 = so.NonlinearConstraint(n_0, S1_MIN, np.inf)
nlc_1 = so.NonlinearConstraint(n_1, S2_MIN, np.inf)
nlc_2 = so.NonlinearConstraint(n_2, S3_MIN, np.inf)


# Use span instead of proj_span due to extended dihedral limits on tips
def n_3_4(x):
    """span_3_min < span_3"""
    S1 = (x[2] + x[7]) / 2 * x[10]
    S2 = (x[7] + x[13]) / 2 * x[16]
    S3 = S_TOTAL - (S1 + S2)
    proj_span_3 = 2 * S3 / (x[13] + x[19])
    span_3 = proj_span_3 / np.cos(np.radians(x[18]))
    return span_3


nlc_3 = so.NonlinearConstraint(n_3_4, SPAN3_MIN, SPAN3_MAX)


# Max rearward location at centreline
def n_5(x):
    """Max aft wing root location: x_offset + chord_0"""
    x_aft = x[0] + x[2]
    return x_aft


nlc_4 = so.NonlinearConstraint(n_5, -np.inf, X_AFT_MAX)
# lc_4 = so.LinearConstraint(n_5, -np.inf, X_AFT_MAX)


# Assume linear between know stations from OpenVSP model and PW cut
# Assume alpha/twist at 50% chord and c_o & max thickness doesn't vary with alpha
def n_6(x):
    """Min z_0.5c, z_LE, or z_TE of wing at centreline"""

    # z_min(LE/TE) = z - abs(c_0/2 * tan(twist_0))
    z_le_te = x[1] - np.abs(x[2] / 2 * np.tan(np.radians(x[4])))

    # z_min_50% chord = z - 0.5 * t/c * c
    z_50 = x[1] - 0.5 * x[3] * x[2]

    # Check if diamond centre is lower than LE/TE
    # If 50% chord is z_min, check that x location
    if z_50 < z_le_te:
        z_i = z_50
        x_i = x[0] + 0.5 * x[2]  # x_LE + 0.5 * c_0 = x_50% chord

    # Else, check the x location of the LE or TE
    else:
        # For twist_0 > 0 check TE limit, else check LE limit
        if x[4] > 0:
            x_i = x[0] + x[2]  # x_LE + c_0 = x_TE
        else:
            x_i = x[0]
        z_i = z_le_te

    # # For twist_0 > 0 check TE limit, else check LE limit
    # if x[4] > 0:
    #     x_i = x[0] + x[2]  # x_LE + c_0 = x_TE
    # else:
    #     x_i = x[0]
    # z_i = z_le_te

    # Max/min z changes depending on x location
    # Model intermediate x as linear change in z between fuselage stations
    # Fuselage section 2 - 3
    if 8.84 <= x_i < 16.64:
        x_a = 8.84
        x_b = 16.64
        z_min_a = 0.66040 - 2.16 / 2
        z_min_b = 0.66560 - 2.5 / 2
    # Fuselage section 3 - 4
    elif 16.64 <= x_i < 36.4:
        x_a = 16.64
        x_b = 36.4
        z_min_a = z_min_b = 0.66560 - 2.5 / 2
    # Fuselage section 4 - x_42
    elif 36.4 <= x_i <= 42:
        x_a = 36.4
        x_b = 42
        z_min_a = 0.66560 - 2.5 / 2
        z_min_b = -0.46594752  # From PW cut at x = 42
    else:
        return np.inf

    z_min = (z_min_b - z_min_a) / (x_b - x_a) * (x_i - x_a) + z_min_a
    return z_min - z_i


nlc_5 = so.NonlinearConstraint(n_6, -np.inf, 0)


def n_7(x):
    """Max z_0.5c, z_LE, or z_TE of wing at centreline"""

    # z_max(LE/TE) = z + abs(c_0/2 * tan(twist_0))
    z_le_te = x[1] + np.abs(x[2] / 2 * np.tan(np.radians(x[4])))

    # z_max_50% chord = z + 0.5 * t/c * c
    z_50 = x[1] + 0.5 * x[3] * x[2]

    # Check if diamond centre is higher than LE/TE
    # If 50% chord is z_max, check that x location
    if z_50 > z_le_te:
        z_i = z_50
        x_i = x[0] + 0.5 * x[2]  # x_LE + 0.5 * c_0 = x_50% chord

    # Else, check the x location of the LE or TE
    else:
        # For twist_0 < 0 check TE limit, else check LE limit
        if x[4] < 0:
            x_i = x[0] + x[2]  # x_LE + c_0 = x_TE
        else:
            x_i = x[0]
        z_i = z_le_te

    # # For twist_0 < 0 check TE limit, else check LE limit
    # if x[4] < 0:
    #     x_i = x[0] + x[2]  # x_LE + c_0 = x_TE
    # else:
    #     x_i = x[0]
    # z_i = z_le_te

    # Max/min z changes depending on x location
    # Model intermediate x as linear change in z between fuselage stations
    # Fuselage section 2 - 3
    if 8.84 <= x_i < 16.64:
        x_a = 8.84
        x_b = 16.64
        z_max_a = 0.66040 + 2.16 / 2
        z_max_b = 0.66560 + 2.5 / 2
    # Fuselage section 3 - 4
    elif 16.64 <= x_i < 36.4:
        x_a = 16.64
        x_b = 36.4
        z_max_a = z_max_b = 0.66560 + 2.5 / 2
    # Fuselage section 4 - x_42
    elif 36.4 <= x_i <= 42:
        x_a = 36.4
        x_b = 42
        z_max_a = 0.66560 + 2.5 / 2
        z_max_b = 1.7876549  # From PW cut at x = 42
    else:
        # Handled by constraint g_5(x)
        # print(f'x_i: {x_i} is outside of the modelled range')
        return 1

    z_max = (z_max_b - z_max_a) / (x_b - x_a) * (x_i - x_a) + z_max_a
    return z_i - z_max


nlc_6 = so.NonlinearConstraint(n_7, -np.inf, 0)


if __name__ == '__main__':
    # df = pd.read_excel('../1_SETUP/initial_samples_A9.xlsx', sheet_name=0)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('s_ref_class', choices=['A', 'B'],
    #                     help="A or B optimisation. For opt case f'opt_{s_ref_class}{n_opt:02}'")
    parser.add_argument('n_opt', help="For opt case f'opt_{s_ref_class}{n_opt}'")
    # parser.add_argument('samp_csv')
    args = parser.parse_args()
    # df = pd.read_csv(args.samp_csv, sep=',')

    # s_class = args.s_ref_class
    n_opt = args.n_opt
    # opt = f'opt_{s_class}{n_opt:02}'
    # n_opt = 4
    opt = f'opt_A{n_opt}'
    next_out = f'{opt}/next_samples.csv'
    print(f'Loading: {next_out}')
    df = pd.read_csv(next_out, sep=',')
    params = ['x', 'z', 'chord_0', 'tc_0', 'twist_0',
              'le_sweep_1', 'dihedral_1', 'chord_1',
              'tc_1', 'twist_1', 'proj_span_1',
              'le_sweep_2', 'dihedral_2', 'chord_2',
              'tc_2', 'twist_2', 'proj_span_2',
              'le_sweep_3', 'dihedral_3', 'chord_3',
              'tc_3', 'twist_3']
    dvs = df[params]  # [n_samples, n_dvs]
    n_samples = dvs.shape[0]
    print(f'n_samples: {n_samples}, n_dvs: {dvs.shape[1]}')

    # # Geometric constraints
    # if s_class == 'A':
    #     S_TOTAL = 268.56 / 2  # For half aircraft! ==> 268.56 * 1.2 / 2 for B planform
    # elif s_class == 'B':
    #     S_TOTAL = 322.272 / 2
    # else:
    #     raise ValueError(f's_ref_class must be A or B')
    # S1_MIN = 0.3 * S_TOTAL
    # S2_MIN = 0.1 * S_TOTAL
    # S3_MIN = 0.01 * S_TOTAL
    # constants = {'S_TOTAL': S_TOTAL,
    #              'S1_MIN': S1_MIN,
    #              'S2_MIN': S2_MIN,
    #              'S3_MIN': S3_MIN,
    #              'SPAN3_MIN': 0.3,
    #              'SPAN3_MAX': 3,
    #              'X_AFT_MAX': 42}

    cons = [g_0, g_1, g_2, g_3, g_4, g_5, g_6, g_7]  # constraints
    n_cons = len(cons)
    feas = np.zeros([n_samples, n_cons])  # feasibility array

    # Eval constraints for each sample
    for i in range(n_samples):
        dvs_i = dvs.iloc[i].to_numpy()  # the design vars of 1 sample
        for j in range(n_cons):
            feas[i, j] = cons[j](dvs_i)  # eval it for each constraint

    # Now do the same thing in vectorised mode
    feas2 = np.zeros([n_samples, n_cons])
    for j in range(n_cons):
        feas2[:, j] = cons[j](dvs.to_numpy())

    # This is your PoF array of cheap constraints (0 or 1)
    print(feas)
    print('Constraints satisfied:')
    print('\n'.join(f'Sample {i}: {s}' for i, s in enumerate(feas.all(axis=1))))
    breakpoint()
