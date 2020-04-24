"""
Tests to learn about python and minimization

    Author: Elena Queirolo
    email: elena.queirolo@rutgers.edu
"""
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from math import log
from hill_model import HillComponent, ToggleSwitch, SaddleNode, full_newton, unit_phase_condition, \
    diff_unit_phase_condition, find_root


def distance_saddle(parameters):
    decay_loc = parameters[0:2]
    par1 = parameters[2:5]
    par2 = parameters[5:]

    TS_model = ToggleSwitch(decay_loc, [par1, par2])
    Hill_coef = 4.1
    equilibria = TS_model.find_equilibria(Hill_coef, 10)
    # 10 is the grid density

    SN_model = SaddleNode(TS_model, unit_phase_condition, diff_unit_phase_condition)

    v0 = np.array([1, -.7])

    solns = list([full_newton(SN_model.zero_map, SN_model.diff_zero_map,
                              np.concatenate((equilibria[:, j], v0, np.array(Hill_coef)), axis=None))
                  for j in range(np.shape(equilibria)[1])])

    saddles = np.unique(np.round(solns, 7), axis=1)  # remove duplicates

    if saddles.shape[1] > 1:
        return 0
    if saddles.shape[1] < 1:
        return 0

    P = saddles[:, 1]
    Q = saddles[:, 0]

    distance = np.abs(P[0] - Q[0])
    return -distance


def distance_saddle_with_constraint():
    return 1


# create the Toggle Switch problem with only one variable: the first decay
def ToggleFixed(gamma1):
    decay2 = float(2)
    p1 = np.array([1, 5, 3], dtype=float)
    p2 = np.array([1, 6, 3], dtype=float)
    return ToggleSwitch([gamma1, decay2], [p1, p2])


def SaddleProblem_loc(gamma1):
    return SaddleNode(ToggleFixed(gamma1), unit_phase_condition, diff_unit_phase_condition)


def findSaddle(interval, N=300):
    Hill_coef = 5
    gamma_grid = interval[0] + range(N) / (interval[1] - interval[0])
    for gamma in gamma_grid:
        equilibria = ToggleFixed(gamma).find_equilibria(Hill_coef, 10)
        SN_loc = SaddleProblem_loc(gamma)
        v0 = np.random.uniform(0, 10, 2)
        saddle = full_newton(SN_loc.zero_map, SN_loc.diff_zero_map,
                             np.concatenate((equilibria[:, 1], v0, np.array(Hill_coef)), axis=None))

    return gamma_grid


# set some parameters to test using MATLAB toggle switch for ground truth
"""decay2 = float(2)
p1 = np.array([1, 5, 3], dtype=float)
p2 = np.array([1, 6, 3], dtype=float)
x0 = np.array([4, 3]) """
# first_parameters = np.concatenate((decay, p1, p2), axis=None)


# distance_saddle(first_parameters)

# optimize.minimize(distance_saddle, first_parameters, method='nelder-mead')
# should work with at least one fixed parameter.

# if we want to leave all parameters free, we also want to add a request about norm of the vector of parameters = 1
