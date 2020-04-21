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
    diff_unit_phase_condition

# def find_equilibria(self, parameter, gridDensity, uniqueRootDigits=7):
#     """
#     Return equilibria for the Hill Model by uniformly sampling for initial conditions and iterating a Newton variant.
#     INPUT:
#         parameter - A vector of all variable parameters to use for evaluating the root finding algorithm
#         gridDensity - (int) density to sample in each dimension.
#         uniqueRootDigits - (int) Number of digits to use for distinguishing between floats."""
#
#     # Include root finding method as vararg
#     def F(x):
#         """Fix parameter values in the zero finding map"""
#         return self.__call__(x, parameter)
#
#     def DF(x):
#         """Fix parameter values in the zero finding map derivative"""
#         return self.dx(x, parameter)
#
#     # build a grid of initial data for Newton algorithm
#     evalGrid = np.meshgrid(*[np.linspace(*f_i.eq_interval(), num=gridDensity) for f_i in self.coordinates])
#     X = np.row_stack([G_i.flatten() for G_i in evalGrid])
#     solns = list(filter(lambda root: root.success,
#                         [find_root(F, DF, X[:, j])
#                          for j in range(X.shape[1])]))  # return equilibria which converged
#     equilibria = np.column_stack([root.x for root in solns])  # extra equilibria as vectors in R^n
#     equilibria = np.unique(np.round(equilibria, uniqueRootDigits), axis=1)  # remove duplicates
#     return np.column_stack([find_root(F, DF, equilibria[:, j]).x for j in
#                             range(np.shape(equilibria)[1])])  # Iterate Newton again to regain lost digits


def distance_saddle(decay_loc, par1, par2):
    TS_model = ToggleSwitch(decay_loc, [par1, par2])
    Hill_coef = 4.1
    equilibria = TS_model.find_equilibria(Hill_coef, 10)

    SN_model = SaddleNode(TS_model, unit_phase_condition, diff_unit_phase_condition)

    x0 = equilibria[:, 1]
    v0 = np.array([1, -.7])
    u0 = np.concatenate((x0, v0, np.array(Hill_coef)), axis=None)

    SN_solution = full_newton(SN_model.zero_map, SN_model.diff_zero_map, u0)
    # do many many tries to actually find two saddle nodes

    P, Q = search_saddles()
    distance = np.abs(P[0] - Q[0])
    return -distance


def distance_saddle_with_constraint():
    return 1


# set some parameters to test using MATLAB toggle switch for ground truth
decay = np.array([1, 1], dtype=float)
p1 = np.array([1, 5, 3], dtype=float)
p2 = np.array([1, 6, 3], dtype=float)
x0 = np.array([4, 3])

f = ToggleSwitch(decay, [p1, p2])
n = 4.1
n0 = np.array([n])

SN = SaddleNode(f, unit_phase_condition, diff_unit_phase_condition)

eq = f.find_equilibria(n, 10)

# v0 = np.array([1, 1])
v0 = np.array([1, -.7])
# v0 = eq[:, 0] - eq[:, 1]
# v0 = v0 / np.linalg.norm(v0)
x0 = eq[:, 1]
u0 = np.concatenate((x0, v0, np.array(n)), axis=None)

# print(SN.zero_map(u0))
# print('\n')
# print(SN.diff_zero_map(u0))

# sol = find_root(SN.zero_map, SN.diff_zero_map, u0)
# sol = optimize.root(SN.zero_map, u0, method='hybr')  # set root finding algorithm
# SNsol = sol.x
SNsol = full_newton(SN.zero_map, SN.diff_zero_map, u0)

optimize.minimize(distance_saddle(), first_parameters)
# should work with at least one fixed parameter.

# if we want to leave all parameters free, we also want to add a request about norm of the vector of parameters = 1
