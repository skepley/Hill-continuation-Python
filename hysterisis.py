"""
The hysterisis problem, first draft

"""

from hill_model import *
from saddle_node import *


def compute_distance(SN1, fixed_paraeters, parameter1, parameter2, nIter=20):
    # to avoid getting to negative parameter values, we just return infinity if the parameters are biologically not good
    if np.any(parameter1 < 0):
        return np.inf
    if np.any(parameter2 < 0):
        return np.inf

    # order the parameters
    if parameter2 > parameter1:
        temp = parameter2
        parameter2 = parameter1
        parameter1 = temp

    gamma1 = parameter1
    gamma2 = parameter1
    delta = parameter1 - parameter2
    iteration = 0

    while np.abs(gamma1 - gamma2) < 10 ** -7 and iteration < nIter:
        # find the saddle node bifurcaion
        a, saddlenode1 = SN1(fixed_paraeters, parameter1)
        b, saddlenode2 = SN1(fixed_paraeters, parameter2)

        gamma1 = saddlenode1[0]
        gamma2 = saddlenode2[0]

        parameter1 = parameter1 - delta
        parameter2 = parameter2 - delta

        iteration += 1

    if np.abs(gamma1 - gamma2) > 10 ** -7:
        return -np.abs(gamma1 - gamma2)
    else:
        return np.inf


def find_minima(SN, fixed_parameters, parameter1, parameter2):
    """Find a minimizer of a given loss function with respect to remaining parameters via gradient descent"""

    # TODO: Add the ability to return the entire orbit of the gradient descent algorithm

    delta = np.abs(parameter1 - parameter2)

    def local_function(parameter):
        return compute_distance(SN, fixed_parameters, parameter-delta, parameter+delta)

    minima = optimize.minimize(local_function, parameter1+delta, method='nelder-mead', options={'xatol': 1e-2,
                                                                                         'return_all': True})
    return minima



# set some parameters
decay = np.array([np.nan, 1], dtype=float)  # gamma
p1 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_1, delta_1, theta_1)
p2 = np.array([np.nan, np.nan, np.nan], dtype=float)  # (ell_2, delta_2, theta_2)
f = ToggleSwitch(decay, [p1, p2])
n0 = 4.1
SN = SaddleNode(f)

# ==== find saddle node minimizer for some initial parameter choice
p0 = np.array([1, 5, 3, 1, 6, 3],
              dtype=float)  # (gamma_1, ell_1, delta_1, theta_1, gamma_2, ell_2, delta_2, theta_2)
p1 = np.array([np.nan, 4, 3, 1, 5, 3, 5], dtype=float)

localMinimum = find_minima(SN, p1, 1, 1.5)

print(localMinimum)


