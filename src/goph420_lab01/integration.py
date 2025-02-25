# Matthew Davidson UCID: 30182729
# File Function Description: Contains utility functions (integrate_newton, and integrate_gauss)
# NOTE: Integrate_newton is to be called in the python file "driver_seismic.py"
# NOTE: Integrate_gauss is to be called in the python file "driver_probability.py"

import numpy as np
from scipy.special import roots_legendre

# FIRST STEP: Implement a function that performs numerical integration of discrete data using Newton-Cotes rules.
# Function name: integrate_newton
# Parameters:
# - x: array like, same shape as "f"
# - f: array like, same shape as "x"
# - alg: string with a default value of "trap" and another possible value of "simp" that indicate whether trapezoid rule or simpson's rule(s) should be used
#
# Returns:
# - float: return value is a float providing the integral estimate.
#

def integrate_newton(x, f, alg="trap"):
    x = np.asarray(x)
    f = np.asarray(f)
#   Ignoring leading or trailing whitespace in the flag
    alg = alg.strip().lower()
    #
    #   Raising a ValueError if the dimensions of x and f are incompatible
    #
    if x.shape != f.shape:
        raise ValueError("x and f are not the same shape, they must be the same shape.")
    if alg == "trap":
        return np.trapezoid(f,x)
    elif alg == "simp":
        if len(x) % 2 == 0:
            raise ValueError("There is an even number of points. Simpson's rule requires an odd number of points.")
        return np.trapezoid(f,x) if len(x) < 3 else np.sum((x[2::2] - x[:-2:2]) * (f[:-2:2] + 4*f[1::2] + f[2::2]) / 6)
    else:
        raise ValueError("This algorithm is invalid. Only 'trap' or 'simp' are accepted.")

# SECOND STEP: Implement a function that performs numerical integration of a function using Gauss-Legendre quadrature.
# Function Name: integrate_gauss
# Parameters:
# - f: reference to a callable object (a function or a class that implements the __call__() method).
# - lims: an object with len of 2 containing the lower and upper bound of integration (x=a and x=b)
# - npts: an optional int with possible values 1, 2, 3, 4, or 5 (default = 3)
#
# Returns:
# - float: providing the integral estimate.
#

def integrate_gauss(f, lims, npts=3):
    if not callable(f):
        raise TypeError("f is not callable, it must be callable.")
    if len(lims) != 2:
        raise ValueError("lims is only able to have exactly 2 elements.")
    a, b = map(float, lims)
    if npts not in [1, 2, 3, 4, 5]:
        raise ValueError("npts must be one of = [1, 2, 3, 4, 5].")

    points, weights = roots_legendre(npts)

    new_points = 0.5 * (b - a) * points + 0.5 * (b + a)
    new_weights = 0.5 * (b - a) * weights


    return np.sum(new_weights * f(new_points))

