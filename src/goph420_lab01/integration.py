





import numpy as np
from scipy.special import roots_legendre


def integrate_newton(x, f, alg="trap"):
    x = np.asarray(x)
    f = np.asarray(f)

    if x.shape != f.shape:
        raise ValueError("x and f are not the same shape, they must be the same shape.")

    alg = alg.strip().lower()

    if alg == "trap":
        return np.trapezoid(f,x)
    elif alg == "simp":
        if len(x) % 2 == 0:
            raise ValueError("There is an even number of points. Simpson's rule requires an odd number of points.")
        return np.trapezoid(f,x) if len(x) < 3 else np.sum((x[2::2] - x[:-2:2]) * (f[:-2:2] + 4*f[1::2] + f[2::2]) / 6)
    else:
        raise ValueError("This algorithm is invalid. Only 'trap' or 'simp' are accepted.")



def integrate_gauss(f, lims, npts=3):
    if not callable(f):
        raise TypeError("f is not callable, it must be callable.")
    if len(lims) != 2:
        raise ValueError("lims is only able to have exactly 2 elements.")
    a, b = map(float, lims)
    if npts not in [1, 2, 3, 4, 5]:
        raise ValueError("npts must be one of = [1, 2, 3, 4, 5].")

    points, weights = roots_legendre(npts)

    transformed_points = 0.5 * (b - a) * points + 0.5 * (b + a)
    transformed_weights = 0.5 * (b - a) * weights


    return np.sum(transformed_weights * f(transformed_points))

