# Matthew Davidson UCID: 30182729
# File Function Description: Tests the function integrate_newton function
#

import unittest
import numpy as np
from goph420_lab01.integration import integrate_newton

class TestNewtonCotes(unittest.TestCase):
    # Function Purpose: Checks the trapezoidal rule on a linear function
    # Returns: asserts that the computed result is approximately equal to 2
    def test_trapezoidal_linear(self):
        x = np.linspace(0, 1, 100)
        y = 2*x + 1
        expected = 2.0
        result = integrate_newton(x, y, alg="trap")
        self.assertAlmostEqual(result, expected, places=4)

    # Function Purpose: Checks Simpson's rule on a quadratic function with an even number of points
    # Returns: asserts that the computed result is approximately equal to 1/3
    def test_simpson_quadratic_even(self):
        x = np.linspace(0, 1, 101)
        y = x**2
        expected = 1/3
        result = integrate_newton(x, y, alg="simp")
        self.assertAlmostEqual(result, expected, places=4)

    # Function Purpose: Checks Simpson's rule on a quadratic function with an odd number of points
    # Returns: asserts that the computed result is approximately equal to 1/3
    def test_simpson_quadratic_odd(self):
        x = np.linspace(0, 1, 101)
        y = x**2
        expected = 1/3
        result = integrate_newton(x, y, alg="simp")
        self.assertAlmostEqual(result, expected, places=4)

    # Function Purpose: Checks that an invalid algorithm raises a ValueError
    # Returns: asserts that a ValueError is raised
    def test_invalid_algorithm(self):
        x = np.array([0, 1, 2])
        y = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            integrate_newton(x,y,alg="invalid")

if __name__ == "__main__":
    unittest.main()
