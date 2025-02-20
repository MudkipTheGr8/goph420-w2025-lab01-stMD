





import unittest
import numpy as np
from goph420_lab01.integration import integrate_newton

class TestNewtonCotes(unittest.TestCase):
    def test_trapezoidal_linear(self):
        x = np.linspace(0, 1, 100)
        y = 2*x + 1
        expected = 2.0
        result = integrate_newton(x, y, alg="trap")
        self.assertAlmostEqual(result, expected, places=4)

    def test_simpson_quadratic_even(self):
        x = np.linspace(0, 1, 101)
        y = x**2
        expected = 1/3
        result = integrate_newton(x, y, alg="simp")
        self.assertAlmostEqual(result, expected, places=4)

    def test_simpson_quadratic_odd(self):
        x = np.linspace(0, 1, 101)
        y = x**2
        expected = 1/3
        result = integrate_newton(x, y, alg="simp")
        self.assertAlmostEqual(result, expected, places=4)

    def test_invalid_algorithm(self):
        x = np.array([0, 1, 2])
        y = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            integrate_newton(x,y,alg="invalid")

if __name__ == "__main__":
    unittest.main()
