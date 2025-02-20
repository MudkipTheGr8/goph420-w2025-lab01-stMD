





import unittest
import numpy as np
from goph420_lab01.integration import integrate_gauss

def polynomial_4th_order(x):
    return x**4 - 3*x**3 + 2*x**2 - x + 1

class TestGaussLegendre(unittest.TestCase):
    def test_exact_4th_order_polynomial(self):
        a, b = 0, 2
        expected = (1/5)*(b**5 - a**5) - (3/4)*(b**4 - a**4) + (2/3)*(b**3 - a**3) - (1/2)*(b**2 - a**2) + (b - a)
        result = integrate_gauss(polynomial_4th_order, (a, b), npts=3)
        self.assertAlmostEqual(result, expected, places=6)

    def test_invalid_function(self):
        with self.assertRaises(TypeError):
            integrate_gauss(5, (0, 1), npts=3)

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            integrate_gauss(polynomial_4th_order, (0, 1), npts=6)

if __name__ == "__main__":
    unittest.main()
