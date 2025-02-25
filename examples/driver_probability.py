# Matthew Davidson UCID: 30182729
# File Function Description: Calculating Probabilities and plotting convergence of the Gauss-Legendre Quadrature method.
#

import numpy as np
import matplotlib.pyplot as plt
from goph420_lab01.integration import integrate_gauss
from scipy.stats import norm

# Function Purpose: Computes probability density function of a standard normal distribution
# Parameters:
#  x: float, value to evaluate the normal distribution at
# Returns: float, the value of the normal distribution at x
def normal_pdf(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

# Function Purpose: Computes the probability of an earthquake magnitude exceeding a certain threshold
# Parameters:
#  magnitude: float, the threshold magnitude
#  mean: float, the mean of the normal distribution
#  std_dev: float, the standard deviation of the normal distribution
#  npts: int, the number of Gauss-Legendre points to use
# Returns: float, the probability of an earthquake exceeding the threshold magnitude
def probability_exceedance(magnitude, mean=1.5, std_dev=0.5, npts=3):
    z = (magnitude - mean) / std_dev
    upper_limit = 10
    return integrate_gauss(normal_pdf, (z, upper_limit), npts)

# Function Purpose: Computes the probability of a measured distance being within a certain range
# Parameters:
#  lower: float, the lower bound of the range
#  upper: float, the upper bound of the range
#  mean: float, the mean of the normal distribution
#  std_dev: float, the standard deviation of the normal distribution
#  npts: int, the number of Gauss-Legendre points to use
# Returns: float, the probability of the measured distance being within the range
def probability_within_range(lower, upper, mean=10.28, std_dev=0.05, npts=3):
    z_lower = (lower - mean) / std_dev
    z_upper = (upper - mean) / std_dev
    return integrate_gauss(normal_pdf, (z_lower, z_upper), npts)

# Function Purpose: Plots the convergence of the Gauss-Legendre quadrature method
# Parameters:
#  magnitude: float, the threshold magnitude
#  mean: float, the mean of the normal distribution
#  std_dev: float, the standard deviation of the normal distribution
# Returns: None, creates a Graph 
def plot_convergence(magnitude, mean, std_dev):
    npts_values = [1, 2, 3, 4, 5]
    probabilities = [probability_exceedance(magnitude, mean, std_dev, n) for n in npts_values]

    plt.figure(figsize=(8,6))
    plt.semilogy(npts_values, np.abs(probabilities - probabilities[-1]), 'o-', label="Convergence")
    plt.xlabel("Number of Gauss-Legendre Points")
    plt.ylabel("Approximate Relative Error")
    plt.grid(True)
    plt.legend()
    plt.savefig('figures/probability_convergence.png')
    plt.show()

if __name__ == "__main__":
    magnitude_threshold = 4.0
    prob = probability_exceedance(magnitude_threshold)
    print(f"Probability of earthquake magnitude > {magnitude_threshold}: {prob:.5f}")

    lower_bound, upper_bound = 10.25, 10.35
    prob_range = probability_within_range(lower_bound, upper_bound)
    print(f"Probability that measured distance is within ({lower_bound}, {upper_bound}): {prob_range:.5f}")

    plot_convergence(magnitude_threshold, mean=1.5, std_dev=0.5)
    