# Matthew Davidson UCID: 30182729
# File Function Description: Contains code that generates analysis results and generates plots on integrating discrete seismogram data.
#

import numpy as np
import matplotlib.pyplot as plt
from goph420_lab01.integration import integrate_newton

# Function Purpose: Loads seismic data from a file
# Parameters:
#  filepath: string, the path to the file containing the seismic data
# Returns: two numpy arrays, the time and velocity data
def load_velocity_data(filepath):
    data = np.loadtxt(filepath)
    time = data[:,0]
    velocity = data[:,1]
    return time, velocity

# Function Purpose: Estimates the event duration T from seismic data
# Parameters:
#  time: numpy array, the time data
#  velocity: numpy array, the velocity data
# Returns: float, the estimated event duration T
def estimate_T(time, velocity):
    threshold = 0.005 * np.max(np.abs(velocity))
    valid_indices = np.where(np.abs(velocity) > threshold)[0]
    if len(valid_indices) == 0:
        return time [-1]
    return time[valid_indices[-1]]

# Function Purpose: Computes the integrals of the squared velocity data using Newton-Cotes methods
# Parameters:
#  time: numpy array, the time data
#  velocity: numpy array, the velocity data
# Returns: two lists, the sampling intervals and the computed integrals
def compute_integrals(time, velocity):
    dt_values = [0.01, 0.02, 0.03, 0.04, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
    results = {}
    results["trap"] = []
    results["simp"] = []

    for dt in dt_values:
        indices = np.arange(0, len(time), int(dt / 0.01))
        if len(indices) % 2 == 0:
            indices = indices[:-1]
        
        t_downsampled, v_downsampled = time [indices], velocity[indices]
        v_squared = v_downsampled ** 2

        I_trap = integrate_newton(t_downsampled, v_squared, alg="trap")
        I_simp = integrate_newton(t_downsampled, v_squared, alg="simp")

        results["trap"].append(I_trap)
        results["simp"].append(I_simp)

    return dt_values, results

# Function Purpose: Plots the convergence of the Newton-Cotes methods
# Parameters:
#  dt_values: list, the sampling intervals
#  results: dict, the computed integrals
# Returns: None, creates a Graph
def plot_convergence(dt_values, results):
    epsilon = 1e-10
    relative_error_trap = np.abs((results["trap"] - results["trap"][-1]) / (results["trap"][-1] + epsilon))
    relative_error_simp = np.abs((results["simp"] - results["simp"][-1]) / (results["simp"][-1] + epsilon))
    
    plt.figure(figsize=(8,6))
    plt.loglog(dt_values, relative_error_trap, 'o-', label="Trapezoidal Rule")
    plt.loglog(dt_values, relative_error_simp, 's-', label="Simpson's Rule")
    plt.xlabel("Sampling Interval (s)")
    plt.ylabel("Approximate Relative Error")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/seismic_convergence.png")
    plt.show()

if __name__ == "__main__":
    filepath = "s_wave_data.txt"
    time, velocity = load_seismic_data(filepath)

    plt.plot(time, velocity, label="Velocity (mm/s)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Velocity (mm/s)")
    plt.title("Seismic Velocity Data")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/seismic_velocity.png")
    plt.show()

    T = estimate_T(time, velocity)
    print(f"Estimated event duration T: {T:.2f} s")

    dt_values, results = compute_integrals(time, velocity)
    plot_convergence(dt_values, results)