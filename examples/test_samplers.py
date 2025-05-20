import sys
sys.path.insert(1, 'utils')
sys.path.insert(1, '../utils')

import data_generation_simple
import samplers
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------- Test rejection sampler ---------------------------- #
def target(x):
    # Simple normal distribution with standard mean and variance
    # Can compute the maximum value by plugging in mean to gaussian PDF, as
    # the mean is also the mode for Gaussian.
    mu = 0.0
    var = 1.0
    pdf = np.exp(-1 * np.square(x - mu) / (2 * var)) / np.sqrt(2 * np.pi * var)
    return pdf

# Proposal distribution
def proposal(x):
    # Use the laplace distribution, as it's tail decays slower than the Gaussian it will always lie above
    mu = 0.0    # Mean
    b = 1.0     # Scale parameter (not really used here as I set it to 1)
    pdf = 0.5 * np.exp(-np.abs(x-mu))
    return pdf

# Inverse cdf to easily sample from laplace distribution
# Given that I use a univariate distribution, this exists, but for vector valued cases
# this may need to be a numerical method.
def inverse_proposal():
    # The CDF for laplace with scale b=1 is:   0.5 * exp(x-mu) when x <= mu
    # Otherwise, it is 1 - 0.5 * exp(-x+mu) when x >= mu
    # 2u = exp(x-mu) -> log(2u) = x-mu -> x = log(2u)+mu while 0.0 <= u <= 0.5 and mu=0.0
    # u-1 = -0.5 * exp(-x+mu) -> (2-2u) = exp(-x+mu) -> log(2-2u) = -x+mu -> x = mu-log(2-2u) while 0.5 <= u <= 1.0 and mu=0.0
    u = np.random.uniform(0.0, 1.0) # Sample u between 0, 1
    mu = 0.0
    b = 1.0
    if 0.5 <= u <= 1.0:
        inverse = mu - np.log(2-2*u)
    else:
        inverse = np.log(2*u) + mu
    return inverse
# ------------------------------------------------------------------------------------------ #


# ---------------------------- Test Metropolis-Hastings sampler ---------------------------- #
# ------------------------------------------------------------------------------------------ #


# ------------------------------ Test Hamiltonian MC sampler ------------------------------- #
# ------------------------------------------------------------------------------------------ #

# Main program
if __name__ == "__main__":
    coefficient = 2
    samples = samplers.rejection(target, proposal, inverse_proposal, coefficient, n=10000)
    samples.sort()
    
    # Demonstrate that samples are drawn from standard normal
    x_points = np.arange(-4, 4, step=0.001)
    # print(x_points)
    y_points = target(x_points)
    # print(y_points[2000])
    
    # Plot points
    plt.hist(samples, bins=100, density=True)
    plt.plot(x_points, y_points)
    plt.show()
    
    
