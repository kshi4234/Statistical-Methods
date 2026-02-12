import numpy as np
import matplotlib.pyplot as plt
import functools


"""
The fundamental idea behind Metropolis-Hastings MCMC is that by performing a random walk w.r.t a Markov Chain (transition probabilities)
in the long run we are walking according to the target distribution, and each state we go to is in effect sampling from the target distribution.
Therefore we are NOT in any way "updating" any distribution

I will sample from an unnormalized multivariate (2) Gaussian distribution, and then draw samples from the normalized version using the built in sampler
from numpy to double check. I will then plot in 3D to visualize.
"""

def create_exp(mean, covar, x):
    exp = -1 * (x - mean) @ covar @ np.transpose(x-mean)
    return np.exp(exp)

def target_distribution(mean, covar):
    return functools.partial(create_exp, mean, covar)

"""
The proposal distribution I chose is just the normal distribution. This is basically just a random walk.
"""
def proposal_distribution(current_params, variance=1.0):
    # Literally just use a normal distribution
    # For multiparamter cases, sample independently in all directions, e.g., multivariate normal with diagonal covariance
    num_params = len(current_params)
    return np.random.multivariate_normal(current_params, variance * np.eye(num_params))

"""
The acceptance rule for MCMC
"""
def acceptance_rule(target_distribution, transition_distribution, current, proposal):
    prob = (transition_distribution(proposal, current) * target_distribution(proposal)) / (transition_distribution(current, proposal) * target_distribution(current))
    return min(1, prob)

def main(target_distribution=target_distribution, proposal_distribution=proposal_distribution):
    n_params = 2
    mean = np.array([1,2])
    covar = np.array([[0.1, 2], [0.5, 1.7]])
    distribution = target_distribution(mean, covar)

    a = acceptance_rule(target_distribution=distribution, pro)

if __name__ == '__main__':
    main()
