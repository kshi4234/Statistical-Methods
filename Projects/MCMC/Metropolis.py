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

def create_exp(covar, mean, x):
    precision = np.linalg.inv(covar)
    exp = -0.5 * (x - mean) @ precision @ np.transpose(x-mean)
    return np.exp(exp)

def target_distribution(mean, covar):
    return functools.partial(create_exp, covar, mean)

"""
The proposal distribution I chose is just the normal distribution. This is basically just a random walk.
"""
class RandomWalkProposal:
    def sample_proposal(self, covar_multiplier, current_params):
        # Literally just use a normal distribution
        # For multiparamter cases, sample independently in all directions, e.g., multivariate normal with diagonal covariance
        num_params = len(current_params)
        # Return the sample
        sample = np.random.multivariate_normal(current_params, covar_multiplier * np.eye(num_params))
        return sample

    def prob_proposal(self, covar_multiplier, num_params):
        unnormalized = functools.partial(create_exp, covar_multiplier * np.eye(num_params)) # Proposal distribution of going from current_params to new proposed point
        temp = np.linalg.det(2* np.pi * covar_multiplier * np.eye(num_params)) # Assuming diagonal covariance matrix
        normalization_constant = 1 / np.sqrt(temp)  # Normalization constant for multivariate normal distribution
        return lambda current_params, proposal: normalization_constant * unnormalized(current_params, proposal)


"""
The acceptance rule for MCMC
"""
def acceptance_rule(target_distribution, proposal_distribution, current, proposal):
    # Compute the ratio of the target distribution at the proposed point to the current point
    # and multiply by the ratio of proposal probabilities (i.e., transition probabilities)
    # If proposal is symmetric, this simplifies to just the ratio of target distributions
    prob = (proposal_distribution(proposal, current) * target_distribution(proposal)) / (proposal_distribution(current, proposal) * target_distribution(current))
    return min(1, prob)

def main(target_distribution=target_distribution, proposal_class=RandomWalkProposal):
    n_params = 2
    mean = np.array([0,0])
    covar = np.array([[1, 0], [0, 5]])  # I just selected a random ass PSD matrix

    proposal_var_multiplier = np.array([[2, 0], [0, 10]])
    distribution = target_distribution(mean, covar)

    max_samples = 105000
    burn_in = 10000
    curr_point = np.array([0,0])

    proposal_distribution = proposal_class()

    total = 0
    samples = []
    num_samples = 0
    while num_samples < max_samples:
        proposal = proposal_distribution.sample_proposal(proposal_var_multiplier, mean)

        distribution_instance = proposal_distribution.prob_proposal(proposal_var_multiplier, num_params=len(mean))

        a = acceptance_rule(target_distribution=distribution, proposal_distribution=distribution_instance, current=mean, proposal=proposal)
        # print(a)
        accept = np.random.rand() < a
        if accept:
            curr_point = proposal
            num_samples += 1
            samples.append(curr_point)
        total += 1
    
    samples = samples[burn_in:]
    print(num_samples / total)
    
    true_samples = [np.random.multivariate_normal(mean, covar) for _ in range(max_samples - burn_in)]

    # Plot histogram of MCMC samples
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(*zip(*samples), alpha=0.5, label='MCMC Samples')
    plt.title('MCMC Samples')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(*zip(*true_samples), alpha=0.5, label='True Samples')
    plt.title('True Samples')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
