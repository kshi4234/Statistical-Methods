import numpy as np

def rejection(target, proposal, proposal_sampler, coefficient, n):
    """
    target: target density function of interest
    coefficient: constant 
    proposal: function such that coefficinet * proposal >> function.
    proposal_sampler: Sampling from proposal distribution, either by inverse CDF or some other method.
    n: number of samples
    """
    samples = [None] * n
    def sample():
        x = proposal_sampler()
        u = np.random.uniform(0.0, 1.0)
        while u > (target(x) / (coefficient * proposal(x))):
            # print("retry...")
            # print(x)
            # print(u, target(x), coefficient * proposal(x), target(x) / (coefficient * proposal(x)))
            # input("Press Enter to continue...")
            x = proposal_sampler()
            u = np.random.uniform(0.0, 1.0)
        return x
    for i in range(n):
        print("SAMPLING STARTED...")
        samples[i] = sample()
        print("OBTAINED SAMPLE!")
    return samples

def metropolis_hastings():
    return

def hamiltonian_mc():
    return