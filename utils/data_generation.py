import numpy as np
import samplers

# Instead of using provided samplers, I will implement my own samplers using Monte Carlo methods and use those instead
# The only exception will be for the uniform distribution.

def univariate_gen(distribution='uniform', n=100, a=0.0, b=1.0):
    if distribution=='normal':
        pass
    else:   # uniform is default
        data = np.random.uniform(low=a, high=b, size=n)
    return data

def multivariate_gen(distribution='uniform', a=0.0, b=0.0, n=100, dim=2):
    if distribution=='normal':
        pass
    else:   # uniform is default
        data = np.random.uniform(low=a, high=b, size=(n, dim))
    return data