import numpy as np

# Geenrating data from distributions using numpy provided methods

def univariate_gen(distribution='uniform', n=100, a=0.0, b=1.0, mean=0, variance=1):
    """Generate univariate data
    n: number of samples
    a: lower bound (dependent on distribution)
    b: upper bound (dependent on distribution)
    mean: self-explanatory
    variance: self-explanatory
    """
    if distribution=='normal':
        sigma = np.sqrt(variance)
        data = mean + sigma*np.random.standard_normal(size=n)
    else:   # uniform is default
        data = np.random.uniform(low=a, high=b, size=n)
    return data

def multivariate_gen(distribution='uniform', a=0.0, b=0.0, n=100, dim=2, mean=0, covariance=1):
    if distribution=='normal':
        if np.isscalar(mean) and np.isscalar(covariance):
            mean = np.zeros(dim)
            covariance = np.identity(dim)
        data = np.random.multivariate_normal(mean, covariance, size=n)
    else:   # uniform is default
        data = np.random.uniform(low=a, high=b, size=(n, dim))
    return data