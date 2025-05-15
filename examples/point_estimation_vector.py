import sys
sys.path.insert(1, 'utils')
sys.path.insert(1, '../utils')

import data_generation_simple
import numpy as np

# ---------------------- Multivariate estimation ---------------------- #
n = 1000000
dim = 3
# Generate 2 numbers randomly along each dimension, with the larger being the upper bound and vice versa
bounds = np.random.uniform(low=0.0, high=1.0, size=(dim, 2))
bounds = np.sort(bounds, axis=1)
a, b = bounds[:,0], bounds[:,1]

mean = 10 * np.random.uniform(low=0.0, high=1.0, size=dim)
A = np.random.rand(dim, dim)
covar = np.dot(A, A.transpose())    # Needs to be SPD matrix

uniform_data = data_generation_simple.multivariate_gen(distribution='uniform', a=a, b=b, n=n, dim=3)
gaussian_data = data_generation_simple.multivariate_gen(distribution='normal', mean=mean, covariance=covar, n=n, dim=3)
# Let's assume we have absolutely no knowledge of the underlying distribution our data was sampled from. To demonstrate that 
# the sample mean is a good estimator for the true mean, we use the point estimate described in the theory section, which depends
# on the law of large numbers.
uniform_avg = np.average(uniform_data, axis=0)
gaussian_avg = np.average(gaussian_data, axis=0)

print(f'True Uniform Mean: {(b+a)/2} -- Estimated Uniform Mean: {uniform_avg}')
print(f'True Gaussian Mean: {mean} -- Estimated Gaussian Mean: {gaussian_avg}')

print('\n---------------------------------------\n')

centered_uniform = uniform_data - uniform_avg
centered_gaussian = gaussian_data - gaussian_avg

uniform_var = np.sum(np.square(centered_uniform), axis=0) / (n-1)

sample_covariances = np.empty((n, dim, dim))
for i in range(len(centered_gaussian)):
    point = centered_gaussian[i]
    sample_covariances[i] = np.outer(point, point)

gaussian_covar = np.sum(sample_covariances, axis=0) / (n-1)

print(f'True Uniform Variance: {np.square(b-a)/12} -- Estimated Uniform Variance: {uniform_var}')
print(f'True Gaussian Variance: \n{covar} \n Estimated Gaussian Covariance: \n{gaussian_covar}')