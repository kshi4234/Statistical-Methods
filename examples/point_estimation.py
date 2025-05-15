import sys
sys.path.insert(1, 'utils')
sys.path.insert(1, '../utils')

import data_generation_simple
import numpy as np

# ---------------------- Univariate estimation ---------------------- #
n = 10000000
a, b = 0.0, 3.0
mean, variance = 0.0, 1.5
uniform_data = data_generation_simple.univariate_gen(distribution='uniform', a=a, b=b, n=n)
gaussian_data = data_generation_simple.univariate_gen(distribution='normal', mean=mean, variance=variance, n=n)
# Let's assume we have absolutely no knowledge of the underlying distribution our data was sampled from. To demonstrate that 
# the sample mean is a good estimator for the true mean, we use the point estimate described in the theory section, which depends
# on the law of large numbers.
uniform_avg = np.average(uniform_data)
gaussian_avg = np.average(gaussian_data)

print(f'True Uniform Mean: {(b+a)/2} -- Estimated Uniform Mean: {uniform_avg}')
print(f'True Gaussian Mean: {mean} -- Estimated Gaussian Mean: {gaussian_avg}')

print('\n---------------------------------------\n')

centered_uniform = uniform_data - uniform_avg
centered_gaussian = gaussian_data - gaussian_avg

uniform_var = np.sum(np.square(centered_uniform)) / (n-1)
gaussian_var = np.sum(np.square(centered_gaussian)) / (n-1)

print(f'True Uniform Variance: {np.square(b-a)/12} -- Estimated Uniform Variance: {uniform_var}')
print(f'True Gaussian Variance: {variance} -- Estimated Gaussian Variance: {gaussian_var}')