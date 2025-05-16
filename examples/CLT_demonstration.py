import sys
sys.path.insert(1, 'utils')
sys.path.insert(1, '../utils')

import data_generation_simple
import numpy as np
import matplotlib.pyplot as plt

# In this demonstration, I will create 2D and 3D plots for univariate and R^2 vector-valued random variables.
# The idea is that samples from the sample mean distribution approach the standard normal as the number of data points used to compute
# each sample mean increases.

def standard_normal(x):
    return np.exp((-1 * np.square(x))/2) / np.sqrt(2*np.pi)
    

n = 100000
samples = 1000  # Number of samples to compute each sample mean
a, b = 0.0, 3.0
uniform_bars = np.empty(shape=(n))
normal_bars = np.empty(shape=(n))

mean, variance = 0.0, 1.5
for i in range(n):
    uniform_data = data_generation_simple.univariate_gen(distribution='uniform', a=a, b=b, n=samples)
    gaussian_data = data_generation_simple.univariate_gen(distribution='normal', mean=mean, variance=variance, n=samples)
    uniform_avg = (np.sum(uniform_data) - samples*((a+b)/2)) / (np.sqrt(samples) * np.sqrt(np.square(b-a)/12))
    gaussian_avg = np.average(gaussian_data)

    uniform_bars[i] = uniform_avg
    normal_bars[i] = gaussian_avg
    
    # centered_uniform = uniform_data - uniform_avg
    # centered_gaussian = gaussian_data - gaussian_avg
    # uniform_var = np.sum(np.square(centered_uniform)) / (n-1)
    # gaussian_var = np.sum(np.square(centered_gaussian)) / (n-1)

uniform_bars.sort()
normal_bars.sort()

x_points = np.arange(-4, 4, step=0.001)
print(x_points)
y_points = standard_normal(x_points)
print(y_points[2000])

plt.hist(uniform_bars, bins=500, density=True)
plt.plot(x_points, y_points)
plt.show()