import numpy as np
import pandas as pd
from scipy import stats

NSIMS = 10000

# this is simulating all necessary values from kernel density estimation
# data is the original data, written so that data[k] has dimension dim
# N is the number of data points
# bandwidth is the array of chosen variances
# bandwidth[k] is the variance for kth component
# we assume that the Gaussian noise has independent components
# nYears is, as usual, the number of years in simulation
def simKDE(data, N, dim, nYears):
    # simulate randomly chosen index from 0 to N - 1
    # Need this since the function choice works only with 1D arrays
    index = np.random.choice(range(N), size = (NSIMS, nYears), replace = True)
    pick = data[index] # Result is two-dimensional array of vectors in R^dim 
    # Silverman's rule of thumb: Common factor for all 'dim' 
    silvermanFactor = (4/(dim + 2))**(1/(dim + 4))*N**(-1/(dim + 4))
    noise = [] # Here we will write 2D simulated arrays for each of 'dim' components
    for k in range(dim):
        # actual bandwidth for the kth component
        bandwidth = silvermanFactor * min(np.std(data[:, k]), stats.iqr(data[:, k]/1.34))
        # simulate kth component
        # we can simulate them independently since the covariance matrix is diagonal
        component = np.random.normal(0, bandwidth, size = (NSIMS, nYears))
        noise.append(component) # write the 2D simulated array for the current component 
    noise = np.transpose(np.array(noise), (1, 2, 0)) # need to swap coordinates to sum with 'pick'
    return pick + np.array(noise)