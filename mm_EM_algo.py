import math
from scipy.stats import norm
import numpy as np
from numpy.random import normal
import sys


def EM_algo(data, iters = 20, NUM_GAUSSIANS = 2): 
    """Runs EM algorithm
    data --- pairs of datum of shape (obs, error), where obs is the observed value,
                                        and error is the sampling error of the data point.
    """
    
    print 'running EM algorithm for ', iters, ' iterations (slow)'
    
    SAMPLE_SIZE = len(data)
    print SAMPLE_SIZE
    
    current_ws = [1. / NUM_GAUSSIANS] * NUM_GAUSSIANS
    current_ms = np.random.uniform(-2,2,NUM_GAUSSIANS)
    current_vs = np.random.uniform(0,2,NUM_GAUSSIANS)
    
    # matrix that caches the expected class probabilities for each sample
    pkn_matrix = np.zeros(shape = (NUM_GAUSSIANS, SAMPLE_SIZE))

    
    sys.stdout.flush()
    for i in range(iters):
        pkn_matrix = E_step(data, current_ws, current_ms, current_vs, pkn_matrix, SAMPLE_SIZE, NUM_GAUSSIANS)
        current_ws, current_ms, current_vs = M_step(data,current_ws, current_ms, current_vs, pkn_matrix, SAMPLE_SIZE, NUM_GAUSSIANS)
        
        print 'iteration ', i,
        print clip_float_list(current_ws), clip_float_list(current_ms), clip_float_list(current_vs)
        sys.stdout.flush()
    return current_ws, current_ms, current_vs

# pkn matrix represents the class probabilities for each sample
def E_step(data, weights, means, sigmas, pkn_matrix, SAMPLE_SIZE, NUM_GAUSSIANS):
    ret = np.zeros(shape = pkn_matrix.shape)
    for n, datum in enumerate(data):
        likelihoods = [weights[k] * likelihood(datum, means[k], sigmas[k]) for k in range(NUM_GAUSSIANS)]    
        s =  sum(likelihoods)
        for k in range(NUM_GAUSSIANS):
            if s == 0:
                ret[k][n] = 0
            else :
                ret[k][n] = likelihoods[k] / s     
    return ret

def likelihood(data, mean, sigma):
    obs, error = data
    sigma_corr = math.sqrt(error * error + sigma * sigma)
    return norm.pdf(obs,loc=mean, scale=sigma_corr)

def M_step(data, weights, means, sigmas, pkn_matrix, SAMPLE_SIZE, NUM_GAUSSIANS):
    
    new_means = update_means(data, weights, means, sigmas, pkn_matrix, SAMPLE_SIZE, NUM_GAUSSIANS)
    new_sigmas = update_sigmas(data, weights, means, sigmas, pkn_matrix, SAMPLE_SIZE, NUM_GAUSSIANS)
    new_weights = update_weights(data, weights, means, sigmas, pkn_matrix, SAMPLE_SIZE, NUM_GAUSSIANS)
    
    return (new_weights, new_means, new_sigmas)

def update_weights(data, weights, means, sigmas, pkn_matrix, SAMPLE_SIZE, NUM_GAUSSIANS):
    
    return [sum(pkn_matrix[k]) * 1. / SAMPLE_SIZE for k in range(NUM_GAUSSIANS)]

def update_means(data, weights, means, sigmas, pkn_matrix, SAMPLE_SIZE, NUM_GAUSSIANS):
    
    return [sum([(data[n][0] * pkn_matrix[k][n]) / (sigmas[k] ** 2 + data[n][1] ** 2) for n in range(SAMPLE_SIZE)])             / sum([(pkn_matrix[k][n]) / (sigmas[k] ** 2 + data[n][1] ** 2) for n in range(SAMPLE_SIZE)])             for k in range(NUM_GAUSSIANS)]
    
def update_sigmas(data, weights, means, sigmas, pkn_matrix, SAMPLE_SIZE, NUM_GAUSSIANS):
    return [get_optimal_sigma(data, means, sigmas, num, pkn_matrix, SAMPLE_SIZE, NUM_GAUSSIANS) for num in range(NUM_GAUSSIANS)]
    #return actual_sigmas
    
def get_optimal_sigma(data, means, sigmas, k, pkn_matrix, SAMPLE_SIZE, NUM_GAUSSIANS):
    target = sum(pkn_matrix[k])
    num_arr = [pkn_matrix[k][n] * (data[n][0] - means[k]) ** 2 for n in range(SAMPLE_SIZE)]
    denom_arr = [data[n][1] ** 2 for n in range(SAMPLE_SIZE)]
    fun = lambda x :  func2(x, num_arr, denom_arr) - target
    binary =  binary_search(fun)
    return 1. / binary

# helper function for updating the sigma in the M step, I tried a bunch of different rebalanced equations, this 
# seemed to be the fastest converging one
def func2(x, nom_arr, denom_arr):
    return sum([x * x * nom_arr[i] * 1. /( 1 + denom_arr[i] * x * x) for i in range(len(nom_arr))])

# attempts to find minimum of f(x) on range x>=0
# via a binary search, this works for monotone functions, e.g. func2 as above
def binary_search(fun):
    if fun(1) < fun(0):
        return 0
    COUNT_MAX = 10000
    counter = 0
    lo = 0
    hi = 0.1
    init_val = fun(0)
    while fun(hi) < 0 & counter < COUNT_MAX:  
        counter += 1
        hi *= 2
    while (abs(fun(hi)) > 0.000001 or abs(hi - lo) > 0.000001) and counter < COUNT_MAX:
        counter += 1
        mid = (hi + lo)/2
        if fun(mid) > 0:
            hi = mid
        else:
            lo = mid
    if not counter < COUNT_MAX:
        raise Exception("binary search took too long")
    return (hi + lo) /2

def clip_float_list(fl):
    return [str(f)[:5] for f in fl]


def get_random_weights(NUM_GAUSSIANS):
    randoms = np.random.uniform(range(NUM_GAUSSIANS))
    return randoms / sum(randoms)

def sample_mixture(NUM_GAUSSIANS, actual_weights, actual_means, actual_sigmas):
    model = np.random.choice(NUM_GAUSSIANS, p = actual_weights)
    sample_noise_sigma = min(np.random.pareto(1) , 5)  
    return (normal(actual_means[model], actual_sigmas[model]) + normal(0, sample_noise_sigma) , 
            sample_noise_sigma)

def main():
    NUM_GAUSSIANS = 2

    actual_weights = get_random_weights(NUM_GAUSSIANS)
    actual_means = np.random.uniform(-2,2,NUM_GAUSSIANS)
    actual_sigmas = np.random.uniform(0,2,NUM_GAUSSIANS)

    print 'actual weights, means sigmas : ', actual_weights, actual_means, actual_sigmas

    
    SAMPLE_SIZE = 10000

    data = [sample_mixture(NUM_GAUSSIANS, actual_weights, actual_means, actual_sigmas) for i in range(SAMPLE_SIZE)]

    current_ws, current_ms, current_vs = EM_algo(data, iters = 20, NUM_GAUSSIANS= 2)

if __name__ == '__main__':
    main()


