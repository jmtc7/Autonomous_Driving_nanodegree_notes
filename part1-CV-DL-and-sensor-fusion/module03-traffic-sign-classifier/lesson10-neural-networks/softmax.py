import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    # Compute the exponential of every score in the list and the addition of all of them
    exp_L = np.exp(L)
    exp_L_sum = np.sum(exp_L)
    
    # List to store softmax scores
    result = []
    
    # Compute the softmax scores
    for exp_score in exp_L:
        result.append(exp_score/exp_L_sum)
    
    return result
