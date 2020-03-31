import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    result = []
    
    for idx in range(len(Y)):
        y = Y[idx]
        p = P[idx]
        
        result.append(y*np.log(p) + (1-y)*np.log(1-p))
    
    return -np.sum(result)
