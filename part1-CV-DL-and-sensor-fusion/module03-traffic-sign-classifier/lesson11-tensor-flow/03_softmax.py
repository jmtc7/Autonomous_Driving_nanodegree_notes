# Solution is available in the other "solution.py" tab
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    print(x)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=0)
    
    return exp_x/sum_exp_x

logits = [3.0, 1.0, 0.2]
print(softmax(logits))
