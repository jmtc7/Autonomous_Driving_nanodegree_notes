import numpy as np

def sigmoid(x):
    """
    Applies the Sigmoid funtion to a given X value.
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """
    Applies the first derivative of the Sigmoid funtion to a given X value.
    """
    return sigmoid(x) * (1 - sigmoid(x))


learnrate = 0.5
x = np.array([1, 2])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5])

## Calculate one gradient descent step for each weight
# Calculate output of neural network
nn_output = sigmoid(w[0]*x[0] + w[1]*x[1])

# Calculate error of neural network
error = y - nn_output

# Calculate change in weights
del_w = learnrate * (error*nn_output*(1-nn_output)*x)

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
