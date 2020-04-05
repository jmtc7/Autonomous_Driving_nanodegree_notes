def update(mean1, var1, mean2, var2):
	"""
	Combines two gaussians with the received means and variances returning the mean and variance of the output.
	"""
    new_mean = (var2 * mean1 + var1 * mean2) / (var1 + var2)
    new_var = 1/(1/var1 + 1/var2)
    return [new_mean, new_var]

def predict(mean1, var1, mean2, var2):
	"""
	Applies a motion update using the initial robot position (mean1, var1), the applied motion model (mean2 (ideal translation), var2 (uncertainty))
	"""
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]

print predict(10., 4., 12., 4.)
