def update(mean1, var1, mean2, var2):
    """
    Combines two gaussians with the received means and variances returning the mean and variance of the output.
    """
    new_mean = float(var2 * mean1 + var1 * mean2) / (var1 + var2)
    new_var = 1./(1./var1 + 1./var2)
    return [new_mean, new_var]

def predict(mean1, var1, mean2, var2):
    """
    Applies a motion update using the initial robot position (mean1, var1), the applied motion model (mean2 (ideal translation), var2 (uncertainty))
    """
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]


# Lists of measurements and motion commands to be applied
measurements = [5., 6., 7., 9., 10.]
motion = [1., 1., 2., 1., 1.]

measurement_sig = 4. # Measurement uncertainty
motion_sig = 2. # Motion model uncertainty

# Starting (uncertain) robot pose
mu = 0.
sig = 10000.

# Print out final values of mean and variance [mu, sig]. 
for idx, measure in enumerate(measurements):
    # Take measure and update pose
    mu, sig = update(mu, sig, measure, measurement_sig)
    # Move and update pose
    mu, sig = predict(mu, sig, motion[idx], motion_sig)

# Print final result
print [mu, sig]
