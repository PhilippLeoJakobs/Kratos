import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt

# Define the distribution parameters
mean = 0.0
std_dev = 1.0
lower = -2.0
upper = 2.0
distribution = cp.J(cp.TruncNormal(lower, upper, mean, std_dev), cp.TruncNormal(lower, upper, mean, std_dev))

# Generate samples and sample values
num_samples = 1000
samples = generate_samples(distribution, num_samples=num_samples, strategy="random")
sample_value = np.random.rand(num_samples)  # Replace with actual sample values

# Generate orthogonal polynomial expansion
poly_expansion = cp.orth_ttr(self.pce_order, distribution)

# Fit PCE model using least-squares regression
pce_model_value = cp.fit_regression(poly_expansion, samples.T, sample_value, rule="T")

# Check the constant term and other coefficients
print("PCE Model Coefficients: ", pce_model_value)

# Calculate the mean value of the response using the PCE model
value = cp.E(pce_model_value, distribution, rule="T")
print("PCE Model Mean Value: ", value)

# Manual computation of the mean value
manual_mean_value = np.mean([cp.call(pce_model_value, s) for s in samples.T])
print("Manual Mean Value: ", manual_mean_value)

# Compare with other integration rules
expected_value_ols = cp.E(pce_model_value, distribution, rule="ols")
print("Expected Value with Rule OLS: ", expected_value_ols)

# Plot true vs predicted values to validate the fit
predicted_values = [cp.call(pce_model_value, s) for s in samples.T]
plt.scatter(sample_value, predicted_values)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('PCE Model Fit')
plt.show()

# Debugging output for sample values and force vectors
print("Sample Values: ", sample_value)
