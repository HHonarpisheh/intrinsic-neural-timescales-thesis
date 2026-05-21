import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf

# Define single and double exponential decay functions
def exp_decay(x, a, b, c):
    return a * np.exp(-x / b) + c

def exp2_decay(x, a1, a2, b1, b2, c):
    return a1 * np.exp(-x / b1) + a2 * np.exp(-x / b2) + c

# Generate synthetic time series data
np.random.seed(42)
time_series = np.random.normal(0, 0.1, 100)

# Compute the autocorrelation function (ACF)
acf_values = acf(time_series, nlags=30, fft=True)

# Define x_data as the lag indices
x_data = np.arange(len(acf_values))
y_data = acf_values  # Use ACF values as y-data

# Fit single and double exponential decay to ACF
popt_exp, _ = curve_fit(exp_decay, x_data, y_data)
popt_exp2, _ = curve_fit(exp2_decay, x_data, y_data)

# Generate fitted ACF curves
y_fit_exp = exp_decay(x_data, *popt_exp)
y_fit_exp2 = exp2_decay(x_data, *popt_exp2)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, label="ACF Data", alpha=0.6)
plt.plot(x_data, y_fit_exp, label="Single Exp Fit", linestyle="--", color="blue", linewidth=2)
plt.plot(x_data, y_fit_exp2, label="Double Exp Fit", linestyle="-", color="red", linewidth=2)
plt.title("Exponential Decay Fitting to ACF")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.legend()
plt.show()

# Print fitted parameters
print(f"Single Exp Fit: a = {popt_exp[0]:.2f}, b = {popt_exp[1]:.2f}, c = {popt_exp[2]:.2f}")
print(f"Double Exp Fit: a1 = {popt_exp2[0]:.2f}, a2 = {popt_exp2[1]:.2f}, b1 = {popt_exp2[2]:.2f}, b2 = {popt_exp2[3]:.2f}, c = {popt_exp2[4]:.2f}")
