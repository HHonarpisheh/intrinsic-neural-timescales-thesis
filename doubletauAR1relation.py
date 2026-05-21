import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the double exponential decay function
def exp2_decay(t, a1, a2, tau1, tau2, c):
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c

# Generate synthetic time lags
t = np.arange(50)  # 50 time points

# True parameters for tau1 and tau2
true_a1, true_a2 = 1.5, 0.8
true_tau1, true_tau2 = 5, 15  # Faster and slower decay rates
true_c = 0.1  # Offset

# Generate ACF using the model with some noise
acf_values = exp2_decay(t, true_a1, true_a2, true_tau1, true_tau2, true_c) + np.random.normal(0, 0.05, len(t))

# Fit the double exponential model to estimate tau1 and tau2
popt, _ = curve_fit(exp2_decay, t, acf_values, p0=[1, 1, 5, 10, 0], maxfev=10000)

# Extract estimated tau1 and tau2
estimated_tau1, estimated_tau2 = popt[2], popt[3]

print(f"Estimated τ1: {estimated_tau1:.3f}, τ2: {estimated_tau2:.3f}")

# --- Visualization ---

# Plot ACF vs time
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(t, acf_values, color='blue', alpha=0.6, label="Noisy ACF Data")
plt.plot(t, exp2_decay(t, *popt), color='red', label="Fitted Exp2 Decay")
plt.xlabel("Time Lag")
plt.ylabel("ACF")
plt.title("Autocorrelation Function with Double Exponential Fit")
plt.legend()

# Plot estimated tau1 vs tau2
plt.subplot(1, 2, 2)
plt.scatter([true_tau1], [true_tau2], color='green', s=10, label="True Values")
plt.scatter([estimated_tau1], [estimated_tau2], color='red', s=10, label="Estimated Values")
plt.xlabel("Tau1")
plt.ylabel("Tau2")
plt.title("Tau1 vs Tau2 Relationship")
plt.legend()

plt.tight_layout()
plt.show()
