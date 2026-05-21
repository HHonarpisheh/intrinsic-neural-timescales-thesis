import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.tsa.stattools as smt
from scipy.optimize import curve_fit


def exp_decay(x, a, b, c):
    return a * np.exp(-x / b) + c


# Generate a synthetic time series with oscillations and decay
np.random.seed(42)
n = 512
time_series = np.cos(np.linspace(0, 10*np.pi, n)) * np.exp(-np.linspace(0, 40, n))  # Damped oscillations
time_series += np.random.normal(0, 0.1, n)  # Add small noise

# Compute the ACF
nlags = int(n)  # Ensure nlags is an integer
acf_values = smt.acf(time_series, nlags=nlags, fft=True)

# Compute tau using the given equation
#tau_values = -1 / np.log(acf_values)
popt, _ = curve_fit(exp_decay, np.arange(len(acf_values)), acf_values, maxfev=10000)
tau_values = popt[1]

print(tau_values.shape)
print(acf_values.shape)

# Remove invalid tau values (filter out -inf)
valid_mask1 = np.isfinite(acf_values)
valid_mask2 = np.isfinite(tau_values)
valid_mask = valid_mask1 & valid_mask2
star_values = acf_values[valid_mask]
tau_values = tau_values[valid_mask]


# Compute the correlation coefficient
correlation_coefficient, _ = pearsonr(star_values, tau_values)
# Output the correlation coefficient
print(correlation_coefficient)

# Create figure with two subplots for histograms
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Histogram of acf values
ax[0].hist(star_values, bins=50, orientation='horizontal', color='blue', alpha=0.7, edgecolor='black')
ax[0].set_ylabel("Value")
ax[0].set_xlabel("Time Occurrences")
ax[0].set_title("Distribution of ACF (*)")

# Histogram of tau values
ax[1].hist(tau_values, bins=50, orientation='horizontal', color='red', alpha=0.7, edgecolor='black')
ax[1].set_xlabel("Time Occurrences")
ax[1].set_title("Distribution of Tau (-1 / ln(*))")

plt.tight_layout()
plt.show()
