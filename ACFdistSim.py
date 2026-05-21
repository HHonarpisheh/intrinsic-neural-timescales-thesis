import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# Generate different types of time series
np.random.seed(42)
n_samples = 1000

# 1. Exponential Decay (AR(1) process)
ar1 = np.cumsum(np.random.randn(n_samples))  # Random walk (integrated AR(1))
acf_ar1 = acf(ar1, nlags=40, fft=True)

# 2. Power-Law Decay (Long-Range Memory)
power_series = np.cumsum(np.random.randn(n_samples))**0.5
acf_power = acf(power_series, nlags=40, fft=True)

# 3. Oscillatory ACF (Periodic Signal)
osc_signal = np.sin(2 * np.pi * np.arange(n_samples) / 24) + 0.2 * np.random.randn(n_samples)
acf_osc = acf(osc_signal, nlags=40, fft=True)

# 4. White Noise (Random)
white_noise = np.random.randn(n_samples)
acf_white = acf(white_noise, nlags=40, fft=True)

# --- Plot ACF Distributions ---
fig, ax = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

# Histogram of AR(1) ACF
ax[0].hist(acf_ar1, bins=30, color='blue', alpha=0.7, edgecolor='black')
ax[0].set_xlabel("ACF Value")
ax[0].set_ylabel("Frequency")
ax[0].set_title("ACF Histogram - Exponential Decay (AR1)")

# Histogram of Power-Law ACF
ax[1].hist(acf_power, bins=30, color='red', alpha=0.7, edgecolor='black')
ax[1].set_xlabel("ACF Value")
ax[1].set_title("ACF Histogram - Power Law")

# Histogram of Oscillatory ACF
ax[2].hist(acf_osc, bins=30, color='green', alpha=0.7, edgecolor='black')
ax[2].set_xlabel("ACF Value")
ax[2].set_title("ACF Histogram - Oscillatory")

# Histogram of White Noise ACF
ax[3].hist(acf_white, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax[3].set_xlabel("ACF Value")
ax[3].set_title("ACF Histogram - White Noise")

plt.tight_layout()
plt.show()
