import math
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statsmodels.api as sm

# Define the exponential decay function
def exp_decay(x, a, b, c):
    return a * np.exp(-x / b) + c

# Compute ACF
def calc_acf(signal, max_lag):
    acf_values = sm.tsa.acf(signal, nlags=max_lag)
    return acf_values

# Compute the number of significant lags
def calc_zero_cross(acf):
    idx = np.where(acf <= 0)[0]
    if idx.size == 0:
        return len(acf) - 1
    else:
        zero_crossings = idx[0]
        return zero_crossings
    
# Load the time series
TR = 645  # Time resolution in milliseconds
#time_series = nib.load(r"D:\CMI\Timescale\Scripts\sub-A00028266_ses-BAS1_task-rest_acq-645_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii").get_fdata()
#time_series = nib.load(r"D:\CMI\Timescale\Scripts\sub-A00030980_ses-BAS1_task-rest_acq-645VARIANTMultibandAccelerationFactorPartialFourierTotalReadoutTime_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii").get_fdata()
time_series = nib.load(r"D:\CMI\Timescale\Scripts\sub-A00052461_ses-BAS1_task-rest_acq-645_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii").get_fdata()

# Select a region of interest (ROI)
sig = time_series[:60*1000//TR, 100]
#lags = np.arange(0, (sig.shape[0])*TR/1000, TR/1000)  # Lag in s
lags = np.linspace(0, (sig.shape[0] - 1) * TR / 1000, sig.shape[0])
acf = calc_acf(sig, len(lags))
sig_lags = calc_zero_cross(acf)
# Define x_data for ACF
x_data = lags
y_data = acf
print(f"ACF length: {len(acf)}, Lags length: {len(lags)}")
# Perform curve fitting
popt, pcov = curve_fit(exp_decay, x_data, y_data, p0=[acf[0], sig_lags*TR/1000, acf[sig_lags]])  # Initial guess

# Extract fitted parameters
a_fit, b_fit, c_fit = popt
print(f"Estimated Tau: {b_fit:.2f} s")  # Tau represents the neural timescale

# Generate the fitted curve at higher resolution
x_fit = np.linspace(0, max(x_data), num=500)  # High-resolution x-values
y_fit = exp_decay(x_fit, a_fit, b_fit, c_fit)
print("Zero-Crossing Lag:", sig_lags)
print("Estimated Parameters for Curve Fit:", [a_fit, b_fit, c_fit])

# Plot the results
plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, color='blue', label='ACF Data')  # ACF data at original TR resolution
plt.plot(x_fit, y_fit, color='red', label='Fitted Exponential Decay', linewidth=2)  # High-res fit
plt.title('ACF and Fitted Exponential Decay')
plt.xlabel('Time Lag (s)')
plt.ylabel('ACF')
plt.legend()
plt.grid(True)
plt.show()
