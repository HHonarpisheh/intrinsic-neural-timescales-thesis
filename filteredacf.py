import numpy as np
import nibabel as nib
from scipy.signal import butter, lfilter, fftconvolve
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Variables
task = 'rest'
session = 'BAS1'
TR = 645  # in ms

# Load parcellated timeseries data
pts = nib.load(r"D:\CMI\Timescale\Scripts\sub-A00030980_ses-BAS1_task-rest_acq-645VARIANTMultibandAccelerationFactorPartialFourierTotalReadoutTime_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii").get_fdata()
roi = 100
nlags = len(pts[:, roi])

# Band-pass filter parameters (for BPF response only)
lowcut = 0.01
highcut = 0.08
fs = 1000 / TR  # Sampling frequency
order = 4

# Create band-pass filter (for BPF response only)
b, a = butter(order, [lowcut/fs, highcut/fs], btype='band')

# Generate BPF impulse response
impulse = np.zeros(nlags)
impulse[0] = 1  # Impulse at time zero
bpf_response = lfilter(b, a, impulse)
acf_bpf_response = sm.tsa.acf(bpf_response, nlags=nlags)

# Exponential decay function
def exp_decay(x, b):
    return np.exp(-x / b)

# Compute ACF of the first timeseries (fMRI data)
lags = np.linspace(0, (nlags - 1) * TR / 1000, nlags)  # Convert lags to seconds
acf = sm.tsa.acf(pts[:, roi], nlags=nlags)
acf /= np.max(acf)  # Normalize the ACF

# Residuals function for optimization (Exponential Decay Convolved with ACF of BPF Response)
def residuals(params):
    # params contains the decay constant 'b'
    exp_signal = exp_decay(lags, params[0])  # Generate exponential decay signal
    convolved_signal = fftconvolve(exp_signal, acf_bpf_response, mode='full')[:nlags]  # Convolve with ACF of BPF response
    return convolved_signal - acf  # Return the difference (residuals) between convolved signal and actual ACF

# Fit the model using least squares optimization
initial_guess = [fs]  # Initial guess for decay constant (b), can be set to the sampling frequency for a reasonable start
bounds = ([0], [np.inf])  # Bound the decay constant to be non-negative
result = least_squares(residuals, initial_guess, bounds=bounds, max_nfev=200000)

# Extract the fitted decay constant (time constant) from the result
fitted_b = result.x[0]
print(f"Fitted decay constant (b): {fitted_b}")

# Compute the fitted ACF using the optimized decay constant
fitted_exp = exp_decay(lags, fitted_b)  # Recompute exponential decay using the fitted b
fitted_acf = fftconvolve(fitted_exp, acf_bpf_response, mode='full')[:nlags]  # Convolve it with the ACF of BPF response

# Plot the ACF of fMRI Timeseries and Fitted ACF
plt.figure(figsize=(10, 6))
plt.plot(lags, acf, 'r.', label='ACF of fMRI Timeseries')
plt.plot(lags, fitted_acf, 'b-', label='Fitted ACF (Exponential Decay Convolved with BPF ACF)')
plt.plot(lags, fitted_exp, 'y--', label='Fitted Exponential Decay')
plt.xlabel('Lags (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.title('Fitting Exponential Decay Convolved with BPF ACF to ACF of fMRI Timeseries')
plt.show()

