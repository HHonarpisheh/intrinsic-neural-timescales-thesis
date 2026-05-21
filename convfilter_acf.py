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

# Compute ACF of the first timeseries
lags = np.linspace(0, (nlags - 1) * TR / 1000, nlags)
acf = sm.tsa.acf(pts[:, roi], nlags=nlags)
acf /= np.max(acf)

# Manual convolution function
def manual_convolve(signal, kernel):
    signal_len = len(signal)
    kernel_len = len(kernel)
    output_len = signal_len + kernel_len - 1
    output = np.zeros(output_len)
    
    # DO NOT FLIP THE KERNEL
    for i in range(output_len):
        for j in range(kernel_len):
            if 0 <= i - j < signal_len:
                output[i] += signal[i - j] * kernel[j] 
    
    return output[:signal_len]  # Trim to original length


# Fit exponential decay convolved with BPF to ACF
def residuals(params):
    exp_signal = exp_decay(lags, params)
    #convolved_signal = manual_convolve(exp_signal, acf_bpf_response)
    convolved_signal = fftconvolve(exp_signal, acf_bpf_response, mode = 'full')[:nlags]
    return convolved_signal - acf

# Fit the model
initial_guess = [fs]
bounds = ([0], [np.inf])
result = least_squares(residuals, initial_guess, bounds=bounds, max_nfev=200000)
params = result.x
print(f"Fitted time constant: {params}")

# Compute the fitted ACF using manual convolution
#fitted_acf = manual_convolve(exp_decay(lags, *params), acf_bpf_response)
fitted_acf = fftconvolve(exp_decay(lags, *params), acf_bpf_response, mode = 'full')[:nlags]
fitted_exp = exp_decay(lags, *params)

# Plot BPF response and its ACF in one figure
plt.figure(figsize=(12, 6))

# Plot BPF impulse response
plt.subplot(1, 2, 1)
plt.plot(lags, bpf_response, 'g-', label='BPF Impulse Response')
plt.xlabel('Lags')
plt.ylabel('Amplitude')
plt.title('BPF Impulse Response')
plt.grid(True)
plt.legend()

# Plot ACF of BPF response
plt.subplot(1, 2, 2)
plt.plot(lags, acf_bpf_response, 'm-', label='ACF of BPF Response')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('ACF of BPF Response')
plt.grid(True)
plt.legend()

plt.tight_layout()

# Plot ACF and Fitted ACF in another figure
plt.figure(figsize=(10, 6))
plt.plot(lags, acf, 'r.', label='ACF')
plt.plot(lags, fitted_acf, 'b-', label='Fitted ACF')
plt.plot(lags, fitted_exp, 'y--', label='Fitted Exp')
plt.xlabel('Lags')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.show()
