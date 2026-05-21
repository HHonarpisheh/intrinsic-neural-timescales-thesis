import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as sm_tsa
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import math
import nibabel as nib


# Define exponential decay function
def exp_decay(x, a, b, c):
    return a * np.exp(-x / b) + c
   
# Define double exponential decay function
def exp2_decay(x, a1, a2, b1, b2, c):
    return a1 * np.exp((-x / b1)) + a2 * np.exp((-x / b2)) + c
    
def calc_acf(sig, nlags):
    return sm_tsa.acf(sig, nlags=nlags)
    
def calc_pacf(sig, nlags):
    return sm_tsa.pacf(sig, nlags=nlags)

# Compute the number of significant lags
def calc_zero_cross(acf):
    idx = np.where(acf <= 0)[0]
    if idx.size == 0:
        return len(acf) - 1
    else:
        zero_crossings = idx[0]
        return zero_crossings - 1
    
def calc_lag_when_e_1(acf, nlags):
    lags = np.arange(nlags)
    # Compute 1/e threshold
    e_threshold = 1 / math.e  # ~0.3679
    # Find the first lag where ACF drops below 1/e
    below_e = acf <= e_threshold  # Boolean array
    indices = np.where(below_e)[0]  # Get indices where condition is met
    return lags[indices[0]]  # First occurrence of ACF ≤ 1/e

        
def calc_tau(acf, TR):
    x_data = np.arange(len(acf)) * TR  # Convert lag index to time in ms
    # Improved initial guess
    initial_guess = [acf[0], sig_lags, acf[-1]]  
    
    def residuals(params):
        return exp_decay(np.arange(sig_lags), *params) - acf[:sig_lags]
    
    bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])  
    result = least_squares(residuals, initial_guess, bounds=bounds, max_nfev=200000)
    
    plt.figure(figsize=(12, 6))
    plt.stem(np.arange(len(acf)), acf, label="ACF Values", linefmt="pink", markerfmt="m.", basefmt="k-")
    plt.plot(np.arange(len(acf)), exp_decay(np.arange(len(acf)), *result.x),
             label=f"Fitted Exp Decay (τ={result.x[1]:.2f})", color='red')
    plt.legend()
    
    return result.x[1]  # Return decay constant (tau)

def calc_tau2(acf):
    initial_guess2 = [acf[0], acf[0], len(acf)//2, len(acf)//2, acf[-1]] 
    popt2, _ = curve_fit(exp2_decay, np.arange(len(acf)//2), acf[:len(acf)//2], p0=initial_guess2, maxfev=200000)
    fitted_exp2_decay = exp2_decay(np.arange(len(acf)), *popt2)
    # Visualization
    plt.plot(np.arange(len(acf)), fitted_exp2_decay, label=f"Fitted Double Exp Decay (τ1={popt2[2]:.2f}, τ2={popt2[3]:.2f})", color='blue')
    return popt2[2], popt2[3]   # Extract b1, b2
    
    
def calc_sum_ARs(acf):
    first_non_positive_idx = np.where(acf[1:] < 0)[0] - 1
    if len(first_non_positive_idx) == 0:  # No non-positive values found
        return 1
    else:
        sum_ARs = np.sum(acf[1:first_non_positive_idx[0] + 1])
        return sum_ARs

def calc_acf_at_half_lag(acf):
    lags = calc_zero_cross(acf)
    lag_index = lags // 2
    acf_half_max = acf[lag_index]
    return acf_half_max

# Generate a synthetic time series with oscillations and decay
#np.random.seed(42)
#n = 512
#time_series = np.cos(np.linspace(0, 5*np.pi, n)) * np.exp(-np.linspace(0, 25, n))  # Damped oscillations
#time_series += np.random.normal(0, 0.1, n)  # Add small noise

# Test the indices on one time series(of a region) of one of the subjects
time_series = nib.load(r"D:\CMI\Timescale\Scripts\sub-A00030980_ses-BAS1_task-rest_acq-645VARIANTMultibandAccelerationFactorPartialFourierTotalReadoutTime_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii").get_fdata()
#print(time_series.shape) # =num of timepoints*num of rois
sig = time_series[:,0]
nlags = time_series.shape[0]

acf = calc_acf(sig, nlags)
pacf = calc_pacf(sig, nlags // 2 - 1)

AR1 = acf[1]
AR2 = acf[2]
AR3 = acf[3]
PAR0 = pacf[0]
PAR1 = pacf[1]
PAR2 = pacf[2]

sig_lags = calc_zero_cross(acf)
lag_when_e_1 = calc_lag_when_e_1(acf, nlags)
tau = calc_tau(acf, 645)
tau1, tau2 = calc_tau2(acf)
sum_ARs = calc_sum_ARs(acf)
acf_at_half_lag = calc_acf_at_half_lag(acf)

# Annotate key metrics
if not np.isnan(lag_when_e_1):
    plt.axvline(x=lag_when_e_1, color='purple', linestyle='--', label=f"Lag at ACF=1/e ({lag_when_e_1:.2f})")
if not np.isnan(sig_lags):
    plt.axvline(x=sig_lags, color='orange', linestyle='--', label=f"Lag at ACF=0 ({sig_lags:.2f})")
if not np.isnan(acf_at_half_lag):
    plt.axhline(y=acf_at_half_lag, color='cyan', linestyle='--', label=f"ACF at Lag/2 ({acf_at_half_lag:.2f})")

plt.text(10, 0.5, f" AR1={AR1:.2f}\n AR2={AR2:.2f}\n AR3={AR3:.2f}\nSum ARs={sum_ARs:.2f}\n PAR0={PAR0:.2f}\n PAR1={PAR1:.2f}\n PAR2={PAR2:.2f}",
         bbox=dict(facecolor='white', alpha=0.7), fontsize=10)

plt.xlabel("Time Points")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation Metrics with Fitted Decay Curves")
plt.legend()
plt.grid(True)
plt.show()
