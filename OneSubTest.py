import os
import math
import zipfile
import tempfile
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from scipy.optimize import least_squares

# Variables
task = 'rest'
session = 'BAS1'
TR = 2500  # Repetition time in ms

# Load parcellated timeseries data
pts = nib.load(r"D:\CMI\Timescale\Scripts\sub-A00030980_ses-BAS1_task-rest_acq-CAPVARIANTPartialFourierTotalReadoutTime_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii").get_fdata()

def cos_exp_decay(x, a, b, c):
    return np.cos(a * x) * np.exp(-x / b) + c

# Function to calculate intrinsic neural timescale indices for one ROI
def calculate_timescale_indices(roi):
    timeseries = pts[:, roi].T
    nlags = len(timeseries)

    # Calculate ACF
    acf = sm.tsa.acf(timeseries, nlags=nlags)
    if np.isnan(acf).any():
        print(f"NaN values found in ACF for ROI {roi}")
        acf = np.nan_to_num(acf, nan=0)  # Replace NaNs with 0

    # AR1
    AR1 = acf[1]

    # Zero Crossing
    zero_cross = np.where(acf < 0)[0]
    zero_cross = zero_cross[0] if zero_cross.size > 0 else nlags - 1

    # 1/e Threshold
    e_threshold = 1 / math.e
    below_e = acf <= e_threshold
    lag_e_1_max = np.where(below_e)[0][0] * TR / 1000 if below_e.any() else nlags * TR / 1000

    # Sum of ARs
    first_non_positive_idx = np.where(acf[1:] < 0)[0]
    sum_ARs = np.sum(acf[1:first_non_positive_idx[0] + 1]) * TR / 1000 if first_non_positive_idx.size > 0 else 1 * TR / 1000

    # Half-Max Lag
    half_max_lag = np.argmax(acf < 0.5 * acf[0])
    # ACF at half lag
    acf_half_max = acf[zero_cross // 2]

    # Exponential Decay Fitting
    lags = np.linspace(0, (nlags - 1) * TR / 1000, nlags)
    a_initial = acf[0] if acf[0] > 0 else 1e-6
    b_initial = zero_cross
    c_initial = acf[zero_cross]
    initial_guess = [a_initial, b_initial, c_initial]

    def residuals(params):
        return cos_exp_decay(lags[:zero_cross], *params) - acf[:zero_cross]

    bounds = ([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])
    try:
        result = least_squares(residuals, initial_guess, bounds=bounds, max_nfev=200000)
        params = result.x
        tau = result.x[1]
    except ValueError as e:
        print(f"Skipping ROI {roi} due to error: {e}")
        params = np.nan
        tau = np.nan

    return {
        'AR1': AR1,
        'Zero_Cross': zero_cross,
        'Lag_e_1_max': lag_e_1_max,
        'Sum_ARs': sum_ARs,
        'ACF_Half_Lag': acf_half_max,
        'Lag_Half_ACF' : half_max_lag,
        'Params': params,
        'Tau': tau
    }

# Calculate indices for all ROIs
n_rois = pts.shape[1]
results = Parallel(n_jobs=-1)(delayed(calculate_timescale_indices)(roi) for roi in tqdm(range(n_rois)))

# Convert results to DataFrame
results_df = pd.DataFrame(results)

#print(results_df)

# Plot ACF and fitted curve for the first ROI with all timescale indices as lines

# Plot ACF and fitted curve for the first ROI with all timescale indices as lines
roi = 0  # Focus on the first ROI
timeseries = pts[:, roi].T
nlags = len(timeseries)
lags = np.linspace(0, (nlags - 1) * TR / 1000, nlags)
acf = sm.tsa.acf(timeseries, nlags=nlags)
fitted_acf = cos_exp_decay(lags, *results_df.loc[roi, 'Params'])

# Extract timescale indices for the first ROI
AR1 = results_df.loc[roi, 'AR1']
zero_cross = results_df.loc[roi, 'Zero_Cross']
lag_e_1_max = results_df.loc[roi, 'Lag_e_1_max']
sum_ARs = results_df.loc[roi, 'Sum_ARs']
acf_half_max = results_df.loc[roi, 'ACF_Half_Lag']
half_max_lag = results_df.loc[roi, 'Lag_Half_ACF'] * TR / 1000 # convert to seconds
tau = results_df.loc[roi, 'Tau']

# Plot ACF and fitted curve
plt.figure(figsize=(12, 8))
plt.plot(lags, acf, 'ro', label='ACF')
plt.plot(lags, fitted_acf, 'k-', label='Fitted ACF')

# Shade the area under the ACF curve before the zero crossing (sum_ARs)
zero_cross_lag = zero_cross * TR / 1000  # Convert zero crossing to seconds
plt.fill_between(lags[:zero_cross + 1], acf[:zero_cross + 1], color='pink', alpha=0.5, label=f'Sum of ARs: {sum_ARs:.2f} s')

# Add vertical and horizontal lines for each timescale index
plt.axhline(y=AR1, color='g', linestyle='--', label=f'AR1: {AR1:.2f}')
plt.axvline(x=zero_cross_lag, color='m', linestyle='--', label=f'Zero Crossing: {zero_cross_lag:.2f} s')
plt.axvline(x=lag_e_1_max, color='c', linestyle='--', label=f'Lag at 1/e: {lag_e_1_max:.2f} s')
plt.axvline(x=half_max_lag, color='orange', linestyle='--', label=f'Lag at 1/2: {half_max_lag:.2f} s')
plt.axhline(y=acf_half_max, color='y', linestyle='--', label=f'ACF Half Max: {acf_half_max:.2f}')
plt.axvline(x=tau, color='b', linestyle='--', label=f'Tau: {tau:.2f} s')

# Add labels, title, and legend
plt.xlabel('Lag (s)')
plt.ylabel('ACF')
plt.title(f'ACF, Fitted Exponential Decay, and Timescale Indices for ROI {roi}')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
