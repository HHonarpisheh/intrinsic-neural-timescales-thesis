#Calulating Intrinsic Neural Timescale Indices of Human NKI
#state = resting state
#species = Human
#TR = (645, 1400, 2500)ms
#Parcellation = Schaefer400

#imports
import zipfile
import nibabel as nib
import tempfile
import os
import math
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

from joblib import Parallel, delayed
import statsmodels.tsa.api as sm_tsa
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from scipy.optimize import OptimizeWarning
import warnings

#functions
def get_pts(zip_file, task, session, TR): #time series points
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file) as z:
            if TR == 645:
                fname = [f for f in z.namelist() if f.endswith((
                                                            f'ses-{session}_task-{task}_acq-645_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii', 
                                                            f'ses-{session}_task-{task}_acq-645VARIANTMultibandAccelerationFactorPartialFourierTotalReadoutTime_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii'))]            
            elif TR == 1400:
                fname = [f for f in z.namelist() if f.endswith((
                                                            f'ses-{session}_task-{task}_acq-1400_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii', 
                                                            f'ses-{session}_task-{task}_acq-1400VARIANTMultibandAccelerationFactorPartialFourierTotalReadoutTime_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii'))]
                
            elif TR == 2500:
                fname = [f for f in z.namelist() if f.endswith((
                                                            f'ses-{session}_task-{task}_acq-CAP_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii', 
                                                            f'ses-{session}_task-{task}_acq-CAPVARIANTPartialFourierTotalReadoutTime_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii'))]
            if len(fname) == 1:
                fname = fname[0]
                z.extract(fname, temp_dir)
                pts = nib.load(f'{temp_dir}/{fname}').get_fdata()
                return pts

def save_acf_plot(lags, acf, success=True):
    """Save ACF plot, distinguishing between successful and failed cases."""
    # Define directories
    success_plot_dir = "ACF_Success_Plots"
    error_plot_dir = "ACF_Error_Plots"
    
    # Create directories if they don’t exist
    os.makedirs(success_plot_dir, exist_ok=True)
    os.makedirs(error_plot_dir, exist_ok=True)

    # Choose directory based on success/failure
    plot_dir = success_plot_dir if success else error_plot_dir
    status = "success" if success else "failed"

    # Generate filename with unique identifier
    plot_filename = f"{plot_dir}/acf_{status}_{np.random.randint(10000)}.png"

    # Plot the ACF
    plt.figure(figsize=(8, 5))
    plt.plot(lags, acf[:len(lags)], 'bo-' if success else 'ro-', markersize=4, label=f"ACF ({status})")
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Reference line
    plt.xlabel("Lag")
    plt.ylabel("ACF Value")
    plt.title(f"ACF ({status})")
    plt.legend()
    plt.grid(True)

    # Save and close the plot
    plt.savefig(plot_filename)
    plt.close()
    
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
    
# Compute the number of significant lags based on fitted exponential decay
def calc_lags(acf, nlags):
    lags = np.arange(nlags)

    try:
        # Fit an exponential decay model to the ACF values
        popt_lags, _ = curve_fit(exp_decay, lags, acf[:nlags], maxfev=200000)

        # Compute the fitted function values
        fitted_acf = exp_decay(lags, *popt_lags)

        # Compute the gradient (slope)
        slope = np.gradient(fitted_acf, lags)

        # Define a more adaptive threshold
        slope_threshold = 0.1 * np.max(np.abs(slope))  # Dynamic threshold

        # Find the first lag where the slope is below the threshold
        lag_threshold_indices = np.where(np.abs(slope) < slope_threshold)[0]

        if len(lag_threshold_indices) > 0:
            lag_threshold_index = lag_threshold_indices[0]
        else:
            # Fallback: Take the first large drop
            lag_threshold_index = np.argmax(np.abs(slope))  # Max rate of change

        # Ensure valid lag selection
        num_lags = lags[lag_threshold_index] if lag_threshold_index < len(lags) else nlags

        return num_lags

    except Exception as e:
        print(f"Curve fitting error: {e}")
        return nlags  # Safe fallback

# Separate function for calculating the number of significant lags based on fitted double exponential decay
def calc_lags2(acf, nlags):
    lags = np.arange(nlags) 

    try:
        # Fit the double exponential decay model to the ACF values
        popt_lags, _ = curve_fit(exp2_decay, lags, acf[:nlags], maxfev=200000)

        # Compute the fitted function values
        fitted_acf = exp2_decay(lags, *popt_lags)

        # Compute the gradient (slope)
        slope = np.gradient(fitted_acf, lags)

        # Define a more adaptive threshold
        slope_threshold = 0.1 * np.max(np.abs(slope))  # Dynamic threshold

        # Find the first lag where the slope is below the threshold
        lag_threshold_indices = np.where(np.abs(slope) < slope_threshold)[0]

        if len(lag_threshold_indices) > 0:
            lag_threshold_index = lag_threshold_indices[0]
        else:
            lag_threshold_index = np.argmax(np.abs(slope))  # Max rate of change fallback

        # Ensure valid lag selection
        num_lags = lags[lag_threshold_index] if lag_threshold_index < len(lags) else nlags

        return num_lags

    except Exception as e:
        print(f"Curve fitting error (exp2-decay): {e}")
        return nlags  # Safe fallback


def calc_lag_when_e_1(acf, nlags):
    lags = np.arange(nlags)

    # Compute 1/e threshold
    e_threshold = 1 / math.e  # ~0.3679

    # Find the first lag where ACF drops below 1/e
    below_e = acf <= e_threshold  # Boolean array
    indices = np.where(below_e)[0]  # Get indices where condition is met

    if len(indices) > 0:
        return lags[indices[0]]  # First occurrence of ACF ≤ 1/e
    else:
        return 1
        
def calc_tau(acf, nlags):
    num_lags = calc_lags(acf,nlags)
    lags = np.arange(num_lags)
    if num_lags < 3 or len(acf) < 3:
        return 1
    else:
        popt, _ = curve_fit(exp_decay, lags, acf[:num_lags], maxfev=200000)
        return popt[1]

def calc_tau2(acf, nlags):
    num_lags2 = calc_lags2(acf, nlags)
    lags = np.arange(num_lags2)
    if num_lags2 < 5 or len(acf) < 5:
        return 1, 1
    else:
        popt2, _ = curve_fit(exp2_decay, np.arange(num_lags2), acf[:num_lags2], maxfev=200000)
        return popt2[2], popt2[3]  # Extract b1, b2
    
def get_its_tau(pts):
    n_roi, n_vol = pts.shape
    acf = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_acf)(sig, n_vol) for sig in pts)
    acf = np.asarray(acf)
    tau = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_tau)(sig, n_vol) for sig in acf)
    tau = np.asarray(tau)
    return tau

def get_its_tau2(pts):
    n_roi, n_vol = pts.shape
    acf = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_acf)(sig, n_vol) for sig in pts)
    acf = np.asarray(acf)
    tau = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_tau2)(sig, n_vol) for sig in acf)
    tau = np.asarray(tau)
    return tau
    
def get_its_ptau(pts):
    n_roi, n_vol = pts.shape
    # Can only compute partial correlations for lags up to 50% of the sample size.
    pacf = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_pacf)(sig, n_vol // 2 - 1) for sig in pts)
    pacf = np.asarray(pacf)
    ptau = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_tau)(sig, n_vol // 2) for sig in pacf)
    ptau = np.asarray(ptau)
    return ptau

def get_its_AR1(pts):
    n_roi, n_vol = pts.shape
    acf = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_acf)(sig, 1) for sig in pts)
    acf = np.asarray(acf)
    return acf[:, 1]

def get_its_AR2(pts):
    n_roi, n_vol = pts.shape
    acf = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_acf)(sig, 2) for sig in pts)
    acf = np.asarray(acf)
    return acf[:, 2]

def get_its_AR3(pts):
    n_roi, n_vol = pts.shape
    acf = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_acf)(sig, 3) for sig in pts)
    acf = np.asarray(acf)
    return acf[:, 3]

def get_its_num_of_lags(pts):
    n_roi, n_vol = pts.shape
    acf = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_acf)(sig, n_vol) for sig in pts)
    acf = np.asarray(acf)
    lags = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_lags)(sig, n_vol) for sig in acf)
    lags = np.asarray(lags)
    return lags

def get_its_num_of_lags2(pts):
    n_roi, n_vol = pts.shape
    acf = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_acf)(sig, n_vol) for sig in pts)
    acf = np.asarray(acf)
    lags = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_lags2)(sig, n_vol) for sig in acf)
    lags = np.asarray(lags)
    return lags
    
def get_its_sum_ARs(pts):
    n_roi, n_vol = pts.shape
    acf = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_acf)(sig, n_vol) for sig in pts)
    acf = np.asarray(acf)
    # Initialize total_ARs array to store the sum for each ROI
    total_ARs = np.zeros(n_roi)
    # Iterate over each ROI to compute sum up to the first non-positive value in ACF
    for idx, node_acf in enumerate(acf):
        # Find the index of the first non-positive value in acf[node, :]
        first_non_positive_idx = np.where(acf[1:] <= 0)[0]
        sum_ARs = np.sum(acf[1:first_non_positive_idx[0] + 1]) if first_non_positive_idx.size > 0 else np.sum(acf[1:])
    return sum_ARs


def get_its_PAR1(pts):
    n_roi, n_vol = pts.shape
    pacf = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_pacf)(sig, 1) for sig in pts)
    pacf = np.asarray([result for result in pacf if result is not None])
    return pacf[:, 1]

def get_its_lag_e_1(pts):
    n_roi, n_vol = pts.shape
    acf = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_acf)(sig, n_vol) for sig in pts)
    acf = np.asarray(acf)
    lag_e_1 = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_lag_when_e_1)(sig, n_vol) for sig in acf)
    lag_e_1 = np.asarray(lag_e_1)
    return lag_e_1

def get_its_acf_at_half_lag(pts):
    n_roi, n_vol = pts.shape
    acf = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_acf)(sig, n_vol) for sig in pts)
    acf = np.asarray(acf)
    lag_values = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_lags)(sig, n_vol) for sig in acf)
    lag_values = np.asarray(lag_values)
    lags = np.arange(len(lag_values))
    lag_index = np.argmin(np.abs(lags - lag_values))
    acf_half_max = acf[:, lag_index] if lag_index < n_vol else 0
    return acf_half_max

def get_its_zero_cross(pts):
    n_roi, n_vol = pts.shape
    acf = Parallel(n_jobs=num_cores-2, verbose=0)(delayed(calc_acf)(sig, n_vol) for sig in pts)
    acf = np.asarray(acf)
    zero_crossings = np.where(acf <= 0)[0]  # Find indices where ACF ≤ 0
    return zero_crossings[0] if len(zero_crossings) > 0 else len(acf)  # Return full length if no crossing found

def get_its(pts, method):
    if method == 'tau':
        return get_its_tau(pts)
    elif method == 'tau2':
        return get_its_tau2(pts)
    elif method == 'AR1':
        return get_its_AR1(pts)
    elif method == 'AR2':
        return get_its_AR2(pts)
    elif method == 'AR3':
        return get_its_AR3(pts)
    elif method == 'ptau':
        return get_its_ptau(pts)
    elif method == 'PAR1':
        return get_its_PAR1(pts)
    elif method == 'sum_ARs':
        return get_its_sum_ARs(pts)
    elif method == 'lags':
        return get_its_num_of_lags(pts)
    elif method == 'lags2':
        return get_its_num_of_lags2(pts)
    elif method == 'lag_e_1_max':
        return  get_its_lag_e_1(pts)
    elif method == 'ACF_at_half_lag':
        return get_its_acf_at_half_lag(pts)
    elif method == 'zero_cross':
        return get_its_zero_cross(pts)
    elif method == 'half1_AR1':
        n_roi, n_vol = pts.shape 
        return get_its_AR1(pts[:, :n_vol // 2])
    elif method == 'half2_AR1':
        n_roi, n_vol = pts.shape 
        return get_its_AR1(pts[:, n_vol // 2:])
    elif method == 'half1_tau':
        n_roi, n_vol = pts.shape 
        return get_its_tau(pts[:, :n_vol // 2])
    elif method == 'half2_tau':
        n_roi, n_vol = pts.shape 
        return get_its_tau(pts[:, n_vol // 2:])
    elif method == 'half1_total_ARs':
        n_roi, n_vol = pts.shape 
        return get_its_sum_ARs(pts[:, :n_vol // 2])
    elif method == 'half2_total_ARs':
        n_roi, n_vol = pts.shape 
        return get_its_sum_ARs(pts[:, n_vol // 2:])


num_cores =  os.cpu_count()
dir_data = '/ocean/projects/bio220042p/shared/data/fmriprep_xcp_RBC/NKI_XCP_ZIP'
list_zip_file = np.sort(glob(f'{dir_data}/sub-A*.zip'))
dir_output = 'FinalOutput/INTCalcs'

task='rest'
session='BAS1'

list_method = [
    'AR1', 'AR2', 'AR3', 'tau', 'lags', 'PAR1', 'ptau',
    'sum_ARs', 'lag_e_1_max', 'ACF_at_half_lag','zero_cross'
#    'half1_AR1', 'half2_AR1','half1_tau', 'half2_tau',
#    'half1_total_ARs', 'half2_total_ARs'
]
list_TR = [645, 1400, 2500]

for method in list_method:
    for TR in list_TR:
        dir_save = f'{dir_output}/{task}-{session}-{TR}/{method}'
        os.makedirs(dir_save, exist_ok=True)

for zip_file in tqdm(list_zip_file):
    for TR in list_TR:     
        sub_name = os.path.basename(zip_file).split('.')[0]
        pts = get_pts(zip_file, task=task, session=session, TR=TR) 
        if not (pts is None):
            for method in list_method:
                its = get_its(pts.T, method=method) # Transposed to make (ROI x Time)            
                dir_save = f'{dir_output}/{task}-{session}-{TR}/{method}'
                np.save(f'{dir_save}/{sub_name}.npy', its)
    os.system(f'chmod -R 770 {dir_output}/*')