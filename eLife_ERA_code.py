import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import statsmodels.api as sm

# Function to compute ACF
def compute_acf(signal, max_lag):
    acf_values = sm.tsa.acf(signal, nlags=max_lag)
    return acf_values

def calc_zero_cross(acf):
    idx = np.where(acf < 0)[0]
    if idx.size == 0:
        return len(acf) - 1
    else:
        zero_crossings = idx[0]
        return zero_crossings
    
# Load the time series
TR = 645  # Time resolution in milliseconds
#time_series = nib.load(r"D:\CMI\Timescale\Scripts\sub-A00028266_ses-BAS1_task-rest_acq-645_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii").get_fdata()
time_series = nib.load(r"D:\CMI\Timescale\Scripts\sub-A00030980_ses-BAS1_task-rest_acq-645VARIANTMultibandAccelerationFactorPartialFourierTotalReadoutTime_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii").get_fdata()
#time_series = nib.load(r"D:\CMI\Timescale\Scripts\sub-A00052461_ses-BAS1_task-rest_acq-645_space-fsLR_atlas-Schaefer417_den-91k_bold.ptseries.nii").get_fdata()

colors = ['blue', 'green', 'gold', 'red']  # Corresponding colors
# Generate and plot the time series
plt.figure(figsize=(10, 5))

for i in range(4):
    ts = time_series[:, i*100-20]
    plt.plot(np.arange(ts.shape[0])*TR/1000/60, ts+i*25, color=colors[i], label=f'roi = {i*100-20}')

# Format plot
plt.title('Timeseries Examples')
plt.xlabel('Time (minutes)')
plt.ylabel('Amplitude + Offset')
plt.legend(title="ROI")
plt.grid(True)


# Compute ACF for different timescales
lags = np.arange(0, (ts.shape[0])*TR/1000, TR/1000)  # Lag in s

# Generate ACFs for each timeseries
plt.figure(figsize=(10, 5))

for i in range(4):
    ts = time_series[:, i*100]
    acf_values = compute_acf(ts, len(lags))  # Compute ACF
    sig_lags = calc_zero_cross(acf_values)
    plt.plot(lags[:sig_lags], acf_values[:sig_lags], color=colors[i], label=f'{sig_lags*TR/1000}', linewidth=2)

# Format the plot
plt.title('Autocorrelation Decay')
plt.xlabel('Lag Time (s)')
plt.ylabel('Autocorrelation')
plt.legend(title="Tau")
plt.ylim(-0.5, 1.05)
plt.grid(True)
plt.show()
