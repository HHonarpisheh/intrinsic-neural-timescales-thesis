import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_ind

# Parameters
num_regions = 50  # Number of simulated brain regions
num_subjects = 100  # Number of subjects per group
timepoints = 300  # Number of timepoints in the simulated fMRI data

# Simulate fMRI time series
def generate_fmri_data(num_subjects, num_regions, timepoints, scale=1.0):
    data = np.random.randn(num_subjects, num_regions, timepoints) * scale
    # Add autocorrelation to mimic fMRI signals
    for subj in range(num_subjects):
        for roi in range(num_regions):
            for t in range(1, timepoints):
                data[subj, roi, t] += 0.5 * data[subj, roi, t - 1]
    return data

# Compute intrinsic timescale (sum of positive autocorrelations)
def compute_intrinsic_timescale(data):
    timescales = []
    for subj_data in data:
        subj_timescales = []
        for roi_data in subj_data:
            acf = np.correlate(roi_data, roi_data, mode='full') / len(roi_data)
            acf = acf[len(roi_data) - 1:]  # Keep only positive lags
            positive_acf = acf[acf > 0]
            timescale = np.sum(positive_acf)
            subj_timescales.append(timescale)
        timescales.append(subj_timescales)
    return np.array(timescales)

# Simulate data for two groups
control_data = generate_fmri_data(num_subjects, num_regions, timepoints)
asd_data = generate_fmri_data(num_subjects, num_regions, timepoints, scale=1.2)

# Compute intrinsic timescales
control_timescales = compute_intrinsic_timescale(control_data)
asd_timescales = compute_intrinsic_timescale(asd_data)

# Group comparison
group_diff = ttest_ind(control_timescales, asd_timescales, axis=0)

# Simulate symptom severity and correlate with timescale
severity = np.random.uniform(0, 10, size=num_subjects)
correlations = [spearmanr(asd_timescales[:, roi], severity) for roi in range(num_regions)]

# Visualization
plt.figure(figsize=(12, 6))
for roi in range(num_regions):
    plt.bar(roi, group_diff.statistic[roi], alpha=0.7, label=f'Region {roi + 1}')
plt.axhline(y=0, color='k', linestyle='--')
plt.title('Group Differences in Timescale')
plt.xlabel('Brain Regions')
plt.ylabel('t-statistic')
plt.show()

plt.figure(figsize=(12, 6))
for roi, (corr, pval) in enumerate(correlations):
    plt.bar(roi, corr, alpha=0.7, label=f'Region {roi + 1}')
plt.axhline(y=0, color='k', linestyle='--')
plt.title('Correlation Between Timescale and Severity')
plt.xlabel('Brain Regions')
plt.ylabel('Spearman Correlation')
plt.show()
