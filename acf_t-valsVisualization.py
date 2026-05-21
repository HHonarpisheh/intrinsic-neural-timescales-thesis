import os
import numpy as np
import nibabel as nib
import pandas as pd
import re
from scipy.stats import ttest_ind
from nilearn import plotting, datasets, surface

# Define paths
nifti_folder = r"D:\CMI\ABIDE\Outputs\cpac\nofilt_noglobal\reho"
subject_info_path = r"D:\CMI\ABIDE\Phenotypic_V1_0b_preprocessed1.csv"
output_folder = r"D:\CMI\Timescale\Outputs"

# Load subject metadata
subject_info = pd.read_csv(subject_info_path)

# Convert SUB_ID to string and remove leading zeros for consistency
subject_info["SUB_ID"] = subject_info["SUB_ID"].astype(str).str.lstrip("0")

# Create ASD and TD subject lists
asd_subjects = set(subject_info[subject_info["DX_GROUP"] == 2]["SUB_ID"].tolist())
td_subjects = set(subject_info[subject_info["DX_GROUP"] == 1]["SUB_ID"].tolist())

print(f"Total ASD Subjects: {len(asd_subjects)}")
print(f"Total TD Subjects: {len(td_subjects)}")

# Function to extract SUB_ID from filename
def extract_sub_id(filename):
    """Extracts the numeric SUB_ID appearing before '_reho' in the filename."""
    match = re.search(r"(\d+)_reho", filename)
    return re.sub(r"^0+", "", match.group(1)) if match else None

# Separate ASD and TD file paths based on extracted SUB_ID
asd_files = []
td_files = []

for f in os.listdir(nifti_folder):
    if f.endswith(".nii.gz"):
        sub_id = extract_sub_id(f)  # Extract SUB_ID
        full_path = os.path.join(nifti_folder, f)

        if sub_id in asd_subjects:
            asd_files.append(full_path)
        elif sub_id in td_subjects:
            td_files.append(full_path)

print(f"Total ASD NIfTI Files: {len(asd_files)}")
print(f"Total TD NIfTI Files: {len(td_files)}")

# Ensure there are matching files before proceeding
if len(asd_files) == 0 or len(td_files) == 0:
    raise ValueError("Error: No matching NIfTI files found for ASD or TD subjects. Check SUB_ID format.")

# Load NIfTI images safely
def load_nifti_safe(file_list):
    """Load valid NIfTI files and return an array. Skips corrupted/missing files."""
    valid_imgs = []
    for f in file_list:
        try:
            img = nib.load(f).get_fdata()
            valid_imgs.append(img)
        except Exception as e:
            print(f"Skipping file {f} due to error: {e}")
    return np.array(valid_imgs)

# Load ASD and TD NIfTI images
asd_data = load_nifti_safe(asd_files)  # Shape: (num_ASD, X, Y, Z)
td_data = load_nifti_safe(td_files)  # Shape: (num_TD, X, Y, Z)

# Ensure data arrays are not empty
if asd_data.shape[0] == 0 or td_data.shape[0] == 0:
    raise ValueError("Error: One of the groups has no valid NIfTI images.")

print(f"ASD Data Shape: {asd_data.shape}")
print(f"TD Data Shape: {td_data.shape}")

# Perform voxel-wise independent t-test (TD vs. ASD)
t_values, p_values = ttest_ind(td_data, asd_data, axis=0, equal_var=False)

# Load affine from a sample NIfTI file
affine = nib.load(asd_files[0]).affine

# Save TD > ASD (Shorter Timescales in ASD)
t_stat_td_asd = np.where(p_values < 0.05, t_values, 0)  # Apply significance threshold
t_stat_nifti_td_asd = nib.Nifti1Image(t_stat_td_asd, affine)
td_asd_output = os.path.join(output_folder, "TD_vs_ASD.nii.gz")
nib.save(t_stat_nifti_td_asd, td_asd_output)

# Save ASD > TD (Longer Timescale in ASD)
t_stat_asd_td = np.where(p_values < 0.05, -t_values, 0)  # Flip sign for visualization
t_stat_nifti_asd_td = nib.Nifti1Image(t_stat_asd_td, affine)
asd_td_output = os.path.join(output_folder, "ASD_vs_TD.nii.gz")
nib.save(t_stat_nifti_asd_td, asd_td_output)

# Load fsaverage template for surface-based visualization
fsaverage = datasets.fetch_surf_fsaverage()

# Project statistical maps onto the brain surface (voxel -> surface mapping)
td_asd_surf_left = surface.vol_to_surf(t_stat_nifti_td_asd, fsaverage.pial_left)
td_asd_surf_right = surface.vol_to_surf(t_stat_nifti_td_asd, fsaverage.pial_right)

asd_td_surf_left = surface.vol_to_surf(t_stat_nifti_asd_td, fsaverage.pial_left)
asd_td_surf_right = surface.vol_to_surf(t_stat_nifti_asd_td, fsaverage.pial_right)

# Visualize TD > ASD (Shorter Timescales in ASD)
plotting.plot_surf_stat_map(fsaverage.infl_left, td_asd_surf_left, hemi="left", threshold=1,
                            view="lateral", colorbar=True, cmap="rainbow", title="TD > ASD (Left Hemisphere)")
plotting.plot_surf_stat_map(fsaverage.infl_right, td_asd_surf_right, hemi="right", threshold=1,
                            view="lateral", colorbar=True, cmap="rainbow", title="TD > ASD (Right Hemisphere)")

# Visualize ASD > TD (Longer Timescale in ASD)
plotting.plot_surf_stat_map(fsaverage.infl_left, asd_td_surf_left, hemi="left", threshold=1,
                            view="lateral", colorbar=True, cmap="rainbow", title="ASD > TD (Left Hemisphere)")
plotting.plot_surf_stat_map(fsaverage.infl_right, asd_td_surf_right, hemi="right", threshold=1,
                            view="lateral", colorbar=True, cmap="rainbow", title="ASD > TD (Right Hemisphere)")

#plotting.show()

print(f"Analysis complete. Results saved as:\n - {td_asd_output}\n - {asd_td_output}")

# Glass Brain Visualization for TD > ASD (Shorter Timescales in ASD)
plotting.plot_glass_brain(t_stat_nifti_td_asd, 
                          title="TD > ASD (Shorter Timescales in ASD)",
                          display_mode='lyrz',  # Shows Left, Right, Top, Front views
                          threshold=3, 
                          colorbar=True, 
                          cmap="rainbow")

# Glass Brain Visualization for ASD > TD (Longer Timescales in ASD)
plotting.plot_glass_brain(t_stat_nifti_asd_td, 
                          title="ASD > TD (Longer Timescales in ASD)",
                          display_mode='lyrz',  
                          threshold=3, 
                          colorbar=True, 
                          cmap="rainbow")

#plotting.show()


# Define output paths
acf_td_output_path = os.path.join(output_folder, "acf_timescale_map_TD.nii.gz")
acf_asd_output_path = os.path.join(output_folder, "acf_timescale_map_ASD.nii.gz")

# ACF Function using FFT
def compute_acf(series, n_lags=40, time_resolution=2.0, n_stds=1.96):
    """Compute the Autocorrelation Function (ACF) and Signal Timescale (STS) for a time series."""
    n = len(series)
    if n == 0:  # Avoid zero-length arrays
        return np.zeros(n_lags + 1), 0

    n_fft = 2 ** (np.ceil(np.log2(n)) + 1).astype(int)
    F = np.fft.fft(series - np.mean(series), n=n_fft)
    F = F * np.conj(F)
    acf = np.fft.ifft(F).real[:n_lags + 1]  # Retain only non-negative lags
    acf /= acf[0]  # Normalize

    # Compute confidence bounds
    Q = min(n_lags, n - 1)
    sigma_q = np.sqrt((1 + 2 * np.sum(acf[1:Q + 1] ** 2)) / n)
    bounds = sigma_q * np.array([n_stds, -n_stds])

    # Compute signal timescale (STS)
    positive_acf_sum = np.sum(acf[acf > 0])
    sts = positive_acf_sum * time_resolution

    return acf, sts

# Compute ACF-based timescale map for each voxel (TD & ASD separately)
shape = td_data.shape[1:]  # Get brain shape (3D)
acf_map_td = np.zeros(shape)
acf_map_asd = np.zeros(shape)

for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            voxel_time_series_td = td_data[:, i, j, k]
            voxel_time_series_asd = asd_data[:, i, j, k]
            
            if np.any(voxel_time_series_td):  # Avoid zero voxels
                _, sts_td = compute_acf(voxel_time_series_td, n_lags=40, time_resolution=2.0)
                acf_map_td[i, j, k] = sts_td

            if np.any(voxel_time_series_asd):  # Avoid zero voxels
                _, sts_asd = compute_acf(voxel_time_series_asd, n_lags=40, time_resolution=2.0)
                acf_map_asd[i, j, k] = sts_asd

# Save ACF-based timescale maps for TD and ASD
affine = nib.load(td_files[0]).affine  # Use affine from the first TD subject

acf_nifti_td = nib.Nifti1Image(acf_map_td, affine)
acf_nifti_asd = nib.Nifti1Image(acf_map_asd, affine)

nib.save(acf_nifti_td, acf_td_output_path)
nib.save(acf_nifti_asd, acf_asd_output_path)

print(f"ACF timescale map (TD) saved to: {acf_td_output_path}")
print(f"ACF timescale map (ASD) saved to: {acf_asd_output_path}")

# ---- VISUALIZATION ----

# Load fsaverage template for surface-based visualization
fsaverage = datasets.fetch_surf_fsaverage()

# Project ACF-based timescale maps onto the brain surface
acf_surf_td_left = surface.vol_to_surf(acf_nifti_td, fsaverage.pial_left)
acf_surf_td_right = surface.vol_to_surf(acf_nifti_td, fsaverage.pial_right)

acf_surf_asd_left = surface.vol_to_surf(acf_nifti_asd, fsaverage.pial_left)
acf_surf_asd_right = surface.vol_to_surf(acf_nifti_asd, fsaverage.pial_right)


# ---- Visualizing ACF Maps on Cortical Surface ----

# TD Group ACF Map
plotting.plot_surf_stat_map(fsaverage.infl_left, acf_surf_td_left, hemi="left",
                            view="lateral", colorbar=True, cmap="rainbow",
                            title="ACF Timescale Map (TD - Left Hemisphere)")
plotting.plot_surf_stat_map(fsaverage.infl_right, acf_surf_td_right, hemi="right",
                            view="lateral", colorbar=True, cmap="rainbow",
                            title="ACF Timescale Map (TD - Right Hemisphere)")

# ASD Group ACF Map
plotting.plot_surf_stat_map(fsaverage.infl_left, acf_surf_asd_left, hemi="left",
                            view="lateral", colorbar=True, cmap="rainbow",
                            title="ACF Timescale Map (ASD - Left Hemisphere)")
plotting.plot_surf_stat_map(fsaverage.infl_right, acf_surf_asd_right, hemi="right",
                            view="lateral", colorbar=True, cmap="rainbow",
                            title="ACF Timescale Map (ASD - Right Hemisphere)")

plotting.show()