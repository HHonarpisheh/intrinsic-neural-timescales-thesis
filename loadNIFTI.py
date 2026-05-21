import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, datasets, surface

output_folder = r"D:\CMI\Timescale\Outputs"

# Load fsaverage template for surface-based visualization
fsaverage = datasets.fetch_surf_fsaverage()

# Define file paths
td_acf_path = os.path.join(output_folder, "acf_timescale_map_TD.nii.gz")
asd_acf_path = os.path.join(output_folder, "acf_timescale_map_ASD.nii.gz")
td_vs_asd_path = os.path.join(output_folder, "TD_vs_ASD.nii.gz")
asd_vs_td_path = os.path.join(output_folder, "ASD_vs_TD.nii.gz")

# Load NIfTI images
td_acf_img = nib.load(td_acf_path)
asd_acf_img = nib.load(asd_acf_path)
td_vs_asd_img = nib.load(td_vs_asd_path)
asd_vs_td_img = nib.load(asd_vs_td_path)

# Project statistical maps onto the brain surface
td_surf_left = surface.vol_to_surf(td_acf_img, fsaverage.pial_left)
td_surf_right = surface.vol_to_surf(td_acf_img, fsaverage.pial_right)

asd_surf_left = surface.vol_to_surf(asd_acf_img, fsaverage.pial_left)
asd_surf_right = surface.vol_to_surf(asd_acf_img, fsaverage.pial_right)

td_vs_asd_surf_left = surface.vol_to_surf(td_vs_asd_img, fsaverage.pial_left)
td_vs_asd_surf_right = surface.vol_to_surf(td_vs_asd_img, fsaverage.pial_right)

asd_vs_td_surf_left = surface.vol_to_surf(asd_vs_td_img, fsaverage.pial_left)
asd_vs_td_surf_right = surface.vol_to_surf(asd_vs_td_img, fsaverage.pial_right)

# Define colormap
custom_cmap = "rainbow"

# --- (a) TD and ASD ACF Surface Maps ---
plotting.plot_surf_stat_map(fsaverage.infl_left, td_surf_left, hemi="left",
                            view="lateral", colorbar=True, cmap=custom_cmap,
                            title="TD Intrinsic Timescale")
plotting.plot_surf_stat_map(fsaverage.infl_left, asd_surf_left, hemi="left",
                            view="lateral", colorbar=True, cmap=custom_cmap,
                            title="ASD Intrinsic Timescale")

# --- (b, c) Group Comparisons TD > ASD and ASD > TD ---

plotting.plot_surf_stat_map(fsaverage.infl_left, td_vs_asd_surf_left, hemi="left",
                            view="lateral", colorbar=True, cmap="hot",
                            title="TD > ASD (Intrinsic Timescale)")
plotting.plot_surf_stat_map(fsaverage.infl_left, asd_vs_td_surf_left, hemi="left",
                            view="lateral", colorbar=True, cmap="hot",
                            title="ASD > TD (Intrinsic Timescale)")
plt.show()

# --- (d, e) Slice-Based Visualization ---
plotting.plot_stat_map(td_vs_asd_img, title="TD > ASD (Voxel-wise)", threshold=3, 
                       display_mode="z", cut_coords=[-6, 32, 44], cmap="hot")
plotting.plot_stat_map(asd_vs_td_img, title="ASD > TD (Voxel-wise)", threshold=3, 
                       display_mode="z", cut_coords=[12], cmap="hot")
plotting.show()
