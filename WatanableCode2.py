import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting, datasets, surface
from nilearn.image import load_img
import nibabel as nib

# Load a sample brain surface (fsaverage template)
fsaverage = datasets.fetch_surf_fsaverage()

# Generate synthetic intrinsic timescale data (simulated)
n_vertices = 10242  # Number of vertices in fsaverage5
intrinsic_timescale = np.random.uniform(1, 2.5, size=n_vertices)  # Random values for demonstration

# Generate synthetic statistical t-values (TD > ASD and ASD > TD)
t_values_TD_ASD = np.random.uniform(1, 5, size=n_vertices) * (np.random.rand(n_vertices) > 0.95)
t_values_ASD_TD = np.random.uniform(1, 4.5, size=n_vertices) * (np.random.rand(n_vertices) > 0.95)

# Plot intrinsic timescale maps (TD & ASD)
fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

plotting.plot_surf_stat_map(fsaverage.infl_left, intrinsic_timescale, hemi='left',
                            view='lateral', colorbar=True, cmap='hot', axes=axes[0], title="TD Intrinsic Timescale")

plotting.plot_surf_stat_map(fsaverage.infl_right, intrinsic_timescale, hemi='right',
                            view='lateral', colorbar=True, cmap='hot', axes=axes[1], title="ASD Intrinsic Timescale")

plt.show()

# Plot statistical maps (TD > ASD)
plotting.plot_surf_stat_map(fsaverage.infl_left, t_values_TD_ASD, hemi='left',
                            view='lateral', colorbar=True, cmap='hot', title="TD > ASD")

plotting.plot_surf_stat_map(fsaverage.infl_right, t_values_TD_ASD, hemi='right',
                            view='lateral', colorbar=True, cmap='hot', title="TD > ASD")

plt.show()

# Plot statistical maps (ASD > TD)
plotting.plot_surf_stat_map(fsaverage.infl_left, t_values_ASD_TD, hemi='left',
                            view='lateral', colorbar=True, cmap='hot', title="ASD > TD")

plotting.plot_surf_stat_map(fsaverage.infl_right, t_values_ASD_TD, hemi='right',
                            view='lateral', colorbar=True, cmap='hot', title="ASD > TD")

plt.show()
