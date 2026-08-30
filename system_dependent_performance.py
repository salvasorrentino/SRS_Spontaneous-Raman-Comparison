import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Script_for_Figures.utils_fig2 import (
    crop_to_overlap,
    interpolate_raman_to_srs_grid,
    set_publication_style,
)

from Script_for_Figures.utils_fig3 import (
    make_foreground_mask_from_cubes,
)

from Script_for_Figures.system_dependent_performance_utils import (
    compute_spectral_continuity,
    plot_spectral_continuity,
)

set_publication_style()


# ============================================================
# Paths
# ============================================================

path_fold = (
    r"C:\Users\salva\OneDrive - Massachusetts Institute of Technology\Coregistration "
    r"SRS-confocal\data\skin_small_hyperspectral"
)

path_img_srs = os.path.join(
    path_fold,
    "Project_srs_skin_small_hyper_hyperspectral_processed_coregistered_w_confocal.pickle"
)

path_calib_srs = os.path.join(
    path_fold,
    "srs_Raman_shift.pickle"
)

path_calib_confocal = os.path.join(
    path_fold,
    "arr_calibration_confocal.pickle"
)

path_img_confocal = os.path.join(
    path_fold,
    "arr_confocal_raman_processed.pickle"
)

output_dir = os.path.join(
    path_fold,
    "spectral_continuity"
)

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# Load data
# ============================================================

# raman_cube = pd.read_pickle(path_img_confocal)[10:-10, :, :]

raman_cube = np.rot90(pd.read_pickle(path_img_confocal), k=2)#[:, 75:275, :]

raman_wn = pd.read_pickle(
    path_calib_confocal
)[300:]

srs_cube = pd.read_pickle(path_img_srs)#[:, 75:275, :]#[10:-10, :, :]

srs_wn = pd.read_pickle(
    path_calib_srs
)

sample = {
    "name": "skin",
    "srs_cube": srs_cube,
    "srs_wn": srs_wn,
    "raman_cube": raman_cube,
    "raman_wn": raman_wn,
}


# ============================================================
# Put SRS and Raman on common spectral grid
# ============================================================

(
    srs_cube_common,
    srs_wn_common,
    raman_cube_common,
    raman_wn_common,
) = crop_to_overlap(
    sample["srs_cube"],
    sample["srs_wn"],
    sample["raman_cube"],
    sample["raman_wn"],
)

raman_interp = interpolate_raman_to_srs_grid(
    raman_cube_common,
    raman_wn_common,
    srs_wn_common,
)


plt.figure()
vmin2 = np.percentile(sample["raman_cube"][:, :, 690], 25)
vmax2 = np.percentile(sample["raman_cube"][:, :, 690], 99.5)
plt.imshow(sample["raman_cube"][:, :, 690], vmin=vmin2, vmax=vmax2)
plt.show()
plt.figure()
vmin2 = np.percentile(sample["srs_cube"][:, :, 80], 25)
vmax2 = np.percentile(sample["srs_cube"][:, :, 80], 99.5)
plt.imshow(sample["srs_cube"][:, :, 80], vmin=vmin2, vmax=vmax2)
plt.show()

# ============================================================
# Foreground mask
# ============================================================

foreground_mask = make_foreground_mask_from_cubes(
    srs_cube_common,
    raman_cube_interp=raman_interp,
    method="mean",
    threshold_method="percentile",
    min_size=500,
    opening_radius=2,
)


# ============================================================
# Spectral-continuity analysis
# ============================================================

band_results, summary = compute_spectral_continuity(
    srs_cube=srs_cube_common,
    raman_cube_interp=raman_interp,
    wn=srs_wn_common,
    mask=foreground_mask,
    sample_name=sample["name"],
    percentile=90,
)


# ============================================================
# Export results
# ============================================================

band_results.to_csv(
    os.path.join(
        output_dir,
        f"{sample['name']}_spectral_continuity_by_band.csv",
    ),
    index=False,
)

summary.to_csv(
    os.path.join(
        output_dir,
        f"{sample['name']}_spectral_continuity_summary.csv",
    ),
    index=False,
)


# ============================================================
# Plot results
# ============================================================

fig = plot_spectral_continuity(
    band_results=band_results,
    summary=summary,
    sample_name=sample["name"],
    percentile=90,
    savepath=os.path.join(
        output_dir,
        f"{sample['name']}_spectral_continuity.png",
    ),
)

plt.show()


# ============================================================
# Print summary
# ============================================================

print(summary.T)