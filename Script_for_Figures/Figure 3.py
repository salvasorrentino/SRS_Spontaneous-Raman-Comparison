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
    plot_figure3_mean_and_roi_spectra_stacked,
    export_mean_and_roi_spectra_to_csv,
)

set_publication_style()


# ============================================================
# Paths
# ============================================================

path_fold = (
    r"C:\Users\salva\OneDrive - Massachusetts Institute of Technology\Coregistration "
    r"SRS-confocal\data\20260518_colon"
)

path_img_srs = os.path.join(
    path_fold,
    "Project_srs_colon_hyperspectral_processed_coregistered_w_confocal_no_correction.pickle"
)

path_calib_srs = os.path.join(
    path_fold,
    "srs_Raman_shift.pickle"
)

path_calib_confocal = os.path.join(
    path_fold,
    "arr_calibration202601.pickle"
)

path_img_confocal = os.path.join(
    path_fold,
    "colon_processed_confocal_fill_over_srs.pickle"
)

output_dir = os.path.join(path_fold, "paper_figures")
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# Load data
# ============================================================

raman_cube = pd.read_pickle(path_img_confocal)[10:-10, :, :]
# raman_cube = np.rot90(pd.read_pickle(path_img_confocal), k=2)[:, 75:275, :]
raman_wn = pd.read_pickle(path_calib_confocal)[300:]

srs_cube = pd.read_pickle(path_img_srs)[10:-10, :, :]#[:, 75:275, :]
srs_wn = pd.read_pickle(path_calib_srs)

sample = {
    "name": "colon",
    "tissue": "colon",
    "srs_cube": srs_cube,
    "srs_wn": srs_wn,
    "raman_cube": raman_cube,
    "raman_wn": raman_wn,
}


# ============================================================
# Put SRS and Raman on common spectral grid
# ============================================================

srs_cube_common, srs_wn_common, raman_cube_common, raman_wn_common = crop_to_overlap(
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
# Manual ROI rectangles
# Format: (y0, y1, x0, x1)
# Change these based on the image.
# ============================================================

roi_rects = {
    "ROI 1": (50, 100, 50, 100),
    "ROI 2": (100, 150, 120, 170),
    "ROI 3": (130, 180, 280, 330),
}

# Se non vuoi ROI, metti:
# roi_rects = None


# ============================================================
# Physical calibration
# ============================================================

# Use the real value if known.
# If your coregistered image is 600 x 600 um:
fov_um = 660

# Alternative:
# pixel_size_um = fov_um / srs_cube_common.shape[1]
pixel_size_um = 2


# ============================================================
# Generate Figure 3
# ============================================================

fig = plot_figure3_mean_and_roi_spectra_stacked(
    srs_cube=srs_cube_common,
    raman_cube_interp=raman_interp,
    wn=srs_wn_common,
    mask=None,
    roi_rects=roi_rects,
    overview_band=1445,
    overview_width=12,
    overview_source="raman",
    sample_name=sample["name"],
    spectrum_normalization="minmax",
    curve_normalization="minmax",
    statistic="mean",
    cmap="inferno",
    pmin=10,
    pmax=99.5,
    add_colorbar=False,
    add_scalebar_flag=True,
    scale_bar_um=100,
    fov_um=600,
    roi_colors=["tab:blue", "tab:green", "tab:purple"],
    srs_color="crimson",
    # savepath=None
    savepath=os.path.join(output_dir, "Figure_3_mean_and_roi_spectra_stacked.png"),
)

plt.show()


# ============================================================
# Export spectra used in the figure
# ============================================================

spectra_df = export_mean_and_roi_spectra_to_csv(
    srs_cube=srs_cube_common,
    raman_cube_interp=raman_interp,
    wn=srs_wn_common,
    output_csv=os.path.join(output_dir, "Figure_3_mean_and_roi_spectra.csv"),
    mask=foreground_mask,
    roi_rects=roi_rects,
    spectrum_normalization="minmax",
    statistic="mean",
)

print(spectra_df.head())
