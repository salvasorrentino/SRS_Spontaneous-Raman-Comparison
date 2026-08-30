import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Script_for_Figures.utils_fig2 import (
    crop_to_overlap,
    interpolate_raman_to_srs_grid,
    set_publication_style,
)

from Script_for_Figures.utils_fig4 import (
    make_foreground_mask_for_similarity,
)

from Script_for_Figures.utils_fig5 import (
    compute_bandwise_spatial_metrics,
    summarize_bandwise_metrics,
    plot_figure6_bandwise_spatial_agreement,
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
    "paper_figures",
)

os.makedirs(
    output_dir,
    exist_ok=True,
)


# ============================================================
# Load data
# ============================================================

raman_cube = np.rot90(pd.read_pickle(path_img_confocal), k=2,)#[:-30, 75:275, :]
# raman_cube = pd.read_pickle(path_img_confocal)[15:-15, :, :]

raman_wn = pd.read_pickle(
    path_calib_confocal
)[300:]

srs_cube = pd.read_pickle(path_img_srs)#[15:-15, :, :]#[:-30, 75:275, :]

srs_wn = pd.read_pickle(
    path_calib_srs
)

sample = {
    "name": "breast",
    "tissue": "breast",
    "srs_cube": srs_cube,
    "srs_wn": srs_wn,
    "raman_cube": raman_cube,
    "raman_wn": raman_wn,
}


# ============================================================
# Common spectral range and interpolation
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

print(
    "SRS common shape:",
    srs_cube_common.shape,
)

print(
    "Raman interp shape:",
    raman_interp.shape,
)

print(
    "Common Raman shift range:",
    srs_wn_common.min(),
    srs_wn_common.max(),
)


# ============================================================
# Foreground mask
# ============================================================

foreground_mask = make_foreground_mask_for_similarity(
    srs_cube_common,
    raman_cube_interp=raman_interp,
    projection="mean",
    threshold_method="percentile",
    threshold_percentile=1,
    min_size=40,
    opening_radius=0,
)


# ============================================================
# Compute band-wise spatial metrics
# ============================================================

bands_to_evaluate = srs_wn_common

band_metrics_df = compute_bandwise_spatial_metrics(
    srs_cube=srs_cube_common,
    raman_cube_interp=raman_interp,
    wn=srs_wn_common,
    mask=foreground_mask,
    bands=bands_to_evaluate,
    band_width=12,
    image_normalization="robust_minmax",
    pmin=1,
    pmax=99.8,
    min_valid_pixels=100,
)

band_metrics_df.insert(
    0,
    "sample",
    sample["name"],
)

print("\nBand-wise metrics:")
print(band_metrics_df.head())


# ============================================================
# New analysis: variability across Raman bands
# ============================================================

band_summary_df = summarize_bandwise_metrics(
    band_metrics_df=band_metrics_df,
    sample_name=sample["name"],
    metrics=("bicor", "cosine"),
)

print("\nBand-wise distribution summary:")
print(band_summary_df.to_string(index=False))


# ============================================================
# Export results
# ============================================================

band_metrics_df.to_csv(
    os.path.join(
        output_dir,
        "Figure_5_bandwise_spatial_metrics.csv",
    ),
    index=False,
)

band_summary_df.to_csv(
    os.path.join(
        output_dir,
        "Figure_5_bandwise_distribution_summary.csv",
    ),
    index=False,
)


# ============================================================
# Generate Figure 6
# ============================================================

selected_bands = [
    785,
    1003,
    1445,
    1660,
]

fov_um = 600
pixel_size_um = 2

fig = plot_figure6_bandwise_spatial_agreement(
    srs_cube=srs_cube_common,
    raman_cube_interp=raman_interp,
    wn=srs_wn_common,
    band_metrics_df=band_metrics_df,
    mask=foreground_mask,
    selected_bands=selected_bands,
    band_width=12,
    metric1_to_plot="bicor",
    metric2_to_plot="cosine",
    sample_name=sample["name"],
    image_normalization="robust_minmax",
    cmap_images="afmhot",
    cmap_diff="magma",
    pmin=10,
    pmax=99.8,
    metric1_ylim=(-0.1, 0.5),
    metric2_ylim=(0.35, 1),
    add_colorbar=True,
    add_scalebar_flag=True,
    scale_bar_um=100,
    pixel_size_um=pixel_size_um,
    fov_um=fov_um,
    # savepath=os.path.join(
    #     output_dir,
    #     "Figure_5_bandwise_spatial_agreement_bicor_cosine.png",
    # ),
)

plt.show()