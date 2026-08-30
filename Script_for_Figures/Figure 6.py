import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from Script_for_Figures.utils_fig2 import (
    crop_to_overlap,
    interpolate_raman_to_srs_grid,
    set_publication_style,
)

from Script_for_Figures.utils_fig4 import (make_foreground_mask_for_similarity, compute_pixelwise_similarity_maps)
from Script_for_Figures.utils_fig6 import (select_random_pixels_from_metric_quantiles, replace_selected_pixel,
                                           extract_selected_pixel_spectra, plot_figure6_selected_pixel_spectra)


set_publication_style()


# ============================================================
# USER SETTINGS
# ============================================================

# Metric used to define top/bottom pixels in Figure 6.
#
# Allowed:
#     "cosine"
#     "pearson"
#     "spearman"
#     "bicor"

selection_metric = "pearson"

# Quantile definition
top_quantile = 75
bottom_quantile = 25

# Number of pixels selected in each group
n_points_per_group = 1


# Normalization used for pixel-wise comparison
spectrum_normalization = "minmax"


# ============================================================
# Paths
# ============================================================

path_fold = (
    r"C:\Users\salva\OneDrive - Massachusetts Institute of Technology\Coregistration"
    r" SRS-confocal\data\20260511_skin")

path_img_srs = os.path.join(path_fold, "Project_srs_skin_hyperspectral_processed_coregistered_w_confocal_no_correction.pickle")

path_calib_srs = os.path.join(
    path_fold,
    "srs_Raman_shift.pickle",
)

path_calib_confocal = os.path.join(
    path_fold,
    "arr_calibration202601.pickle",
)

path_img_confocal = os.path.join(
    path_fold,
    "skin_processed_confocal.pickle",
)

output_dir = os.path.join(
    path_fold,
    "paper_figures",
)


os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Load data
# ============================================================

# Previous alternative crop:
#
raman_cube = pd.read_pickle(path_img_confocal)[15:-15, :-15, :]
# raman_cube = np.rot90(pd.read_pickle(path_img_confocal), k=2)[15:-15, :-15, :]#[:-70, 75:275, :]

raman_wn = pd.read_pickle(path_calib_confocal)[300:]

srs_cube = pd.read_pickle(path_img_srs)[15:-15, :-15, :]#[:-70, 75:275, :]

srs_wn = pd.read_pickle(path_calib_srs)

# ============================================================
# Sample information
# ============================================================

sample = {
    "name": "breast",
    "tissue": "breast",
    "srs_cube": srs_cube,
    "srs_wn": srs_wn,
    "raman_cube": raman_cube,
    "raman_wn": raman_wn,
}


# ============================================================
# Common spectral range
# ============================================================

srs_cube_common, srs_wn_common, raman_cube_common, raman_wn_common = (
    crop_to_overlap(
        sample["srs_cube"],
        sample["srs_wn"],
        sample["raman_cube"],
        sample["raman_wn"],
))

# ============================================================
# Interpolate spontaneous Raman onto SRS spectral grid
# ============================================================

raman_interp = interpolate_raman_to_srs_grid(
    raman_cube_common,
    raman_wn_common,
    srs_wn_common,
)


print("SRS common shape:", srs_cube_common.shape)
print("Raman interp shape:", raman_interp.shape)

print("Common Raman shift range:", srs_wn_common.min(), srs_wn_common.max())


# ============================================================
# Quick visual coregistration check
# ============================================================

plt.figure()

vmin2 = np.percentile(sample["raman_cube"][:, :, 690], 25)
vmax2 = np.percentile(sample["raman_cube"][:, :, 690],99.5)
plt.imshow(sample["raman_cube"][:, :, 690], vmin=vmin2, vmax=vmax2)

plt.title("Confocal Raman registration check")
plt.show()

plt.figure()
vmin2 = np.percentile(sample["srs_cube"][:, :, 116],25)
vmax2 = np.percentile(sample["srs_cube"][:, :, 116],99.5)
plt.imshow(sample["srs_cube"][:, :, 116], vmin=vmin2, vmax=vmax2)

plt.title("SRS registration check")
plt.show()


# ============================================================
# Foreground mask
# ============================================================

foreground_mask = make_foreground_mask_for_similarity(
    srs_cube_common,
    raman_cube_interp=raman_interp,
    projection="mean",
    threshold_method="percentile",
    threshold_percentile=20,
    min_size=50,
    opening_radius=0,
)


print("Foreground pixels:", np.sum(foreground_mask))

# ============================================================
# Pixel-wise similarity maps
# ============================================================

metric_maps, summary_df, normalized_cubes = compute_pixelwise_similarity_maps(
    srs_cube=srs_cube_common,
    raman_cube_interp=raman_interp,
    mask=foreground_mask,
    spectrum_normalization=spectrum_normalization,
    clip_negative=False,
    min_signal_percentile=5,
    compute_pearson=True,
    compute_spearman=True,
    compute_bicor=True,
    bicor_c=9.0,
)


# ============================================================
# Save similarity summary
# ============================================================

summary_df.insert(0, "sample", sample["name"])

summary_df.to_csv(os.path.join(output_dir, "Figure_6_pixelwise_similarity_summary.csv"), index=False,)

print("\nPixel-wise similarity summary:")
print(summary_df)
print("\nAvailable metric maps:")
print(list(metric_maps.keys()))


# ============================================================
# Random selection from chosen metric quantiles
# ============================================================

random_state = 1044

selected_points_df = (
    select_random_pixels_from_metric_quantiles(
        metric_maps=metric_maps,
        metric=selection_metric,
        mask=foreground_mask,
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
        n_points_per_group=n_points_per_group,
        random_state=random_state,
    )
)


print(f"\nSelected points using {selection_metric}:")
print(selected_points_df)


# ============================================================
# OPTIONAL MANUAL PIXEL REPLACEMENT
# ============================================================

# IMPORTANT:
#
# Do NOT manually modify:
#
# selected_points_df.iloc[1, 1] = ...
# selected_points_df.iloc[1, 2] = ...
#
# because metric_value would still correspond to the old pixel.
#
#
# If you really need to replace one selected example,
# use replace_selected_pixel().
#
#
# Example:
#
# selected_points_df = replace_selected_pixel(
#     selected_df=selected_points_df,
#     metric_maps=metric_maps,
#     group="bottom",
#     y=119,
#     x=120,
#     mask=foreground_mask,
#     require_same_quantile=True,
# )
#
#
# require_same_quantile=True ensures that the manually
# selected pixel really belongs to the bottom/top group.


# ============================================================
# Save selected pixel information
# ============================================================

selected_points_filename = (
    f"Figure_6_selected_points_"
    f"{selection_metric}.csv"
)


selected_points_df.to_csv(
    os.path.join(
        output_dir,
        selected_points_filename,
    ),
    index=False,
)


# ============================================================
# Extract spectra at selected points
# ============================================================

spectra_df = extract_selected_pixel_spectra(
    srs_cube=srs_cube_common,
    raman_cube_interp=raman_interp,
    wn=srs_wn_common,
    selected_df=selected_points_df,
    spectrum_normalization=spectrum_normalization,
)


# ============================================================
# Save selected spectra
# ============================================================

spectra_filename = f"Figure_6_selected_spectra_{selection_metric}.csv"

spectra_df.to_csv(
    os.path.join(
        output_dir,
        spectra_filename,
    ),
    index=False,
)


# ============================================================
# Physical calibration
# ============================================================

fov_um = 600


pixel_size_um = None


# ============================================================
# Generate Figure 6
# ============================================================

figure_filename = (
    f"Figure_6_selected_pixel_spectra_"
    f"{selection_metric}.png"
)


fig = plot_figure6_selected_pixel_spectra(
    srs_cube=srs_cube_common,
    raman_cube_interp=raman_interp,
    wn=srs_wn_common,
    metric_maps=metric_maps,
    selected_df=selected_points_df,
    spectra_df=spectra_df,
    mask=foreground_mask,
    sample_name=sample["name"],
    selection_metric=selection_metric,
    overview_band=1445,
    overview_width=12,
    overview_source="raman",
    cmap_overview="afmhot",
    pmin_overview=1,
    pmax_overview=99.8,
    add_colorbar=True,
    add_scalebar_flag=True,
    scale_bar_um=100,
    pixel_size_um=pixel_size_um,
    fov_um=fov_um,
    random_state=random_state,
    plot_normalized=True,

    # Keep None while tuning the figure:
    savepath=None,
    # When ready to save, replace the previous line with:
    #
    # savepath=os.path.join(
    #     output_dir,
    #     figure_filename,
    # ),
)

plt.show()