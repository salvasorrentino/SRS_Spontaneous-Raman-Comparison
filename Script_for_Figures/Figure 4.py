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
    compute_pixelwise_similarity_maps,
    flatten_metric_maps_to_dataframe,
    plot_figure4_pixelwise_similarity,
)

set_publication_style()


# ============================================================
# Paths
# ============================================================

path_fold = (
    r"C:\Users\salva\OneDrive - Massachusetts Institute of Technology\Coregistration "
    r"SRS-confocal\data\20260511_skin"
)

path_img_srs = os.path.join(
    path_fold,
    "Project_srs_skin_hyperspectral_processed_coregistered_w_confocal_no_correction.pickle"
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
    "skin_processed_confocal.pickle"
)

output_dir = os.path.join(path_fold, "paper_figures")
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# Load data
# ============================================================

# Adjust crop if needed
# raman_cube = pd.read_pickle(path_img_confocal)#[20:-20, :-15, :]
raman_cube = np.rot90(pd.read_pickle(path_img_confocal), k=2)#[:-50, 75:275, :]#[15:-15, :-15, :]#[:-50, 75:275, :]
raman_wn = pd.read_pickle(path_calib_confocal)[300:]

srs_cube = pd.read_pickle(path_img_srs)#[:-50, 75:275, :]#[20:-20, :-15, :]#
srs_wn = pd.read_pickle(path_calib_srs)

sample = {
    "name": "skin",
    "tissue": "skin",
    "srs_cube": srs_cube,
    "srs_wn": srs_wn,
    "raman_cube": raman_cube,
    "raman_wn": raman_wn,
}

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

print("SRS common shape:", srs_cube_common.shape)
print("Raman interp shape:", raman_interp.shape)
print("Common Raman shift range:", srs_wn_common.min(), srs_wn_common.max())


# ============================================================
# Foreground mask
# ============================================================

foreground_mask = make_foreground_mask_for_similarity(
    srs_cube_common,
    raman_cube_interp=raman_interp,
    projection="mean",
    threshold_method="percentile",
    threshold_percentile=5,
    min_size=40,
    opening_radius=0,
)

# ============================================================
# Compute pixel-wise metrics
# ============================================================

metric_maps, summary_df, normalized_cubes = compute_pixelwise_similarity_maps(
    srs_cube=srs_cube_common,
    raman_cube_interp=raman_interp,
    mask=foreground_mask,

    # Main recommendation:
    # each pixel spectrum is independently scaled between 0 and 1.
    spectrum_normalization="minmax",

    # Usually keep this False if baseline correction was already applied.
    clip_negative=False,

    # Removes the weakest pixels where min-max normalization would amplify noise.
    min_signal_percentile=10,

    compute_pearson=True,
    bicor_c=9
)

summary_df.insert(0, "sample", sample["name"])

print(summary_df)

summary_csv = os.path.join(output_dir, "Figure_4_pixelwise_similarity_summary.csv")
summary_df.to_csv(summary_csv, index=False)


# Optional: save flattened metric values.
# This can be useful later for boxplots across all samples.
metric_df = flatten_metric_maps_to_dataframe(
    metric_maps,
    sample_name=sample["name"]
)

metric_df.to_csv(
    os.path.join(output_dir, "Figure_4_pixelwise_similarity_values.csv"),
    index=False
)


# Optional: save maps as compressed numpy file
np.savez_compressed(
    os.path.join(output_dir, "Figure_4_metric_maps.npz"),
    cosine=metric_maps["cosine"],
    sam_deg=metric_maps["sam_deg"],
    medae=metric_maps["medae"],
    mae=metric_maps["mae"],
    rmse=metric_maps["rmse"],
    pearson=metric_maps["pearson"],
    valid_mask=metric_maps["valid_mask"],
    srs_wn_common=srs_wn_common,
)


# ============================================================
# Physical calibration
# ============================================================

# Use the real value for this specific image if known.
# If the final coregistered image is approximately 600 x 600 um:
fov_um = 600

# Alternative if you know pixel size:
# pixel_size_um = fov_um / srs_cube_common.shape[1]
pixel_size_um = 2


# ============================================================
# Generate Figure 4
# ============================================================

fig = plot_figure4_pixelwise_similarity(
    srs_cube=srs_cube_common,
    raman_cube_interp=raman_interp,
    wn=srs_wn_common,
    metric_maps=metric_maps,
    mask=foreground_mask,

    metrics_to_plot=("cosine", "pearson", "spearman", "bicor", "medae"),
    # metrics_to_plot=("cosine", "bicor", "medae"),

    metric_vlims={
        "cosine": (0.50, 1.00),
        "pearson": (-1.00, 1.00),
        "spearman": (-1.00, 1.00),
        "bicor": (-1.00, 1.00),
        "medae": (0.00, None),
    },

    metric_cmaps={
        "cosine": "viridis",
        "pearson": "coolwarm",
        "spearman": "coolwarm",
        "bicor": "coolwarm",
        "medae": "magma",
    },

    overview_band=1445,
    overview_width=12,
    overview_source="raman",
    sample_name=sample["name"],
    spectrum_normalization="minmax",

    add_colorbar=True,
    add_scalebar_flag=True,
    scale_bar_um=100,
    pixel_size_um=pixel_size_um,
    fov_um=fov_um,

    # savepath=os.path.join(
    #     output_dir,
    #     "Figure_4_similarity_all_correlations.png"
    # ),
)

plt.show()

# plt.figure()
# plt.plot(srs_wn, srs_cube[145, 117, :]/srs_cube[145, 117, :].max())
# plt.plot(raman_wn, raman_cube[145, 117, :]/raman_cube[145, 117, :].max())
# plt.show()