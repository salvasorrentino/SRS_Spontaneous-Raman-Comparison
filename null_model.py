import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Project utilities
# ============================================================

from Script_for_Figures.utils_fig2 import (
    crop_to_overlap,
    interpolate_raman_to_srs_grid,
    set_publication_style,
)

from Script_for_Figures.utils_fig4 import (
    make_foreground_mask_for_similarity,
    compute_pixelwise_similarity_maps,
    normalize_cube_spectra,
)

from Script_for_Figures.utils_null_model import (
    compute_similarity_permutation_null,
    summarize_similarity_null,
    benjamini_hochberg,
    plot_similarity_permutation_test,
)


set_publication_style()


# ============================================================
# GLOBAL SETTINGS
# ============================================================

# Use 200 for an initial complete run.
#
# With 200 permutations, the smallest possible empirical
# p-value is:
#
#     1 / (200 + 1) = 0.00498
#
# If runtime permits, use 999 for the final manuscript.
n_null_permutations = 200


random_state = 5008


# Must match the normalization used in the main pixel-wise
# analysis.
spectrum_normalization = "minmax"


# Foreground-mask parameters.
mask_projection = "mean"
mask_threshold_method = "percentile"
mask_threshold_percentile = 10
mask_min_size = 50
mask_opening_radius = 0


# Similarity filtering.
min_signal_percentile = 5
clip_negative = False


save_reference_pixel_distributions = True
save_permutation_plots = True


# ============================================================
# ROOT PATH
# ============================================================

root_path = (
    r"C:\Users\salva\OneDrive - Massachusetts Institute of Technology"
    r"\Coregistration SRS-confocal\data"
)


# ============================================================
# SAMPLE CONFIGURATION
# ============================================================

SAMPLES = [

    # --------------------------------------------------------
    # SKIN 1
    # --------------------------------------------------------
    {
        "name": "skin_1",
        "tissue": "skin",

        "path_fold": os.path.join(
            root_path,
            r"20260511_skin",
        ),

        "srs_filename":
            r"Project_srs_skin_hyperspectral_processed_coregistered_w_confocal_no_correction.pickle",

        "srs_calibration_filename":
            r"srs_Raman_shift.pickle",

        "raman_filename":
            r"skin_processed_confocal.pickle",

        "raman_calibration_filename":
            r"arr_calibration202601.pickle",

        "process_raman":
            lambda cube:
            cube,

        "process_srs":
            lambda cube:
            np.rot90(
                cube,
                k=2,
            ),

        "process_raman_wn":
            lambda wn:
            wn[300:],
    },


    # --------------------------------------------------------
    # SKIN 2
    # --------------------------------------------------------
    {
        "name": "skin_2",
        "tissue": "skin",

        "path_fold": os.path.join(
            root_path,
            r"skin_small_hyperspectral",
        ),

        "srs_filename":
            r"Project_srs_skin_small_hyper_hyperspectral_processed_coregistered_w_confocal.pickle",

        "srs_calibration_filename":
            r"srs_Raman_shift.pickle",

        "raman_filename":
            r"arr_confocal_raman_processed.pickle",

        "raman_calibration_filename":
            r"arr_calibration_confocal.pickle",

        "process_raman":
            lambda cube:
            cube,

        "process_srs":
            lambda cube:
            np.rot90(
                cube,
                k=2,
            ),

        "process_raman_wn":
            lambda wn:
            wn[300:],
    },


    # --------------------------------------------------------
    # COLON
    # --------------------------------------------------------
    {
        "name": "colon",
        "tissue": "colon",

        "path_fold": os.path.join(
            root_path,
            r"20260518_colon",
        ),

        "srs_filename":
            r"Project_srs_colon_hyperspectral_processed_coregistered_w_confocal_no_correction.pickle",

        "srs_calibration_filename":
            r"srs_Raman_shift.pickle",

        "raman_filename":
            r"colon_processed_confocal_fill_over_srs.pickle",

        "raman_calibration_filename":
            r"arr_calibration202601.pickle",

        "process_raman":
            lambda cube:
            cube[10:-10, :, :],

        "process_srs":
            lambda cube:
            cube[10:-10, :, :],

        "process_raman_wn":
            lambda wn:
            wn[300:],
    },


    # --------------------------------------------------------
    # BREAST
    # --------------------------------------------------------
    {
        "name": "breast",
        "tissue": "breast",

        "path_fold": os.path.join(
            root_path,
            r"20260520_breast",
        ),

        "srs_filename":
            r"Project_srs_breast_hyperspectral_processed_coregistered_w_confocal_no_correction.pickle",

        "srs_calibration_filename":
            r"srs_Raman_shift.pickle",

        "raman_filename":
            r"breast_processed_confocal.pickle",

        "raman_calibration_filename":
            r"arr_calibration202601.pickle",

        # Match this crop exactly to the final Figure 6 script.
        "process_raman": (
            lambda cube:
            np.rot90(
                cube,
                k=2,
            )[:-30, 75:275, :]
        ),

        "process_srs": (
            lambda cube:
            cube[:-30, 75:275, :]
        ),

        "process_raman_wn":
            lambda wn:
            wn[300:],
    },
]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = os.path.join(
    root_path,
    "similarity_null_model",
)

os.makedirs(
    output_dir,
    exist_ok=True,
)


# ============================================================
# Load and preprocess one sample
# ============================================================

def load_sample(
    sample_config,
):
    """
    Load one paired SRS / spontaneous Raman dataset and apply
    the preprocessing specified in SAMPLES.
    """

    path_fold = sample_config[
        "path_fold"
    ]

    path_srs = os.path.join(
        path_fold,
        sample_config[
            "srs_filename"
        ],
    )

    path_srs_wn = os.path.join(
        path_fold,
        sample_config[
            "srs_calibration_filename"
        ],
    )

    path_raman = os.path.join(
        path_fold,
        sample_config[
            "raman_filename"
        ],
    )

    path_raman_wn = os.path.join(
        path_fold,
        sample_config[
            "raman_calibration_filename"
        ],
    )

    srs_cube = np.asarray(
        pd.read_pickle(
            path_srs
        )
    )

    srs_wn = np.asarray(
        pd.read_pickle(
            path_srs_wn
        )
    )

    raman_cube = np.asarray(
        pd.read_pickle(
            path_raman
        )
    )

    raman_wn = np.asarray(
        pd.read_pickle(
            path_raman_wn
        )
    )

    srs_cube = sample_config[
        "process_srs"
    ](
        srs_cube
    )

    raman_cube = sample_config[
        "process_raman"
    ](
        raman_cube
    )

    raman_wn = sample_config[
        "process_raman_wn"
    ](
        raman_wn
    )

    return (
        srs_cube,
        srs_wn,
        raman_cube,
        raman_wn,
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

all_summary_rows = []
all_permutation_rows = []


for sample_index, sample_config in enumerate(
    SAMPLES
):

    sample_name = sample_config[
        "name"
    ]

    tissue_name = sample_config[
        "tissue"
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Processing sample: "
        f"{sample_name}"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # Load
    # ========================================================

    (
        srs_cube,
        srs_wn,
        raman_cube,
        raman_wn,
    ) = load_sample(
        sample_config
    )

    print(
        "Original SRS shape:",
        srs_cube.shape,
    )

    print(
        "Original Raman shape:",
        raman_cube.shape,
    )


    # ========================================================
    # Common spectral range
    # ========================================================

    (
        srs_cube_common,
        srs_wn_common,
        raman_cube_common,
        raman_wn_common,
    ) = crop_to_overlap(
        srs_cube,
        srs_wn,
        raman_cube,
        raman_wn,
    )


    # ========================================================
    # Interpolate Raman onto SRS spectral grid
    # ========================================================

    raman_interp = interpolate_raman_to_srs_grid(
        raman_cube_common,
        raman_wn_common,
        srs_wn_common,
    )

    print(
        "Common SRS shape:",
        srs_cube_common.shape,
    )

    print(
        "Interpolated Raman shape:",
        raman_interp.shape,
    )

    print(
        "Common spectral range:",
        f"{srs_wn_common.min():.1f}",
        "-",
        f"{srs_wn_common.max():.1f}",
        "cm^-1",
    )


    # ========================================================
    # Foreground mask
    # ========================================================

    foreground_mask = (
        make_foreground_mask_for_similarity(
            srs_cube_common,

            raman_cube_interp=(
                raman_interp
            ),

            projection=(
                mask_projection
            ),

            threshold_method=(
                mask_threshold_method
            ),

            threshold_percentile=(
                mask_threshold_percentile
            ),

            min_size=(
                mask_min_size
            ),

            opening_radius=(
                mask_opening_radius
            ),
        )
    )

    print(
        "Foreground pixels:",
        int(
            np.sum(
                foreground_mask
            )
        ),
    )


    # ========================================================
    # Observed cosine and Pearson maps
    # ========================================================

    (
        metric_maps,
        observed_summary_df,
        normalized_cubes,
    ) = compute_pixelwise_similarity_maps(
        srs_cube=(
            srs_cube_common
        ),

        raman_cube_interp=(
            raman_interp
        ),

        mask=(
            foreground_mask
        ),

        spectrum_normalization=(
            spectrum_normalization
        ),

        clip_negative=(
            clip_negative
        ),

        min_signal_percentile=(
            min_signal_percentile
        ),

        compute_pearson=True,
        compute_spearman=False,
        compute_bicor=False,
    )

    valid_mask = np.asarray(
        metric_maps[
            "valid_mask"
        ],
        dtype=bool,
    )

    observed_cosine_map = np.asarray(
        metric_maps[
            "cosine"
        ],
        dtype=float,
    )

    observed_pearson_map = np.asarray(
        metric_maps[
            "pearson"
        ],
        dtype=float,
    )

    observed_cosine_values = (
        observed_cosine_map[
            valid_mask
            & np.isfinite(
                observed_cosine_map
            )
        ]
    )

    observed_pearson_values = (
        observed_pearson_map[
            valid_mask
            & np.isfinite(
                observed_pearson_map
            )
        ]
    )

    print(
        "Valid cosine pixels:",
        observed_cosine_values.size,
    )

    print(
        "Observed cosine median:",
        np.median(
            observed_cosine_values
        ),
    )

    print(
        "Valid Pearson pixels:",
        observed_pearson_values.size,
    )

    print(
        "Observed Pearson median:",
        np.median(
            observed_pearson_values
        ),
    )


    # ========================================================
    # Normalize cubes for the null model
    # ========================================================
    #
    # This uses exactly the same normalization as the observed
    # pixel-wise analysis.
    # ========================================================

    srs_norm = normalize_cube_spectra(
        srs_cube_common,

        method=(
            spectrum_normalization
        ),

        mask=None,
    )

    raman_norm = normalize_cube_spectra(
        raman_interp,

        method=(
            spectrum_normalization
        ),

        mask=None,
    )


    # ========================================================
    # Permutation null model
    # ========================================================

    sample_random_state = (
        random_state
        + sample_index * 10000
    )

    print(
        f"Running "
        f"{n_null_permutations} "
        f"null permutations..."
    )

    (
        null_reference_df,
        null_permutation_df,
    ) = compute_similarity_permutation_null(
        srs_cube=(
            srs_norm
        ),

        raman_cube=(
            raman_norm
        ),

        mask=(
            valid_mask
        ),

        n_permutations=(
            n_null_permutations
        ),

        random_state=(
            sample_random_state
        ),
    )


    # ========================================================
    # Statistical summary
    # ========================================================

    summary_df = summarize_similarity_null(
        observed_cosine_map=(
            observed_cosine_map
        ),

        observed_pearson_map=(
            observed_pearson_map
        ),

        null_permutation_df=(
            null_permutation_df
        ),

        mask=(
            valid_mask
        ),
    )

    summary_df.insert(
        0,
        "sample",
        sample_name,
    )

    summary_df.insert(
        1,
        "tissue",
        tissue_name,
    )

    summary_df[
        "n_spectral_bands"
    ] = (
        srs_cube_common.shape[-1]
    )


    # ========================================================
    # Add sample information to null outputs
    # ========================================================

    null_reference_df.insert(
        0,
        "sample",
        sample_name,
    )

    null_reference_df.insert(
        1,
        "tissue",
        tissue_name,
    )

    null_permutation_df.insert(
        0,
        "sample",
        sample_name,
    )

    null_permutation_df.insert(
        1,
        "tissue",
        tissue_name,
    )


    # ========================================================
    # Print results
    # ========================================================

    print(
        "\nStatistical results:"
    )

    for _, row in summary_df.iterrows():

        print(
            f"\nMetric: "
            f"{row['metric']}"
        )

        print(
            f"Observed median: "
            f"{row['observed_median']:.4f}"
        )

        print(
            f"Null median: "
            f"{row['null_field_median_center']:.4f}"
        )

        print(
            f"Typical pixel null q95: "
            f"{row['null_pixel_q95_typical']:.4f}"
        )

        print(
            f"Null field-median q95: "
            f"{row['null_field_median_q95']:.4f}"
        )

        print(
            f"Median difference: "
            f"{row['median_difference']:.4f}"
        )

        print(
            f"Exceedances: "
            f"{row['null_exceedances']}/"
            f"{row['n_permutations']}"
        )

        print(
            f"Empirical p: "
            f"{row['empirical_p_greater']:.5f}"
        )


    # ========================================================
    # Sample output directory
    # ========================================================

    sample_output_dir = os.path.join(
        output_dir,
        sample_name,
    )

    os.makedirs(
        sample_output_dir,
        exist_ok=True,
    )


    # ========================================================
    # Save sample-level tables
    # ========================================================

    summary_df.to_csv(
        os.path.join(
            sample_output_dir,
            "similarity_null_summary.csv",
        ),
        index=False,
    )

    null_permutation_df.to_csv(
        os.path.join(
            sample_output_dir,
            "similarity_null_permutations.csv",
        ),
        index=False,
    )

    if save_reference_pixel_distributions:

        null_reference_df.to_csv(
            os.path.join(
                sample_output_dir,
                "similarity_null_reference_pixels.csv",
            ),
            index=False,
        )


    # ========================================================
    # Plot permutation tests
    # ========================================================

    if save_permutation_plots:

        fig = plot_similarity_permutation_test(
            null_permutation_df=(
                null_permutation_df
            ),

            summary_df=(
                summary_df
            ),

            sample_name=(
                sample_name
            ),

            savepath=os.path.join(
                sample_output_dir,
                "similarity_permutation_tests.png",
            ),
        )

        plt.close(
            fig
        )


    # ========================================================
    # Collect results
    # ========================================================

    all_summary_rows.append(
        summary_df
    )

    all_permutation_rows.append(
        null_permutation_df
    )


# ============================================================
# FINAL TABLES
# ============================================================

final_summary_df = pd.concat(
    all_summary_rows,
    ignore_index=True,
)

final_permutation_df = pd.concat(
    all_permutation_rows,
    ignore_index=True,
)


# ============================================================
# Multiple-testing correction
# ============================================================
#
# There are two metrics for each sample. The raw empirical
# p-value and the Benjamini-Hochberg-adjusted value are both
# retained.
# ============================================================

final_summary_df[
    "empirical_p_fdr_bh"
] = benjamini_hochberg(
    final_summary_df[
        "empirical_p_greater"
    ].to_numpy()
)


# ============================================================
# Compact publication-oriented table
# ============================================================

publication_table = final_summary_df[
    [
        "sample",
        "tissue",
        "metric",

        "observed_median",
        "observed_q25",
        "observed_q75",

        "null_field_median_center",
        "null_field_median_q95",

        "null_pixel_q95_typical",

        "median_difference",
        "observed_minus_null_pixel_q95",

        "null_exceedances",
        "n_permutations",

        "empirical_p_greater",
        "empirical_p_fdr_bh",

        "observed_percentile_in_null",

        "observed_n",
        "n_spectral_bands",
    ]
].copy()


publication_table = publication_table.rename(
    columns={
        "sample":
            "Sample",

        "tissue":
            "Tissue",

        "metric":
            "Metric",

        "observed_median":
            "Observed median",

        "observed_q25":
            "Observed Q25",

        "observed_q75":
            "Observed Q75",

        "null_field_median_center":
            "Null field-median center",

        "null_field_median_q95":
            "Null field-median 95th percentile",

        "null_pixel_q95_typical":
            "Typical pixel-level null 95th percentile",

        "median_difference":
            "Observed - null median",

        "observed_minus_null_pixel_q95":
            "Observed median - pixel null Q95",

        "null_exceedances":
            "Null exceedances",

        "n_permutations":
            "Null permutations",

        "empirical_p_greater":
            "Empirical one-sided p",

        "empirical_p_fdr_bh":
            "BH-adjusted p",

        "observed_percentile_in_null":
            "Observed percentile in null",

        "observed_n":
            "Foreground pixels",

        "n_spectral_bands":
            "Spectral bands",
    }
)


# ============================================================
# Save complete results
# ============================================================

final_summary_df.to_csv(
    os.path.join(
        output_dir,
        "similarity_null_model_full_summary.csv",
    ),
    index=False,
)

final_permutation_df.to_csv(
    os.path.join(
        output_dir,
        "similarity_null_model_all_permutations.csv",
    ),
    index=False,
)

publication_table.to_csv(
    os.path.join(
        output_dir,
        "similarity_null_model_table.csv",
    ),
    index=False,
)

publication_table.to_excel(
    os.path.join(
        output_dir,
        "similarity_null_model_table.xlsx",
    ),
    index=False,
)


# ============================================================
# Print final table
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FINAL COSINE AND PEARSON NULL-MODEL TABLE"
)

print(
    "=" * 100
)

print(
    publication_table.to_string(
        index=False
    )
)

print(
    "\nSaved results to:"
)

print(
    output_dir
)