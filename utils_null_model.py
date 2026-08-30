import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Cosine similarity
# ============================================================

def cosine_similarity_pixelwise(
    cube1,
    cube2,
    mask=None,
    eps=1e-12,
):
    """
    Compute pixel-wise cosine similarity between two
    hyperspectral cubes.

    Parameters
    ----------
    cube1, cube2 : ndarray
        Shape (ny, nx, n_bands).

    mask : ndarray or None
        Optional boolean spatial mask with shape (ny, nx).

    eps : float
        Numerical stability constant.

    Returns
    -------
    cosine_map : ndarray
        Shape (ny, nx), NaN outside valid pixels.
    """

    cube1 = np.asarray(
        cube1,
        dtype=float,
    )

    cube2 = np.asarray(
        cube2,
        dtype=float,
    )

    if cube1.shape != cube2.shape:

        raise ValueError(
            f"Cube shapes differ: "
            f"{cube1.shape} vs {cube2.shape}"
        )

    dot_product = np.sum(
        cube1 * cube2,
        axis=-1,
    )

    norm1 = np.linalg.norm(
        cube1,
        axis=-1,
    )

    norm2 = np.linalg.norm(
        cube2,
        axis=-1,
    )

    denominator = norm1 * norm2

    cosine_map = np.full(
        cube1.shape[:2],
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(dot_product)
        & np.isfinite(denominator)
        & (denominator > eps)
    )

    if mask is not None:

        valid &= np.asarray(
            mask,
            dtype=bool,
        )

    cosine_map[valid] = (
        dot_product[valid]
        / denominator[valid]
    )

    return cosine_map


# ============================================================
# Pearson correlation
# ============================================================

def pearson_correlation_pixelwise(
    cube1,
    cube2,
    mask=None,
    eps=1e-12,
):
    """
    Compute pixel-wise Pearson correlation between two
    hyperspectral cubes.

    Pearson correlation measures the mean-centered
    co-variation of the two spectra.

    Parameters
    ----------
    cube1, cube2 : ndarray
        Shape (ny, nx, n_bands).

    mask : ndarray or None
        Optional boolean spatial mask with shape (ny, nx).

    eps : float
        Numerical stability constant.

    Returns
    -------
    pearson_map : ndarray
        Shape (ny, nx), NaN outside valid pixels.
    """

    cube1 = np.asarray(
        cube1,
        dtype=float,
    )

    cube2 = np.asarray(
        cube2,
        dtype=float,
    )

    if cube1.shape != cube2.shape:

        raise ValueError(
            f"Cube shapes differ: "
            f"{cube1.shape} vs {cube2.shape}"
        )

    cube1_centered = (
        cube1
        - np.mean(
            cube1,
            axis=-1,
            keepdims=True,
        )
    )

    cube2_centered = (
        cube2
        - np.mean(
            cube2,
            axis=-1,
            keepdims=True,
        )
    )

    numerator = np.sum(
        cube1_centered
        * cube2_centered,
        axis=-1,
    )

    denominator = (
        np.linalg.norm(
            cube1_centered,
            axis=-1,
        )
        * np.linalg.norm(
            cube2_centered,
            axis=-1,
        )
    )

    pearson_map = np.full(
        cube1.shape[:2],
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > eps)
    )

    if mask is not None:

        valid &= np.asarray(
            mask,
            dtype=bool,
        )

    pearson_map[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return pearson_map


# ============================================================
# Permute spectral channels
# ============================================================

def permute_spectral_channels(
    spectra,
    rng=None,
):
    """
    Independently shuffle the spectral channels of every
    spectrum.

    Parameters
    ----------
    spectra : ndarray
        Shape (n_pixels, n_bands).

    rng : np.random.Generator or None
        Random-number generator.

    Returns
    -------
    permuted_spectra : ndarray
        Spectra with independently shuffled channels.
    """

    spectra = np.asarray(
        spectra,
        dtype=float,
    )

    if spectra.ndim != 2:

        raise ValueError(
            "spectra must have shape "
            "(n_pixels, n_bands)."
        )

    if rng is None:

        rng = np.random.default_rng()

    # Generator.permuted independently permutes every row.
    return rng.permuted(
        spectra,
        axis=1,
    )


# ============================================================
# Distribution summary helper
# ============================================================

def _summarize_distribution(
    values,
    prefix,
):
    """
    Return descriptive statistics for one metric distribution.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:

        raise ValueError(
            f"No finite values found for {prefix}."
        )

    return {
        f"{prefix}_n":
            values.size,

        f"{prefix}_mean":
            np.mean(values),

        f"{prefix}_median":
            np.median(values),

        f"{prefix}_q25":
            np.percentile(
                values,
                25,
            ),

        f"{prefix}_q75":
            np.percentile(
                values,
                75,
            ),

        f"{prefix}_q95":
            np.percentile(
                values,
                95,
            ),

        f"{prefix}_q99":
            np.percentile(
                values,
                99,
            ),
    }


# ============================================================
# Cosine and Pearson permutation null model
# ============================================================

def compute_similarity_permutation_null(
    srs_cube,
    raman_cube,
    mask=None,
    n_permutations=200,
    random_state=0,
):
    """
    Construct permutation null distributions for cosine
    similarity and Pearson correlation.

    The spectral channels of the Raman spectrum are
    independently permuted at every pixel. This destroys the
    correspondence between intensity and Raman shift while
    preserving the exact intensity distribution of each
    spectrum.

    Permuting one modality is mathematically equivalent to
    independently permuting both modalities for cosine and
    Pearson, because only their relative channel ordering
    determines the resulting metric. It is substantially faster.

    Two outputs are produced:

    1. null_reference_df:
       pixel-wise values from the first null realization,
       useful for plotting and inspecting the null distribution.

    2. null_permutation_df:
       one row per permutation, containing the field-level
       distribution summaries used for the statistical test.

    Parameters
    ----------
    srs_cube, raman_cube : ndarray
        Shape (ny, nx, n_bands). The cubes must already be
        normalized and expressed on the same spectral grid.

    mask : ndarray or None
        Foreground / valid-pixel mask.

    n_permutations : int
        Number of permutation realizations.

    random_state : int
        Seed for reproducibility.

    Returns
    -------
    null_reference_df : pd.DataFrame
        Pixel-wise cosine and Pearson values from the first
        permutation.

    null_permutation_df : pd.DataFrame
        One row per permutation, containing the median and
        percentile summaries of both metrics.
    """

    srs_cube = np.asarray(
        srs_cube,
        dtype=float,
    )

    raman_cube = np.asarray(
        raman_cube,
        dtype=float,
    )

    if srs_cube.shape != raman_cube.shape:

        raise ValueError(
            f"SRS and Raman cubes must have the same shape. "
            f"Got {srs_cube.shape} and {raman_cube.shape}."
        )

    if srs_cube.ndim != 3:

        raise ValueError(
            "Input cubes must have shape "
            "(ny, nx, n_bands)."
        )

    ny, nx, n_bands = srs_cube.shape

    if mask is None:

        mask = np.ones(
            (ny, nx),
            dtype=bool,
        )

    else:

        mask = np.asarray(
            mask,
            dtype=bool,
        )

    if mask.shape != (ny, nx):

        raise ValueError(
            f"Mask shape {mask.shape} does not match "
            f"cube spatial shape {(ny, nx)}."
        )

    # --------------------------------------------------------
    # Extract masked spectra
    # --------------------------------------------------------

    coordinates = np.argwhere(
        mask
    )

    srs_pixels = srs_cube[
        mask
    ]

    raman_pixels = raman_cube[
        mask
    ]

    # Remove spectra containing non-finite values.
    finite_rows = (
        np.all(
            np.isfinite(srs_pixels),
            axis=1,
        )
        & np.all(
            np.isfinite(raman_pixels),
            axis=1,
        )
    )

    coordinates = coordinates[
        finite_rows
    ]

    srs_pixels = srs_pixels[
        finite_rows
    ]

    raman_pixels = raman_pixels[
        finite_rows
    ]

    if srs_pixels.shape[0] == 0:

        raise ValueError(
            "No valid foreground spectra were found."
        )

    rng = np.random.default_rng(
        random_state
    )

    permutation_rows = []

    null_reference_df = None

    # ========================================================
    # Permutation loop
    # ========================================================

    for permutation_idx in range(
        n_permutations
    ):

        # ----------------------------------------------------
        # Break Raman-shift correspondence
        # ----------------------------------------------------

        raman_permuted = permute_spectral_channels(
            spectra=raman_pixels,
            rng=rng,
        )

        # Add a singleton spatial dimension so that the same
        # pixel-wise metric functions can be reused.
        srs_for_metric = srs_pixels[
            :,
            np.newaxis,
            :,
        ]

        raman_for_metric = raman_permuted[
            :,
            np.newaxis,
            :,
        ]

        # ----------------------------------------------------
        # Cosine null values
        # ----------------------------------------------------

        cosine_null = (
            cosine_similarity_pixelwise(
                cube1=srs_for_metric,
                cube2=raman_for_metric,
            )
            .ravel()
        )

        # ----------------------------------------------------
        # Pearson null values
        # ----------------------------------------------------

        pearson_null = (
            pearson_correlation_pixelwise(
                cube1=srs_for_metric,
                cube2=raman_for_metric,
            )
            .ravel()
        )

        # ----------------------------------------------------
        # One field-level summary per permutation
        # ----------------------------------------------------

        row = {
            "permutation":
                permutation_idx,
        }

        row.update(
            _summarize_distribution(
                values=cosine_null,
                prefix="cosine_null",
            )
        )

        row.update(
            _summarize_distribution(
                values=pearson_null,
                prefix="pearson_null",
            )
        )

        permutation_rows.append(
            row
        )

        # ----------------------------------------------------
        # Save one pixel-wise null realization
        # ----------------------------------------------------

        if permutation_idx == 0:

            null_reference_df = pd.DataFrame(
                {
                    "permutation":
                        permutation_idx,

                    "y":
                        coordinates[:, 0],

                    "x":
                        coordinates[:, 1],

                    "cosine_null":
                        cosine_null,

                    "pearson_null":
                        pearson_null,
                }
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        progress_interval = max(
            1,
            n_permutations // 10,
        )

        if (
            permutation_idx == 0
            or (permutation_idx + 1)
            % progress_interval == 0
            or permutation_idx
            == n_permutations - 1
        ):

            print(
                f"Permutation "
                f"{permutation_idx + 1}/"
                f"{n_permutations}"
            )

    null_permutation_df = pd.DataFrame(
        permutation_rows
    )

    return (
        null_reference_df,
        null_permutation_df,
    )


# ============================================================
# Statistical summary
# ============================================================

def summarize_similarity_null(
    observed_cosine_map,
    observed_pearson_map,
    null_permutation_df,
    mask=None,
):
    """
    Compare the observed field median with the distribution of
    field medians generated by the null model.

    The empirical p-value is one-sided:

        H0: observed agreement is not greater than expected
            after random spectral-channel assignment.

        H1: observed agreement is greater than expected under
            the permutation null model.

    The pixel-wise null percentiles remain descriptive and are
    not used as the inferential p-value.
    """

    observed_cosine_map = np.asarray(
        observed_cosine_map,
        dtype=float,
    )

    observed_pearson_map = np.asarray(
        observed_pearson_map,
        dtype=float,
    )

    if mask is None:

        mask = np.ones(
            observed_cosine_map.shape,
            dtype=bool,
        )

    else:

        mask = np.asarray(
            mask,
            dtype=bool,
        )

    metric_information = {
        "cosine": {
            "observed_map":
                observed_cosine_map,

            "null_prefix":
                "cosine_null",
        },

        "pearson": {
            "observed_map":
                observed_pearson_map,

            "null_prefix":
                "pearson_null",
        },
    }

    summary_rows = []

    for metric_name, information in (
        metric_information.items()
    ):

        observed_map = information[
            "observed_map"
        ]

        null_prefix = information[
            "null_prefix"
        ]

        observed_valid = (
            mask
            & np.isfinite(
                observed_map
            )
        )

        observed_values = observed_map[
            observed_valid
        ]

        if observed_values.size == 0:

            raise ValueError(
                f"No observed values found for "
                f"{metric_name}."
            )

        observed_median = np.median(
            observed_values
        )

        # ----------------------------------------------------
        # Distribution of field-level null medians
        # ----------------------------------------------------

        null_field_medians = (
            null_permutation_df[
                f"{null_prefix}_median"
            ]
            .to_numpy(
                dtype=float
            )
        )

        null_field_medians = (
            null_field_medians[
                np.isfinite(
                    null_field_medians
                )
            ]
        )

        n_permutations = (
            null_field_medians.size
        )

        # ----------------------------------------------------
        # Empirical one-sided permutation p-value
        # ----------------------------------------------------

        n_exceedances = np.sum(
            null_field_medians
            >= observed_median
        )

        empirical_p = (
            n_exceedances + 1
        ) / (
            n_permutations + 1
        )

        # Percentile position of the observed median within
        # the distribution of null field medians.
        observed_percentile = (
            100
            * (
                np.sum(
                    null_field_medians
                    < observed_median
                )
                + 0.5
                * np.sum(
                    null_field_medians
                    == observed_median
                )
            )
            / n_permutations
        )

        # ----------------------------------------------------
        # Typical pixel-level null percentiles
        # ----------------------------------------------------
        #
        # Every permutation contains many pixel-wise null
        # values. We take the median of the percentile estimate
        # obtained from each permutation. This avoids pooling
        # millions of correlated pixel values into one table.
        # ----------------------------------------------------

        null_pixel_q25_typical = np.median(
            null_permutation_df[
                f"{null_prefix}_q25"
            ]
        )

        null_pixel_q75_typical = np.median(
            null_permutation_df[
                f"{null_prefix}_q75"
            ]
        )

        null_pixel_q95_typical = np.median(
            null_permutation_df[
                f"{null_prefix}_q95"
            ]
        )

        null_pixel_q99_typical = np.median(
            null_permutation_df[
                f"{null_prefix}_q99"
            ]
        )

        null_field_median_center = np.median(
            null_field_medians
        )

        null_field_median_q05 = np.percentile(
            null_field_medians,
            5,
        )

        null_field_median_q95 = np.percentile(
            null_field_medians,
            95,
        )

        null_field_median_q99 = np.percentile(
            null_field_medians,
            99,
        )

        summary_rows.append(
            {
                "metric":
                    metric_name,

                # Observed distribution
                "observed_n":
                    observed_values.size,

                "observed_mean":
                    np.mean(
                        observed_values
                    ),

                "observed_median":
                    observed_median,

                "observed_q25":
                    np.percentile(
                        observed_values,
                        25,
                    ),

                "observed_q75":
                    np.percentile(
                        observed_values,
                        75,
                    ),

                # Descriptive pixel-level null
                "null_pixel_q25_typical":
                    null_pixel_q25_typical,

                "null_pixel_q75_typical":
                    null_pixel_q75_typical,

                "null_pixel_q95_typical":
                    null_pixel_q95_typical,

                "null_pixel_q99_typical":
                    null_pixel_q99_typical,

                # Inferential field-median null
                "null_field_median_center":
                    null_field_median_center,

                "null_field_median_q05":
                    null_field_median_q05,

                "null_field_median_q95":
                    null_field_median_q95,

                "null_field_median_q99":
                    null_field_median_q99,

                # Effect sizes
                "median_difference":
                    (
                        observed_median
                        - null_field_median_center
                    ),

                "observed_minus_null_pixel_q95":
                    (
                        observed_median
                        - null_pixel_q95_typical
                    ),

                # Permutation test
                "null_exceedances":
                    int(
                        n_exceedances
                    ),

                "n_permutations":
                    n_permutations,

                "empirical_p_greater":
                    empirical_p,

                "observed_percentile_in_null":
                    observed_percentile,

                "observed_above_null_field_q95":
                    bool(
                        observed_median
                        > null_field_median_q95
                    ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# Benjamini-Hochberg correction
# ============================================================

def benjamini_hochberg(
    p_values,
):
    """
    Benjamini-Hochberg false-discovery-rate correction.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        p_values.shape,
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(
        p_values
    )

    valid_indices = np.where(
        valid
    )[0]

    valid_p = p_values[
        valid
    ]

    if valid_p.size == 0:

        return adjusted

    order = np.argsort(
        valid_p
    )

    sorted_p = valid_p[
        order
    ]

    n_tests = sorted_p.size

    ranks = np.arange(
        1,
        n_tests + 1,
    )

    adjusted_sorted = (
        sorted_p
        * n_tests
        / ranks
    )

    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0,
        1,
    )

    adjusted_valid = np.empty_like(
        adjusted_sorted
    )

    adjusted_valid[
        order
    ] = adjusted_sorted

    adjusted[
        valid_indices
    ] = adjusted_valid

    return adjusted


# ============================================================
# Plot permutation tests
# ============================================================

def plot_similarity_permutation_test(
    null_permutation_df,
    summary_df,
    sample_name,
    savepath=None,
):
    """
    Plot the null distribution of field medians for cosine
    similarity and Pearson correlation.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.2),
    )

    plot_information = [
        {
            "metric":
                "cosine",

            "column":
                "cosine_null_median",

            "xlabel":
                "Field median cosine similarity",
        },

        {
            "metric":
                "pearson",

            "column":
                "pearson_null_median",

            "xlabel":
                "Field median Pearson r",
        },
    ]

    for ax, information in zip(
        axes,
        plot_information,
    ):

        metric_name = information[
            "metric"
        ]

        null_values = (
            null_permutation_df[
                information["column"]
            ]
            .to_numpy(
                dtype=float
            )
        )

        null_values = null_values[
            np.isfinite(
                null_values
            )
        ]

        metric_summary = (
            summary_df[
                summary_df["metric"]
                == metric_name
            ]
            .iloc[0]
        )

        observed_median = metric_summary[
            "observed_median"
        ]

        null_center = metric_summary[
            "null_field_median_center"
        ]

        null_q95 = metric_summary[
            "null_field_median_q95"
        ]

        empirical_p = metric_summary[
            "empirical_p_greater"
        ]

        n_bins = min(
            40,
            max(
                12,
                int(
                    np.sqrt(
                        null_values.size
                    )
                ),
            ),
        )

        ax.hist(
            null_values,
            bins=n_bins,
            density=True,
            color="0.65",
            edgecolor="0.25",
            alpha=0.8,
            label="Permutation null",
        )

        ax.axvline(
            null_center,
            color="0.25",
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Null median = "
                f"{null_center:.3f}"
            ),
        )

        ax.axvline(
            null_q95,
            color="0.40",
            linestyle=":",
            linewidth=1.5,
            label=(
                f"Null 95th pct = "
                f"{null_q95:.3f}"
            ),
        )

        ax.axvline(
            observed_median,
            color="#1f77b4",
            linestyle="-",
            linewidth=2.0,
            label=(
                f"Observed = "
                f"{observed_median:.3f}"
            ),
        )

        ax.set_xlabel(
            information["xlabel"]
        )

        ax.set_ylabel(
            "Probability density"
        )

        ax.set_title(
            f"{metric_name.capitalize()}: "
            f"p = {empirical_p:.4f}"
        )

        ax.legend(
            frameon=False,
            fontsize=8,
        )

    fig.suptitle(
        sample_name
    )

    fig.tight_layout()

    if savepath is not None:

        fig.savefig(
            savepath,
            dpi=600,
            bbox_inches="tight",
        )

    return fig