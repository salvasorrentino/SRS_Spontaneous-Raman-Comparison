import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Basic utilities
# ============================================================

def normalize_pixel_spectra(pixels, eps=1e-12):
    """
    Min-max normalize each pixel spectrum independently.

    Parameters
    ----------
    pixels : ndarray
        Array with shape n_pixels x n_bands.

    Returns
    -------
    pixels_norm : ndarray
        Min-max-normalized spectra.
    """
    spec_min = np.min(pixels, axis=1, keepdims=True)
    spec_max = np.max(pixels, axis=1, keepdims=True)

    return (pixels - spec_min) / (spec_max - spec_min + eps)


def neighbor_prediction_error(pixels, wn):
    """
    Predict every internal spectral band using linear interpolation
    between the two adjacent bands.

    Parameters
    ----------
    pixels : ndarray
        Normalized spectra, shape n_pixels x n_bands.

    wn : ndarray
        Raman shifts, shape n_bands.

    Returns
    -------
    error : ndarray
        Median absolute neighbor-prediction error for every internal
        Raman shift.
    """
    wn = np.asarray(wn, dtype=float)

    denominator = wn[2:] - wn[:-2]

    weight_left = (wn[2:] - wn[1:-1]) / denominator
    weight_right = (wn[1:-1] - wn[:-2]) / denominator

    predicted = (
        pixels[:, :-2] * weight_left[None, :]
        + pixels[:, 2:] * weight_right[None, :]
    )

    residual = np.abs(pixels[:, 1:-1] - predicted)

    return np.median(residual, axis=0)


# ============================================================
# Spectral-continuity analysis
# ============================================================

def compute_spectral_continuity(
    srs_cube,
    raman_cube_interp,
    wn,
    mask=None,
    sample_name="sample",
    percentile=90,
    eps=1e-12,
):
    """
    Compare the local spectral continuity of SRS and spontaneous Raman.

    Each internal spectral band is predicted from its two adjacent
    bands. The median prediction error is calculated across foreground
    pixels.

    Parameters
    ----------
    srs_cube : ndarray
        SRS cube, shape H x W x N.

    raman_cube_interp : ndarray
        Raman cube interpolated on the SRS grid, shape H x W x N.

    wn : ndarray
        Common Raman-shift axis.

    mask : ndarray or None
        Boolean foreground mask.

    sample_name : str
        Sample name used in the summary table.

    percentile : float
        Percentile used to summarize the upper tail of the excess
        prediction error.

    Returns
    -------
    band_results : pandas.DataFrame
        Band-wise SRS, Raman, and excess errors.

    summary : pandas.DataFrame
        Sample-level continuity metrics.
    """
    srs_cube = np.asarray(srs_cube, dtype=float)
    raman_cube_interp = np.asarray(raman_cube_interp, dtype=float)
    wn = np.asarray(wn, dtype=float).ravel()

    if srs_cube.shape != raman_cube_interp.shape:
        raise ValueError("SRS and Raman cubes must have the same shape.")

    if srs_cube.shape[-1] != len(wn):
        raise ValueError("The spectral axis does not match the cube.")

    if np.any(np.diff(wn) <= 0):
        raise ValueError("Raman shifts must be in increasing order.")

    if mask is None:
        mask = np.ones(srs_cube.shape[:2], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    # Keep pixels with finite spectra in both modalities
    valid_mask = (
        mask
        & np.all(np.isfinite(srs_cube), axis=-1)
        & np.all(np.isfinite(raman_cube_interp), axis=-1)
    )

    srs_pixels = srs_cube[valid_mask]
    raman_pixels = raman_cube_interp[valid_mask]

    # Remove flat spectra
    valid_pixels = (
        np.ptp(srs_pixels, axis=1) > eps
    ) & (
        np.ptp(raman_pixels, axis=1) > eps
    )

    srs_pixels = srs_pixels[valid_pixels]
    raman_pixels = raman_pixels[valid_pixels]

    if len(srs_pixels) == 0:
        raise ValueError("No valid foreground spectra were found.")

    # Independent pixel-wise min-max normalization
    srs_pixels = normalize_pixel_spectra(srs_pixels, eps=eps)
    raman_pixels = normalize_pixel_spectra(raman_pixels, eps=eps)

    # Leave-one-band-out prediction errors
    srs_error = neighbor_prediction_error(srs_pixels, wn)
    raman_error = neighbor_prediction_error(raman_pixels, wn)

    # Absolute excess error specifically present in SRS
    # Symmetric absolute difference between the two error profiles
    absolute_difference = np.abs(
        srs_error - raman_error
    )

    # Symmetric relative mismatch
    # Zero means identical local-continuity errors.
    # Larger values mean greater disagreement in either direction.
    relative_mismatch = (
            absolute_difference
            / (srs_error + raman_error + eps)
    )

    # Global similarity between the complete continuity profiles
    global_continuity_similarity = 1 - (
            np.sum(absolute_difference)
            / (
                    np.sum(srs_error + raman_error)
                    + eps
            )
    )

    # Upper-tail mismatch used to identify problematic bands
    relative_mismatch_percentile = np.percentile(
        relative_mismatch,
        percentile,
    )

    tail_continuity_score = (
            1 - relative_mismatch_percentile
    )

    internal_wn = wn[1:-1]

    worst_index = np.argmax(relative_mismatch)

    band_results = pd.DataFrame(
        {
            "raman_shift_cm-1": internal_wn,
            "srs_prediction_error": srs_error,
            "raman_prediction_error": raman_error,
            "absolute_continuity_difference": (
                absolute_difference
            ),
            "relative_continuity_mismatch": (
                relative_mismatch
            ),
        }
    )

    summary = pd.DataFrame(
        {
            "sample": [sample_name],
            "n_foreground_pixels": [len(srs_pixels)],
            "n_bands_tested": [len(internal_wn)],
            "median_srs_error": [
                np.median(srs_error)
            ],
            "median_raman_error": [
                np.median(raman_error)
            ],
            "median_absolute_difference": [
                np.median(absolute_difference)
            ],
            "mean_relative_mismatch": [
                np.mean(relative_mismatch)
            ],
            "global_continuity_similarity": [
                global_continuity_similarity
            ],
            f"relative_mismatch_q{percentile}": [
                relative_mismatch_percentile
            ],
            "tail_continuity_score": [
                tail_continuity_score
            ],
            "worst_raman_shift_cm-1": [
                internal_wn[worst_index]
            ],
            "worst_relative_mismatch": [
                relative_mismatch[worst_index]
            ],
            "fraction_srs_error_above_raman": [
                np.mean(srs_error > raman_error)
            ],
        }
    )

    return band_results, summary

# ============================================================
# Plot
# ============================================================

def plot_spectral_continuity(
    band_results,
    summary,
    sample_name="sample",
    percentile=90,
    savepath=None,
):
    """
    Plot the neighbor-prediction errors and the symmetric
    continuity mismatch between SRS and Raman.
    """
    wn = band_results[
        "raman_shift_cm-1"
    ].values

    srs_error = band_results[
        "srs_prediction_error"
    ].values

    raman_error = band_results[
        "raman_prediction_error"
    ].values

    relative_mismatch = band_results[
        "relative_continuity_mismatch"
    ].values

    mismatch_threshold = summary[
        f"relative_mismatch_q{percentile}"
    ].iloc[0]

    global_similarity = summary[
        "global_continuity_similarity"
    ].iloc[0]

    tail_score = summary[
        "tail_continuity_score"
    ].iloc[0]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # Panel 1: neighbor-prediction errors
    # --------------------------------------------------------

    axes[0].plot(
        wn,
        srs_error,
        color="tab:blue",
        linewidth=2,
        label="SRS",
    )

    axes[0].plot(
        wn,
        raman_error,
        color="0.35",
        linewidth=2,
        label="Spontaneous Raman",
    )

    axes[0].set_xlabel(
        r"Raman shift (cm$^{-1}$)"
    )

    axes[0].set_ylabel(
        "Median neighbor-prediction error"
    )

    axes[0].set_title(
        "Local spectral continuity"
    )

    axes[0].legend(frameon=False)

    # --------------------------------------------------------
    # Panel 2: symmetric relative mismatch
    # --------------------------------------------------------

    axes[1].plot(
        wn,
        relative_mismatch,
        color="tab:red",
        linewidth=2,
    )

    axes[1].fill_between(
        wn,
        0,
        relative_mismatch,
        color="tab:red",
        alpha=0.20,
    )

    axes[1].axhline(
        mismatch_threshold,
        color="0.35",
        linestyle="--",
        linewidth=1.5,
        label=f"{percentile}th percentile",
    )

    high_mismatch = (
        relative_mismatch
        >= mismatch_threshold
    )

    axes[1].scatter(
        wn[high_mismatch],
        relative_mismatch[high_mismatch],
        color="tab:red",
        edgecolor="black",
        s=35,
        zorder=3,
    )

    axes[1].set_xlabel(
        r"Raman shift (cm$^{-1}$)"
    )

    axes[1].set_ylabel(
        "Relative continuity mismatch"
    )

    axes[1].set_title(
        f"Global similarity = {global_similarity:.3f}; "
        f"tail score = {tail_score:.3f}"
    )

    axes[1].set_ylim(
        bottom=0,
        top=max(
            1.05 * np.max(relative_mismatch),
            0.05,
        ),
    )

    axes[1].legend(frameon=False)

    fig.suptitle(sample_name)

    if savepath is not None:
        fig.savefig(
            savepath,
            dpi=600,
            bbox_inches="tight",
        )

    return fig