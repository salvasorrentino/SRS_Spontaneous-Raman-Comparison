import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from matplotlib.gridspec import GridSpec

from Script_for_Figures.utils_fig2 import (
    add_scalebar,
)


# ============================================================
# Basic helpers
# ============================================================

def get_band_image(cube, wn, center, width=12):
    """
    Extract a 2D band image by averaging channels around
    center +/- width/2.
    """
    wn = np.asarray(wn)

    keep = (
        (wn >= center - width / 2)
        & (wn <= center + width / 2)
    )

    if keep.sum() == 0:
        idx = np.argmin(np.abs(wn - center))
        return cube[..., idx]

    return np.nanmean(cube[..., keep], axis=-1)


def robust_minmax_masked(
    img,
    mask=None,
    pmin=1,
    pmax=99.8,
    eps=1e-12,
    clip=True,
):
    """
    Robust min-max normalization for a 2D image.
    """
    img = img.astype(float).copy()

    if mask is not None:
        vals = img[
            mask & np.isfinite(img)
        ]
    else:
        vals = img[np.isfinite(img)]

    if vals.size == 0:
        return np.full_like(
            img,
            np.nan,
            dtype=float,
        )

    vmin = np.nanpercentile(vals, pmin)
    vmax = np.nanpercentile(vals, pmax)

    out = (
        (img - vmin)
        / (vmax - vmin + eps)
    )

    if clip:
        out = np.clip(out, 0, 1)

    if mask is not None:
        out[~mask] = np.nan

    return out


def _pearson_1d(x, y, eps=1e-12):
    """
    Pearson correlation between two 1D arrays.
    """
    keep = np.isfinite(x) & np.isfinite(y)

    x = x[keep]
    y = y[keep]

    if x.size < 3:
        return np.nan

    x0 = x - np.nanmean(x)
    y0 = y - np.nanmean(y)

    denominator = (
        np.sqrt(np.nansum(x0 ** 2))
        * np.sqrt(np.nansum(y0 ** 2))
        + eps
    )

    return np.nansum(x0 * y0) / denominator


def _spearman_1d(x, y, eps=1e-12):
    """
    Spearman correlation as Pearson correlation on ranks.
    """
    keep = np.isfinite(x) & np.isfinite(y)

    x = x[keep]
    y = y[keep]

    if x.size < 3:
        return np.nan

    return _pearson_1d(
        rankdata(x),
        rankdata(y),
        eps=eps,
    )


def _cosine_1d(x, y, eps=1e-12):
    """
    Cosine similarity between two 1D arrays.
    """
    keep = np.isfinite(x) & np.isfinite(y)

    x = x[keep]
    y = y[keep]

    if x.size < 3:
        return np.nan

    denominator = (
        np.sqrt(np.nansum(x ** 2))
        * np.sqrt(np.nansum(y ** 2))
        + eps
    )

    return np.nansum(x * y) / denominator


def _bicor_1d(x, y, c=9.0, eps=1e-12):
    """
    Biweight midcorrelation between two 1D arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    keep = np.isfinite(x) & np.isfinite(y)

    x = x[keep]
    y = y[keep]

    if x.size < 3:
        return np.nan

    median_x = np.nanmedian(x)
    median_y = np.nanmedian(y)

    x_centered = x - median_x
    y_centered = y - median_y

    mad_x = np.nanmedian(
        np.abs(x_centered)
    )

    mad_y = np.nanmedian(
        np.abs(y_centered)
    )

    if mad_x < eps or mad_y < eps:
        return np.nan

    ux = x_centered / (
        c * mad_x + eps
    )

    uy = y_centered / (
        c * mad_y + eps
    )

    wx = (1 - ux ** 2) ** 2
    wy = (1 - uy ** 2) ** 2

    wx[np.abs(ux) >= 1] = 0
    wy[np.abs(uy) >= 1] = 0

    x_weighted = x_centered * wx
    y_weighted = y_centered * wy

    denominator = (
        np.sqrt(np.nansum(x_weighted ** 2))
        * np.sqrt(np.nansum(y_weighted ** 2))
        + eps
    )

    return (
        np.nansum(x_weighted * y_weighted)
        / denominator
    )


def _safe_cmap(
    cmap_name,
    bad_color="white",
):
    """
    Return a colormap with NaNs shown as bad_color.
    """
    cmap = plt.get_cmap(
        cmap_name
    ).copy()

    cmap.set_bad(bad_color)

    return cmap


# ============================================================
# Band-wise spatial metrics
# ============================================================

def compute_bandwise_spatial_metrics(
    srs_cube,
    raman_cube_interp,
    wn,
    mask=None,
    bands=None,
    band_width=12,
    image_normalization="robust_minmax",
    pmin=1,
    pmax=99.8,
    min_valid_pixels=100,
    eps=1e-12,
    compute_bicor=True,
    bicor_c=9.0,
):
    """
    Compute band-wise spatial agreement between SRS and
    spontaneous Raman.

    Each metric compares the spatial patterns of the SRS and
    Raman images at one Raman shift.
    """
    if srs_cube.shape != raman_cube_interp.shape:
        raise ValueError(
            "SRS and Raman cubes must have the same shape. "
            f"Got {srs_cube.shape} and "
            f"{raman_cube_interp.shape}."
        )

    height, width, n_bands = srs_cube.shape
    wn = np.asarray(wn)

    if bands is None:
        bands = wn.copy()

    bands = np.asarray(
        bands,
        dtype=float,
    )

    if mask is None:
        mask_use = np.ones(
            (height, width),
            dtype=bool,
        )
    else:
        mask_use = mask.astype(bool)

    rows = []

    for band in bands:
        srs_img_raw = get_band_image(
            srs_cube,
            wn,
            center=band,
            width=band_width,
        )

        raman_img_raw = get_band_image(
            raman_cube_interp,
            wn,
            center=band,
            width=band_width,
        )

        if image_normalization == "robust_minmax":
            srs_img = robust_minmax_masked(
                srs_img_raw,
                mask=mask_use,
                pmin=pmin,
                pmax=pmax,
                eps=eps,
                clip=True,
            )

            raman_img = robust_minmax_masked(
                raman_img_raw,
                mask=mask_use,
                pmin=pmin,
                pmax=pmax,
                eps=eps,
                clip=True,
            )

        elif image_normalization == "none":
            srs_img = srs_img_raw.astype(float)
            raman_img = raman_img_raw.astype(float)

            srs_img[~mask_use] = np.nan
            raman_img[~mask_use] = np.nan

        else:
            raise ValueError(
                "image_normalization must be "
                "'robust_minmax' or 'none'"
            )

        valid = (
            mask_use
            & np.isfinite(srs_img)
            & np.isfinite(raman_img)
        )

        n_valid = int(
            np.sum(valid)
        )

        if n_valid < min_valid_pixels:
            pearson = np.nan
            spearman = np.nan
            bicor = np.nan
            cosine = np.nan
            medae = np.nan
            mae = np.nan
            rmse = np.nan

        else:
            x = srs_img[valid].ravel()
            y = raman_img[valid].ravel()

            pearson = _pearson_1d(
                x,
                y,
                eps=eps,
            )

            spearman = _spearman_1d(
                x,
                y,
                eps=eps,
            )

            cosine = _cosine_1d(
                x,
                y,
                eps=eps,
            )

            if compute_bicor:
                bicor = _bicor_1d(
                    x,
                    y,
                    c=bicor_c,
                    eps=eps,
                )
            else:
                bicor = np.nan

            absolute_error = np.abs(
                x - y
            )

            medae = np.nanmedian(
                absolute_error
            )

            mae = np.nanmean(
                absolute_error
            )

            rmse = np.sqrt(
                np.nanmean((x - y) ** 2)
            )

        rows.append(
            {
                "band_cm-1": band,
                "n_valid_pixels": n_valid,
                "pearson": pearson,
                "spearman": spearman,
                "bicor": bicor,
                "cosine": cosine,
                "medae": medae,
                "mae": mae,
                "rmse": rmse,
                "band_width": band_width,
                "image_normalization": (
                    image_normalization
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# New analysis: distribution across Raman bands
# ============================================================

def summarize_bandwise_metrics(
    band_metrics_df,
    sample_name="sample",
    metrics=("bicor", "cosine"),
):
    """
    Summarize the distribution of band-wise agreement values.

    This analysis quantifies how strongly spatial agreement varies
    across the fingerprint region without defining an arbitrary
    pass/fail threshold.

    The interdecile range, q90 - q10, describes the difference
    between relatively high- and low-agreement spectral regions.

    The best- and worst-quartile medians provide representative
    agreement values for the upper and lower parts of the
    distribution.

    Notes
    -----
    These are descriptive summaries. Adjacent Raman-shift points
    can contain overlapping spectral windows and should not be
    treated as statistically independent observations.
    """
    rows = []

    for metric in metrics:
        if metric not in band_metrics_df.columns:
            raise KeyError(
                f"Metric '{metric}' was not found. "
                f"Available columns: "
                f"{list(band_metrics_df.columns)}"
            )

        values = np.asarray(
            band_metrics_df[metric],
            dtype=float,
        )

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            rows.append(
                {
                    "sample": sample_name,
                    "metric": metric,
                    "n_bands": 0,
                    "q10": np.nan,
                    "q25": np.nan,
                    "median": np.nan,
                    "q75": np.nan,
                    "q90": np.nan,
                    "iqr": np.nan,
                    "interdecile_range": np.nan,
                    "worst_quartile_median": np.nan,
                    "best_quartile_median": np.nan,
                    "best_worst_difference": np.nan,
                    "fraction_nonpositive": np.nan,
                }
            )
            continue

        q10, q25, median, q75, q90 = (
            np.nanpercentile(
                values,
                [10, 25, 50, 75, 90],
            )
        )

        worst_quartile = values[
            values <= q25
        ]

        best_quartile = values[
            values >= q75
        ]

        worst_quartile_median = np.nanmedian(
            worst_quartile
        )

        best_quartile_median = np.nanmedian(
            best_quartile
        )

        rows.append(
            {
                "sample": sample_name,
                "metric": metric,
                "n_bands": values.size,
                "q10": q10,
                "q25": q25,
                "median": median,
                "q75": q75,
                "q90": q90,
                "iqr": q75 - q25,
                "interdecile_range": q90 - q10,
                "worst_quartile_median": (
                    worst_quartile_median
                ),
                "best_quartile_median": (
                    best_quartile_median
                ),
                "best_worst_difference": (
                    best_quartile_median
                    - worst_quartile_median
                ),
                "fraction_nonpositive": np.mean(
                    values <= 0
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Selected band images
# ============================================================

def extract_selected_band_images(
    srs_cube,
    raman_cube_interp,
    wn,
    selected_bands,
    mask=None,
    band_width=12,
    image_normalization="robust_minmax",
    pmin=1,
    pmax=99.8,
):
    """
    Extract normalized SRS, Raman, and absolute-difference images
    for selected Raman bands.
    """
    band_images = {}

    for band in selected_bands:
        srs_raw = get_band_image(
            srs_cube,
            wn,
            center=band,
            width=band_width,
        )

        raman_raw = get_band_image(
            raman_cube_interp,
            wn,
            center=band,
            width=band_width,
        )

        if image_normalization == "robust_minmax":
            srs_img = robust_minmax_masked(
                srs_raw,
                mask=mask,
                pmin=pmin,
                pmax=pmax,
                clip=True,
            )

            raman_img = robust_minmax_masked(
                raman_raw,
                mask=mask,
                pmin=pmin,
                pmax=pmax,
                clip=True,
            )

        elif image_normalization == "none":
            srs_img = srs_raw.astype(float)
            raman_img = raman_raw.astype(float)

            if mask is not None:
                srs_img[~mask] = np.nan
                raman_img[~mask] = np.nan

        else:
            raise ValueError(
                "image_normalization must be "
                "'robust_minmax' or 'none'"
            )

        absolute_difference = np.abs(
            srs_img - raman_img
        )

        band_images[float(band)] = {
            "srs": srs_img,
            "raman": raman_img,
            "absdiff": absolute_difference,
        }

    return band_images


# ============================================================
# Plot Figure 6
# ============================================================

def plot_figure6_bandwise_spatial_agreement(
    srs_cube,
    raman_cube_interp,
    wn,
    band_metrics_df,
    mask=None,
    selected_bands=(785, 1003, 1445, 1660),
    band_width=12,
    metric1_to_plot="bicor",
    metric2_to_plot="cosine",
    sample_name="sample",
    image_normalization="robust_minmax",
    cmap_images="afmhot",
    cmap_diff="magma",
    pmin=1,
    pmax=99.8,
    metric1_ylim=None,
    metric2_ylim=None,
    add_colorbar=True,
    add_scalebar_flag=True,
    scale_bar_um=100,
    pixel_size_um=None,
    fov_um=None,
    savepath=None,
):
    """
    Plot band-wise spatial agreement and selected band images.
    """
    metric_label_map = {
        "pearson": "Pearson correlation",
        "spearman": "Spearman correlation",
        "bicor": "Biweight midcorrelation",
        "cosine": "Cosine similarity",
        "medae": "Median absolute error",
        "mae": "Mean absolute error",
        "rmse": "Root mean square error",
    }

    allowed_metrics = list(
        metric_label_map.keys()
    )

    if metric1_to_plot not in allowed_metrics:
        raise ValueError(
            f"Unknown metric: {metric1_to_plot}"
        )

    if metric2_to_plot not in allowed_metrics:
        raise ValueError(
            f"Unknown metric: {metric2_to_plot}"
        )

    if metric1_to_plot not in band_metrics_df.columns:
        raise KeyError(
            f"'{metric1_to_plot}' not found in dataframe."
        )

    if metric2_to_plot not in band_metrics_df.columns:
        raise KeyError(
            f"'{metric2_to_plot}' not found in dataframe."
        )

    selected_bands = list(
        selected_bands
    )

    band_images = extract_selected_band_images(
        srs_cube=srs_cube,
        raman_cube_interp=raman_cube_interp,
        wn=wn,
        selected_bands=selected_bands,
        mask=mask,
        band_width=band_width,
        image_normalization=image_normalization,
        pmin=pmin,
        pmax=pmax,
    )

    n_selected_bands = len(
        selected_bands
    )

    figure_height = (
        3.1
        + 2.25 * n_selected_bands
    )

    fig = plt.figure(
        figsize=(12, figure_height),
        constrained_layout=True,
    )

    grid = GridSpec(
        n_selected_bands + 1,
        3,
        figure=fig,
        height_ratios=(
            [1.15]
            + [1.0] * n_selected_bands
        ),
        width_ratios=[1, 1, 1],
    )

    ax_metric1 = fig.add_subplot(
        grid[0, 0:2]
    )

    ax_metric2 = fig.add_subplot(
        grid[0, 2]
    )

    image_axes = []

    for index in range(
        n_selected_bands
    ):
        image_axes.append(
            [
                fig.add_subplot(
                    grid[index + 1, 0]
                ),
                fig.add_subplot(
                    grid[index + 1, 1]
                ),
                fig.add_subplot(
                    grid[index + 1, 2]
                ),
            ]
        )

    df = band_metrics_df.sort_values(
        "band_cm-1"
    ).copy()

    # Metric 1
    ax_metric1.plot(
        df["band_cm-1"],
        df[metric1_to_plot],
        linewidth=1.5,
    )

    ax_metric1.scatter(
        df["band_cm-1"],
        df[metric1_to_plot],
        s=10,
        alpha=0.65,
    )

    # Metric 2
    ax_metric2.plot(
        df["band_cm-1"],
        df[metric2_to_plot],
        linewidth=1.5,
    )

    ax_metric2.scatter(
        df["band_cm-1"],
        df[metric2_to_plot],
        s=10,
        alpha=0.65,
    )

    for band in selected_bands:
        ax_metric1.axvline(
            band,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

        ax_metric2.axvline(
            band,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    ax_metric1.set_xlabel(
        "Raman shift (cm$^{-1}$)"
    )

    ax_metric1.set_ylabel(
        metric_label_map[metric1_to_plot]
    )

    ax_metric1.set_title(
        "Band-wise spatial agreement:\n"
        f"{metric_label_map[metric1_to_plot]}"
    )

    ax_metric2.set_xlabel(
        "Raman shift (cm$^{-1}$)"
    )

    ax_metric2.set_ylabel(
        metric_label_map[metric2_to_plot]
    )

    ax_metric2.set_title(
        "Band-wise spatial agreement:\n"
        f"{metric_label_map[metric2_to_plot]}"
    )

    if metric1_ylim is not None:
        ax_metric1.set_ylim(
            *metric1_ylim
        )

    if metric2_ylim is not None:
        ax_metric2.set_ylim(
            *metric2_ylim
        )

    # Selected image rows
    for index, band in enumerate(
        selected_bands
    ):
        ax_raman, ax_srs, ax_difference = (
            image_axes[index]
        )

        raman_img = band_images[
            float(band)
        ]["raman"]

        srs_img = band_images[
            float(band)
        ]["srs"]

        difference_img = band_images[
            float(band)
        ]["absdiff"]

        if image_normalization == "robust_minmax":
            image_vmin = 0
            image_vmax = 1
        else:
            values = np.concatenate(
                [
                    raman_img[
                        np.isfinite(raman_img)
                    ].ravel(),
                    srs_img[
                        np.isfinite(srs_img)
                    ].ravel(),
                ]
            )

            image_vmin = np.nanpercentile(
                values,
                pmin,
            )

            image_vmax = np.nanpercentile(
                values,
                pmax,
            )

        difference_values = difference_img[
            np.isfinite(difference_img)
        ]

        difference_vmin = 0

        if difference_values.size > 0:
            difference_vmax = np.nanpercentile(
                difference_values,
                99,
            )
        else:
            difference_vmax = 1

        image_raman = ax_raman.imshow(
            raman_img,
            cmap=_safe_cmap(cmap_images),
            vmin=image_vmin,
            vmax=image_vmax,
        )

        image_srs = ax_srs.imshow(
            srs_img,
            cmap=_safe_cmap(cmap_images),
            vmin=image_vmin,
            vmax=image_vmax,
        )

        image_difference = ax_difference.imshow(
            difference_img,
            cmap=_safe_cmap(cmap_diff),
            vmin=difference_vmin,
            vmax=difference_vmax,
        )

        ax_raman.set_title(
            f"Raman {band:.0f} cm$^{{-1}}$"
        )

        ax_srs.set_title(
            f"SRS {band:.0f} cm$^{{-1}}$"
        )

        ax_difference.set_title(
            f"|SRS - Raman| {band:.0f}"
        )

        for axis in [
            ax_raman,
            ax_srs,
            ax_difference,
        ]:
            axis.axis("off")

        if add_scalebar_flag:
            add_scalebar(
                ax_raman,
                raman_img.shape,
                pixel_size_um=pixel_size_um,
                fov_um=fov_um,
                scale_bar_um=scale_bar_um,
                location="lower right",
                linewidth=1.5,
                fontsize=7,
            )

        if add_colorbar:
            colorbar = fig.colorbar(
                image_difference,
                ax=ax_difference,
                fraction=0.046,
                pad=0.04,
            )

            colorbar.ax.tick_params(
                labelsize=7
            )

    fig.suptitle(
        f"{sample_name} | "
        "band-wise SRS/Raman spatial agreement",
        y=1.01,
        fontsize=10,
    )

    if savepath is not None:
        fig.savefig(
            savepath,
            dpi=600,
            bbox_inches="tight",
        )

    return fig