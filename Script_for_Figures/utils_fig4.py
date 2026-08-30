import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, disk
from scipy.stats import rankdata

from Script_for_Figures.utils_fig2 import (
    robust_minmax,
    percentile_vmin_vmax,
    add_scalebar,
)


# ============================================================
# Basic spectral utilities
# ============================================================

def get_band_image(cube, wn, center, width=10):
    """
    Extract a 2D band image by averaging channels around center +/- width/2.
    """
    wn = np.asarray(wn)

    keep = (wn >= center - width / 2) & (wn <= center + width / 2)

    if keep.sum() == 0:
        idx = np.argmin(np.abs(wn - center))
        return cube[..., idx]

    return np.nanmean(cube[..., keep], axis=-1)


def normalize_cube_spectra(
    cube,
    method="minmax",
    mask=None,
    eps=1e-12,
    clip_negative=False,
):
    """
    Normalize each pixel spectrum independently.

    Parameters
    ----------
    cube : ndarray
        Hyperspectral cube with shape H x W x N.

    method : {"none", "minmax", "l1", "l2", "zscore"}
        "none"   : no normalization.
        "minmax" : each pixel spectrum scaled to [0, 1].
        "l1"     : each pixel spectrum divided by sum(abs(spectrum)).
        "l2"     : each pixel spectrum divided by sqrt(sum(spectrum^2)).
        "zscore" : each pixel spectrum centered and divided by std.

    mask : ndarray or None
        Boolean foreground mask. Pixels outside mask are set to NaN.

    clip_negative : bool
        If True, negative values are set to zero before normalization.

    Returns
    -------
    cube_norm : ndarray
        Normalized cube.
    """

    cube = cube.astype(float).copy()

    if clip_negative:
        cube = np.maximum(cube, 0)

    if method == "none":
        cube_norm = cube

    elif method == "minmax":
        spec_min = np.nanmin(cube, axis=-1, keepdims=True)
        spec_max = np.nanmax(cube, axis=-1, keepdims=True)
        dyn = spec_max - spec_min
        cube_norm = (cube - spec_min) / (dyn + eps)
        cube_norm[dyn[..., 0] <= eps, :] = np.nan

    elif method == "l1":
        norm = np.nansum(np.abs(cube), axis=-1, keepdims=True)
        cube_norm = cube / (norm + eps)
        cube_norm[norm[..., 0] <= eps, :] = np.nan

    elif method == "l2":
        norm = np.sqrt(np.nansum(cube ** 2, axis=-1, keepdims=True))
        cube_norm = cube / (norm + eps)
        cube_norm[norm[..., 0] <= eps, :] = np.nan

    elif method == "zscore":
        mean = np.nanmean(cube, axis=-1, keepdims=True)
        std = np.nanstd(cube, axis=-1, keepdims=True)
        cube_norm = (cube - mean) / (std + eps)
        cube_norm[std[..., 0] <= eps, :] = np.nan

    else:
        raise ValueError("method must be 'none', 'minmax', 'l1', 'l2', or 'zscore'")

    if mask is not None:
        cube_norm[~mask, :] = np.nan

    return cube_norm


# ============================================================
# Foreground mask
# ============================================================

def make_foreground_mask_for_similarity(
    srs_cube,
    raman_cube_interp=None,
    projection="mean",
    threshold_method="otsu",
    threshold_percentile=20,
    min_size=500,
    opening_radius=2,
):
    """
    Create a foreground mask from SRS and optionally Raman total signal.

    This is important because per-pixel min-max normalization can make
    weak background/noise pixels look artificially structured.
    """

    def project_cube(cube):
        cube_pos = np.maximum(cube, 0)

        if projection == "mean":
            return np.nanmean(cube_pos, axis=-1)
        elif projection == "median":
            return np.nanmedian(cube_pos, axis=-1)
        elif projection == "max":
            return np.nanmax(cube_pos, axis=-1)
        else:
            raise ValueError("projection must be 'mean', 'median', or 'max'")

    srs_summary = robust_minmax(project_cube(srs_cube))

    if raman_cube_interp is not None:
        ram_summary = robust_minmax(project_cube(raman_cube_interp))
        summary = srs_summary + ram_summary
    else:
        summary = srs_summary

    vals = summary[np.isfinite(summary)]

    if threshold_method == "otsu":
        thr = threshold_otsu(vals)
    elif threshold_method == "percentile":
        thr = np.nanpercentile(vals, threshold_percentile)
    else:
        raise ValueError("threshold_method must be 'otsu' or 'percentile'")

    mask = summary > thr

    if opening_radius is not None and opening_radius > 0:
        mask = binary_opening(mask, disk(opening_radius))

    if min_size is not None and min_size > 0:
        mask = remove_small_objects(mask, min_size=min_size)

    return mask


# ============================================================
# Pixel-wise spectral metrics
# ============================================================

def _rowwise_pearson(x, y, eps=1e-12):
    """
    Fast row-wise Pearson correlation.

    x, y shape:
        n_pixels x n_wavenumbers
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x0 = x - np.nanmean(x, axis=1, keepdims=True)
    y0 = y - np.nanmean(y, axis=1, keepdims=True)

    num = np.nansum(x0 * y0, axis=1)

    den = (
        np.sqrt(np.nansum(x0 ** 2, axis=1))
        * np.sqrt(np.nansum(y0 ** 2, axis=1))
        + eps
    )

    return num / den


def _rowwise_biweight_midcorrelation(x, y, c=9.0, eps=1e-12):
    """
    Fast row-wise biweight midcorrelation.

    This is a robust Pearson-like correlation.
    It down-weights spectral points that are outliers relative to the
    median and MAD of each spectrum.

    x, y shape:
        n_pixels x n_wavenumbers
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mx = np.nanmedian(x, axis=1, keepdims=True)
    my = np.nanmedian(y, axis=1, keepdims=True)

    x_centered = x - mx
    y_centered = y - my

    mad_x = np.nanmedian(np.abs(x_centered), axis=1, keepdims=True)
    mad_y = np.nanmedian(np.abs(y_centered), axis=1, keepdims=True)

    valid = (mad_x[:, 0] > eps) & (mad_y[:, 0] > eps)

    ux = x_centered / (c * mad_x + eps)
    uy = y_centered / (c * mad_y + eps)

    wx = (1 - ux ** 2) ** 2
    wy = (1 - uy ** 2) ** 2

    wx[np.abs(ux) >= 1] = 0
    wy[np.abs(uy) >= 1] = 0

    xw = x_centered * wx
    yw = y_centered * wy

    num = np.nansum(xw * yw, axis=1)

    den = (
        np.sqrt(np.nansum(xw ** 2, axis=1))
        * np.sqrt(np.nansum(yw ** 2, axis=1))
        + eps
    )

    bicor = num / den
    bicor[~valid] = np.nan

    return bicor

def compute_pixelwise_similarity_maps(
    srs_cube,
    raman_cube_interp,
    mask=None,
    spectrum_normalization="minmax",
    clip_negative=False,
    min_signal_percentile=5,
    eps=1e-12,
    compute_pearson=True,
    compute_spearman=True,
    compute_bicor=True,
    bicor_c=9.0,
):
    """
    Compute pixel-wise spectral similarity maps between SRS and Raman.

    Recommended main mode:
        spectrum_normalization="minmax"

    Meaning:
        Each pixel spectrum is independently normalized to [0, 1]
        over the common spectral range before metrics are computed.

    Metrics returned:
        cosine     : cosine similarity, higher is better.
        sam_deg    : spectral angle in degrees, lower is better.
        medae      : median absolute error, lower is better.
        mae        : mean absolute error, lower is better.
        rmse       : root mean square error, lower is better.
        pearson    : standard Pearson correlation.
        spearman   : rank-based correlation, more robust to amplitude outliers.
        bicor      : biweight midcorrelation, robust Pearson-like correlation.
    """

    if srs_cube.shape != raman_cube_interp.shape:
        raise ValueError(
            f"SRS and Raman cubes must have the same shape. "
            f"Got {srs_cube.shape} and {raman_cube_interp.shape}."
        )

    H, W, N = srs_cube.shape

    # ------------------------------------------------------------
    # Normalize each pixel spectrum
    # ------------------------------------------------------------
    srs_norm = normalize_cube_spectra(
        srs_cube,
        method=spectrum_normalization,
        mask=None,
        clip_negative=clip_negative,
        eps=eps,
    )

    ram_norm = normalize_cube_spectra(
        raman_cube_interp,
        method=spectrum_normalization,
        mask=None,
        clip_negative=clip_negative,
        eps=eps,
    )

    srs_flat = srs_norm.reshape(-1, N)
    ram_flat = ram_norm.reshape(-1, N)

    # ------------------------------------------------------------
    # Valid pixels
    # ------------------------------------------------------------
    if mask is None:
        valid = np.ones(H * W, dtype=bool)
    else:
        valid = mask.ravel().copy()

    valid &= np.isfinite(srs_flat).all(axis=1)
    valid &= np.isfinite(ram_flat).all(axis=1)

    # Remove very weak pixels before min-max normalization becomes misleading
    raw_signal = (
        np.nanmean(np.abs(srs_cube), axis=-1)
        + np.nanmean(np.abs(raman_cube_interp), axis=-1)
    )

    raw_signal_flat = raw_signal.ravel()

    if np.any(valid) and min_signal_percentile is not None and min_signal_percentile > 0:
        signal_thr = np.nanpercentile(
            raw_signal_flat[valid],
            min_signal_percentile
        )
        valid &= raw_signal_flat > signal_thr

    # ------------------------------------------------------------
    # Allocate metric maps
    # ------------------------------------------------------------
    cosine = np.full(H * W, np.nan)
    sam_deg = np.full(H * W, np.nan)
    medae = np.full(H * W, np.nan)
    mae = np.full(H * W, np.nan)
    rmse = np.full(H * W, np.nan)
    pearson = np.full(H * W, np.nan)
    spearman = np.full(H * W, np.nan)
    bicor = np.full(H * W, np.nan)

    # ------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------
    if np.any(valid):
        x = srs_flat[valid]
        y = ram_flat[valid]

        # ----------------------------
        # Cosine similarity
        # ----------------------------
        dot = np.sum(x * y, axis=1)
        nx = np.sqrt(np.sum(x ** 2, axis=1))
        ny = np.sqrt(np.sum(y ** 2, axis=1))

        cos = dot / (nx * ny + eps)
        cos = np.clip(cos, -1, 1)

        cosine[valid] = cos

        # Spectral angle mapper, in degrees
        sam_deg[valid] = np.degrees(np.arccos(cos))

        # ----------------------------
        # Error metrics
        # ----------------------------
        abs_err = np.abs(x - y)

        medae[valid] = np.nanmedian(abs_err, axis=1)
        mae[valid] = np.nanmean(abs_err, axis=1)
        rmse[valid] = np.sqrt(np.nanmean((x - y) ** 2, axis=1))

        # ----------------------------
        # Standard Pearson
        # ----------------------------
        if compute_pearson:
            pearson[valid] = _rowwise_pearson(x, y, eps=eps)

        # ----------------------------
        # Spearman correlation
        # ----------------------------
        if compute_spearman:
            # Spearman = Pearson correlation computed on ranks.
            # rankdata works row-wise with axis=1 in recent SciPy versions.
            x_rank = rankdata(x, axis=1)
            y_rank = rankdata(y, axis=1)

            spearman[valid] = _rowwise_pearson(
                x_rank,
                y_rank,
                eps=eps
            )

        # ----------------------------
        # Biweight midcorrelation
        # ----------------------------
        if compute_bicor:
            bicor[valid] = _rowwise_biweight_midcorrelation(
                x,
                y,
                c=bicor_c,
                eps=eps
            )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------
    metric_maps = {
        "cosine": cosine.reshape(H, W),
        "sam_deg": sam_deg.reshape(H, W),
        "medae": medae.reshape(H, W),
        "mae": mae.reshape(H, W),
        "rmse": rmse.reshape(H, W),
        "pearson": pearson.reshape(H, W),
        "spearman": spearman.reshape(H, W),
        "bicor": bicor.reshape(H, W),
        "valid_mask": valid.reshape(H, W),
    }

    summary_df = summarize_metric_maps(metric_maps)

    normalized_cubes = {
        "srs_norm": srs_norm,
        "raman_norm": ram_norm,
    }

    return metric_maps, summary_df, normalized_cubes

def summarize_metric_maps(metric_maps):
    """
    Summarize each metric map with robust statistics.
    """

    rows = []

    for name, img in metric_maps.items():
        if name == "valid_mask":
            continue

        vals = img[np.isfinite(img)]

        if vals.size == 0:
            row = {
                "metric": name,
                "n_pixels": 0,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "q10": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "q90": np.nan,
            }
        else:
            row = {
                "metric": name,
                "n_pixels": vals.size,
                "mean": np.nanmean(vals),
                "median": np.nanmedian(vals),
                "std": np.nanstd(vals),
                "q10": np.nanpercentile(vals, 10),
                "q25": np.nanpercentile(vals, 25),
                "q75": np.nanpercentile(vals, 75),
                "q90": np.nanpercentile(vals, 90),
            }

        rows.append(row)

    return pd.DataFrame(rows)


def flatten_metric_maps_to_dataframe(metric_maps, sample_name=None):
    """
    Flatten metric maps to a dataframe.
    Useful for boxplots or external analysis.
    """

    valid = metric_maps["valid_mask"]

    out = {}

    for name, img in metric_maps.items():
        if name == "valid_mask":
            continue
        out[name] = img[valid]

    df = pd.DataFrame(out)

    if sample_name is not None:
        df.insert(0, "sample", sample_name)

    return df


# ============================================================
# Plot utilities
# ============================================================

def _safe_cmap(cmap_name, bad_color="white"):
    """
    Return a copy of a colormap with NaNs shown as bad_color.
    """
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(bad_color)
    return cmap


def _imshow_with_colorbar(
    fig,
    ax,
    img,
    title,
    cmap,
    vmin=None,
    vmax=None,
    cbar_label=None,
    add_colorbar=True,
):
    """
    Helper for consistent image panels.
    """

    im = ax.imshow(
        img,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.axis("off")

    if add_colorbar:
        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        cbar.ax.tick_params(labelsize=7)

        if cbar_label is not None:
            cbar.set_label(cbar_label, fontsize=8)

    return im


def _plot_histogram(
    ax,
    values,
    title,
    xlabel,
    bins=50,
    xlim=None,
):
    """
    Plot metric distribution.
    """

    values = values[np.isfinite(values)]

    ax.hist(values, bins=bins, alpha=0.85)

    if values.size > 0:
        med = np.nanmedian(values)
        ax.axvline(med, linestyle="--", linewidth=1.0)
        ax.text(
            0.97,
            0.95,
            f"median = {med:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pixel count")

    if xlim is not None:
        ax.set_xlim(*xlim)

    return ax


# ============================================================
# Figure 4
# ============================================================

def plot_figure4_pixelwise_similarity(
    srs_cube,
    raman_cube_interp,
    wn,
    metric_maps,
    mask=None,
    overview_band=1445,
    overview_width=12,
    overview_source="srs",
    sample_name="sample",
    spectrum_normalization="minmax",

    # Choose which metrics to show.
    # Possible options:
    # "cosine", "pearson", "spearman", "bicor", "sam_deg", "medae", "mae", "rmse"
    metrics_to_plot=("cosine", "pearson", "spearman", "bicor", "medae"),

    # Optional overrides
    metric_vlims=None,
    metric_cmaps=None,

    # Overview image display
    cmap_overview="afmhot",
    pmin_overview=1,
    pmax_overview=99.8,

    # Figure options
    add_colorbar=True,
    add_scalebar_flag=True,
    scale_bar_um=100,
    pixel_size_um=None,
    fov_um=None,
    scalebar_on_metric_maps=False,
    savepath=None,
):
    """
    Create Figure 4-style plot with user-selected pixel-wise similarity/error metrics.

    Layout:
        Top row:
            A) overview image
            B...) selected metric maps

        Bottom row:
            info panel
            corresponding metric distributions

    Metrics can be selected dynamically using metrics_to_plot.

    Examples
    --------
    metrics_to_plot=("cosine", "pearson", "spearman", "bicor", "medae")
    metrics_to_plot=("cosine", "bicor", "medae")
    metrics_to_plot=("cosine", "pearson", "medae")
    metrics_to_plot=("cosine", "sam_deg", "medae")

    Notes
    -----
    The function only plots metrics already present in metric_maps.
    Therefore, if you want to plot "spearman" or "bicor", they must be computed
    beforehand in compute_pixelwise_similarity_maps.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if metric_vlims is None:
        metric_vlims = {}

    if metric_cmaps is None:
        metric_cmaps = {}

    # ------------------------------------------------------------
    # Default display settings for known metrics
    # ------------------------------------------------------------
    default_metric_settings = {
        "cosine": {
            "title": "Pixel-wise cosine similarity",
            "cbar_label": "cosine",
            "hist_title": "Cosine distribution",
            "hist_xlabel": "cosine similarity",
            "cmap": "viridis",
            "vmin": 0.70,
            "vmax": 1.00,
            "median_label": "Median cosine",
            "median_fmt": ".2f",
        },

        "pearson": {
            "title": "Pixel-wise Pearson correlation",
            "cbar_label": "Pearson r",
            "hist_title": "Pearson distribution",
            "hist_xlabel": "Pearson r",
            "cmap": "coolwarm",
            "vmin": -1.00,
            "vmax": 1.00,
            "median_label": "Median Pearson",
            "median_fmt": ".2f",
        },

        "spearman": {
            "title": "Pixel-wise Spearman correlation",
            "cbar_label": "Spearman ρ",
            "hist_title": "Spearman distribution",
            "hist_xlabel": "Spearman ρ",
            "cmap": "coolwarm",
            "vmin": -1.00,
            "vmax": 1.00,
            "median_label": "Median Spearman",
            "median_fmt": ".2f",
        },

        "bicor": {
            "title": "Pixel-wise robust correlation",
            "cbar_label": "biweight r",
            "hist_title": "Biweight correlation distribution",
            "hist_xlabel": "biweight midcorrelation",
            "cmap": "coolwarm",
            "vmin": -1.00,
            "vmax": 1.00,
            "median_label": "Median biweight r",
            "median_fmt": ".2f",
        },

        "sam_deg": {
            "title": "Spectral angle",
            "cbar_label": "degrees",
            "hist_title": "Spectral angle distribution",
            "hist_xlabel": "degrees",
            "cmap": "magma",
            "vmin": 0.00,
            "vmax": None,
            "median_label": "Median angle",
            "median_fmt": ".1f",
        },

        "medae": {
            "title": "Median absolute spectral error",
            "cbar_label": "a.u.",
            "hist_title": "MedAE distribution",
            "hist_xlabel": "median absolute error",
            "cmap": "magma",
            "vmin": 0.00,
            "vmax": None,
            "median_label": "Median MedAE",
            "median_fmt": ".3f",
        },

        "mae": {
            "title": "Mean absolute spectral error",
            "cbar_label": "a.u.",
            "hist_title": "MAE distribution",
            "hist_xlabel": "mean absolute error",
            "cmap": "magma",
            "vmin": 0.00,
            "vmax": None,
            "median_label": "Median MAE",
            "median_fmt": ".3f",
        },

        "rmse": {
            "title": "Root mean square spectral error",
            "cbar_label": "a.u.",
            "hist_title": "RMSE distribution",
            "hist_xlabel": "RMSE",
            "cmap": "magma",
            "vmin": 0.00,
            "vmax": None,
            "median_label": "Median RMSE",
            "median_fmt": ".3f",
        },
    }

    metrics_to_plot = tuple(metrics_to_plot)

    if len(metrics_to_plot) < 1:
        raise ValueError("metrics_to_plot must contain at least one metric.")

    for metric in metrics_to_plot:
        if metric not in default_metric_settings:
            raise KeyError(
                f"No display settings found for metric '{metric}'. "
                f"Available metrics are: {list(default_metric_settings.keys())}"
            )

        if metric not in metric_maps:
            raise KeyError(
                f"Metric '{metric}' not found in metric_maps. "
                f"Available keys are: {list(metric_maps.keys())}. "
                f"If you want to plot '{metric}', compute it first in "
                f"compute_pixelwise_similarity_maps."
            )

    # ------------------------------------------------------------
    # Overview image
    # ------------------------------------------------------------
    if overview_source == "srs":
        overview_cube = srs_cube
        overview_title = f"SRS {overview_band} cm$^{{-1}}$"
    elif overview_source == "raman":
        overview_cube = raman_cube_interp
        overview_title = f"Raman {overview_band} cm$^{{-1}}$"
    else:
        raise ValueError("overview_source must be 'srs' or 'raman'")

    overview_img = get_band_image(
        overview_cube,
        wn,
        center=overview_band,
        width=overview_width,
    )

    overview_vmin, overview_vmax = percentile_vmin_vmax(
        overview_img,
        pmin=pmin_overview,
        pmax=pmax_overview,
    )

    # ------------------------------------------------------------
    # Valid mask
    # ------------------------------------------------------------
    if "valid_mask" in metric_maps:
        valid_mask = metric_maps["valid_mask"].astype(bool)
    elif mask is not None:
        valid_mask = mask.astype(bool)
    else:
        valid_mask = np.ones(srs_cube.shape[:2], dtype=bool)

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------
    n_metrics = len(metrics_to_plot)
    n_cols = n_metrics + 1

    # Dynamic width: good for 3-5 metrics
    fig_width = max(10.0, 2.85 * n_cols + 0.8)

    fig = plt.figure(
        figsize=(fig_width, 6.5),
        constrained_layout=True
    )

    width_ratios = [0.90] + [1.00] * n_metrics

    gs = GridSpec(
        2,
        n_cols,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=[1.00, 0.72],
    )

    ax_overview = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[1, 0])

    metric_axes = {}
    hist_axes = {}

    for j, metric in enumerate(metrics_to_plot, start=1):
        metric_axes[metric] = fig.add_subplot(gs[0, j])
        hist_axes[metric] = fig.add_subplot(gs[1, j])

    # ------------------------------------------------------------
    # Panel A: overview
    # ------------------------------------------------------------
    _imshow_with_colorbar(
        fig,
        ax_overview,
        overview_img,
        overview_title,
        cmap=_safe_cmap(cmap_overview),
        vmin=overview_vmin,
        vmax=overview_vmax,
        cbar_label=None,
        add_colorbar=add_colorbar,
    )

    if add_scalebar_flag:
        add_scalebar(
            ax_overview,
            overview_img.shape,
            pixel_size_um=pixel_size_um,
            fov_um=fov_um,
            scale_bar_um=scale_bar_um,
            location="lower right",
            linewidth=1.5,
            fontsize=7,
        )

    # ------------------------------------------------------------
    # Metric panels and histograms
    # ------------------------------------------------------------
    median_lines = []
    metric_values = {}

    for metric in metrics_to_plot:
        settings = default_metric_settings[metric].copy()

        # Colormap override
        cmap = metric_cmaps.get(metric, settings["cmap"])

        # vmin/vmax override
        if metric in metric_vlims:
            vmin, vmax = metric_vlims[metric]
        else:
            vmin = settings["vmin"]
            vmax = settings["vmax"]

        metric_map = metric_maps[metric].astype(float).copy()
        metric_map[~valid_mask] = np.nan

        vals = metric_map[np.isfinite(metric_map)]
        metric_values[metric] = vals

        # Auto vmax if None
        if vmax is None:
            if vals.size > 0:
                vmax = np.nanpercentile(vals, 99)
            else:
                vmax = 1.0

        # Auto vmin if None
        if vmin is None:
            if vals.size > 0:
                vmin = np.nanpercentile(vals, 1)
            else:
                vmin = 0.0

        # Image map
        _imshow_with_colorbar(
            fig,
            metric_axes[metric],
            metric_map,
            settings["title"],
            cmap=_safe_cmap(cmap),
            vmin=vmin,
            vmax=vmax,
            cbar_label=settings["cbar_label"],
            add_colorbar=add_colorbar,
        )

        if add_scalebar_flag and scalebar_on_metric_maps:
            add_scalebar(
                metric_axes[metric],
                metric_map.shape,
                pixel_size_um=pixel_size_um,
                fov_um=fov_um,
                scale_bar_um=scale_bar_um,
                location="lower right",
                linewidth=1.5,
                fontsize=7,
            )

        # Histogram
        _plot_histogram(
            hist_axes[metric],
            vals,
            settings["hist_title"],
            settings["hist_xlabel"],
            bins=50,
            xlim=(vmin, vmax),
        )

        if vals.size > 0:
            med = np.nanmedian(vals)
            fmt = settings["median_fmt"]
            median_lines.append(
                f"{settings['median_label']}: {med:{fmt}}"
            )
        else:
            median_lines.append(
                f"{settings['median_label']}: NaN"
            )

    # ------------------------------------------------------------
    # Info panel
    # ------------------------------------------------------------
    n_valid = int(np.sum(valid_mask))

    info_text = (
        f"{sample_name}\n"
        f"Spectral norm.: {spectrum_normalization}\n"
        f"Valid pixels: {n_valid:,}\n\n"
        + "\n".join(median_lines)
    )

    ax_info.axis("off")
    ax_info.text(
        0.02,
        0.98,
        info_text,
        ha="left",
        va="top",
        fontsize=8,
        transform=ax_info.transAxes,
    )

    # ------------------------------------------------------------
    # Suptitle
    # ------------------------------------------------------------
    fig.suptitle(
        f"{sample_name} | pixel-wise SRS/Raman spectral-shape agreement",
        y=1.02,
        fontsize=10,
    )

    if savepath is not None:
        fig.savefig(savepath, dpi=600, bbox_inches="tight")

    return fig
# ============================================================
# Optional: multi-sample summary plot
# ============================================================

def plot_similarity_boxplot_across_samples(
    metric_df,
    metric="cosine",
    sample_col="sample",
    savepath=None,
):
    """
    Plot boxplot of one metric across multiple samples.

    metric_df should be made by concatenating outputs from:
        flatten_metric_maps_to_dataframe(...)
    """

    samples = metric_df[sample_col].unique()
    data = [
        metric_df.loc[metric_df[sample_col] == s, metric].dropna().values
        for s in samples
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.boxplot(
        data,
        labels=samples,
        showfliers=False,
    )

    ax.set_ylabel(metric)
    ax.set_title(f"Pixel-wise {metric} across samples")
    ax.tick_params(axis="x", rotation=35)

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=600, bbox_inches="tight")

    return fig