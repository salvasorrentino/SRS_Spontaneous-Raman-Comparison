import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from Script_for_Figures.utils_fig2 import (
    add_scalebar,
    percentile_vmin_vmax,
)

from Script_for_Figures.utils_fig4 import (
    normalize_cube_spectra,
)


# ============================================================
# Metric definitions
# ============================================================

METRIC_ALIASES = {
    "cosine": "cosine",
    "cosine similarity": "cosine",

    "pearson": "pearson",
    "pearson correlation": "pearson",

    "spearman": "spearman",
    "spearman correlation": "spearman",

    "bicor": "bicor",
    "biweight": "bicor",
    "biweight correlation": "bicor",
}


METRIC_LABELS = {
    "cosine": "Cosine similarity",
    "pearson": "Pearson correlation",
    "spearman": "Spearman correlation",
    "bicor": "Biweight correlation",
}


def resolve_metric(metric_maps, metric):
    """
    Resolve a user-provided metric name to the corresponding key
    in metric_maps.

    Allowed canonical metrics:
        cosine
        pearson
        spearman
        bicor
    """

    metric_input = str(metric).lower().strip()

    if metric_input not in METRIC_ALIASES:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Choose among: cosine, pearson, spearman, bicor."
        )

    metric_key = METRIC_ALIASES[metric_input]

    if metric_key not in metric_maps:
        raise KeyError(
            f"Metric '{metric_key}' not found in metric_maps.\n"
            f"Available keys: {list(metric_maps.keys())}"
        )

    return metric_key


def get_metric_label(metric):
    """
    Return publication-friendly name for a metric.
    """

    metric_input = str(metric).lower().strip()

    if metric_input in METRIC_ALIASES:
        metric_key = METRIC_ALIASES[metric_input]
    else:
        metric_key = metric_input

    return METRIC_LABELS.get(metric_key, metric_key)


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


def _safe_cmap(cmap_name, bad_color="white"):
    """
    Return a copy of a matplotlib colormap with NaNs displayed
    using bad_color.
    """

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(bad_color)

    return cmap


def _get_valid_metric_values(
    metric_maps,
    metric,
    mask=None,
):
    """
    Return:
        metric_key,
        metric_map,
        valid_mask,
        valid metric values.
    """

    metric_key = resolve_metric(
        metric_maps,
        metric,
    )

    if "valid_mask" not in metric_maps:
        raise KeyError(
            "metric_maps must contain 'valid_mask'."
        )

    metric_map = np.asarray(
        metric_maps[metric_key],
        dtype=float,
    )

    valid_mask = np.asarray(
        metric_maps["valid_mask"],
        dtype=bool,
    )

    if metric_map.shape != valid_mask.shape:
        raise ValueError(
            "Metric map and valid_mask have different shapes: "
            f"{metric_map.shape} vs {valid_mask.shape}"
        )

    if mask is not None:

        mask = np.asarray(
            mask,
            dtype=bool,
        )

        if mask.shape != valid_mask.shape:
            raise ValueError(
                "Additional mask and valid_mask have different shapes: "
                f"{mask.shape} vs {valid_mask.shape}"
            )

        valid_mask = valid_mask & mask

    finite_valid_mask = (
        valid_mask
        & np.isfinite(metric_map)
    )

    metric_vals = metric_map[
        finite_valid_mask
    ]

    if metric_vals.size == 0:
        raise ValueError(
            f"No valid values are available for metric "
            f"'{metric_key}'."
        )

    return (
        metric_key,
        metric_map,
        finite_valid_mask,
        metric_vals,
    )


# ============================================================
# Random selection from arbitrary metric quantiles
# ============================================================

def select_random_pixels_from_metric_quantiles(
    metric_maps,
    metric="cosine",
    mask=None,
    top_quantile=75,
    bottom_quantile=25,
    n_points_per_group=1,
    random_state=0,
):
    """
    Randomly select pixels from the upper and lower quantiles
    of a chosen pixel-wise similarity/correlation metric.

    Parameters
    ----------
    metric_maps : dict
        Output of compute_pixelwise_similarity_maps(...).

        Expected to contain:
            "valid_mask"

        and the requested metric, e.g.:
            "cosine"
            "pearson"
            "spearman"
            "bicor"

    metric : str
        Metric used for pixel selection.

        Allowed:
            "cosine"
            "pearson"
            "spearman"
            "bicor"

    mask : ndarray or None
        Optional additional boolean mask.

    top_quantile : float
        Percentile defining the high-similarity group.

        Example:
            75 -> pixels >= 75th percentile,
            i.e. top 25%.

    bottom_quantile : float
        Percentile defining the low-similarity group.

        Example:
            25 -> pixels <= 25th percentile,
            i.e. bottom 25%.

    n_points_per_group : int
        Number of randomly selected pixels per group.

    random_state : int
        Seed for reproducible random selection.

    Returns
    -------
    selected_df : pd.DataFrame

        Columns:
            group
            y
            x
            metric
            metric_value
            lower_threshold
            upper_threshold
            bottom_quantile
            top_quantile
    """

    (
        metric_key,
        metric_map,
        valid_mask,
        metric_vals,
    ) = _get_valid_metric_values(
        metric_maps=metric_maps,
        metric=metric,
        mask=mask,
    )

    # --------------------------------------------------------
    # Quantile thresholds
    # --------------------------------------------------------

    lower_threshold = np.nanpercentile(
        metric_vals,
        bottom_quantile,
    )

    upper_threshold = np.nanpercentile(
        metric_vals,
        top_quantile,
    )

    # --------------------------------------------------------
    # Candidate pixels
    # --------------------------------------------------------

    bottom_candidates = np.argwhere(
        valid_mask
        & (metric_map <= lower_threshold)
    )

    top_candidates = np.argwhere(
        valid_mask
        & (metric_map >= upper_threshold)
    )

    if bottom_candidates.shape[0] < n_points_per_group:
        raise ValueError(
            f"Not enough bottom candidates for metric "
            f"'{metric_key}'. "
            f"Found {bottom_candidates.shape[0]}, "
            f"requested {n_points_per_group}."
        )

    if top_candidates.shape[0] < n_points_per_group:
        raise ValueError(
            f"Not enough top candidates for metric "
            f"'{metric_key}'. "
            f"Found {top_candidates.shape[0]}, "
            f"requested {n_points_per_group}."
        )

    # --------------------------------------------------------
    # Reproducible random sampling
    # --------------------------------------------------------

    rng = np.random.default_rng(
        random_state
    )

    bottom_idx = rng.choice(
        bottom_candidates.shape[0],
        size=n_points_per_group,
        replace=False,
    )

    top_idx = rng.choice(
        top_candidates.shape[0],
        size=n_points_per_group,
        replace=False,
    )

    rows = []

    # --------------------------------------------------------
    # Bottom group
    # --------------------------------------------------------

    for idx in bottom_idx:

        y, x = bottom_candidates[idx]

        rows.append(
            {
                "group": "bottom",
                "y": int(y),
                "x": int(x),
                "metric": metric_key,
                "metric_value": float(
                    metric_map[y, x]
                ),
                "lower_threshold": float(
                    lower_threshold
                ),
                "upper_threshold": float(
                    upper_threshold
                ),
                "bottom_quantile": float(
                    bottom_quantile
                ),
                "top_quantile": float(
                    top_quantile
                ),
            }
        )

    # --------------------------------------------------------
    # Top group
    # --------------------------------------------------------

    for idx in top_idx:

        y, x = top_candidates[idx]

        rows.append(
            {
                "group": "top",
                "y": int(y),
                "x": int(x),
                "metric": metric_key,
                "metric_value": float(
                    metric_map[y, x]
                ),
                "lower_threshold": float(
                    lower_threshold
                ),
                "upper_threshold": float(
                    upper_threshold
                ),
                "bottom_quantile": float(
                    bottom_quantile
                ),
                "top_quantile": float(
                    top_quantile
                ),
            }
        )

    selected_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Put top examples first for plotting
    # --------------------------------------------------------

    selected_df["group_order"] = (
        selected_df["group"].map(
            {
                "top": 0,
                "bottom": 1,
            }
        )
    )

    selected_df = (
        selected_df
        .sort_values(
            [
                "group_order",
                "y",
                "x",
            ]
        )
        .drop(
            columns="group_order"
        )
        .reset_index(
            drop=True
        )
    )

    return selected_df


# ============================================================
# Optional manual replacement of a selected pixel
# ============================================================

def replace_selected_pixel(
    selected_df,
    metric_maps,
    group,
    y,
    x,
    mask=None,
    require_same_quantile=True,
):
    """
    Safely replace the selected pixel of a specified group.

    Unlike manually changing y and x in selected_df, this
    function also recalculates metric_value.

    Parameters
    ----------
    selected_df : pd.DataFrame
        Output of select_random_pixels_from_metric_quantiles().

    metric_maps : dict
        Pixel-wise metric maps.

    group : str
        "top" or "bottom".

    y, x : int
        Coordinates of the replacement pixel.

    mask : ndarray or None
        Optional foreground mask.

    require_same_quantile : bool
        If True, verify that the manually selected pixel still
        belongs to the requested top/bottom quantile group.

    Returns
    -------
    updated_df : pd.DataFrame
        Copy of selected_df with the pixel replaced.
    """

    if group not in {"top", "bottom"}:
        raise ValueError(
            "group must be 'top' or 'bottom'."
        )

    if selected_df.empty:
        raise ValueError(
            "selected_df is empty."
        )

    matches = (
        selected_df["group"] == group
    )

    if matches.sum() == 0:
        raise ValueError(
            f"No '{group}' point exists in selected_df."
        )

    # Assume a single metric is used for the whole dataframe
    metric_key = selected_df.iloc[0]["metric"]

    (
        metric_key,
        metric_map,
        valid_mask,
        _,
    ) = _get_valid_metric_values(
        metric_maps=metric_maps,
        metric=metric_key,
        mask=mask,
    )

    y = int(y)
    x = int(x)

    if (
        y < 0
        or x < 0
        or y >= metric_map.shape[0]
        or x >= metric_map.shape[1]
    ):
        raise IndexError(
            f"Pixel ({y}, {x}) is outside the metric map "
            f"with shape {metric_map.shape}."
        )

    if not valid_mask[y, x]:
        raise ValueError(
            f"Pixel ({y}, {x}) is not valid according to "
            f"the similarity mask."
        )

    metric_value = float(
        metric_map[y, x]
    )

    lower_threshold = float(
        selected_df.loc[
            matches,
            "lower_threshold"
        ].iloc[0]
    )

    upper_threshold = float(
        selected_df.loc[
            matches,
            "upper_threshold"
        ].iloc[0]
    )

    if require_same_quantile:

        if (
            group == "top"
            and metric_value < upper_threshold
        ):
            raise ValueError(
                f"Pixel ({y}, {x}) has {metric_key} = "
                f"{metric_value:.4f}, which is below the "
                f"upper-quantile threshold "
                f"{upper_threshold:.4f}."
            )

        if (
            group == "bottom"
            and metric_value > lower_threshold
        ):
            raise ValueError(
                f"Pixel ({y}, {x}) has {metric_key} = "
                f"{metric_value:.4f}, which is above the "
                f"lower-quantile threshold "
                f"{lower_threshold:.4f}."
            )

    updated_df = selected_df.copy()

    idx = updated_df.index[
        matches
    ][0]

    updated_df.loc[
        idx,
        "y"
    ] = y

    updated_df.loc[
        idx,
        "x"
    ] = x

    updated_df.loc[
        idx,
        "metric_value"
    ] = metric_value

    return updated_df


# ============================================================
# Extract spectra at selected pixels
# ============================================================

def extract_selected_pixel_spectra(
    srs_cube,
    raman_cube_interp,
    wn,
    selected_df,
    spectrum_normalization="minmax",
    eps=1e-12,
):
    """
    Extract raw and normalized spectra at selected pixel
    positions.

    Parameters
    ----------
    srs_cube : ndarray
        SRS cube on common spectral grid.

    raman_cube_interp : ndarray
        Interpolated spontaneous Raman cube on the SRS grid.

    wn : ndarray
        Common Raman-shift axis.

    selected_df : pd.DataFrame
        Output of
        select_random_pixels_from_metric_quantiles(...).

    spectrum_normalization : str
        Normalization applied independently to every spectrum.

    eps : float
        Numerical stability constant.

    Returns
    -------
    spectra_df : pd.DataFrame
        Long-format dataframe.
    """

    # --------------------------------------------------------
    # Normalize entire cubes exactly as used for comparison
    # --------------------------------------------------------

    srs_norm = normalize_cube_spectra(
        srs_cube,
        method=spectrum_normalization,
        mask=None,
        eps=eps,
    )

    ram_norm = normalize_cube_spectra(
        raman_cube_interp,
        method=spectrum_normalization,
        mask=None,
        eps=eps,
    )

    rows = []

    # --------------------------------------------------------
    # Extract each selected pixel
    # --------------------------------------------------------

    for _, row in selected_df.iterrows():

        group = row["group"]

        y = int(
            row["y"]
        )

        x = int(
            row["x"]
        )

        metric = row["metric"]

        metric_value = float(
            row["metric_value"]
        )

        srs_raw = srs_cube[
            y,
            x,
            :
        ]

        ram_raw = raman_cube_interp[
            y,
            x,
            :
        ]

        srs_n = srs_norm[
            y,
            x,
            :
        ]

        ram_n = ram_norm[
            y,
            x,
            :
        ]

        for (
            w,
            sr,
            rr,
            sn,
            rn,
        ) in zip(
            wn,
            srs_raw,
            ram_raw,
            srs_n,
            ram_n,
        ):

            rows.append(
                {
                    "group": group,
                    "y": y,
                    "x": x,
                    "metric": metric,
                    "metric_value": metric_value,
                    "wn": float(w),
                    "srs_raw": float(sr),
                    "raman_raw": float(rr),
                    "srs_norm": float(sn),
                    "raman_norm": float(rn),
                }
            )

    spectra_df = pd.DataFrame(
        rows
    )

    return spectra_df


# ============================================================
# Plot Figure 6
# ============================================================

def plot_figure6_selected_pixel_spectra(
    srs_cube,
    raman_cube_interp,
    wn,
    metric_maps,
    selected_df,
    spectra_df,
    mask=None,
    sample_name="sample",
    selection_metric="cosine",
    overview_band=1445,
    overview_width=12,
    overview_source="srs",
    cmap_overview="afmhot",
    pmin_overview=1,
    pmax_overview=99.8,
    add_colorbar=True,
    add_scalebar_flag=True,
    scale_bar_um=100,
    pixel_size_um=None,
    fov_um=None,
    random_state=None,
    plot_normalized=True,
    savepath=None,
):
    """
    Create Figure 6:

        A) overview image with selected pixels
        B) distribution of the selected similarity metric
        C) spectrum from a random top-quantile pixel
        D) spectrum from a random bottom-quantile pixel

    The metric is controlled by selection_metric and can be:

        cosine
        pearson
        spearman
        bicor
    """

    # ========================================================
    # Basic validation
    # ========================================================

    if selected_df.empty:
        raise ValueError(
            "selected_df is empty."
        )

    metric_key = resolve_metric(
        metric_maps,
        selection_metric,
    )

    metric_label = get_metric_label(
        metric_key
    )

    # Verify that selected_df was actually generated
    # using the same metric requested for plotting.
    selected_metrics = (
        selected_df["metric"]
        .astype(str)
        .unique()
    )

    if (
        len(selected_metrics) != 1
        or selected_metrics[0] != metric_key
    ):
        raise ValueError(
            "selected_df and selection_metric are inconsistent.\n"
            f"selection_metric = '{metric_key}'\n"
            f"selected_df metric(s) = {selected_metrics}"
        )

    # ========================================================
    # Choose overview image
    # ========================================================

    if overview_source == "srs":

        overview_cube = srs_cube

        overview_title = (
            f"SRS {overview_band} cm$^{{-1}}$"
        )

    elif overview_source == "raman":

        overview_cube = raman_cube_interp

        overview_title = (
            f"Raman {overview_band} cm$^{{-1}}$"
        )

    else:

        raise ValueError(
            "overview_source must be 'srs' or 'raman'."
        )

    overview_img = get_band_image(
        overview_cube,
        wn,
        center=overview_band,
        width=overview_width,
    )

    (
        overview_vmin,
        overview_vmax,
    ) = percentile_vmin_vmax(
        overview_img,
        pmin=pmin_overview,
        pmax=pmax_overview,
    )

    # ========================================================
    # Metric values and quantile thresholds
    # ========================================================

    (
        metric_key,
        metric_map,
        valid_mask,
        metric_vals,
    ) = _get_valid_metric_values(
        metric_maps=metric_maps,
        metric=selection_metric,
        mask=mask,
    )

    # Use thresholds stored during selection.
    lower_threshold = float(
        selected_df[
            "lower_threshold"
        ].iloc[0]
    )

    upper_threshold = float(
        selected_df[
            "upper_threshold"
        ].iloc[0]
    )

    bottom_quantile = float(
        selected_df[
            "bottom_quantile"
        ].iloc[0]
    )

    top_quantile = float(
        selected_df[
            "top_quantile"
        ].iloc[0]
    )

    bottom_fraction = int(
        round(bottom_quantile)
    )

    top_fraction = int(
        round(100 - top_quantile)
    )

    # ========================================================
    # Plot layout
    # ========================================================

    fig = plt.figure(
        figsize=(11.5, 7.2),
        constrained_layout=True,
    )

    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[
            1.0,
            1.1,
        ],
        height_ratios=[
            1.0,
            1.0,
        ],
    )

    ax_img = fig.add_subplot(
        gs[0, 0]
    )

    ax_hist = fig.add_subplot(
        gs[0, 1]
    )

    ax_top = fig.add_subplot(
        gs[1, 0]
    )

    ax_bottom = fig.add_subplot(
        gs[1, 1]
    )

    # ========================================================
    # Panel A: overview image
    # ========================================================

    im = ax_img.imshow(
        overview_img,
        cmap=_safe_cmap(
            cmap_overview
        ),
        vmin=overview_vmin,
        vmax=overview_vmax,
    )

    ax_img.set_title(
        overview_title
    )

    ax_img.axis(
        "off"
    )

    if add_colorbar:

        cbar = fig.colorbar(
            im,
            ax=ax_img,
            fraction=0.046,
            pad=0.04,
        )

        cbar.ax.tick_params(
            labelsize=7
        )

    if add_scalebar_flag:

        add_scalebar(
            ax_img,
            overview_img.shape,
            pixel_size_um=pixel_size_um,
            fov_um=fov_um,
            scale_bar_um=scale_bar_um,
            location="lower right",
            linewidth=1.5,
            fontsize=7,
        )

    # --------------------------------------------------------
    # Marker styles
    # --------------------------------------------------------

    marker_specs = {
        "top": {
            "color": "limegreen",
            "marker": "o",
        },
        "bottom": {
            "color": "magenta",
            "marker": "X",
        },
    }

    # --------------------------------------------------------
    # Selected pixel markers
    # --------------------------------------------------------

    for _, row in selected_df.iterrows():

        group = row["group"]

        y = int(
            row["y"]
        )

        x = int(
            row["x"]
        )

        spec = marker_specs[
            group
        ]

        ax_img.scatter(
            x,
            y,
            s=90,
            facecolors="none",
            edgecolors=spec["color"],
            linewidths=2.0,
            marker=spec["marker"],
            zorder=20,
        )

        ax_img.text(
            x + 5,
            y - 5,
            f"{group}\n({y}, {x})",
            color=spec["color"],
            fontsize=7,
            ha="left",
            va="bottom",
            bbox=dict(
                facecolor="black",
                edgecolor="none",
                alpha=0.45,
                pad=2,
            ),
            zorder=21,
        )

    # --------------------------------------------------------
    # Manual legend
    # --------------------------------------------------------

    ax_img.text(
        0.02,
        0.98,
        (
            f"Green circle: top {top_fraction}%\n"
            f"Magenta X: bottom {bottom_fraction}%"
        ),
        transform=ax_img.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
            pad=2,
        ),
    )

    # ========================================================
    # Panel B: metric histogram
    # ========================================================

    ax_hist.hist(
        metric_vals,
        bins=50,
        alpha=0.85,
    )

    ax_hist.axvline(
        lower_threshold,
        linestyle="--",
        linewidth=1.0,
        label=f"{bottom_quantile:g}th percentile",
    )

    ax_hist.axvline(
        upper_threshold,
        linestyle="--",
        linewidth=1.0,
        label=f"{top_quantile:g}th percentile",
    )

    # --------------------------------------------------------
    # Highlight selected examples
    # --------------------------------------------------------

    for _, row in selected_df.iterrows():

        group = row["group"]

        metric_value = float(
            row["metric_value"]
        )

        spec = marker_specs[
            group
        ]

        ax_hist.axvline(
            metric_value,
            color=spec["color"],
            linewidth=1.5,
            alpha=0.9,
        )

        ax_hist.scatter(
            [metric_value],
            [0],
            color=spec["color"],
            s=40,
            zorder=20,
            label=f"{group} example",
        )

    ax_hist.set_xlabel(
        f"Pixel-wise {metric_label.lower()}"
    )

    ax_hist.set_ylabel(
        "Pixel count"
    )

    ax_hist.set_title(
        f"{metric_label} distribution"
    )

    # Remove duplicate legend entries
    handles, labels = (
        ax_hist.get_legend_handles_labels()
    )

    unique = dict(
        zip(
            labels,
            handles,
        )
    )

    ax_hist.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        fontsize=7,
    )

    # ========================================================
    # Panels C and D: spectra
    # ========================================================

    def _plot_single_selected_spectrum(
        ax,
        group_name,
        title_prefix,
    ):

        sub = selected_df[
            selected_df["group"] == group_name
        ]

        if sub.empty:

            ax.axis(
                "off"
            )

            return

        row = sub.iloc[0]

        y = int(
            row["y"]
        )

        x = int(
            row["x"]
        )

        metric_value = float(
            row["metric_value"]
        )

        spec_df = spectra_df[
            (
                spectra_df["group"]
                == group_name
            )
            & (
                spectra_df["y"]
                == y
            )
            & (
                spectra_df["x"]
                == x
            )
        ].sort_values(
            "wn"
        )

        if spec_df.empty:

            raise ValueError(
                f"No spectrum found for selected pixel "
                f"{group_name}: ({y}, {x})."
            )

        # ----------------------------------------------------
        # Plot normalized spectra
        # ----------------------------------------------------

        if plot_normalized:

            ax.plot(
                spec_df["wn"],
                spec_df["raman_norm"],
                label="Confocal spontaneous Raman",
                linewidth=1.5,
                color="0.25",
            )

            ax.plot(
                spec_df["wn"],
                spec_df["srs_norm"],
                label="Hyperspectral SRS",
                linewidth=1.6,
                linestyle="--",
                color="crimson",
            )

            ylabel = (
                "Normalized intensity"
            )

        # ----------------------------------------------------
        # Plot raw spectra
        # ----------------------------------------------------

        else:

            ax.plot(
                spec_df["wn"],
                spec_df["raman_raw"],
                label="Confocal spontaneous Raman",
                linewidth=1.5,
                color="0.25",
            )

            ax.plot(
                spec_df["wn"],
                spec_df["srs_raw"],
                label="Hyperspectral SRS",
                linewidth=1.6,
                linestyle="--",
                color="crimson",
            )

            ylabel = (
                "Intensity (a.u.)"
            )

        ax.set_xlabel(
            "Raman shift (cm$^{-1}$)"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            f"{title_prefix}\n"
            f"(y={y}, x={x}) | "
            f"{metric_label} = {metric_value:.2f}"
        )

        ax.legend(
            frameon=False,
            fontsize=7,
        )

    # --------------------------------------------------------
    # Top spectrum
    # --------------------------------------------------------

    _plot_single_selected_spectrum(
        ax_top,
        group_name="top",
        title_prefix=(
            f"Random example from "
            f"top {top_fraction}%"
        ),
    )

    # --------------------------------------------------------
    # Bottom spectrum
    # --------------------------------------------------------

    _plot_single_selected_spectrum(
        ax_bottom,
        group_name="bottom",
        title_prefix=(
            f"Random example from "
            f"bottom {bottom_fraction}%"
        ),
    )

    # ========================================================
    # Suptitle
    # ========================================================

    extra_title = ""

    if random_state is not None:

        extra_title = (
            f" | random seed: "
            f"{random_state}"
        )

    fig.suptitle(
        (
            f"{sample_name} | "
            f"selected pixel spectra from "
            f"{metric_label.lower()} quantiles"
            f"{extra_title}"
        ),
        y=1.02,
        fontsize=10,
    )

    # ========================================================
    # Save
    # ========================================================

    if savepath is not None:

        fig.savefig(
            savepath,
            dpi=600,
            bbox_inches="tight",
        )

    return fig


# ============================================================
# Backward-compatible wrapper
# ============================================================

def select_random_pixels_from_cosine_quantiles(
    metric_maps,
    mask=None,
    top_quantile=75,
    bottom_quantile=25,
    n_points_per_group=1,
    random_state=0,
):
    """
    Backward-compatible wrapper for older scripts.

    Equivalent to calling:
        select_random_pixels_from_metric_quantiles(
            metric="cosine"
        )
    """

    return select_random_pixels_from_metric_quantiles(
        metric_maps=metric_maps,
        metric="cosine",
        mask=mask,
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
        n_points_per_group=n_points_per_group,
        random_state=random_state,
    )