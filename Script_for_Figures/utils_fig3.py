import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

from scipy.stats import pearsonr
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, disk

from Script_for_Figures.utils_fig2 import (
    robust_minmax,
    percentile_vmin_vmax,
    symmetric_percentile_vmin_vmax,
    add_scalebar,
    save_publication_figure,
)


# ============================================================
# Basic spectral utilities
# ============================================================

def get_band_image(cube, wn, center, width=10):
    """
    Integrate or average a spectral band around center +/- width/2.

    Parameters
    ----------
    cube : ndarray
        Hyperspectral cube, shape H x W x N.
    wn : ndarray
        Raman shift axis, shape N.
    center : float
        Band center in cm^-1.
    width : float
        Band width in cm^-1.

    Returns
    -------
    img : ndarray
        2D band image.
    """
    wn = np.asarray(wn)

    keep = (wn >= center - width / 2) & (wn <= center + width / 2)

    if keep.sum() == 0:
        idx = np.argmin(np.abs(wn - center))
        return cube[..., idx]

    return np.nanmean(cube[..., keep], axis=-1)


def normalize_cube_spectra(cube, method="minmax", mask=None, eps=1e-12):
    """
    Normalize each pixel spectrum independently.

    Parameters
    ----------
    cube : ndarray
        Hyperspectral cube, shape H x W x N.

    method : {"none", "minmax", "l1", "l2", "zscore"}
        "none"   : no spectral normalization.
        "minmax" : each pixel spectrum scaled to [0, 1].
        "l1"     : each pixel spectrum divided by sum(abs(spectrum)).
        "l2"     : each pixel spectrum divided by sqrt(sum(spectrum^2)).
        "zscore" : each pixel spectrum centered and divided by std.

    mask : ndarray or None
        Optional boolean foreground mask, shape H x W.
        Pixels outside the mask are set to NaN.

    Returns
    -------
    cube_norm : ndarray
        Normalized cube.
    """
    cube = cube.astype(float).copy()

    if method == "none":
        cube_norm = cube

    elif method == "minmax":
        spec_min = np.nanmin(cube, axis=-1, keepdims=True)
        spec_max = np.nanmax(cube, axis=-1, keepdims=True)
        cube_norm = (cube - spec_min) / (spec_max - spec_min + eps)

    elif method == "l1":
        norm = np.nansum(np.abs(cube), axis=-1, keepdims=True)
        cube_norm = cube / (norm + eps)

    elif method == "l2":
        norm = np.sqrt(np.nansum(cube ** 2, axis=-1, keepdims=True))
        cube_norm = cube / (norm + eps)

    elif method == "zscore":
        mean = np.nanmean(cube, axis=-1, keepdims=True)
        std = np.nanstd(cube, axis=-1, keepdims=True)
        cube_norm = (cube - mean) / (std + eps)

    else:
        raise ValueError("method must be 'none', 'minmax', 'l1', 'l2', or 'zscore'")

    if mask is not None:
        cube_norm[~mask, :] = np.nan

    return cube_norm


def normalize_1d_spectrum(y, method="none", eps=1e-12):
    """
    Normalize a 1D spectrum for plotting.

    Parameters
    ----------
    method : {"none", "max", "minmax", "area", "l2"}
    """
    y = np.asarray(y, dtype=float).copy()

    if method == "none":
        return y

    elif method == "max":
        return y / (np.nanmax(np.abs(y)) + eps)

    elif method == "minmax":
        ymin = np.nanmin(y)
        ymax = np.nanmax(y)
        return (y - ymin) / (ymax - ymin + eps)

    elif method == "area":
        return y / (np.nansum(np.abs(y)) + eps)

    elif method == "l2":
        return y / (np.sqrt(np.nansum(y ** 2)) + eps)

    else:
        raise ValueError("method must be 'none', 'max', 'minmax', 'area', or 'l2'")


def mean_spectrum(
    cube,
    mask=None,
    statistic="mean",
    return_std=True,
):
    """
    Compute mean or median spectrum over masked pixels.

    Parameters
    ----------
    cube : ndarray
        Hyperspectral cube, shape H x W x N.
    mask : ndarray or None
        Boolean mask, shape H x W.
    statistic : {"mean", "median"}
        Statistic used across pixels.
    return_std : bool
        If True, also returns standard deviation across pixels.

    Returns
    -------
    spec : ndarray
        Mean or median spectrum.
    std : ndarray
        Standard deviation spectrum, if return_std=True.
    """
    if mask is None:
        pixels = cube.reshape(-1, cube.shape[-1])
    else:
        pixels = cube[mask]

    pixels = pixels[np.isfinite(pixels).all(axis=1)]

    if pixels.size == 0:
        spec = np.full(cube.shape[-1], np.nan)
        std = np.full(cube.shape[-1], np.nan)
        return spec, std

    if statistic == "mean":
        spec = np.nanmean(pixels, axis=0)
    elif statistic == "median":
        spec = np.nanmedian(pixels, axis=0)
    else:
        raise ValueError("statistic must be 'mean' or 'median'")

    std = np.nanstd(pixels, axis=0)

    if return_std:
        return spec, std

    return spec


def compute_curve_metrics(y1, y2, eps=1e-12):
    """
    Compute simple similarity metrics between two 1D spectra.
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    keep = np.isfinite(y1) & np.isfinite(y2)

    y1 = y1[keep]
    y2 = y2[keep]

    if y1.size < 3:
        return {
            "pearson": np.nan,
            "cosine": np.nan,
            "rmse": np.nan,
        }

    if np.nanstd(y1) > eps and np.nanstd(y2) > eps:
        pearson = pearsonr(y1, y2)[0]
    else:
        pearson = np.nan

    cosine = np.sum(y1 * y2) / (
        np.sqrt(np.sum(y1 ** 2)) * np.sqrt(np.sum(y2 ** 2)) + eps
    )

    rmse = np.sqrt(np.mean((y1 - y2) ** 2))

    return {
        "pearson": pearson,
        "cosine": cosine,
        "rmse": rmse,
    }


# ============================================================
# Mask and ROI utilities
# ============================================================

def make_foreground_mask_from_cubes(
    srs_cube,
    raman_cube_interp=None,
    method="mean",
    threshold_method="otsu",
    min_size=500,
    opening_radius=2,
):
    """
    Create a simple foreground mask using spectral summary images.

    Parameters
    ----------
    srs_cube : ndarray
        SRS cube, shape H x W x N.
    raman_cube_interp : ndarray or None
        Raman cube interpolated on SRS grid, shape H x W x N.
    method : {"mean", "median", "max"}
        Spectral projection used to build the mask.
    threshold_method : {"otsu", "percentile"}
        Thresholding method.
    min_size : int
        Remove small objects below this area.
    opening_radius : int
        Radius for binary opening.

    Returns
    -------
    mask : ndarray
        Boolean foreground mask.
    """

    def project(cube):
        if method == "mean":
            return np.nanmean(np.maximum(cube, 0), axis=-1)
        elif method == "median":
            return np.nanmedian(np.maximum(cube, 0), axis=-1)
        elif method == "max":
            return np.nanmax(cube, axis=-1)
        else:
            raise ValueError("method must be 'mean', 'median', or 'max'")

    srs_img = robust_minmax(project(srs_cube))

    if raman_cube_interp is not None:
        ram_img = robust_minmax(project(raman_cube_interp))
        summary = srs_img + ram_img
    else:
        summary = srs_img

    vals = summary[np.isfinite(summary)]

    if threshold_method == "otsu":
        thr = threshold_otsu(vals)
    elif threshold_method == "percentile":
        thr = np.nanpercentile(vals, 20)
    else:
        raise ValueError("threshold_method must be 'otsu' or 'percentile'")

    mask = summary > thr

    if opening_radius is not None and opening_radius > 0:
        mask = binary_opening(mask, disk(opening_radius))

    if min_size is not None and min_size > 0:
        mask = remove_small_objects(mask, min_size=min_size)

    return mask


def make_rect_mask(image_shape, rect):
    """
    Create a boolean rectangular ROI mask.

    Parameters
    ----------
    image_shape : tuple
        Shape of 2D image or cube, e.g. (H, W) or (H, W, N).
    rect : tuple
        Rectangle in the format (y0, y1, x0, x1).

    Returns
    -------
    mask : ndarray
        Boolean mask, shape H x W.
    """
    H, W = image_shape[:2]
    y0, y1, x0, x1 = rect

    y0 = max(0, int(y0))
    y1 = min(H, int(y1))
    x0 = max(0, int(x0))
    x1 = min(W, int(x1))

    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True

    return mask


def make_rect_roi_masks(image_shape, roi_rects):
    """
    Create ROI masks from a dictionary of rectangles.

    Parameters
    ----------
    roi_rects : dict
        Example:
        {
            "ROI 1": (50, 150, 70, 180),
            "ROI 2": (250, 350, 300, 420),
        }

    Returns
    -------
    roi_masks : list
    roi_names : list
    """
    roi_names = []
    roi_masks = []

    for name, rect in roi_rects.items():
        roi_names.append(name)
        roi_masks.append(make_rect_mask(image_shape, rect))

    return roi_masks, roi_names


def add_roi_boxes(
    ax,
    roi_rects,
    image_shape=None,
    labels=True,
    linewidth=1.5,
    text_size=7,
):
    """
    Add rectangular ROI boxes to an image axis.

    Parameters
    ----------
    ax : matplotlib axis
        Axis containing the image.
    roi_rects : dict
        Dictionary with ROI rectangles:
        {"ROI name": (y0, y1, x0, x1)}
    image_shape : tuple or None
        Shape of the image, e.g. (H, W). If provided, ROI boxes are clipped
        to image boundaries and the axis limits are restored.
    labels : bool
        If True, add ROI labels.
    """

    if image_shape is not None:
        H, W = image_shape[:2]
    else:
        H, W = None, None

    for idx, (name, rect) in enumerate(roi_rects.items()):
        y0, y1, x0, x1 = rect

        if image_shape is not None:
            # Clip ROI coordinates to image boundaries
            y0c = max(0, int(y0))
            y1c = min(H, int(y1))
            x0c = max(0, int(x0))
            x1c = min(W, int(x1))

            # Skip ROIs completely outside the image
            if y1c <= y0c or x1c <= x0c:
                print(f"Skipping {name}: ROI outside image boundaries.")
                continue
        else:
            y0c, y1c, x0c, x1c = int(y0), int(y1), int(x0), int(x1)

        color = f"C{idx}"

        box = patches.Rectangle(
            (x0c, y0c),
            x1c - x0c,
            y1c - y0c,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            zorder=20,
            clip_on=True,
        )

        ax.add_patch(box)

        if labels:
            text_y = max(0, y0c - 5) if image_shape is not None else y0c - 5

            txt = ax.text(
                x0c,
                text_y,
                name,
                color=color,
                fontsize=text_size,
                ha="left",
                va="bottom",
                zorder=21,
                clip_on=True,
            )

            txt.set_path_effects([
                pe.Stroke(linewidth=2, foreground="black"),
                pe.Normal()
            ])

    # Important: prevent Matplotlib from expanding the axis
    # when ROI boxes are close to or outside the image boundary.
    if image_shape is not None:
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)

    return ax


# ============================================================
# Figure 3 plotting
# ============================================================

def prepare_cubes_for_mean_spectra(
    srs_cube,
    raman_cube_interp,
    mask=None,
    spectrum_normalization="minmax",
):
    """
    Apply per-pixel spectral normalization before computing mean spectra.

    This is the recommended mode for spectral-shape comparison:
    each pixel spectrum is normalized first, then spectra are averaged.
    """
    srs_norm = normalize_cube_spectra(
        srs_cube,
        method=spectrum_normalization,
        mask=mask
    )

    ram_norm = normalize_cube_spectra(
        raman_cube_interp,
        method=spectrum_normalization,
        mask=mask
    )

    return srs_norm, ram_norm


def export_mean_and_roi_spectra_to_csv(
    srs_cube,
    raman_cube_interp,
    wn,
    output_csv,
    mask=None,
    roi_masks=None,
    roi_names=None,
    roi_rects=None,
    spectrum_normalization="minmax",
    statistic="mean",
):
    """
    Export global and ROI mean spectra to CSV.
    Useful for reproducibility and later plotting.
    """

    if roi_rects is not None:
        roi_masks, roi_names = make_rect_roi_masks(
            srs_cube.shape,
            roi_rects
        )

    srs_spec_cube, ram_spec_cube = prepare_cubes_for_mean_spectra(
        srs_cube,
        raman_cube_interp,
        mask=mask,
        spectrum_normalization=spectrum_normalization,
    )

    out = pd.DataFrame({"wn": wn})

    srs_mean, _ = mean_spectrum(
        srs_spec_cube,
        mask=mask,
        statistic=statistic,
        return_std=True
    )

    ram_mean, _ = mean_spectrum(
        ram_spec_cube,
        mask=mask,
        statistic=statistic,
        return_std=True
    )

    out["global_raman"] = ram_mean
    out["global_srs"] = srs_mean

    if roi_masks is not None and roi_names is not None:
        for roi_mask, roi_name in zip(roi_masks, roi_names):
            if mask is not None:
                roi_mask_use = roi_mask & mask
            else:
                roi_mask_use = roi_mask

            srs_roi_mean, _ = mean_spectrum(
                srs_spec_cube,
                mask=roi_mask_use,
                statistic=statistic,
                return_std=True
            )

            ram_roi_mean, _ = mean_spectrum(
                ram_spec_cube,
                mask=roi_mask_use,
                statistic=statistic,
                return_std=True
            )

            safe_name = roi_name.replace(" ", "_")
            out[f"{safe_name}_raman"] = ram_roi_mean
            out[f"{safe_name}_srs"] = srs_roi_mean

    out.to_csv(output_csv, index=False)

    return out

import numpy as np
from scipy.stats import wasserstein_distance


def _to_probability_spectrum(y, eps=1e-12):
    """
    Convert a 1D spectrum into a non-negative probability distribution
    for Wasserstein distance.
    """
    y = np.asarray(y, dtype=float).copy()
    y[~np.isfinite(y)] = np.nan

    # clip negatives
    y = np.maximum(y, 0)

    s = np.nansum(y)
    if s <= eps:
        return np.full_like(y, np.nan)

    return y / s


def compute_curve_metrics(
    y1,
    y2,
    wn=None,
    eps=1e-12,
):
    """
    Compute similarity / distance metrics between two 1D spectra.

    Returned metrics:
    - cosine
    - wasserstein
    - medae
    - rmse
    - pearson (optional diagnostic)
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    keep = np.isfinite(y1) & np.isfinite(y2)

    if wn is not None:
        wn = np.asarray(wn)
        keep &= np.isfinite(wn)

    y1 = y1[keep]
    y2 = y2[keep]

    if wn is not None:
        wn_use = wn[keep]
    else:
        wn_use = np.arange(len(y1), dtype=float)

    if y1.size < 3:
        return {
            "cosine": np.nan,
            "wasserstein": np.nan,
            "medae": np.nan,
            "rmse": np.nan,
            "pearson": np.nan,
        }

    # cosine
    cosine = np.sum(y1 * y2) / (
        np.sqrt(np.sum(y1 ** 2)) * np.sqrt(np.sum(y2 ** 2)) + eps
    )

    # robust absolute errors
    abs_err = np.abs(y1 - y2)
    medae = np.nanmedian(abs_err)
    rmse = np.sqrt(np.nanmean((y1 - y2) ** 2))

    # pearson kept only as diagnostic
    if np.nanstd(y1) > eps and np.nanstd(y2) > eps:
        pearson = np.corrcoef(y1, y2)[0, 1]
    else:
        pearson = np.nan

    # Wasserstein distance on non-negative probability spectra
    p1 = _to_probability_spectrum(y1, eps=eps)
    p2 = _to_probability_spectrum(y2, eps=eps)

    if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
        wasser = np.nan
    else:
        wasser = wasserstein_distance(
            wn_use,
            wn_use,
            u_weights=p1,
            v_weights=p2
        )

    return {
        "cosine": cosine,
        "wasserstein": wasser,
        "medae": medae,
        "rmse": rmse,
        "pearson": pearson,
    }

def plot_figure3_mean_and_roi_spectra_stacked(
    srs_cube,
    raman_cube_interp,
    wn,
    mask=None,
    roi_masks=None,
    roi_names=None,
    roi_rects=None,
    overview_band=1445,
    overview_width=12,
    overview_source="srs",
    sample_name="sample",
    spectrum_normalization="minmax",
    statistic="mean",
    curve_normalization="none",
    cmap="afmhot",
    pmin=1,
    pmax=99.8,
    add_colorbar=True,
    add_scalebar_flag=True,
    scale_bar_um=100,
    pixel_size_um=None,
    fov_um=None,
    roi_colors=None,
    srs_color="crimson",
    global_raman_color="0.25",
    global_srs_color="crimson",
    show_cosine=True,
    savepath=None,
):
    """
    Figure 3 layout:
        Top row:
            A) smaller overview image with ROI boxes
            B) global mean Raman/SRS spectrum

        Bottom row:
            C-E) one separate ROI spectrum plot per ROI

    Spectral interpretation:
        Each pixel spectrum can be normalized first, then spectra are averaged.
        With spectrum_normalization="minmax", the plotted spectra represent
        average spectral shapes, not absolute intensities.

    Color convention:
        - Global Raman: dark gray
        - Global SRS: red dashed
        - ROI Raman: ROI-specific color
        - ROI SRS: red dashed
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.patheffects as pe
    from matplotlib.gridspec import GridSpec

    # ------------------------------------------------------------
    # ROI masks from rectangles, if provided
    # ------------------------------------------------------------
    if roi_rects is not None:
        roi_masks, roi_names = make_rect_roi_masks(
            srs_cube.shape,
            roi_rects
        )

    has_rois = (
        roi_masks is not None
        and roi_names is not None
        and len(roi_masks) > 0
    )

    if has_rois:
        n_rois = len(roi_masks)
    else:
        n_rois = 0

    # ------------------------------------------------------------
    # ROI colors
    # ------------------------------------------------------------
    if roi_colors is None:
        # Avoid orange because SRS red/orange can become visually too similar.
        default_colors = [
            "tab:blue",
            "tab:green",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:cyan",
        ]
        roi_colors = default_colors[:max(n_rois, 1)]

    # ------------------------------------------------------------
    # Helper to draw ROI boxes with matching colors
    # ------------------------------------------------------------
    def _add_roi_boxes_matching_colors(
        ax,
        roi_rects,
        image_shape,
        roi_colors,
        labels=True,
        linewidth=1.5,
        text_size=7,
    ):
        H, W = image_shape[:2]

        for idx, (name, rect) in enumerate(roi_rects.items()):
            y0, y1, x0, x1 = rect

            # Clip coordinates to image boundaries
            y0c = max(0, int(y0))
            y1c = min(H, int(y1))
            x0c = max(0, int(x0))
            x1c = min(W, int(x1))

            if y1c <= y0c or x1c <= x0c:
                print(f"Skipping {name}: ROI outside image boundaries.")
                continue

            color = roi_colors[idx % len(roi_colors)]

            box = patches.Rectangle(
                (x0c, y0c),
                x1c - x0c,
                y1c - y0c,
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
                zorder=20,
                clip_on=True,
            )
            ax.add_patch(box)

            if labels:
                text_y = max(0, y0c - 5)

                txt = ax.text(
                    x0c,
                    text_y,
                    name,
                    color=color,
                    fontsize=text_size,
                    ha="left",
                    va="bottom",
                    zorder=21,
                    clip_on=True,
                    weight="bold",
                )

                txt.set_path_effects([
                    pe.Stroke(linewidth=2, foreground="black"),
                    pe.Normal()
                ])

        # Prevent Matplotlib from expanding axes because of ROI boxes
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)

        return ax

    # ------------------------------------------------------------
    # Normalize spectra before computing mean spectra
    # ------------------------------------------------------------
    srs_spec_cube, ram_spec_cube = prepare_cubes_for_mean_spectra(
        srs_cube,
        raman_cube_interp,
        mask=mask,
        spectrum_normalization=spectrum_normalization,
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
    elif overview_source == "mean":
        overview_cube = 0.5 * (
            normalize_cube_spectra(srs_cube, method="minmax") +
            normalize_cube_spectra(raman_cube_interp, method="minmax")
        )
        overview_title = f"Mean SRS/Raman {overview_band} cm$^{{-1}}$"
    else:
        raise ValueError("overview_source must be 'srs', 'raman', or 'mean'")

    overview_img = get_band_image(
        overview_cube,
        wn,
        center=overview_band,
        width=overview_width
    )

    overview_vmin, overview_vmax = percentile_vmin_vmax(
        overview_img,
        pmin=pmin,
        pmax=pmax
    )

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------
    if has_rois:
        fig = plt.figure(figsize=(13.5, 6.8), constrained_layout=True)

        # 4 columns:
        # first column is narrow for the overview image,
        # remaining columns are used by the large global spectrum.
        gs = GridSpec(
            2,
            4,
            figure=fig,
            width_ratios=[0.80, 1.20, 1.20, 1.20],
            height_ratios=[0.88, 1.00],
        )

        ax_img = fig.add_subplot(gs[0, 0])
        ax_global = fig.add_subplot(gs[0, 1:])

        # Bottom row: three equal ROI panels across the full figure width
        bottom_gs = gs[1, :].subgridspec(1, 3, wspace=0.20)
        roi_axes = [fig.add_subplot(bottom_gs[0, i]) for i in range(3)]

    else:
        fig = plt.figure(figsize=(10.0, 3.8), constrained_layout=True)
        gs = GridSpec(
            1,
            4,
            figure=fig,
            width_ratios=[0.80, 1.20, 1.20, 1.20],
        )

        ax_img = fig.add_subplot(gs[0, 0])
        ax_global = fig.add_subplot(gs[0, 1:])
        roi_axes = []

    # ------------------------------------------------------------
    # Panel A: overview image
    # ------------------------------------------------------------
    im = ax_img.imshow(
        overview_img,
        cmap=cmap,
        vmin=overview_vmin,
        vmax=overview_vmax
    )

    ax_img.set_title(overview_title, fontsize=8)
    ax_img.axis("off")

    if add_colorbar:
        cbar = fig.colorbar(
            im,
            ax=ax_img,
            fraction=0.046,
            pad=0.04
        )
        cbar.ax.tick_params(labelsize=7)

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

    if roi_rects is not None:
        _add_roi_boxes_matching_colors(
            ax_img,
            roi_rects,
            image_shape=overview_img.shape,
            roi_colors=roi_colors,
            labels=True,
            linewidth=1.5,
            text_size=7,
        )

    # ------------------------------------------------------------
    # Panel B: global mean spectrum
    # ------------------------------------------------------------
    srs_mean, _ = mean_spectrum(
        srs_spec_cube,
        mask=mask,
        statistic=statistic,
        return_std=True
    )

    ram_mean, _ = mean_spectrum(
        ram_spec_cube,
        mask=mask,
        statistic=statistic,
        return_std=True
    )

    srs_plot = normalize_1d_spectrum(
        srs_mean,
        method=curve_normalization
    )

    ram_plot = normalize_1d_spectrum(
        ram_mean,
        method=curve_normalization
    )

    global_metrics = compute_curve_metrics(
        ram_plot,
        srs_plot,
        wn=wn
    )

    ax_global.plot(
        wn,
        ram_plot,
        label="Confocal spontaneous Raman",
        linewidth=1.5,
        linestyle="-",
        color=global_raman_color,
    )

    ax_global.plot(
        wn,
        srs_plot,
        label="Hyperspectral SRS",
        linewidth=1.6,
        linestyle="--",
        color=global_srs_color,
    )

    ax_global.set_xlabel("Raman shift (cm$^{-1}$)")
    ax_global.set_ylabel("Normalized intensity")
    ax_global.set_title("Mean tissue spectrum")
    ax_global.legend(
        frameon=False,
        fontsize=8,
        loc="upper left"
    )

    if show_cosine:
        txt = f"cos = {global_metrics['cosine']:.2f}"

        ax_global.text(
            0.98,
            0.05,
            txt,
            transform=ax_global.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.80,
                pad=2
            )
        )

    # ------------------------------------------------------------
    # Bottom row: ROI spectra
    # ------------------------------------------------------------
    if has_rois:
        n_plot = min(3, len(roi_masks))

        for idx, (ax, roi_mask, roi_name) in enumerate(
            zip(roi_axes[:n_plot], roi_masks[:n_plot], roi_names[:n_plot])
        ):
            roi_color = roi_colors[idx % len(roi_colors)]

            if mask is not None:
                roi_mask_use = roi_mask & mask
            else:
                roi_mask_use = roi_mask

            srs_roi_mean, _ = mean_spectrum(
                srs_spec_cube,
                mask=roi_mask_use,
                statistic=statistic,
                return_std=True
            )

            ram_roi_mean, _ = mean_spectrum(
                ram_spec_cube,
                mask=roi_mask_use,
                statistic=statistic,
                return_std=True
            )

            srs_roi_plot = normalize_1d_spectrum(
                srs_roi_mean,
                method=curve_normalization
            )

            ram_roi_plot = normalize_1d_spectrum(
                ram_roi_mean,
                method=curve_normalization
            )

            roi_metrics = compute_curve_metrics(
                ram_roi_plot,
                srs_roi_plot,
                wn=wn
            )

            # Confocal/Raman: ROI-specific color
            ax.plot(
                wn,
                ram_roi_plot,
                linewidth=1.4,
                linestyle="-",
                color=roi_color,
            )

            # SRS: red dashed
            ax.plot(
                wn,
                srs_roi_plot,
                linewidth=1.5,
                linestyle="--",
                color=srs_color,
            )

            ax.set_title(
                roi_name,
                color=roi_color,
                fontsize=9,
                weight="bold"
            )

            ax.set_xlabel("Raman shift (cm$^{-1}$)")

            if idx == 0:
                ax.set_ylabel("Norm. intensity")
            else:
                ax.set_ylabel("")

            # No repeated legend inside ROI panels.
            # Add only a very small modality cue in the first ROI panel.
            if idx == 0:
                ax.text(
                    0.03,
                    0.96,
                    "solid: Raman\nred dashed: SRS",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.75,
                        pad=2
                    )
                )

            if show_cosine:
                txt = f"cos = {roi_metrics['cosine']:.2f}"

                ax.text(
                    0.98,
                    0.05,
                    txt,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.80,
                        pad=2
                    )
                )

        for ax in roi_axes[n_plot:]:
            ax.axis("off")

    # ------------------------------------------------------------
    # Suptitle
    # ------------------------------------------------------------
    fig.suptitle(
        f"{sample_name} | spectral normalization: {spectrum_normalization}",
        y=1.02,
        fontsize=10,
    )

    if savepath is not None:
        fig.savefig(savepath, dpi=600, bbox_inches="tight")

    return fig