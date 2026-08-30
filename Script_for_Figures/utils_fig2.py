import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.metrics import structural_similarity as ssim


# -----------------------------
# Utility
# -----------------------------

def ensure_ascending_wn(cube, wn):
    """
    Ensures wavenumber axis is ascending.
    cube shape: H x W x N
    wn shape: N
    """
    wn = np.asarray(wn)
    if wn[0] > wn[-1]:
        wn = wn[::-1]
        cube = cube[..., ::-1]
    return cube, wn


def crop_to_overlap(srs_cube, srs_wn, raman_cube, raman_wn):
    """
    Keeps only SRS shifts within the Raman spontaneous spectral range.
    Raman will later be interpolated onto this SRS grid.
    """
    srs_cube, srs_wn = ensure_ascending_wn(srs_cube, srs_wn)
    raman_cube, raman_wn = ensure_ascending_wn(raman_cube, raman_wn)

    lo = max(np.min(srs_wn), np.min(raman_wn))
    hi = min(np.max(srs_wn), np.max(raman_wn))

    keep = (srs_wn >= lo) & (srs_wn <= hi)

    return srs_cube[..., keep], srs_wn[keep], raman_cube, raman_wn


def interpolate_raman_to_srs_grid(raman_cube, raman_wn, srs_wn):
    """
    Interpolates Raman spontaneous cube onto SRS wavenumber grid.
    Input Raman cube: H x W x Nraman
    Output Raman interp cube: H x W x Nsrs
    """
    H, W, N = raman_cube.shape
    flat = raman_cube.reshape(-1, N)

    f = interp1d(
        raman_wn,
        flat,
        kind="linear",
        axis=1,
        bounds_error=False,
        fill_value=np.nan
    )

    interp_flat = f(srs_wn)
    return interp_flat.reshape(H, W, len(srs_wn))


def robust_minmax(x, p_low=1, p_high=99):
    """
    Robust normalization to [0, 1].
    Useful for visualization of images.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanpercentile(x, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)


def vector_normalize_spectra(cube, eps=1e-12):
    """
    Per-pixel L2 normalization.
    Use for spectral-shape comparison, not for absolute intensity comparison.
    """
    norm = np.sqrt(np.nansum(cube ** 2, axis=-1, keepdims=True))
    return cube / (norm + eps)


def area_normalize_spectra(cube, eps=1e-12):
    """
    Per-pixel area normalization.
    More intensity-preserving than min-max, but still shape-oriented.
    """
    area = np.nansum(np.abs(cube), axis=-1, keepdims=True)
    return cube / (area + eps)


def percentile_baseline_subtract(cube, percentile=5):
    """
    Very simple per-pixel baseline subtraction.
    Conservative option: subtract a low percentile instead of fitting a polynomial.
    """
    baseline = np.nanpercentile(cube, percentile, axis=-1, keepdims=True)
    corrected = cube - baseline
    corrected[corrected < 0] = 0
    return corrected


def make_foreground_mask(srs_cube, raman_cube_interp, min_size=200):
    """
    Builds a foreground mask from combined SRS and Raman total signal.
    This avoids artificially high similarity in empty/background regions.
    """
    srs_total = np.nanmean(srs_cube, axis=-1)
    ram_total = np.nanmean(raman_cube_interp, axis=-1)

    combined = robust_minmax(srs_total) + robust_minmax(ram_total)

    thresh = threshold_otsu(combined[np.isfinite(combined)])
    mask = combined > thresh

    mask = binary_opening(mask, disk(2))
    mask = remove_small_objects(mask, min_size=min_size)

    return mask


# -----------------------------
# Main preprocessing function
# -----------------------------

def prepare_pair(
        srs_cube,
        srs_wn,
        raman_cube,
        raman_wn,
        baseline=True,
        normalization="vector"
):
    """
    Returns SRS cube and Raman cube on same SRS spectral grid.
    Normalization is applied per pixel.

    normalization options:
        "vector" : best for cosine similarity
        "area"   : useful sensitivity check
        "none"   : no per-pixel normalization
    """
    srs_cube, srs_wn, raman_cube, raman_wn = crop_to_overlap(
        srs_cube, srs_wn, raman_cube, raman_wn
    )

    raman_interp = interpolate_raman_to_srs_grid(raman_cube, raman_wn, srs_wn)

    valid = np.isfinite(srs_cube).all(axis=-1) & np.isfinite(raman_interp).all(axis=-1)

    if baseline:
        srs_cube = percentile_baseline_subtract(srs_cube, percentile=5)
        raman_interp = percentile_baseline_subtract(raman_interp, percentile=5)

    if normalization == "vector":
        srs_proc = vector_normalize_spectra(srs_cube)
        raman_proc = vector_normalize_spectra(raman_interp)
    elif normalization == "area":
        srs_proc = area_normalize_spectra(srs_cube)
        raman_proc = area_normalize_spectra(raman_interp)
    elif normalization == "none":
        srs_proc = srs_cube.copy()
        raman_proc = raman_interp.copy()
    else:
        raise ValueError("normalization must be 'vector', 'area', or 'none'")

    mask = make_foreground_mask(srs_cube, raman_interp)
    mask = mask & valid

    return srs_proc, raman_proc, srs_wn, mask

# utils_fig2.py

from pathlib import Path
import matplotlib.pyplot as plt


def set_publication_style(
    notebook_dpi=150,
    save_dpi=300,
    font_size=8,
    axes_label_size=8,
    axes_title_size=9,
    tick_label_size=7,
    legend_size=7,
):
    """
    Set Matplotlib parameters for publication-quality figures.

    Parameters
    ----------
    notebook_dpi : int
        Resolution used for displaying figures in notebooks.
    save_dpi : int
        Default resolution used when saving raster figures.
    font_size : int
        Base font size.
    axes_label_size : int
        Axis label font size.
    axes_title_size : int
        Axis title font size.
    tick_label_size : int
        Tick label font size.
    legend_size : int
        Legend font size.
    """

    plt.rcParams.update({
        "figure.dpi": notebook_dpi,
        "savefig.dpi": save_dpi,

        "font.size": font_size,
        "axes.labelsize": axes_label_size,
        "axes.titlesize": axes_title_size,
        "xtick.labelsize": tick_label_size,
        "ytick.labelsize": tick_label_size,
        "legend.fontsize": legend_size,

        # Keeps text editable in Illustrator/Inkscape for PDF/PS exports
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Cleaner default output
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_publication_figure(
    fig,
    filename,
    output_dir="figures",
    dpi=300,
    formats=("png", "pdf"),
    bbox_inches="tight",
    pad_inches=0.02,
    close=False,
):
    """
    Save a Matplotlib figure in publication-ready formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    filename : str
        Base filename without extension, or with extension if only one format is needed.
    output_dir : str or Path
        Folder where the figure will be saved.
    dpi : int
        Resolution for raster formats such as PNG or TIFF.
    formats : tuple
        Output formats, for example ("png", "pdf") or ("tif",).
    bbox_inches : str
        Usually "tight" to remove excess whitespace.
    pad_inches : float
        Padding around the figure.
    close : bool
        If True, closes the figure after saving.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(filename)

    # Case 1: user passes filename with extension, e.g. "figure_1.png"
    if filename.suffix:
        save_path = output_dir / filename
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches
        )

    # Case 2: user passes filename without extension, e.g. "figure_1"
    else:
        for fmt in formats:
            save_path = output_dir / f"{filename.name}.{fmt}"
            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches
            )

    if close:
        plt.close(fig)

def l2_normalize_cube(cube, mask=None, eps=1e-12):
    """
    L2-normalize each pixel spectrum over the full spectral axis.

    Parameters
    ----------
    cube : ndarray
        Hyperspectral cube with shape H x W x N.
    mask : ndarray or None
        Boolean foreground mask with shape H x W.
        If provided, pixels outside the mask are set to NaN.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    cube_norm : ndarray
        L2-normalized cube.
    """

    cube = cube.astype(float).copy()

    norm = np.sqrt(np.nansum(cube ** 2, axis=-1, keepdims=True))
    cube_norm = cube / (norm + eps)

    if mask is not None:
        cube_norm[~mask, :] = np.nan

    return cube_norm

def area_normalize_cube(cube, mask=None, eps=1e-12):
    """
    Area-normalize each pixel spectrum.
    Less aggressive than L2 in some cases.
    """

    cube = cube.astype(float).copy()

    area = np.nansum(np.abs(cube), axis=-1, keepdims=True)
    cube_norm = cube / (area + eps)

    if mask is not None:
        cube_norm[~mask, :] = np.nan

    return cube_norm

def percentile_vmin_vmax(img, pmin=1, pmax=99):
    """
    Compute vmin/vmax from image percentiles.
    This affects only visualization if passed to imshow.
    """

    vals = img[np.isfinite(img)]

    if vals.size == 0:
        return None, None

    vmin, vmax = np.percentile(vals, [pmin, pmax])

    if vmax <= vmin:
        vmax = vmin + 1e-12

    return vmin, vmax


def symmetric_percentile_vmin_vmax(img, p=99):
    """
    Symmetric color limits around zero, useful for difference maps.
    """

    vals = img[np.isfinite(img)]

    if vals.size == 0:
        return None, None

    vmax = np.percentile(np.abs(vals), p)

    if vmax == 0:
        vmax = 1e-12

    return -vmax, vmax


def l1_normalize_cube(cube, mask=None, eps=1e-12):
    """
    L1-normalize each pixel spectrum over the full spectral axis.

    The L1 norm is defined as sum(abs(spectrum)).
    For non-negative baseline-corrected spectra, this is equivalent to
    area normalization.

    Parameters
    ----------
    cube : ndarray
        Hyperspectral cube with shape H x W x N.
    mask : ndarray or None
        Boolean foreground mask with shape H x W.
        If provided, pixels outside the mask are set to NaN.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    cube_norm : ndarray
        L1-normalized cube.
    """

    cube = cube.astype(float).copy()

    norm = np.nansum(np.abs(cube), axis=-1, keepdims=True)
    cube_norm = cube / (norm + eps)

    if mask is not None:
        cube_norm[~mask, :] = np.nan

    return cube_norm

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def add_scalebar(
    ax,
    image_shape,
    pixel_size_um=None,
    fov_um=None,
    scale_bar_um=100,
    location="lower right",
    color="white",
    outline_color="black",
    linewidth=2,
    fontsize=5,
    pad_fraction=0.05,
    text_pad_fraction=0.03,
    label=None,
):
    """
    Add a scale bar to an imshow axis.

    Parameters
    ----------
    ax : matplotlib axis
        Axis where the image is displayed.
    image_shape : tuple
        Shape of the 2D image, usually img.shape.
    pixel_size_um : float or None
        Pixel size in micrometers/pixel.
    fov_um : float or None
        Field of view in micrometers along x. Used only if pixel_size_um is None.
    scale_bar_um : float
        Length of the scale bar in micrometers.
    location : str
        "lower right", "lower left", "upper right", or "upper left".
    """

    H, W = image_shape[:2]

    if pixel_size_um is None:
        if fov_um is None:
            raise ValueError("Provide either pixel_size_um or fov_um.")
        pixel_size_um = fov_um / W

    bar_px = scale_bar_um / pixel_size_um

    if label is None:
        label = f"{scale_bar_um:g} µm"

    pad_x = W * pad_fraction
    pad_y = H * pad_fraction
    text_pad = H * text_pad_fraction

    if "right" in location:
        x1 = W - pad_x
        x0 = x1 - bar_px
    else:
        x0 = pad_x
        x1 = x0 + bar_px

    if "lower" in location:
        y = H - pad_y
        text_y = y - text_pad
        va = "bottom"
    else:
        y = pad_y
        text_y = y + text_pad
        va = "top"

    # Black/white outline behind the scale bar for visibility
    ax.plot(
        [x0, x1],
        [y, y],
        color=outline_color,
        linewidth=linewidth + 2,
        solid_capstyle="butt",
        zorder=10,
    )

    ax.plot(
        [x0, x1],
        [y, y],
        color=color,
        linewidth=linewidth,
        solid_capstyle="butt",
        zorder=11,
    )

    txt = ax.text(
        (x0 + x1) / 2,
        text_y,
        label,
        color=color,
        ha="center",
        va=va,
        fontsize=fontsize,
        zorder=12,
    )

    txt.set_path_effects([
        pe.Stroke(linewidth=2, foreground=outline_color),
        pe.Normal()
    ])

    return ax