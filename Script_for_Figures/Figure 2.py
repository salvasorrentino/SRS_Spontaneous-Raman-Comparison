import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.metrics import structural_similarity as ssim
from Script_for_Figures.utils_fig2 import (robust_minmax, crop_to_overlap, interpolate_raman_to_srs_grid,
                                           set_publication_style, save_publication_figure, l2_normalize_cube,
                                           area_normalize_cube, symmetric_percentile_vmin_vmax, percentile_vmin_vmax,
                                           l1_normalize_cube, add_scalebar)

set_publication_style()

path_fold = (r'C:\Users\salva\OneDrive - Massachusetts Institute of Technology\Coregistration '
             r'SRS-confocal\data\20260520_breast')

path_img_srs = os.path.join(path_fold, "Project_srs_breast_hyperspectral_processed_coregistered_w_confocal_no_correction.pickle")
path_calib_srs = os.path.join(path_fold, r"srs_Raman_shift.pickle")

path_calib_confocal = os.path.join(path_fold, "arr_calibration202601.pickle")
path_img_confocal = os.path.join(path_fold, "breast_processed_confocal.pickle")

raman_cube = np.rot90(pd.read_pickle(path_img_confocal), k=2)[:, 75:275, :]
# raman_cube = pd.read_pickle(path_img_confocal)[15:-15, :, :]
raman_wn = pd.read_pickle(path_calib_confocal)[300:]

srs_cube = pd.read_pickle(path_img_srs)[:, 75:275, :]#[15:-15, :, :]
srs_wn = pd.read_pickle(path_calib_srs)

sample = {
    "name": "skin_small_hyper",
    "tissue": "skin",
    "srs_cube": srs_cube,        # shape: H x W x Nsrs
    "srs_wn": srs_wn,            # shape: Nsrs
    "raman_cube": raman_cube,    # shape: H x W x Nraman
    "raman_wn": raman_wn         # shape: Nraman
}

# plt.figure()
# plt.imshow(sample["raman_cube"][:, :, 100])
# plt.show()
# plt.figure()
# plt.imshow(sample["srs_cube"][:, :, 100])
# plt.show()

def get_band_image(cube, wn, center, width=10):
    """
    Integrates a spectral band around center +/- width/2.
    """
    keep = (wn >= center - width / 2) & (wn <= center + width / 2)
    if keep.sum() == 0:
        idx = np.argmin(np.abs(wn - center))
        return cube[..., idx]
    return np.nanmean(cube[..., keep], axis=-1)


def plot_band_comparison(
    srs_cube,
    raman_cube_interp,
    wn,
    bands=(785, 1003, 1445, 1660),
    width=12,
    cmap="afmhot",
    sample_name="sample",
    savepath=None,
    mask=None,
    normalization="l2",
    pmin=1,
    pmax=99.0,
    diff_p=99.0,
    shared_scale=True,
    add_colorbar=True,
    add_scalebar_flag=True,
    scale_bar_um=100,
    pixel_size_um=2,
    fov_um=None,
    scalebar_location="lower right",
    show_fov_label=False,
):
    """
    Creates a direct SRS vs Raman band-image comparison.

    Parameters
    ----------
    normalization : {"l2", "area", "none"}
        "l2" normalizes each pixel spectrum by its L2 norm.
        "area" normalizes each pixel spectrum by total absolute area.
        "none" uses the input cubes as provided.

    shared_scale : bool
        If True, Raman and SRS images for the same band use the same
        percentile-based vmin/vmax.
    """

    if normalization == "l2":
        srs_plot_cube = l2_normalize_cube(srs_cube, mask=mask)
        ram_plot_cube = l2_normalize_cube(raman_cube_interp, mask=mask)

    elif normalization in ["l1", "area"]:
        srs_plot_cube = l1_normalize_cube(srs_cube, mask=mask)
        ram_plot_cube = l1_normalize_cube(raman_cube_interp, mask=mask)

    elif normalization in ["none", 'robust']:
        srs_plot_cube = srs_cube.astype(float).copy()
        ram_plot_cube = raman_cube_interp.astype(float).copy()

        if mask is not None:
            srs_plot_cube[~mask, :] = np.nan
            ram_plot_cube[~mask, :] = np.nan

    else:
        raise ValueError("normalization must be 'l2', 'l1', 'area', or 'none'")

    n_bands = len(bands)
    fig, axes = plt.subplots(n_bands, 4, figsize=(12, 3 * n_bands))

    if n_bands == 1:
        axes = axes[None, :]

    for i, band in enumerate(bands):
        srs_img = get_band_image(srs_plot_cube, wn, band, width=width)
        ram_img = get_band_image(ram_plot_cube, wn, band, width=width)

        if normalization == "robust":
            srs_img = robust_minmax(srs_img)
            ram_img = robust_minmax(ram_img)

        diff = ram_img - srs_img

        if shared_scale:
            combined = np.concatenate([
                ram_img.ravel(),
                srs_img.ravel()
            ])
            combined = combined[np.isfinite(combined)]

            if combined.size > 0:
                image_vmin, image_vmax = np.percentile(combined, [pmin, pmax])
            else:
                image_vmin, image_vmax = None, None

            ram_vmin, ram_vmax = image_vmin, image_vmax
            srs_vmin, srs_vmax = image_vmin, image_vmax

        else:
            ram_vmin, ram_vmax = percentile_vmin_vmax(
                ram_img,
                pmin=pmin,
                pmax=pmax
            )
            srs_vmin, srs_vmax = percentile_vmin_vmax(
                srs_img,
                pmin=pmin,
                pmax=pmax
            )

        diff_vmin, diff_vmax = symmetric_percentile_vmin_vmax(
            diff,
            p=diff_p
        )

        im0 = axes[i, 0].imshow(
            ram_img,
            cmap=cmap,
            vmin=ram_vmin,
            vmax=ram_vmax
        )
        axes[i, 0].set_title(f"Raman {band} cm$^{{-1}}$")
        axes[i, 0].axis("off")

        if add_colorbar:
            fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        if add_scalebar_flag:
            add_scalebar(
                axes[i, 0],
                ram_img.shape,
                pixel_size_um=pixel_size_um,
                fov_um=fov_um,
                scale_bar_um=scale_bar_um,
                location=scalebar_location,
                color="white",
                outline_color="black",
            )

        im1 = axes[i, 1].imshow(
            srs_img,
            cmap=cmap,
            vmin=srs_vmin,
            vmax=srs_vmax
        )
        axes[i, 1].set_title(f"SRS {band} cm$^{{-1}}$")
        axes[i, 1].axis("off")

        if add_colorbar:
            fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        if add_scalebar_flag:
            add_scalebar(
                axes[i, 1],
                srs_img.shape,
                pixel_size_um=pixel_size_um,
                fov_um=fov_um,
                scale_bar_um=scale_bar_um,
                location=scalebar_location,
                color="white",
                outline_color="black",
            )

        im2 = axes[i, 2].imshow(
            diff,
            cmap="bwr",
            vmin=diff_vmin,
            vmax=diff_vmax
        )
        axes[i, 2].set_title("Raman - SRS")
        axes[i, 2].axis("off")

        if add_colorbar:
            fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        x = ram_img.ravel()
        y = srs_img.ravel()
        keep = np.isfinite(x) & np.isfinite(y)

        axes[i, 3].scatter(
            x[keep],
            y[keep],
            s=1,
            alpha=0.15
        )

        axes[i, 3].set_xlabel("Raman")
        axes[i, 3].set_ylabel("SRS")
        axes[i, 3].set_title("Pixel scatter")

        # Same axis limits for interpretability
        if shared_scale and image_vmin is not None and image_vmax is not None:
            axes[i, 3].set_xlim(image_vmin, image_vmax)
            axes[i, 3].set_ylim(image_vmin, image_vmax)
            axes[i, 3].plot(
                [image_vmin, image_vmax],
                [image_vmin, image_vmax],
                linestyle="--",
                linewidth=0.8
            )

    fig.suptitle(
        f"{sample_name} | normalization: {normalization}",
        y=1.01
    )

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=600, bbox_inches="tight")

    return fig

plot_band_comparison(
    sample["srs_cube"],
    interpolate_raman_to_srs_grid(sample["raman_cube"], sample["raman_wn"], sample["srs_wn"]),
    sample["srs_wn"],
    sample_name=sample["name"],
    cmap="afmhot",
    bands=(1003, 1445, 1660),
    width=12,
    # savepath=os.path.join(path_fold, "Figure 2.png"),
    normalization="robust",
    pmin=1,
    pmax=98.5,
    shared_scale=False,
    add_colorbar=True
)