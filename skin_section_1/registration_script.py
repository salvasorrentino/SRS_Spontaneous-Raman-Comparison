import os
import numpy as np
import cv2
import tifffile as tiff
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import inspect
from matplotlib.widgets import RectangleSelector


# ============================================================
# 0) USER PARAMETERS
# ============================================================

RAMAN_CALIB_PATH = None                        # optional .pickle with Raman shifts

# Raman band selection if cube
RAMAN_INDEX_RANGE = (365, 367)   # start inclusive, end exclusive
RAMAN_WN_RANGE = None
RAMAN_SPECTRAL_AXIS = 2  # (H,W,nW)->2 ; (nW,H,W)->0

# Scale
MOSAIC_PIXEL_SIZE_UM = 2  # e.g. 0.25
RAMAN_PIXEL_SIZE_UM  = 2  # e.g. 1.0
MOSAIC_DOWNSAMPLE_FACTOR = 4.0

# Rotation search (we rotate IHC mosaic, not Raman)
ANGLE_RANGE_DEG = (-45.0, 45.0)
ANGLE_STEP_DEG  = 0.1

# Prior strictness
PRIOR_MARGIN_PIXELS = 3   # <= make this small if your prior is reliable

# Preprocessing / feature extraction (more similar to real images)
USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

HP_SIGMA = 10          # high-pass: remove smooth background
SMOOTH_SIGMA = 0.7    # light smoothing before gradients
# SMOOTH_SIGMA = 0.5

# Feature mixing (keep close to original intensities)
ALPHA_GRAD = 0.8   # weight for gradient magnitude
BETA_INT  = 0.1       # weight for intensity (after preprocessing)

# Matching
TM_METHOD = cv2.TM_CCORR_NORMED  # often more stable across modalities than CCOEFF_NORMED
DO_PHASECORR_REFINEMENT = True
PHASECORR_HIGHPASS_SIGMA = 10

# Small local similarity refinement (AFTER coarse rigid registration)
LOCAL_ANGLE_DELTA_DEG = 1
LOCAL_ANGLE_STEP_DEG  = 0.05

LOCAL_SCALE_RANGE = (0.97, 1.03)
LOCAL_SCALE_STEP  = 0.001

LOCAL_SEARCH_RADIUS_PX = 20
LOCAL_MAX_ALLOWED_SHIFT_PX = 15

SHOW_DEBUG_PLOTS = True

# --- Fine tuning using DNA band (Raman) ---
DNA_RAMAN_INDEX_RANGE = (855, 865)   # <-- METTI QUI la banda DNA che ti interessa
FINE_ANGLE_DELTA_DEG  = 0.6          # max change around current angle
FINE_ANGLE_STEP_DEG   = 0.05         # finer step
FINE_SEARCH_RADIUS_PX = 20           # search around current (x0,y0)
MAX_ALLOWED_SHIFT_PX  = 30           # reject crazy shifts

DO_ECC_TRANSLATION_REFINE = True     # optional subpixel refinement
ECC_MAX_ITERS = 80
ECC_EPS       = 1e-6

# ============================================================
# 1) UTILITIES
# ============================================================

def load_array(path: str) -> np.ndarray:
    p = str(path).lower()
    if p.endswith(".pickle") or p.endswith(".pkl"):
        return pd.read_pickle(path)
    return tiff.imread(path)


def to_gray_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = img[..., :3].mean(axis=2)
    return img.astype(np.float32, copy=False)


def robust01(img: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    lo = float(np.percentile(img, p_low))
    hi = float(np.percentile(img, p_high))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def resize_by_factor(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp).astype(np.float32)


def extract_raman_map(raman: np.ndarray, spectral_axis: int,
                      index_range, wn_range, calib: np.ndarray | None) -> np.ndarray:
    if raman.ndim == 2:
        return raman
    if raman.ndim != 3:
        raise ValueError("Raman must be 2D or 3D.")

    if spectral_axis != 2:
        raman = np.moveaxis(raman, spectral_axis, 2)  # (H,W,nW)

    if wn_range is not None:
        if calib is None:
            raise ValueError("wn_range set but calib is None.")
        wn0, wn1 = float(wn_range[0]), float(wn_range[1])
        i0 = int(np.argmin(np.abs(calib - wn0)))
        i1 = int(np.argmin(np.abs(calib - wn1))) + 1
    else:
        i0, i1 = int(index_range[0]), int(index_range[1])

    return raman[:, :, i0:i1].sum(axis=2)


def clahe01(img01: np.ndarray, clip=2.0, tile=(8,8)) -> np.ndarray:
    u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tile))
    out = clahe.apply(u8)
    return (out.astype(np.float32) / 255.0)


def highpass(img01: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img01
    low = gaussian_filter(img01, sigma=float(sigma))
    hp = img01 - low
    # rescale robustly to [0,1]
    return robust01(hp, 1, 99)


def feature_repr(img01: np.ndarray) -> np.ndarray:
    """
    Feature representation closer to the original image:
      - optional CLAHE
      - mild high-pass to remove background
      - gradient magnitude (soft)
      - mix gradient + intensity so it remains "image-like"
    """
    x = img01.astype(np.float32, copy=False)

    if USE_CLAHE:
        x = clahe01(x, clip=CLAHE_CLIP, tile=CLAHE_TILE)

    x = highpass(x, HP_SIGMA)

    if SMOOTH_SIGMA > 0:
        x_s = gaussian_filter(x, sigma=float(SMOOTH_SIGMA)).astype(np.float32)
    else:
        x_s = x

    gx = cv2.Sobel(x_s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x_s, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)
    gmag = robust01(gmag, 1, 99)

    # Mix gradient and intensity -> stays more similar to real structure
    out = ALPHA_GRAD * gmag + BETA_INT * x
    return robust01(out, 1, 99)


def rotate_image_keep_all(img: np.ndarray, angle_deg: float, scale: float = 1.0):
    """
    Rotate + isotropically scale image while keeping all content by expanding canvas.
    Returns:
      - transformed image
      - 2x3 affine matrix mapping original coords -> transformed coords
    """
    h, w = img.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)

    # Bounding box of rotated+scaled image
    a = abs(M[0, 0])  # = scale * cos(theta)
    b = abs(M[0, 1])  # = scale * sin(theta)

    new_w = int(np.ceil(h * b + w * a))
    new_h = int(np.ceil(h * a + w * b))

    # Re-center on expanded canvas
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    out = cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0
    )
    return out.astype(np.float32), M.astype(np.float32)


def rotate_point(M: np.ndarray, x: float, y: float):
    """Apply 2x3 affine M to point (x,y)."""
    xr = M[0, 0]*x + M[0, 1]*y + M[0, 2]
    yr = M[1, 0]*x + M[1, 1]*y + M[1, 2]
    return float(xr), float(yr)


def apply_affine_to_point(M: np.ndarray, x: float, y: float):
    """Apply 2x3 affine matrix to one point."""
    xo = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    yo = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return float(xo), float(yo)


def invert_affine(M: np.ndarray) -> np.ndarray:
    """Inverse of a 2x3 affine matrix."""
    return cv2.invertAffineTransform(M.astype(np.float32)).astype(np.float32)


def phasecorr_refine(patch: np.ndarray, templ: np.ndarray):
    """
    Refine translation between patch and template using phase correlation.
    Returns dx,dy such that patch shifted by (dx,dy) aligns better to template.
    """
    def hp(z):
        z = z.astype(np.float32, copy=False)
        blur = cv2.GaussianBlur(z, (0,0), PHASECORR_HIGHPASS_SIGMA)
        return (z - blur).astype(np.float32)

    p = hp(patch)
    t = hp(templ)
    (dx, dy), resp = cv2.phaseCorrelate(p, t)
    return float(dx), float(dy), float(resp)

def _local_match_around_xy(big_img, templ, x0, y0, radius):
    """
    Match template only in a local window around (x0,y0).
    Returns (best_score, best_y, best_x) in big_img coordinates.
    """
    Ht, Wt = templ.shape
    H, W = big_img.shape

    # Local search bounds for top-left
    x_min = max(0, x0 - radius)
    y_min = max(0, y0 - radius)
    x_max = min(W - Wt, x0 + radius)
    y_max = min(H - Ht, y0 + radius)
    if x_max < x_min or y_max < y_min:
        return -np.inf, y0, x0

    # Crop region where matchTemplate can slide
    crop = big_img[y_min:y_max + Ht, x_min:x_max + Wt].astype(np.float32, copy=False)
    res = cv2.matchTemplate(crop, templ.astype(np.float32, copy=False), method=TM_METHOD)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    best_x = x_min + int(max_loc[0])
    best_y = y_min + int(max_loc[1])
    return float(max_val), best_y, best_x


def ecc_refine_translation(template01, image01, max_iters=80, eps=1e-6):
    """
    Subpixel refinement of translation only using ECC.
    template01 and image01 must have same shape and be float32 in [0,1].
    Returns (dx, dy, cc) where (dx,dy) is translation to apply to image to align to template.
    """
    templ = template01.astype(np.float32, copy=False)
    img   = image01.astype(np.float32, copy=False)

    warp = np.array([[1, 0, 0],
                     [0, 1, 0]], dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iters), float(eps))
    try:
        cc, warp = cv2.findTransformECC(templ, img, warp, cv2.MOTION_TRANSLATION, criteria, None, 1)
        dx = float(warp[0, 2])
        dy = float(warp[1, 2])
        return dx, dy, float(cc)
    except cv2.error:
        # ECC can fail if images are too different; just skip
        return 0.0, 0.0, -1.0


def fine_tune_with_dna_band(
    mosaic_ds,          # 2D float mosaic downsampled (same as used in coarse)
    raman_cube_or_map,  # Raman 2D or 3D
    calib,              # or None
    current_angle_deg,
    current_x0,
    current_y0,
    *,
    dna_index_range=(230,242),
    angle_delta=0.6,
    angle_step=0.1,
    search_radius=20,
    max_allowed_shift=30,
):
    """
    Fine tune around current (angle,x0,y0) using a DNA Raman band.
    - Keeps changes small by construction.
    - Returns refined (angle, x0, y0, score, debug_dict).
    """

    # 1) Build the DNA Raman map (2D)
    raman_dna_map = extract_raman_map(
        raman_cube_or_map,
        RAMAN_SPECTRAL_AXIS,
        dna_index_range,
        wn_range=None,
        calib=calib
    )
    raman_dna01 = robust01(to_gray_float(raman_dna_map), 1, 99)

    # 2) Feature representation (reuse your feature pipeline)
    raman_dna_feat = feature_repr(raman_dna01)
    Ht, Wt = raman_dna_feat.shape

    best = {
        "score": -np.inf,
        "angle": current_angle_deg,
        "x0": current_x0,
        "y0": current_y0,
        "mosaic_ds_rot": None
    }

    # 3) Small angle sweep around current angle
    angles = np.arange(current_angle_deg - angle_delta,
                       current_angle_deg + angle_delta + 1e-9,
                       angle_step, dtype=np.float32)

    for ang in angles:
        ang = float(ang)

        # rotate mosaic_ds (intensity) and then build features
        mosaic_ds_rot, _ = rotate_image_keep_all(mosaic_ds, ang)
        mosaic_feat_rot = feature_repr(mosaic_ds_rot)

        # Local match around current position only (strong anchoring)
        score, y1, x1 = _local_match_around_xy(
            mosaic_feat_rot, raman_dna_feat,
            current_x0, current_y0,
            radius=search_radius
        )

        # Reject big jumps (extra hard constraint)
        if abs(x1 - current_x0) > max_allowed_shift or abs(y1 - current_y0) > max_allowed_shift:
            continue

        if score > best["score"]:
            best.update({"score": score, "angle": ang, "x0": x1, "y0": y1, "mosaic_ds_rot": mosaic_ds_rot})

    # 4) Optional subpixel refinement on the best candidate using ECC translation
    # Use features (or you can use intensity) — features usually more robust.
    debug = {"ecc_cc": None, "ecc_dx": None, "ecc_dy": None}

    if best["mosaic_ds_rot"] is not None:
        mosaic_feat_rot = feature_repr(best["mosaic_ds_rot"])
        patch = mosaic_feat_rot[best["y0"]:best["y0"]+Ht, best["x0"]:best["x0"]+Wt].copy()

        if DO_ECC_TRANSLATION_REFINE:
            dx, dy, cc = ecc_refine_translation(raman_dna_feat, patch, ECC_MAX_ITERS, ECC_EPS)
            debug.update({"ecc_cc": cc, "ecc_dx": dx, "ecc_dy": dy})

            # Update x0,y0 with small subpixel shift (and clip)
            # dx,dy moves PATCH to align to TEMPLATE -> so top-left should be shifted by dx,dy
            new_x0 = int(np.round(best["x0"] + dx))
            new_y0 = int(np.round(best["y0"] + dy))

            # Keep it conservative: do not allow big moves
            if abs(new_x0 - best["x0"]) <= max_allowed_shift and abs(new_y0 - best["y0"]) <= max_allowed_shift:
                best["x0"], best["y0"] = new_x0, new_y0

    return best["angle"], best["x0"], best["y0"], best["score"], debug

# ============================================================
# 2) INTERACTIVE PRIOR UI
# ============================================================

class PriorSelectionUI:
    """
    Left: Raman (click ONE point)
    Right: Mosaic_ds (draw ONE rectangle where that Raman point must lie)
    Confirm: Enter or Space. Reset: r. Close: Esc.
    """
    def __init__(self, raman_display, mosaic_display):
        self.raman_point = None
        self.rect = None

        self.fig, (self.ax_r, self.ax_m) = plt.subplots(1, 2, figsize=(12, 6))
        self.ax_r.set_title("Raman (click ONE point)")
        self.ax_r.imshow(raman_display, cmap="inferno")
        self.ax_r.axis("off")

        self.ax_m.set_title("IHC mosaic_ds (draw ONE rectangle)")
        self.ax_m.imshow(mosaic_display, cmap="gray")
        self.ax_m.axis("off")

        self.point_artist = None

        selector_kwargs = dict(useblit=True, button=[1], interactive=True)
        style = dict(facecolor="none", edgecolor="lime", linewidth=2)
        sig = inspect.signature(RectangleSelector.__init__)
        if "props" in sig.parameters:
            selector_kwargs["props"] = style
        elif "rectprops" in sig.parameters:
            selector_kwargs["rectprops"] = style
        if "drawtype" in sig.parameters:
            selector_kwargs["drawtype"] = "box"

        self.rect_selector = RectangleSelector(self.ax_m, self._onselect_rect, **selector_kwargs)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        print("\nINTERACTIVE STEP")
        print("1) Click ONE point in Raman (left).")
        print("2) Draw ONE rectangle in mosaic_ds (right).")
        print("3) Press Enter or Space to proceed. ('r' reset, Esc close)\n")

    def _on_click(self, event):
        if event.inaxes != self.ax_r or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        self.raman_point = (x, y)
        if self.point_artist is not None:
            self.point_artist.remove()
        self.point_artist = self.ax_r.plot([x], [y], marker="x", markersize=10, color="cyan")[0]
        self.fig.canvas.draw_idle()
        print(f"Raman point selected: (x={x:.1f}, y={y:.1f})")

    def _onselect_rect(self, eclick, erelease):
        if None in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata):
            return
        x0, y0 = float(eclick.xdata), float(eclick.ydata)
        x1, y1 = float(erelease.xdata), float(erelease.ydata)
        xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
        ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
        self.rect = (xa, ya, xb, yb)
        print(f"Mosaic rectangle: x=[{xa:.1f},{xb:.1f}], y=[{ya:.1f},{yb:.1f}]")

    def _on_key(self, event):
        if event.key in ("r", "R"):
            self.raman_point = None
            self.rect = None
            if self.point_artist is not None:
                self.point_artist.remove()
                self.point_artist = None
            self.fig.canvas.draw_idle()
            print("Selections reset.")
            return
        if event.key in ("enter", "return", " "):
            if self.raman_point is None or self.rect is None:
                print("Select BOTH point and rectangle before confirming.")
                return
            plt.close(self.fig)
            return
        if event.key in ("escape",):
            plt.close(self.fig)
            return

    def run(self):
        plt.ioff()
        plt.show(block=True)
        return self.raman_point, self.rect


# ============================================================
# 3) REGISTRATION: rotate/translate IHC to match Raman, with HARD prior
# ============================================================

def register_with_hard_prior(raman_feat: np.ndarray,
                             mosaic_feat: np.ndarray,
                             raman_point_xy: tuple[float, float],
                             mosaic_rect_xyxy: tuple[float, float, float, float],
                             angle_range=(-5, 5),
                             angle_step=0.25,
                             prior_margin=10):
    """
    Coarse robust search: rotation + translation only.
    This is your old behavior, which was working.
    """
    Ht, Wt = raman_feat.shape
    px, py = raman_point_xy
    rx0, ry0, rx1, ry1 = mosaic_rect_xyxy

    rect_corners = np.array([
        [rx0, ry0],
        [rx1, ry0],
        [rx1, ry1],
        [rx0, ry1]
    ], dtype=np.float32)

    best = {"score": -np.inf, "angle": None, "pos": None, "M_rot": None, "mosaic_rot": None}

    angles = np.arange(angle_range[0], angle_range[1] + 1e-9, angle_step, dtype=np.float32)

    for ang in angles:
        ang = float(ang)
        mosaic_rot, M = rotate_image_keep_all(mosaic_feat, ang, 1.0)
        Hm, Wm = mosaic_rot.shape

        corners_rot = []
        for (x, y) in rect_corners:
            xr, yr = rotate_point(M, float(x), float(y))
            corners_rot.append([xr, yr])
        corners_rot = np.array(corners_rot, dtype=np.float32)

        rrx0 = float(np.min(corners_rot[:, 0])) - prior_margin
        rrx1 = float(np.max(corners_rot[:, 0])) + prior_margin
        rry0 = float(np.min(corners_rot[:, 1])) - prior_margin
        rry1 = float(np.max(corners_rot[:, 1])) + prior_margin

        x0_min = int(np.ceil(rrx0 - px))
        x0_max = int(np.floor(rrx1 - px))
        y0_min = int(np.ceil(rry0 - py))
        y0_max = int(np.floor(rry1 - py))

        x0_min = max(0, x0_min)
        y0_min = max(0, y0_min)
        x0_max = min(Wm - Wt, x0_max)
        y0_max = min(Hm - Ht, y0_max)

        if x0_max < x0_min or y0_max < y0_min:
            continue

        crop = mosaic_rot[y0_min:y0_max + Ht, x0_min:x0_max + Wt].astype(np.float32, copy=False)
        templ = raman_feat.astype(np.float32, copy=False)

        res = cv2.matchTemplate(crop, templ, method=TM_METHOD)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        x0 = x0_min + int(max_loc[0])
        y0 = y0_min + int(max_loc[1])
        score = float(max_val)

        if score > best["score"]:
            best.update({"score": score, "angle": ang, "pos": (y0, x0), "M_rot": M, "mosaic_rot": mosaic_rot})

    return best

def refine_local_similarity(
    mosaic_ds: np.ndarray,
    raman_feat: np.ndarray,
    current_angle_deg: float,
    current_x0: int,
    current_y0: int,
    current_M_coarse: np.ndarray,
    *,
    current_scale: float = 1.0,
    angle_delta=0.8,
    angle_step=0.05,
    scale_range=(0.985, 1.015),
    scale_step=0.0025,
    search_radius=15,
    max_allowed_shift=20,
):
    """
    Local refinement around an already-good coarse solution.

    Correct geometry:
    - current_x0,current_y0 live in the coarse transformed canvas
    - each candidate (angle, scale) has its own transformed canvas
    - therefore we:
        coarse transformed coords -> original mosaic coords -> candidate transformed coords
      and only then do local matching around the predicted candidate position
    """
    Ht, Wt = raman_feat.shape

    # ------------------------------------------------------------
    # Anchor = center of the matched Raman patch in coarse canvas
    # ------------------------------------------------------------
    cx_coarse = float(current_x0 + 0.5 * Wt)
    cy_coarse = float(current_y0 + 0.5 * Ht)

    M_coarse_inv = invert_affine(current_M_coarse)
    cx_orig, cy_orig = apply_affine_to_point(M_coarse_inv, cx_coarse, cy_coarse)

    # ------------------------------------------------------------
    # Build default result = coarse solution itself
    # so the function always returns something valid
    # ------------------------------------------------------------
    mosaic_ds_coarse, M_check = rotate_image_keep_all(mosaic_ds, current_angle_deg, current_scale)
    mosaic_feat_coarse = feature_repr(mosaic_ds_coarse)

    best = {
        "rank_score": -np.inf,
        "match_score": -np.inf,
        "angle": float(current_angle_deg),
        "scale": float(current_scale),
        "x0": int(current_x0),
        "y0": int(current_y0),
        "M_rot": M_check,
        "mosaic_ds_tf": mosaic_ds_coarse,
        "mosaic_feat_tf": mosaic_feat_coarse,
    }

    # Optional: evaluate the current coarse candidate score once
    if (0 <= current_x0 <= mosaic_feat_coarse.shape[1] - Wt) and (0 <= current_y0 <= mosaic_feat_coarse.shape[0] - Ht):
        patch0 = mosaic_feat_coarse[current_y0:current_y0 + Ht, current_x0:current_x0 + Wt]
        if patch0.shape == raman_feat.shape:
            res0 = cv2.matchTemplate(
                patch0.astype(np.float32, copy=False),
                raman_feat.astype(np.float32, copy=False),
                method=TM_METHOD
            )
            score0 = float(res0[0, 0])
            best["match_score"] = score0
            best["rank_score"] = score0

    scales = np.arange(scale_range[0], scale_range[1] + 1e-12, scale_step, dtype=np.float32)
    if scales.size == 0:
        scales = np.array([current_scale], dtype=np.float32)

    angles = np.arange(
        current_angle_deg - angle_delta,
        current_angle_deg + angle_delta + 1e-9,
        angle_step,
        dtype=np.float32
    )

    for scl in scales:
        scl = float(scl)

        for ang in angles:
            ang = float(ang)

            # Candidate transformed image and its affine
            mosaic_ds_tf, M_cand = rotate_image_keep_all(mosaic_ds, ang, scl)
            mosaic_feat_tf = feature_repr(mosaic_ds_tf)

            # ------------------------------------------------------------
            # Predict where the SAME physical patch center lands
            # in the candidate transformed canvas
            # ------------------------------------------------------------
            cx_pred, cy_pred = apply_affine_to_point(M_cand, cx_orig, cy_orig)

            x_pred = int(round(cx_pred - 0.5 * Wt))
            y_pred = int(round(cy_pred - 0.5 * Ht))

            score, y1, x1 = _local_match_around_xy(
                mosaic_feat_tf,
                raman_feat,
                x_pred,
                y_pred,
                radius=search_radius
            )

            # Deviation from the geometrically predicted position
            dx = x1 - x_pred
            dy = y1 - y_pred

            if abs(dx) > max_allowed_shift or abs(dy) > max_allowed_shift:
                continue

            # Conservative ranking:
            # prefer high score, small correction, scale close to coarse
            dist = float(np.hypot(dx, dy))
            rank_score = float(score) - 0.0015 * dist - 0.02 * abs(scl - current_scale)

            if rank_score > best["rank_score"]:
                best.update({
                    "rank_score": rank_score,
                    "match_score": float(score),
                    "angle": ang,
                    "scale": scl,
                    "x0": int(x1),
                    "y0": int(y1),
                    "M_rot": M_cand,
                    "mosaic_ds_tf": mosaic_ds_tf,
                    "mosaic_feat_tf": mosaic_feat_tf,
                })

    return best


def build_raman_to_mosaicrot_affine(x0: int, y0: int) -> np.ndarray:
    """
    Build affine matrix to place the Raman image into the rotated mosaic canvas.

    Raman coordinates:
        (u, v)

    Rotated mosaic coordinates:
        (x, y) = (u + x0, v + y0)

    This is the matrix to use with:
        cv2.warpAffine(raman, M, dsize=(W_mosaic_rot, H_mosaic_rot))
    """
    return np.array([[1, 0, x0],
                     [0, 1, y0]], dtype=np.float32)


# ============================================================
# 4) MAIN
# ============================================================

def coregistration_algo(mosaic, raman, path):
    plt.ioff()

    # Load images
    # mosaic = load_array(MOSAIC_PATH)
    mosaic01 = robust01(to_gray_float(mosaic), 1, 99)

    # raman = load_array(RAMAN_PATH)
    calib = None
    if RAMAN_CALIB_PATH is not None:
        calib = load_array(RAMAN_CALIB_PATH).astype(np.float32)

    raman_map = extract_raman_map(raman, RAMAN_SPECTRAL_AXIS, RAMAN_INDEX_RANGE, RAMAN_WN_RANGE, calib)
    raman01 = robust01(to_gray_float(raman_map), 1, 99)

    # Downsample mosaic to approximate Raman sampling
    if MOSAIC_PIXEL_SIZE_UM is not None and RAMAN_PIXEL_SIZE_UM is not None:
        ds_factor = float(RAMAN_PIXEL_SIZE_UM / MOSAIC_PIXEL_SIZE_UM)
    else:
        ds_factor = float(MOSAIC_DOWNSAMPLE_FACTOR)

    mosaic_ds = resize_by_factor(mosaic01, factor=1.0 / ds_factor)

    print("\n=== Inputs ===")
    print("mosaic_ds shape:", mosaic_ds.shape)
    print("raman shape:", raman01.shape)
    print("downsample factor:", ds_factor)

    # Interactive prior
    ui = PriorSelectionUI(raman_display=raman01, mosaic_display=mosaic_ds)
    raman_point, mosaic_rect = ui.run()

    if raman_point is None or mosaic_rect is None:
        raise RuntimeError("Selezione incompleta: seleziona punto + rettangolo, poi Enter/Space.")

    # Build features closer to real images
    raman_feat = feature_repr(raman01)
    mosaic_feat = feature_repr(mosaic_ds)

    print("\n=== Feature stats ===")
    print("mosaic_feat std:", float(np.std(mosaic_feat)))
    print("raman_feat  std:", float(np.std(raman_feat)))

    # Registration: rotate/translate IHC (mosaic), keep Raman fixed
    best = register_with_hard_prior(
        raman_feat=raman_feat,
        mosaic_feat=mosaic_feat,
        raman_point_xy=raman_point,
        mosaic_rect_xyxy=mosaic_rect,
        angle_range=ANGLE_RANGE_DEG,
        angle_step=ANGLE_STEP_DEG,
        prior_margin=PRIOR_MARGIN_PIXELS
    )

    if best["pos"] is None:
        raise RuntimeError(
            "Nessun match trovato. Prova: aumentare ANGLE_RANGE_DEG o PRIOR_MARGIN_PIXELS, oppure cambiare band Raman."
        )

    best_angle = best["angle"]
    best_score = best["score"]
    (y0, x0) = best["pos"]

    print("\n=== Coarse registration result (rigid) ===")
    print(f"best angle (deg): {best_angle:.3f}")
    print(f"best score:       {best_score:.4f}")
    print(f"template top-left in mosaic_rot: x0={x0}, y0={y0}")

    # ------------------------------------------------------------
    # Local refinement: small scale + small angle + small shift
    # ------------------------------------------------------------
    best_local = refine_local_similarity(
        mosaic_ds=mosaic_ds,
        raman_feat=raman_feat,
        current_angle_deg=best_angle,
        current_x0=x0,
        current_y0=y0,
        current_M_coarse=best["M_rot"],
        current_scale=1.0,
        angle_delta=LOCAL_ANGLE_DELTA_DEG,
        angle_step=LOCAL_ANGLE_STEP_DEG,
        scale_range=LOCAL_SCALE_RANGE,
        scale_step=LOCAL_SCALE_STEP,
        search_radius=LOCAL_SEARCH_RADIUS_PX,
        max_allowed_shift=LOCAL_MAX_ALLOWED_SHIFT_PX,
    )

    best_angle = best_local["angle"]
    best_scale = best_local["scale"]
    best_score = best_local["match_score"]
    x0 = best_local["x0"]
    y0 = best_local["y0"]
    M_rot = best_local["M_rot"]
    mosaic_ds_rot = best_local["mosaic_ds_tf"]
    mosaic_feat_rot = best_local["mosaic_feat_tf"]

    print("\n=== Local similarity refinement ===")
    print(f"best angle (deg): {best_angle:.3f}")
    print(f"best scale:       {best_scale:.5f}")
    print(f"best score:       {best_score:.4f}")
    print(f"template top-left: x0={x0}, y0={y0}")
    print("mosaic_ds_rot shape:", mosaic_ds_rot.shape)

    print("\n=== Registration result (mosaic_ds rotated+scaled) ===")
    print(f"best angle (deg): {best_angle:.3f}")
    print(f"best scale:       {best_scale:.5f}")
    print(f"best score:       {best_score:.4f}")
    print(f"template top-left in mosaic_rot: x0={x0}, y0={y0}")
    print("mosaic_rot shape:", mosaic_feat_rot.shape)

    # Optional refinement: phase correlation on matched patch
    Ht, Wt = raman_feat.shape
    patch = mosaic_feat_rot[y0:y0 + Ht, x0:x0 + Wt].copy()

    if DO_PHASECORR_REFINEMENT:
        dx, dy, resp = phasecorr_refine(patch, raman_feat)
        x0 = int(np.clip(int(round(x0 - dx)), 0, mosaic_feat_rot.shape[1] - Wt))
        y0 = int(np.clip(int(round(y0 - dy)), 0, mosaic_feat_rot.shape[0] - Ht))
        patch = mosaic_feat_rot[y0:y0 + Ht, x0:x0 + Wt].copy()
        print(f"[Refine] resp={resp:.3f}, dx={dx:.2f}, dy={dy:.2f}")

    # Final local similarity score after all refinements
    final_similarity_score = float(
        cv2.matchTemplate(
            patch.astype(np.float32, copy=False),
            raman_feat.astype(np.float32, copy=False),
            method=TM_METHOD
        )[0, 0]
    )
    print(f"[Final similarity] score={final_similarity_score:.6f}")

    # Warp Raman intensity into mosaic_rot coordinates (for overlay)
    M_raman_to_mosaicrot = build_raman_to_mosaicrot_affine(x0, y0)

    # (this maps Raman(u,v) -> mosaic_rot(x,y) = (u+x0, v+y0))
    raman_on_mosaic_rot = cv2.warpAffine(
        raman01.astype(np.float32),
        M_raman_to_mosaicrot,
        dsize=(mosaic_feat_rot.shape[1], mosaic_feat_rot.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=0.0
    )

    # Also rotate the *original* mosaic_ds intensity for a nicer overlay

    if SHOW_DEBUG_PLOTS:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Raman feature")
        plt.imshow(raman_feat, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Matched patch (mosaic_rot feature)")
        plt.imshow(patch, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Overlay patch vs Raman feature")
        plt.imshow(patch, cmap="Blues")
        plt.imshow(raman_feat, cmap="Reds", alpha=0.4)
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=False)

        plt.figure(figsize=(9, 7))
        plt.title("Overlay Raman intensity on rotated mosaic_ds")
        plt.imshow(mosaic_ds_rot, cmap="gray")
        plt.imshow(raman_on_mosaic_rot, cmap="inferno", alpha=0.45)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # Save state
    np.savez(
        os.path.join(path, "rot_raman_fixed_registration_cv2.npz"),
        downsample_factor=ds_factor,
        best_angle_deg=best_angle,
        best_scale=best_scale,
        best_score=best_score,
        x0=x0, y0=y0,
        raman_point=np.array(raman_point, dtype=np.float32),
        mosaic_rect=np.array(mosaic_rect, dtype=np.float32),
        M_rot=M_rot
    )
    print("\nSaved: rot_raman_fixed_registration_cv2.npz")

# ============================================================
# FINE TUNING with DNA band (small adjustments only)
# ============================================================

# angle_ft, x0_ft, y0_ft, score_ft, dbg = fine_tune_with_dna_band(
#     mosaic_ds=mosaic_ds,  # IMPORTANT: use the same mosaic_ds used for coarse
#     raman_cube_or_map=raman,  # the Raman you loaded (2D or 3D)
#     calib=calib,  # can be None if you use index ranges
#     current_angle_deg=best_angle,  # coarse result
#     current_x0=x0,
#     current_y0=y0,
#     dna_index_range=DNA_RAMAN_INDEX_RANGE,
#     angle_delta=FINE_ANGLE_DELTA_DEG,
#     angle_step=FINE_ANGLE_STEP_DEG,
#     search_radius=FINE_SEARCH_RADIUS_PX,
#     max_allowed_shift=MAX_ALLOWED_SHIFT_PX
# )
#
# print("\n=== Fine-tuning (DNA band) ===")
# print(f"angle: {best_angle:.3f} -> {angle_ft:.3f}")
# print(f"x0,y0: ({x0},{y0}) -> ({x0_ft},{y0_ft})")
# print(f"score: {score_ft:.4f}")
# print(f"ECC debug: {dbg}")
#
# # overwrite with refined values (or keep both if you prefer)
# best_angle = angle_ft
# x0, y0 = x0_ft, y0_ft
#
# # Save state
# np.savez(
#     "ihc_rot_raman_fixed_registration_cv2.npz",
#     downsample_factor=ds_factor,
#     best_angle_deg=best_angle,
#     best_score=best_score,
#     x0=x0, y0=y0,
#     raman_point=np.array(raman_point, dtype=np.float32),
#     mosaic_rect=np.array(mosaic_rect, dtype=np.float32),
#     M_rot=M_rot
# )
# print("\nSaved: ihc_rot_raman_fixed_registration_cv2.npz")


def raman_hw(arr: np.ndarray):
    """
    Estrae (H, W) per croppare:
    - se 2D: (H,W)
    - se 3D: assume (H,W,C) o (H,W,nW) e prende i primi due
    """
    if arr.ndim < 2:
        raise ValueError("Raman array must have at least 2 dimensions.")
    return int(arr.shape[0]), int(arr.shape[1])


def resize_by_factor(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    interp = cv2.INTER_NEAREST
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def rotate_scale_image_keep_all(img: np.ndarray, angle_deg: float, scale: float = 1.0):
    """
    Rotate + isotropically scale image while keeping all content by expanding canvas.
    Works for grayscale or multi-channel images.
    Returns transformed image and 2x3 affine matrix mapping original coords -> transformed coords.
    """
    h, w = img.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)

    a = abs(M[0, 0])  # scale * cos(theta)
    b = abs(M[0, 1])  # scale * sin(theta)

    new_w = int(np.ceil(h * b + w * a))
    new_h = int(np.ceil(h * a + w * b))

    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    if img.ndim == 2:
        border_value = 0.0
    else:
        border_value = tuple([0.0] * img.shape[2])

    out = cv2.warpAffine(
        img,
        M.astype(np.float32),
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return out, M.astype(np.float32)