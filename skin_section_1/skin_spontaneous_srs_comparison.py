import os
import numpy as np
import pandas as pd

from utils_registration import *

import matplotlib.pyplot as plt


path_fold = (r'C:\Users\salva\OneDrive - Massachusetts Institute of Technology\Coregistration '
             r'SRS-confocal\data\skin_section_1')
path_img_srs = os.path.join(path_fold, "skin_srs_hyperspectral.pickle")

path_calib_srs = os.path.join(path_fold, r"srs_Raman_shift.pickle")

path_calib_confocal = os.path.join(path_fold, "arr_calibration_confocal.pickle")
path_img_confocal = os.path.join(path_fold, "skin_spontaneous_raman.pickle")

image_confocal = np.rot90(pd.read_pickle(path_img_confocal), k=2)
calib_confocal = pd.read_pickle(path_calib_confocal)[300:]

image_srs = pd.read_pickle(path_img_srs)
calib_srs = pd.read_pickle(path_calib_srs)

img_1450_srs = image_srs.T[:, :, 74:84].sum(axis=2)
img_1450_confocal = image_confocal[:, :, 678:703].sum(axis=2)

from skin_comparison.registration_script import (coregistration_algo, resize_by_factor, rotate_scale_image_keep_all,
                                                 raman_hw)


coregistration_algo(img_1450_srs, image_confocal, path_fold)

REG_NPZ = os.path.join(path_fold, "rot_raman_fixed_registration_cv2.npz")

reg = np.load(REG_NPZ, allow_pickle=True)
ds_factor  = float(reg["downsample_factor"])
angle_deg  = float(reg["best_angle_deg"])
best_scale = float(reg["best_scale"]) if "best_scale" in reg else 1.0
x0         = int(reg["x0"])
y0         = int(reg["y0"])

print("Loaded registration:")
print(f"  ds_factor  = {ds_factor}")
print(f"  angle_deg  = {angle_deg}")
print(f"  best_scale = {best_scale}")
print(f"  x0, y0     = ({x0}, {y0})")

H_r, W_r = raman_hw(np.asarray(image_confocal))
orig_dtype = img_1450_srs.dtype

# ---------------------------
# 2) Load images
# ---------------------------
srs_rot_best = image_srs.T[:image_confocal.shape[0], :image_confocal.shape[1], :].copy()
for wn in range(image_srs.T.shape[2]):
    srs_resize_best = resize_by_factor(image_srs.T[:, :, wn], factor=1.0 / ds_factor).astype(np.float32)

    srs_rot_best_tmp, M_rot = rotate_scale_image_keep_all(
        srs_resize_best,
        angle_deg=angle_deg,
        scale=best_scale
    )
    if srs_rot_best_tmp.ndim == 2:
        srs_rot_best[:, :, wn] = srs_rot_best_tmp[y0:y0 + H_r, x0:x0 + W_r]
    else:
        srs_rot_best[:, :, wn] = srs_rot_best_tmp[y0:y0 + H_r, x0:x0 + W_r, :]

    if srs_rot_best.shape[0] != H_r or srs_rot_best.shape[1] != W_r:
        raise RuntimeError(
            f"Crop fuori bounds o mismatch shape. "
            f"Got {srs_rot_best.shape[:2]} vs Raman {(H_r, W_r)}. "
            f"Check x0, y0, angle_deg, best_scale, ds_factor."
        )

    print(f"patch shape:           {srs_rot_best.shape}")

img_1450_srs_rot = srs_rot_best[:, :, 74:84].sum(axis=2)
img_1003_srs_rot = srs_rot_best[:, :, 201:207].sum(axis=2)
img_1003_confocal = image_confocal[:, :, 363:370].sum(axis=2)

plt.figure()
vmin2 = np.percentile(img_1450_srs_rot, 25)
vmax2 = np.percentile(img_1450_srs_rot, 99)
plt.imshow(img_1450_srs_rot[:, :], cmap='afmhot', vmin=vmin2, vmax=vmax2)
plt.show()

plt.figure()
vmin1 = np.percentile(img_1450_confocal, 10)
vmax1 = np.percentile(img_1450_confocal, 99.99)
plt.imshow(img_1450_confocal, cmap='afmhot', vmin=vmin1, vmax=vmax1, alpha=1)
plt.show()

plt.figure()
plt.plot(calib_confocal, image_confocal[250, 300, :]/image_confocal[250, 300, :].max())
plt.plot(calib_srs, (srs_rot_best[250, 300, :]/srs_rot_best[250, 300, :].max()) -
         (srs_rot_best[250, 300, :]/srs_rot_best[250, 300, :].max()).min())
plt.show()

pd.to_pickle(srs_rot_best, os.path.join(path_fold, "skin_srs_hyperspectral_registered_confocal"))
