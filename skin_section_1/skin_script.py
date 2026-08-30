import pandas as pd
from matplotlib import patches

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import tifffile as tiff
from scipy.signal import savgol_filter


path = (r'C:\Users\salva\OneDrive - Massachusetts Institute of Technology\Coregistration'
        r' SRS-confocal\data\20260511_skin\Project_srs_skin_hyperspectral_merged.tif')

image = tiff.imread(path)
print(image.shape)  # probably (125, 480, 448)

plt.figure()
vmin = np.percentile(image, 60)
vmax = np.percentile(image, 99)
plt.imshow(image[118, :, :], cmap='gray', vmin=vmin, vmax=vmax)
plt.show(block=False)

wn_start = 1750 #cm-1

stokes = 1031.2 #nm
steps = 296
step_size = 0.3 #nm

# convert pump in nanometers
def convert_pump_nm(wn, stokes):
    pump = 1 / (1 / (stokes) + wn/1e7)
    return pump

def convert_raman_cm(pump, stokes):
    raman = 1e7 * (1 / (pump) - 1/stokes)
    return raman

wn_start_nm = convert_pump_nm(wn_start, stokes)
wn_stop_nm = wn_start_nm + step_size * steps

wns = np.linspace(wn_start_nm, wn_stop_nm, steps)

raman_shifts = convert_raman_cm(wns, stokes)

idx = 208

plt.figure()
vmin = np.percentile(image[idx, :, :], 10)
vmax = np.percentile(image[idx, :, :], 99.9)
raman_shift_use = raman_shifts[idx]
plt.imshow(image[idx, :, :], cmap='inferno', vmin=vmin, vmax=vmax)
plt.title(f'Raman shift: {raman_shift_use}')
plt.show()

plt.figure()
plt.plot(raman_shifts, image[:, 520:570, 259:309].mean(axis=(1, 2)))

back = image[:, 520:570, 259:309].mean(axis=(1, 2))
img_fil = image - back[:, None, None]

plt.figure()
plt.plot(raman_shifts, back)

plt.figure()
plt.plot(raman_shifts, img_fil[:, 235, 238])

plt.figure()
plt.plot(raman_shifts, image[:, 426, 248])

idx2 =204
plt.figure()
vmin2 = np.percentile(image[idx2, :, :], 10)
vmax2 = np.percentile(image[idx2, :, :], 99)
plt.imshow(img_fil[idx2, :, :], cmap='inferno', vmin=vmin2, vmax=vmax2)

plt.figure()
plt.imshow(img_fil[idx, :, :], cmap='inferno', vmin=vmin, vmax=vmax, alpha=1)

idx3 = 79
plt.figure()
vmin2 = np.percentile(image[idx, :, :], 10)
vmax2 = np.percentile(image[idx, :, :], 99.9)
vmin3 = np.percentile(image[idx3, :, :], 10)
vmax3 = np.percentile(image[idx3, :, :], 99.9)
plt.imshow(img_fil[idx, :, :], cmap='Blues', vmin=vmin2, vmax=vmax2)
plt.imshow(img_fil[idx3, :, :], cmap='Reds', vmin=vmin3, vmax=vmax3, alpha=0.5)



arr_data_proc = savgol_filter(img_fil.T, 7, 3)
arr_use = arr_data_proc.copy()

plt.figure()
plt.plot(raman_shifts, arr_data_proc.T[:, 235, 238])


idx3 = 79
plt.figure()
vmin2 = np.percentile(image[idx, :, :], 10)
vmax2 = np.percentile(image[idx, :, :], 99.9)
vmin3 = np.percentile(image[idx3, :, :], 10)
vmax3 = np.percentile(image[idx3, :, :], 99.9)
plt.imshow(arr_data_proc.T[idx, :, :], cmap='inferno', vmin=vmin2, vmax=vmax2)
plt.imshow(arr_data_proc.T[idx3, :, :], cmap='Reds', vmin=vmin3, vmax=vmax3, alpha=0.5)

idx4 = 275

plt.figure()
vmin4 = np.percentile(arr_data_proc.T[idx4, :, :], 80)
vmax4 = np.percentile(arr_data_proc.T[idx4, :, :], 99)
plt.imshow(arr_data_proc.T[idx4, :, :], cmap='inferno', vmin=vmin4, vmax=vmax4)

import numpy as np
import rampy as rp
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

arr_use = arr_data_proc.T.copy()
method = "als"  # Asymmetric Least Squares
lam = 1e4
p = 0.01
niter = 10

wv_ind = np.arange(arr_use.shape[0])[:]
roi = np.array([[np.min(wv_ind), np.max(wv_ind)]])

pixels = np.ascontiguousarray(
    arr_use.reshape(arr_use.shape[0], -1).astype(np.float32, copy=False)
).T
N, n_w = pixels.shape

corr_flat = np.empty((N, n_w), dtype=np.float32)

chunk = 512  # try 128/256/512

def process_chunk(start):
    stop = min(start + chunk, N)
    corr_blk = np.empty((stop - start, n_w), dtype=np.float32)
    base_blk = np.empty((stop - start, n_w), dtype=np.float32)

    # evita che librerie numeriche interne aprano troppi thread per processo
    with threadpool_limits(limits=1):
        for k, i in enumerate(range(start, stop)):
            y = pixels[i].reshape(-1)
            corr, baseline = rp.baseline(
                wv_ind, y, roi, method=method, lam=lam, p=p, niter=niter, polynomial_order=3
            )
            corr_blk[k] = np.asarray(corr, dtype=np.float32).reshape(-1)
            base_blk[k] = np.asarray(baseline, dtype=np.float32).reshape(-1)

    return start, stop, corr_blk, base_blk

starts = list(range(0, N, chunk))

results = []

results = Parallel(n_jobs=15, backend="loky", pre_dispatch="2*n_jobs", verbose=10)(
        delayed(process_chunk)(start) for start in starts
    )

for start, stop, corr_blk, base_blk in results:
    corr_flat[start:stop] = corr_blk
shifted_basesub_resh = np.array(corr_flat).reshape(559, 557, -1)

shifted_basesub_resh2 = shifted_basesub_resh - shifted_basesub_resh.min(axis=2)[:, :, None]
plt.figure()
plt.plot(raman_shifts, shifted_basesub_resh2.T[:, 235, 238])

pd.to_pickle(shifted_basesub_resh2.T, r'C:\Users\salva\OneDrive - Massachusetts Institute of Technology\Coregistration'
            r' SRS-confocal\data\20260511_skin\skin_srs_hyperspectral.pickle')

pd.to_pickle(raman_shifts, r'C:\Users\salva\OneDrive - Massachusetts Institute of Technology\Coregistration'
        r' SRS-confocal\data\20260511_skin\srs_Raman_shift.pickle')