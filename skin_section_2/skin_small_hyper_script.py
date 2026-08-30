import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

import tifffile as tiff
import rampy as rp
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits


path = (r'C:\Users\salva\OneDrive - Massachusetts Institute of Technology\Coregistration'
        r' SRS-confocal\data\20260518_colon\colon_hyperspectral_srs.tif')

image = tiff.imread(path)
print(image.shape)  # probably (125, 480, 448)

plt.figure()
vmin = np.percentile(image, 60)
vmax = np.percentile(image, 99)
plt.imshow(image[118, :, :], cmap='gray', vmin=vmin, vmax=vmax)
plt.show(block=False)

wn_start = 1750 #cm-1

stokes = 1031.8 #nm
steps = 125
step_size = 0.7 #nm

image = tiff.imread(path)
print(image.shape)  # probably (125, 480, 448)

plt.figure()
vmin = np.percentile(image, 20)
vmax = np.percentile(image, 99)
plt.imshow(image[118, :, :], cmap='gray', vmin=vmin, vmax=vmax)
plt.show(block=False)

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

idx = 24

plt.figure()
vmin = np.percentile(image[idx, :, :], 10)
vmax = np.percentile(image[idx, :, :], 99.9)
raman_shift_use = raman_shifts[idx]
plt.imshow(image[idx, :, :], cmap='inferno', vmin=vmin, vmax=vmax)
plt.title(f'Raman shift: {raman_shift_use}')
plt.show()

back = image[:, 450:, 360:].mean(axis=(1, 2))
img_fil = image - back[:, None, None]

plt.figure()
plt.plot(raman_shifts, back)

plt.figure()
plt.plot(raman_shifts, img_fil[:, 400, 308])

plt.figure()
plt.plot(raman_shifts, image[:, 400, 308])

idx2 =10
plt.figure()
vmin2 = np.percentile(image[idx2, :, :], 10)
vmax2 = np.percentile(image[idx2, :, :], 99.9)
plt.imshow(img_fil[idx2, :, :], cmap='inferno', vmin=vmin2, vmax=vmax2)
plt.imshow(img_fil[idx, :, :], cmap='Reds', vmin=vmin, vmax=vmax, alpha=0.5)

arr_data_proc = img_fil.T.copy()
arr_use = arr_data_proc.copy()

plt.figure()
plt.plot(raman_shifts, arr_data_proc.T[:, 400, 308])

idx4 = 45

plt.figure()
vmin4 = np.percentile(arr_data_proc.T[idx4, :, :], 80)
vmax4 = np.percentile(arr_data_proc.T[idx4, :, :], 99)
plt.imshow(arr_data_proc.T[idx4, :, :], cmap='inferno', vmin=vmin4, vmax=vmax4)

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
shifted_basesub_resh = np.array(corr_flat).reshape(468, 485, -1)

shifted_basesub_resh2 = shifted_basesub_resh - shifted_basesub_resh.min(axis=2)[:, :, None]
plt.figure()
plt.plot(raman_shifts, shifted_basesub_resh2.T[:, 400, 308])

pd.to_pickle(shifted_basesub_resh2, r'C:\Users\salva\OneDrive - Massachusetts Institute of Technology\Coregistration'
            r' SRS-confocal\data\skin_small_hyperspectral\skin_srs_small_sampling_hyperspectral')
pd.to_pickle(raman_shifts, r'C:\Users\salva\OneDrive - Massachusetts Institute of Technology\Coregistration'
        r' SRS-confocal\data\skin_small_hyperspectral\srs_Raman_shift.pickle')
