import time

from constants import *
from functions import *
from imports import *

st = time.time()

MVIR = np.load(f"{CACHE_PATH}/halofunc_points.npy", mmap_mode='r')
X = np.load(f"{CACHE_PATH}/x_points.npy", mmap_mode='r')
Y = np.load(f"{CACHE_PATH}/y_points.npy", mmap_mode='r')
Z = np.load(f"{CACHE_PATH}/z_points.npy", mmap_mode='r')
RVIR = np.load(f"{CACHE_PATH}/rvir_points.npy", mmap_mode='r')
RS = np.load(f"{CACHE_PATH}/rs_points.npy", mmap_mode='r')
ARR_POINTS = np.load(f"{CACHE_PATH}/arr_points.npy", mmap_mode='r')

print("Loaded Cache Files")
print(f"Time taken to load cache files: {time.time()-st} s")

np.random.seed(10)
