import time

from constants import *
from imports import *

st = time.time()
data_points = np.load(f"{CACHE_PATH}/halofunc_points.npy")
x = np.load(f"{CACHE_PATH}/x_points.npy")
y = np.load(f"{CACHE_PATH}/y_points.npy")
z = np.load(f"{CACHE_PATH}/z_points.npy")
rvir = np.load(f"{CACHE_PATH}/rvir_points.npy")
rs = np.load(f"{CACHE_PATH}/rs_points.npy")
arr_points = np.load(f"{CACHE_PATH}/arr_points.npy")
print("Loaded Cache Files")
print(f"Time taken to load cache files: {time.time()-st:2.2f}")

np.random.seed(10)