import time

from constants import *
from functions import *
from imports import *

st = time.time()
mvir = np.load(f"{CACHE_PATH}/halofunc_points.npy", mmap_mode='r')
x = np.load(f"{CACHE_PATH}/x_points.npy", mmap_mode='r')
y = np.load(f"{CACHE_PATH}/y_points.npy", mmap_mode='r')
z = np.load(f"{CACHE_PATH}/z_points.npy", mmap_mode='r')
rvir = np.load(f"{CACHE_PATH}/rvir_points.npy", mmap_mode='r')
rs = np.load(f"{CACHE_PATH}/rs_points.npy", mmap_mode='r')
arr_points = np.load(f"{CACHE_PATH}/arr_points.npy", mmap_mode='r')
print("Loaded Cache Files")
print(f"Time taken to load cache files: {time.time()-st} s")

np.random.seed(10)
