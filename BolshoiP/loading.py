from constants import *
from imports import *

data_points = np.load(f"{CACHE_PATH}/halofunc_points.npy")
r_vir = np.load(f"{CACHE_PATH}/rvir_points.npy")
r_sk = np.load(f"{CACHE_PATH}/rsk_points.npy")
hmass_scale = np.load(f"{CACHE_PATH}/hmscale_points.npy")
mean_age = np.load(f"{CACHE_PATH}/mean_age.npy")
mean_mass = np.load(f"{CACHE_PATH}/mean_mass.npy")
std_age = np.load(f"{CACHE_PATH}/std_mass.npy")
