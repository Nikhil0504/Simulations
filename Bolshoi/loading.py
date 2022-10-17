from constants import *
from functions import *
from imports import *

data_points = np.load(f'{CACHE_PATH}/halofunc_points.npy')
x = np.load(f'{CACHE_PATH}/x_points.npy')
y = np.load(f'{CACHE_PATH}/y_points.npy')
z = np.load(f'{CACHE_PATH}/z_points.npy')
rvir = np.load(f'{CACHE_PATH}/rvir_points.npy')
rs = np.load(f'{CACHE_PATH}/rs_points.npy')
arr_points = np.load(f'{CACHE_PATH}/arr_points.npy')

np.random.seed(10)