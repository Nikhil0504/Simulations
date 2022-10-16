from imports import os

# CONSTANTS
PERCENT = 10
BIN_NO = 25
UPPER_LIMIT = 1e15
LOWER_LIMIT = 1e11

# PATHS
if os.name != "posix":
    ABS_PATH = "/Users/nikhilgaruda/Desktop/Simulations/Bolshoi"
else:
    ABS_PATH = "/spiff/nikhilgaruda"

HALOS_PATH = os.path.join(ABS_PATH, "hlist_1.00035.list")
POINTS_PATH = os.path.join(ABS_PATH, "sim_points_10p.csv")

if not os.path.exists(f'{ABS_PATH}/cache/'):
    os.mkdir(f'{ABS_PATH}/cache/')

CACHE_PATH = os.path.join(ABS_PATH, "cache/Bolshoi")