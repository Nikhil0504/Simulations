import platform

from imports import np, os, plt

# MAIN CONSTANTS
PERCENT = 10
BIN_NO = 25
UPPER_LIMIT = 1e15
LOWER_LIMIT = 1e11

RADIUS_BINS = np.logspace(-2, 1, BIN_NO + 1)
MASS_BINS = np.logspace(11, 14, 11)

MASS = 1.35 * (10 ** 8)  # from Bolshoi website -> Msun/h
RADIUS = (RADIUS_BINS[1:] + RADIUS_BINS[:-1]) / 2.0

# PATHS
if platform.system() == "Darwin":
    ABS_PATH = "/Users/nikhilgaruda/Desktop/Simulations/Bolshoi"
else:
    ABS_PATH = "/spiff/nikhilgaruda"

HALOS_PATH = os.path.join(ABS_PATH, "hlist_1.00035.list")
POINTS_PATH = os.path.join(ABS_PATH, "sim_points_10p.csv")

if not os.path.exists(f"{ABS_PATH}/cache/"):
    os.mkdir(f"{ABS_PATH}/cache/")

CACHE_PATH = os.path.join(ABS_PATH, "cache/Bolshoi")
