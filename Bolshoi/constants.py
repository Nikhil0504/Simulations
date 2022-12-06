import platform

from imports import np, os, plt

# MAIN CONSTANTS
PERCENT = 10
BIN_NO = 25
UPPER_LIMIT = 1e15
LOWER_LIMIT = 1e11

RADIUS_BINS = np.logspace(-2, 1, BIN_NO + 1)
MASS_BINS = np.logspace(11, 14, 11)

MASS = 1.35 * (10**8)
RADIUS = (RADIUS_BINS[1:] + RADIUS_BINS[:-1]) / 2.0

# PLOTTING SPECS
plt.style.use(["science"])  # type: ignore
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = (5, 5)
plt.rcParams["xtick.major.size"] = 12
plt.rcParams["xtick.minor.size"] = 7
plt.rcParams["ytick.major.size"] = 12
plt.rcParams["ytick.minor.size"] = 7
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["font.size"] = 15
plt.rcParams["legend.fontsize"] = 12
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 300


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
