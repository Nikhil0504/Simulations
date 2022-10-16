from imports import np, os, plt

# PLOTTING SPECS
plt.style.use(["science"])  # type: ignore
# plt.rcParams.update({'figure.dpi': '200'})
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = (20, 20)
plt.rcParams["xtick.major.size"] = 15
plt.rcParams["xtick.minor.size"] = 7
plt.rcParams["ytick.major.size"] = 15
plt.rcParams["ytick.minor.size"] = 7
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["font.size"] = 30
plt.rcParams["legend.fontsize"] = 20

# MAIN CONSTANTS
MASS_BINS = np.logspace(11, 15, 20)

# PATHS
if os.name != "posix":
    ABS_PATH = "/Users/nikhilgaruda/Desktop/Simulations/BolshoiP"
else:
    ABS_PATH = "/spiff/nikhilgaruda"

HALOS_PATH = os.path.join(ABS_PATH, "hlist_1.00231.list")

if not os.path.exists(f'{ABS_PATH}/cache/'):
    os.mkdir(f'{ABS_PATH}/cache/')

CACHE_PATH = os.path.join(ABS_PATH, "cache/BolshoiP")