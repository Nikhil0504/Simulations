# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
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
plt.rcParams["font.size"] = 25
plt.rcParams["legend.fontsize"] = 30
plt.rcParams['xtick.direction'] = 'inout'
plt.rcParams['ytick.direction'] = 'inout'

# %%
BIN_NO = 25
RADIUS_BINS = np.logspace(-2, 1, BIN_NO + 1)
RADIUS = (RADIUS_BINS[1:] + RADIUS_BINS[:-1]) / 2.0

# %%
fi = np.loadtxt('inds.txt')

# %%
slices = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1497, 1545, 1568]

# %%
for ind, halo in enumerate(fi):
    for i in range(9):
        if i == 0:
            plt.plot(RADIUS, halo[3+(i*BIN_NO):3+((i+1)*BIN_NO)], linewidth=2, label=f"Full Sample", color="red", zorder=10)
        else:
            plt.plot(RADIUS, halo[3+(i*BIN_NO):3+((i+1)*BIN_NO)], linewidth=0.5, color="black", zorder=1)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"$r (\mathrm{Mpc}/h)$")
        plt.ylabel(r"$\rho (M_{\odot} h^2 / \mathrm{Mpc}^3)$")
        plt.title(f'Halo Mass: {np.log10(halo[1]):.2f}')
    plt.plot(
            [], [], color="black", linewidth=3, marker="None", label="Jackknife Samples"
        )
    plt.axvspan(halo[2], RADIUS_BINS[-1], color='gray', alpha=0.3)
    plt.legend()
    plt.savefig(f'figures/cache_test/{ind}_hm_{np.log10(halo[1]):.0f}.png')
    plt.clf()

# %%
avg = np.average(fi[:, 3:], axis=0)

# %%
for i in range(10):
    avg = np.average(fi[slices[i]:slices[i+1] + 1, 3:], axis=0)
    jack = 0
    if jack == 0:
        plt.plot(RADIUS, avg[(jack*BIN_NO):((jack+1)*BIN_NO)], linewidth=2, color=f"red", zorder=10, alpha=(i+1)/10)
    else:
        plt.plot(RADIUS, avg[(jack*BIN_NO):((jack+1)*BIN_NO)], linewidth=0.5, color="black", zorder=1)
    plt.xscale('log')
    plt.yscale('log')
plt.xlabel(r"$r (\mathrm{Mpc}/h)$")
plt.ylabel(r"$\rho (M_{\odot} h^2 / \mathrm{Mpc}^3)$")
plt.plot(
    [], [], color="red", linewidth=3, marker="None", label="Average Density Profiles"
)
plt.legend()
plt.savefig('figures/avg_density_prof.png')

# %%



