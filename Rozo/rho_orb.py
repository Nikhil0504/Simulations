from loading import *

from os.path import join

plt.style.use(['science', 'grid'])

NIKHIL_PATH = '/home/nikhilgaruda/Simulations/Rozo/out'

MBINEDGES_RAW = np.array([13.40, 13.55, 13.70, 13.85, 14.00, 14.15, 14.30, 14.45, 14.65, 15.00])
MBINEDGES = 10 ** MBINEDGES_RAW

# open h5 file to get the halo information
hdf = h5.File(join(NIKHIL_PATH, 'catalogue/halo_densities_full.h5'), 'r')
print(hdf.keys())

# create a figure to plot the results
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# iterate over each bin
for i in range(len(MBINEDGES) - 1):
    rho = rho_orb(np.median(MBINEDGES[i:i + 2]))

    # constrain the h5 file to the bin
    mask = (hdf['Morb'][:] >= MBINEDGES[i]) & (hdf['Morb'][:] < MBINEDGES[i + 1])
    h5_rho = hdf['rho_o'][:][mask] # type: ignore

    # take mean of the density values
    h5_rho_mean = np.mean(h5_rho, axis=0)

    # plot r vs r**2 * rho in log scale with the color of the bin
    ax.plot(RADIUS, RADIUS**2 * rho, label=f'{MBINEDGES_RAW[i]}')

    ax.scatter(RADIUS, RADIUS**2 * h5_rho_mean, s=3)

    # set the x and y axis labels
    ax.set_xlabel(r'$r [h^{-1} \rm{Mpc}]$')
    ax.set_ylabel(r'$r^2 \rho_{\rm{orb}}(r|M)$')

    # set the x and y axis limits
    ax.set_ylim(1e9, 1e15)
    ax.set_xlim(1e-2, 10 ** 0.7)

    # set the x and y axis scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # set the legend
    ax.legend()

# save the figure
fig.savefig(join('/home/nikhilgaruda/Simulations/Rozo/out', 'figures/rho_orb.pdf'), dpi=300)



