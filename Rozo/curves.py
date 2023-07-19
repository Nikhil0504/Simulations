import tqdm
from loading import *

from scipy import integrate

plt.style.use(['science', 'grid'])
plt.rcParams['legend.frameon'] = False


def M200_densities(file, save=True):
    M200m = file['M200m'][:]
    R200m = file['R200m'][:]
    rho = file['densities'][:]

    plt.figure(figsize=(8, 6), dpi=300)

    for bin in range(len(MBINEDGES) - 1):
        plt.subplot(3, 3, bin + 1)

        print(MBINEDGES[bin], MBINEDGES[bin + 1])

        mask = (M200m > MBINEDGES[bin]) & (M200m < MBINEDGES[bin + 1])
        mask_den = rho[mask]
        mask_R200m = R200m[mask]
        mask_M200m = M200m[mask]

        print(mask_den.shape)

        for halo in tqdm.tqdm(range(250)):
            density = mask_den[halo]
            # mask where density is not zero
            h_mask = density != 0

            x = RADIUS[h_mask] / mask_R200m[halo]
            rho_x = density[h_mask] * (mask_R200m[halo]**3 / mask_M200m[halo])
            
            # plot density profile
            # plt.loglog(RADIUS[h_mask] / mask_R200m[halo], 
            #            RADIUS[h_mask]**2 * density[h_mask] * (mask_R200m[halo]**3  / m), 
            #            color='k', alpha=0.05, linewidth=0.5)
            plt.loglog(x, x**2 * rho_x, color='k', alpha=0.05, linewidth=0.5)

        plt.plot([], [], 
                 label=f'log10(M200) = {MBINEDGES_RAW[bin]} - {MBINEDGES_RAW[bin + 1]}', 
                 alpha=0, linestyle=' ')

        if bin == 6:
            plt.xlabel(r'$x = r / R_{200}$')
            plt.ylabel(r'$x^2 \rho({x})$')

        # grey region from start of x axis to 6 * RSOFT
        # plt.axvspan(RADIUS_BINS[0], 6 * RSOFT, color='grey', alpha=0.5)

        plt.legend(fontsize=7)
        # plt.ylim(1e10, 1e14)
        plt.tight_layout()
    
    if save:
        plt.savefig(join(NIKHIL_PATH, 'figures', 'density_profiles.png'), dpi=300)
        plt.clf()


def Morb_densities(file, save=True):
    morb = file['Morb'][:]
    rh = file['rh'][:]
    rho = file['rho'][:]

    plt.figure(figsize=(8, 6), dpi=300)

    for bin in range(len(MBINEDGES) - 1):
        plt.subplot(3, 3, bin + 1)

        print(MBINEDGES[bin], MBINEDGES[bin + 1])

        mask = (morb > MBINEDGES[bin]) & (morb < MBINEDGES[bin + 1])
        mask_den = rho[mask]
        mask_rh = rh[mask]
        mask_m = morb[mask]

        print(mask_den.shape)

        for halo in tqdm.tqdm(range(250)):
            density = mask_den[halo]
            # mask where density is not zero
            mask = density != 0


            RH = (843.8 / 1000) * (mask_m[halo] / PIVOT_MASS)**(0.223)
            x = RADIUS[mask] / RH
            rho_x = density[mask] * (RH**3 / mask_m[halo])
            # plot density profile
            # plt.loglog(RADIUS[mask] / RH, 
            #            RADIUS[mask]**2 * (density[mask] * RH**3 / m), 
            #            color='k', alpha=0.05, linewidth=0.5)
            plt.loglog(x, x**2 * rho_x, color='k', alpha=0.05, linewidth=0.5)

        plt.plot([], [], 
                 label=f'log10(Morb) = {MBINEDGES_RAW[bin]} - {MBINEDGES_RAW[bin + 1]}', 
                 alpha=0, linestyle=' ')

        if bin == 6:
            plt.xlabel(r'$x = \frac{r}{r_h}$')
            plt.ylabel(r'$x^2 \rho({x})$')
        
        # grey region from start of x axis to 6 * RSOFT
        # plt.axvspan(RADIUS_BINS[0], 6 * RSOFT, color='grey', alpha=0.5)

        plt.legend(fontsize=7)
        plt.xlim(1e-2, 1e1)
        # plt.ylim(1e10, 1e14)
        plt.tight_layout()
    
    if save:
        plt.savefig(join(NIKHIL_PATH, 'figures', 'density_profiles_morb.png'), dpi=300)
        plt.clf()

def rh_hist(file=join(NIKHIL_PATH, 'rh_fits.h5'), save=True):
    with h5.File(file, 'r') as hdf:
        rh_x = hdf['rh'][:] / 1000 # type: ignore
        morb = hdf['Morb'][:] # type: ignore
        rh = rh_x * (morb / PIVOT_MASS)**(0.223) # type: ignore

    fig = plt.figure(figsize=(8, 6), dpi=300)

    for bin in range(MBINEDGES.shape[0] - 1):
        plt.subplot(3, 3, bin + 1)

        mask = (MBINEDGES[bin] < morb) & (morb < MBINEDGES[bin + 1])

        plt.hist(rh[mask], bins=100) # type: ignore
        plt.plot([], [], ' ', label=r'$\log_{10}(M_{orb}): %.2f-%.2f$' % (MBINEDGES_RAW[bin], MBINEDGES_RAW[bin + 1]))

        plt.legend(prop={'size': 6})

        if bin == 6:
            plt.xlabel(r'$r_{\rm{h}} (\mathrm{Mpc})$')
    
    if save:
        plt.savefig(join(NIKHIL_PATH, 'figures', 'rh_hist.png'), dpi=300)
        plt.clf()
    # plt.show()

if __name__ == '__main__':
    MBINEDGES_RAW = [13.40, 13.55, 13.70, 13.85, 14.00, 14.15, 14.30, 14.45, 14.65, 15.00]
    # take 10th power of it
    MBINEDGES = np.power(10, MBINEDGES_RAW)

    # file = h5.File(join(NIKHIL_PATH, 'catalogue/m200_densities.h5'), 'r')
    # M200_densities(file, save=True)
    # file.close()

    rh_hist(save=True)
    
    file = h5.File(join(NIKHIL_PATH, 'rh_fits.h5'), 'r')
    Morb_densities(file, save=True)
    file.close()
