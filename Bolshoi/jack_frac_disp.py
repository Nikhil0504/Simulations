# plots for jackknife (cached points)
from loading import *

sim_points = np.load(f'{CACHE_PATH}/simulation_points.npy')

# Hard coding the desired locations to get
# 4 ranges of halos
locations = np.array([[2, 31, 114]])
m_location = np.argmax(data_points)
locations = np.append(locations, m_location)

a = np.arange(1, 10)

for loc in locations:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plt.tight_layout()

    M = data_points[loc]
    Rvir = rvir[loc] / 1000
    Rs = rs[loc] / 1000
    cvir = Rvir / Rs
    X, Y, Z = x[loc], y[loc], z[loc]

    arr_points_2 = get_points(X, Y, Z, arr_points)

    opts = []
    for i in a:
        array = arrays(arr_points_2, X, Y, Z, i=i)
        R = compute_R2(array)

        pairs, _ = np.histogram(R, bins=RADIUS_BINS)
        total_mass = np.array(pairs) * MASS * (100 / PERCENT)

        if i == 1:
            volume = Volume(1)
        else:
            volume = Volume(7./8.)
        
        mask = np.where(RADIUS < rvir[loc] / 1000)
        obs = total_mass / volume  # type: ignore
        obs = obs[mask]
        c_inv = cinv(obs)

        optres = iminuit.minimize(cost, [10], args=(obs, c_inv, M, Rvir))
        opts.append(optres.x)

        oRs = Rvir / optres.x

        ocos = cost(optres.x, obs, c_inv, M, Rvir)
        orad, orhos = rho_r(oRs, M, Rvir)

        if i == 1:
            # plt.plot(RADIUS[mask], obs, linewidth=1, label=f'Full Sample', color='red', zorder=10)
            ax1.plot(orad, orhos, linewidth=1, label=f'Full Sample', color='red', zorder=10)
        else:
            # plt.plot(RADIUS[mask], obs, linewidth=0.5, color='black', zorder=1)
            ax1.plot(orad, orhos, linewidth=0.5, color='black', zorder=1)
        # plt.plot(orad, orhos, linewidth=1, marker='*', label=f'array{i+1} {b[i]}') 
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    
    ax1.plot([],[],color='black',linewidth=3,marker='None',label="Jackknife Samples")
    ax1.axvline(x=Rs, color='darkgreen', label='Rs location')
    #plt.axvline(x=Rvir, color='g', label='Rvir location')
    # plt.axvspan(Rvir, bins2[-1], color='gray', alpha=0.3)
    ax1.set_xlabel(r'$r (\mathrm{Mpc}/h)$')
    ax1.set_ylabel(r'$\rho (M_{\odot} h^2 / \mathrm{Mpc}^3)$')
    ax1.legend(loc='upper right')

    ax2.hist(opts[1:]/opts[1], alpha=0.5)
    ax2.set_xlabel('Fractional Concentration')

    plt.suptitle(f'Halo Mass: {np.log10(M):.2f}', y=1.002)
    plt.savefig(f'figures/jack_frac_{np.log10(M):.2f}.png')
    