from loading import *

sim_points = np.load(f'{CACHE_PATH}/simulation_points.npy')

# Hard coding the desired locations to get
# 4 ranges of halos
locations = np.array([[2, 31, 114]])
m_location = np.argmax(data_points)
locations = np.append(locations, m_location)

for loc in range(len(locations)):
    plt.subplot(2, 2, loc + 1)

    M = data_points[locations[loc]]
    Rvir = rvir[locations[loc]] / 1000
    Rs = rs[locations[loc]] / 1000
    cvir = Rvir / Rs

    mask = np.where(RADIUS < rvir[locations[loc]] / 1000)
    obs = sim_points[loc, :]
    obs = obs[mask]
    c_inv = cinv(obs)

    rad, rhos = rho_r(Rs, M, Rvir)
    cos = cost(cvir, obs, c_inv, M, Rvir)

    optres = iminuit.minimize(cost, [10], args=(obs, c_inv, M, Rvir))
    oRs = Rvir/optres.x
    ocos = cost(optres.x, obs, c_inv, M, Rvir)
    orad, orhos = rho_r(oRs, M, Rvir)

    # plt.plot(RADIUS, sim_points[loc, :], linewidth=0.5, label=f'simulation')
    plt.plot(RADIUS[mask], sim_points[loc, :][mask], linewidth=0.5, label=f'simulation')
    plt.plot(rad, rhos, linewidth=2, label=f'Bolshoi NFW $\\rightarrow$ {-0.5 * cos:.2f}')
    plt.plot(orad, orhos, linewidth=2, label=f'Optimised NFW $\\rightarrow$  {-0.5 * ocos:.2f}')
    plt.axvline(x=Rs, color='r', label='Rs location')
    plt.axvline(x=oRs, color='m', label='Optimised Rs location')
    plt.axvline(x=Rvir, color='g', label='Rvir location')
    # plt.axvspan(Rvir, RADIUS_BINS[-1], color='gray', alpha=0.3)

    plt.title(f'Halo Mass: {np.log10(M):.2f}')
    plt.xlabel(r'$r (\mathrm{Mpc}/h)$')
    plt.ylabel(r'$\rho (M_{\odot} h^2 / \mathrm{Mpc}^3)$')

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

plt.savefig('figures/NFW_opti_vs_actual_2')
