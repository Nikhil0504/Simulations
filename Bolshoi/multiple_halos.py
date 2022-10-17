from loading import *

mean_ms = np.array([])
mean_cvirs = np.array([])
std_errs_cvirs = np.array([])

bins = MASS_BINS
print('Starting Bins')
for ind in range(bins.size - 1):
    m_ind = np.nonzero((data_points <= bins[ind + 1]) & (data_points > bins[ind]))[0]
    N_samples = min([len(m_ind), 100])

    r_ind = np.random.choice(m_ind, N_samples, replace=False)

    mean_m = np.mean(data_points[r_ind])
    mean_ms = np.append(mean_ms, mean_m)

    opts = np.array([])
    
    for r in r_ind:
        M = data_points[r]
        Rvir = rvir[r] / 1000
        Rs = rs[r] / 1000
        cvir = (Rvir) / Rs

        X, Y, Z = x[r], y[r], z[r]
        arr_points_2 = get_points(X, Y, Z, arr_points)

        R = compute_R(X, Y, Z, arr_points_2)
        pairs, _ = np.histogram(R, bins=RADIUS_BINS)
        total_mass = np.array(pairs) * MASS * (100 / PERCENT)

        volume = Volume(1)

        obs = total_mass / volume

        mask = np.where(RADIUS < rvir[r] / 1000)

        obs = obs[mask]
        c_inv = cinv(obs)

        optres = iminuit.minimize(cost, [10], args=(obs, c_inv, M, Rvir))
        opts = np.append(opts, optres.x)

    mean_opts = np.mean(opts)
    std_err = np.std(opts)

    mean_cvirs = np.append(mean_cvirs, mean_opts)
    std_errs_cvirs = np.append(std_errs_cvirs, std_err)

plt.errorbar(mean_ms, mean_cvirs, std_errs_cvirs)
plt.xscale("log")
plt.savefig(f"figures/errorbars_halos")
plt.clf()

plt.hist2d(
    mean_ms,
    mean_cvirs,
    bins=[np.logspace(11, 15, 50), np.linspace(0, 30, 25)],
    norm=mpl.colors.LogNorm(),
    cmap="YlGnBu",
)
plt.xscale("log")
plt.colorbar()
plt.savefig(f"figures/hist2d_halos")
