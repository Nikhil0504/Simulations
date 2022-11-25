# plots for jackknife (cached points)
from loading import *

sim_points = np.load(f"{CACHE_PATH}/simulation_points.npy")

# Hard coding the desired locations to get
# 4 ranges of halos
locations = np.array([[2, 31, 114]])
m_location = np.argmax(data_points)
locations = np.append(locations, m_location)

a = np.arange(1, 10)
iloc = 1

plt.subplots()

for loc in locations:
    plt.subplot(2, 2, iloc)
    iloc += 1

    M = data_points[loc]
    Rvir = rvir[loc] / 1000
    Rs = rs[loc] / 1000
    cvir = Rvir / Rs
    X, Y, Z = x[loc], y[loc], z[loc]

    arr_points_2 = get_points(X, Y, Z, arr_points)
    for i in a:
        array = arrays(arr_points_2, X, Y, Z, i=i)
        R = compute_R2(array)

        pairs, _ = np.histogram(R, bins=RADIUS_BINS)
        total_mass = np.array(pairs) * MASS * (100 / PERCENT)

        if i == 1:
            volume = Volume(1)
        else:
            volume = Volume(7.0 / 8.0)

        mask = np.where(RADIUS < rvir[loc] / 1000)
        obs = total_mass / volume  # type: ignore
        obs = obs[mask]
        c_inv = cinv(obs)

        optres = iminuit.minimize(cost, [np.log(10)], args=(obs, c_inv, M, Rvir, 2))
        oRs = Rvir / np.exp(optres.x)
        ocos = cost(np.exp(optres.x), obs, c_inv, M, Rvir)
        orad, orhos = rho_r(oRs, M, Rvir)

        if i == 1:
            # plt.plot(RADIUS[mask], obs, linewidth=1, label=f'Full Sample', color='red', zorder=10)
            plt.plot(
                orad, orhos, linewidth=1, label=f"Full Sample", color="red", zorder=10
            )
        else:
            # plt.plot(RADIUS[mask], obs, linewidth=0.5, color='black', zorder=1)
            plt.plot(orad, orhos, linewidth=0.5, color="black", zorder=1)
        # plt.plot(orad, orhos, linewidth=1, marker='*', label=f'array{i+1} {b[i]}')
        plt.xscale("log")
        plt.yscale("log")

    plt.plot(
        [], [], color="black", linewidth=3, marker="None", label="Jackknife Samples"
    )
    plt.title(f"Halo Mass: {np.log10(M):.2f}")
    plt.axvline(x=Rs, color="darkgreen", label="Rs location")
    # plt.axvline(x=Rvir, color='g', label='Rvir location')
    # plt.axvspan(Rvir, bins2[-1], color='gray', alpha=0.3)
    plt.xlabel(r"$r (\mathrm{Mpc}/h)$")
    plt.ylabel(r"$\rho (M_{\odot} h^2 / \mathrm{Mpc}^3)$")
    plt.legend(loc="upper right", prop={"size": 13})

plt.savefig(f"figures/4_jacks_2.png")
