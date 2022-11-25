from loading import *

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
    # print('actual cvir', cvir)
    # print('actual rs', Rs)

    X, Y, Z = x[locations[loc]], y[locations[loc]], z[locations[loc]]
    arr_points_2 = get_points(X, Y, Z, arr_points)

    # R = compute_R2(arrays(arr_points_2, X, Y, Z, 1))
    R = compute_R(X, Y, Z, arr_points_2)
    pairs, _ = np.histogram(R, bins=RADIUS_BINS)
    total_mass = np.array(pairs) * MASS * (100 / PERCENT)

    volume = Volume(1)

    obs = total_mass / volume

    mask = np.where(RADIUS < rvir[locations[loc]] / 1000)
    obs = obs[mask]

    c_inv = cinv(obs)

    # rad, rhos = rho_r(Rs, M, Rvir)
    # cos = cost(cvir, obs, c_inv, M, Rvir, 0)

    optres = iminuit.minimize(
        cost, [np.log(10)], args=(obs, c_inv, M, Rvir, "gaussian")
    )
    # print('gau cvir', np.exp(optres.x))
    oRs = Rvir / np.exp(optres.x)
    # print('gau rs', oRs)
    # ocos = cost(optres.x, obs, c_inv, M, Rvir, 0)
    orad, orhos = rho_r(oRs, M, Rvir)

    optres2 = iminuit.minimize(
        cost, [np.log(10)], args=(obs, c_inv, M, Rvir, "lorentz")
    )
    # print('lor cvir', np.exp(optres2.x))
    oRs2 = Rvir / np.exp(optres2.x)
    # print('lor rs', oRs2)
    # ocos2 = cost(optres2.x, obs, c_inv, M, Rvir, 1)
    orad2, orhos2 = rho_r(oRs2, M, Rvir)

    optres3 = iminuit.minimize(cost, [np.log(10)], args=(obs, c_inv, M, Rvir, "abs"))
    oRs3 = Rvir / np.exp(optres3.x)
    # ocos2 = cost(optres2.x, obs, c_inv, M, Rvir, 2)
    orad3, orhos3 = rho_r(oRs3, M, Rvir)

    # plt.plot(RADIUS[mask], obs, '--', color='black', linewidth=2, label=f"simulation, cvir={cvir:.5f}")
    # plt.plot(
    #     rad, rhos, linewidth=1, label=f"Bolshoi NFW"
    # )
    plt.plot(
        orad,
        orhos,
        "r",
        linewidth=1,
        label=f"Gaussian Uncertianties, cvir = {np.exp(optres.x)}",
    )
    plt.plot(
        orad2,
        orhos2,
        "g",
        linewidth=1,
        label=f"Lorentz Distribution Uncertianties, cvir = {np.exp(optres2.x)}",
    )
    plt.plot(
        orad3,
        orhos3,
        "b",
        linewidth=1,
        label=f"Absolute Distribution Uncertianties, cvir = {np.exp(optres3.x)}",
    )
    # plt.axvline(x=Rs, color="r", label="Rs location")
    # plt.axvline(x=oRs, color="m", label="Optimised Rs location")
    # plt.axvline(x=Rvir, color="g", label="Rvir location")
    # plt.axvspan(Rvir, RADIUS_BINS[-1], color='gray', alpha=0.3)

    plt.title(f"Halo Mass: {np.log10(M):.2f}")
    plt.xlabel(r"$r (\mathrm{Mpc}/h)$")
    plt.ylabel(r"$\rho (M_{\odot} h^2 / \mathrm{Mpc}^3)$")

    plt.xscale("log")
    plt.yscale("log")
    plt.legend(prop={"size": 13})

plt.savefig("figures/NFW_opti_vs_actual_4")
