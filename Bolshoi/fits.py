from loading import *

# locations = np.array([[2, 31, 114]])
# m_location = np.argmax(data_points)
# locations = np.append(locations, m_location)
bins = np.logspace(11, 11.5, 2)
m_ind = np.nonzero((data_points <= bins[1]) & (data_points > bins[0]))[0]
locations = np.random.choice(m_ind, 4, replace=False)

cvirs = np.linspace(np.log(1), np.log(30), 1000)

i = 1

for loc in locations:
    plt.subplot(2, 2, i)
    i += 1

    M = data_points[loc]
    Rvir = rvir[loc] / 1000
    Rs = rs[loc] / 1000
    cvir = Rvir / Rs

    X, Y, Z = x[loc], y[loc], z[loc]
    arr_points_2 = get_points(X, Y, Z, arr_points)

    R = compute_R2(arrays(arr_points_2, X, Y, Z, 1))
    pairs, _ = np.histogram(R, bins=RADIUS_BINS)
    total_mass = np.array(pairs) * MASS * (100 / PERCENT)

    volume = Volume(1)

    obs = total_mass / volume

    mask = np.where(RADIUS < rvir[loc] / 1000)
    obs = obs[mask]

    c_inv = cinv(obs)

    # generate random costs for random cvirs
    b = np.array([])
    bl = np.array([])
    for c in cvirs:
        cos = cost(c, obs, c_inv, M, Rvir, "gaussian")
        b = np.append(b, cos)
        cos1 = cost(c, obs, c_inv, M, Rvir, "lorentz")
        bl = np.append(bl, cos1)

    optres = iminuit.minimize(
        cost, [np.log(10)], args=(obs, c_inv, M, Rvir, "gaussian")
    )
    optresl = iminuit.minimize(
        cost, [np.log(10)], args=(obs, c_inv, M, Rvir, "lorentz")
    )

    plt.scatter(np.exp(cvirs), bl)
    plt.scatter(np.exp(cvirs), b)
    plt.axvline(cvir, c="g", label="actual cvir")
    plt.axvline(np.exp(optres.x), c="r", label="optimised cvir")
    plt.axvline(np.exp(optresl.x), c="r", label="optimised lorentz cvir")
    plt.axvline(np.exp(cvirs)[np.argmin(b)], label="least chisq cvir")
    plt.axvline(np.exp(cvirs)[np.argmin(bl)], label="least chisq lorentz cvir")
    # plt.ylim(0, 50)
    # plt.xlim(0, 30)

    plt.title(f"Halo Mass: {np.log10(M):.2f}")
    plt.legend(prop={"size": 13})

plt.savefig("figures/fits_10e11.png")
