from loading import *

a = np.arange(9)
total = None

bins = MASS_BINS

for ind in range(bins.size - 1):
    m_ind = np.nonzero((data_points <= bins[ind + 1]) & (data_points > bins[ind]))[0]
    N_samples = min([len(m_ind), 100])
    r_ind = np.random.choice(m_ind, N_samples, replace=False)

    print(f"Samples: {N_samples}")

    main = r_ind.reshape(-1, 1)
    _M = data_points[r_ind].reshape(-1, 1)
    _c = (rvir[r_ind] / rs[r_ind]).reshape(-1, 1)
    main = np.column_stack((main, _M, _c))
    # print(main.shape)
    opts = np.array([])

    for index in a:
        print(f"{bins[ind]}-{bins[ind + 1]} -> J: {index}")
        for r in r_ind:
            M = data_points[r]
            Rvir = rvir[r] / 1000
            Rs = rs[r] / 1000
            cvir = (Rvir) / Rs

            X, Y, Z = x[r], y[r], z[r]
            arr_points_2 = get_points(X, Y, Z, arr_points)

            R = compute_R2(arrays(arr_points_2, X, Y, Z, index + 1))
            pairs, _ = np.histogram(R, bins=RADIUS_BINS)
            total_mass = np.array(pairs) * MASS * (100 / PERCENT)

            if index == 1:
                volume = Volume(1)
            else:
                volume = Volume(7.0 / 8.0)

            obs = total_mass / volume

            mask = np.where(RADIUS < rvir[r] / 1000)

            obs = obs[mask]
            c_inv = cinv(obs)

            optres = iminuit.minimize(cost, [10], args=(obs, c_inv, M, Rvir))
            opts = np.append(opts, optres.x)

        opts = opts.reshape(-1, 1)
        # print(opts.shape)
        main = np.column_stack((main, opts))
        opts = np.array([])
        # print(main.shape)

    if total is None:
        total = main
    else:
        total = np.vstack((total, main))  # type: ignore

m_c = np.mean(total[:, 4:], axis=1)  # type: ignore
total = np.column_stack((total, m_c))
# total - id, M, actual cvir, opt cvir, 8 jacknifes (j_c_i), mean of j_c
np.save(f"{CACHE_PATH}/jackknife_cvirs.npy", total)
np.savetxt(f"{CACHE_PATH}/jackknife_cvirs.out", total)
print(total.shape)
