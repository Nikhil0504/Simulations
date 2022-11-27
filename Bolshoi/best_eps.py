# %%
from loading import *
import concurrent.futures

# %%
a = np.arange(9)
total = None

bins = MASS_BINS[:2]
eps = np.arange(0.01, 0.2, 0.05)
ses = []

# %%
def tst(eps):    
    total = None
    for ind in range(bins.size - 1):
        m_ind = np.nonzero((data_points <= bins[ind + 1]) & (data_points > bins[ind]))[0]
        N_samples = min([len(m_ind), 100])
        r_ind = np.random.choice(m_ind, N_samples, replace=False)

        print(f"{eps} Samples: {N_samples}")

        main = r_ind.reshape(-1, 1)
        _M = data_points[r_ind].reshape(-1, 1)
        _c = (rvir[r_ind] / rs[r_ind]).reshape(-1, 1)
        main = np.column_stack((main, _M, _c))
        opts = np.array([])

        for index in a:
            # print(f"{np.log10(bins[ind])}-{np.log10(bins[ind + 1])} -> J: {index}")
            for r in r_ind:
                # print(r)
                M = data_points[r]
                Rvir = rvir[r] / 1000

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
                c_inv = cinv(obs, eps, 2)

                optres = iminuit.minimize(
                    cost, [np.log(10)], args=(obs, c_inv, M, Rvir, "gaussian")
                )
                opts = np.append(opts, np.exp(optres.x))

            opts = opts.reshape(-1, 1)
            # print(opts.shape)
            main = np.column_stack((main, opts))
            opts = np.array([])
            # print(eps, main.shape)
        
            
        m_c = np.mean(main[:, 4:], axis=1)
        main = np.column_stack((main, m_c))
        # print(eps, main.shape)
        
        if total is None:
            total = main
            # print(eps, total.shape)
        else:
            total = np.vstack((total, main))  # type: ignore
            # print(eps, total.shape)
    
    jacks = total[:, 4:-1] # type: ignore
    meanjk = total[:, -1] # type: ignore
    se_jk = se_jack(jacks, meanjk, jacks.shape[1])
    # ses = np.append(ses, np.mean(se_jk))
    print(eps, np.mean(se_jk))
    return np.mean(se_jk)

# %%
def update_ses(s):
    ses.append(s)

# %%
# executor = concurrent.futures.ProcessPoolExecutor(10)
# futures = [executor.submit(tst, ep) for ep in eps]
# concurrent.futures.wait(futures)

def start_processing(eps):
    with concurrent.futures.ProcessPoolExecutor(9) as executor:
        future_proc = {executor.submit(tst, ep) for ep in eps}
        for future in concurrent.futures.as_completed(future_proc):
            update_ses(future.result())

# %%
start_processing(eps)

# %%
# plt.plot(eps, ses)
mi = np.argmin(ses)
plt.scatter(eps, ses)
plt.axvline(eps[mi], label=r'$\sum \frac{(\rho_{mod} - \rho_{data})^2}{{(\epsilon \rho_{data})}^2}$')
plt.ylabel(r'$< \sigma_{jk} >$')
plt.xlabel(r'$\epsilon$')
plt.title(f'Min $\\epsilon$: {eps[mi]}')
plt.savefig('testing_eps_full_low.png')
