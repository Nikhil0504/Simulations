import multiprocessing as mp

from scipy.stats import median_abs_deviation as mad

from halos import Halo
from loading import *

plt.style.use(["science", "scatter"])


def Eps_Parallel(
    inds: np.ndarray,
    ep: float,
    func: str = "gaussian",
    lib: str = "scipy"
) -> np.ndarray:
    num_processes = mp.cpu_count() - 1  # you may want to adjust this based on your system
    pool = mp.Pool(num_processes)

    results = []
    for id, halo in enumerate(inds):
        results.append(pool.apply_async(Eps_Parallel_Helper, args=(halo, id, ep, func, lib)))

    pool.close()
    pool.join()

    main = np.vstack([r.get() for r in results])
    return main

def Eps_Parallel_Helper(halo, id, ep, func, lib):
    h = Halo(int(halo[0]), MVIR, X, Y, Z, RVIR, RS)

    opts = np.zeros(0)

    for jack in range(9):
        density = halo[3 + (jack * 25):3 + ((jack + 1) * 25)]

        optres = h.minimise_cost(density, ep, func, lib)

        opts = np.append(opts, optres)

    print(f'{func} {ep} {id}')
    return opts


inds = np.loadtxt("out/inds.txt")
epss = np.arange(0.01, 0.26, 0.01)

mad_se = np.array([])

for ep in epss:
    main = Eps_Parallel(inds, ep)
    temp = se_jack(main[:, 1:], np.mean(main[:, 1:], axis=1), 8)

    # slices = np.array(
        # [0, 200, 400, 600, 800, 1000, 1200, 1400, 1497, 1545, 1568])
    slices = np.array([0, 708, 1130, 1332, 1429, 1478, 1501])

    medians = np.array([])

    for i in range(6):
        new = temp[slices[i]:slices[i + 1]]
        med_c = np.median(new)
        sigma = 1.4826 * mad(new)
        temp2 = (new - med_c) / sigma
        # medians = np.append(medians, 1.4826 * mad(new))
        medians = np.append(medians, len(np.where(np.abs(temp2) > 3)[0]) / len(new))

    if mad_se.size == 0:
        mad_se = medians
    else:
        mad_se = np.vstack((mad_se, medians))


np.savetxt(f"out/gaussian_mad_3sigma_iminuit.out", mad_se)
# mad_se = np.loadtxt('out/gaussian_mad_3sigma_iminuit.out')

##########################
plt.style.use(["science"])

MASS_BINS = MASS_BINS[4:]
MASS2 = (MASS_BINS[1:] + MASS_BINS[:-1]) / 2.0

fig = plt.figure(figsize=(17, 5))

for i in range(6):
    plt.subplot(2, 5, i + 1)

    plt.scatter(epss, mad_se[:, i], label="")
    plt.plot([], [], color="black", label=f"MASS: {np.log10(MASS2[i]):1.2f}")

    # plt.ylim(np.mean(mad_se[:, i]) - 0.02, np.mean(mad_se[:, i]) + 0.02)
    if i == 5:
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$3 \sigma (Med(\sigma_{jk}))$")

    plt.legend(prop={"size": 8})

# plt.savefig(f"figures/eps_gaussian_mad_constrains_possion.jpg", dpi=150)
plt.savefig(f"test3.jpg", dpi=150)
