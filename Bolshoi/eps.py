from scipy.stats import median_abs_deviation as mad

from halos import Halo
from loading import *

plt.style.use(["science", "scatter"])


def Eps(
    inds: np.ndarray,
    ep: float,
    func: str = "gaussian",
    lib: str = "scipy"
) -> np.ndarray:
    main = np.zeros(0)

    for id, halo in enumerate(inds):
        print(f'{func} {ep} {id}')
        h = Halo(int(halo[0]), MVIR, X, Y, Z, RVIR, RS)

        opts = np.zeros(0)

        for jack in range(9):
            density = halo[3 + (jack * 25):3 + ((jack + 1) * 25)]

            optres = h.minimise_cost(density, ep, func, lib)

            opts = np.append(opts, optres)

        if main.size == 0:
            main = opts
        else:
            main = np.vstack((main, opts))

    return main


inds = np.loadtxt("out/inds.txt")
epss = np.arange(0.01, 0.26, 0.01)

mean_se = np.array([])

for ep in epss:
    main = Eps(inds, ep)
    temp = se_jack(main[:, 1:], np.mean(main[:, 1:], axis=1), 8)

    slices = np.array(
        [0, 200, 400, 600, 800, 1000, 1200, 1400, 1497, 1545, 1568])

    medians = np.array([])

    for i in range(10):
        new = temp[slices[i]:slices[i + 1]]
        medians = np.append(medians, 1.4826 * mad(new))

    if mean_se.size == 0:
        mean_se = medians
    else:
        mean_se = np.vstack((mean_se, medians))


np.savetxt(f"out/mean_se_gaussian_mad_constrains.out", mean_se)

##########################
plt.style.use(["science"])

MASS2 = (MASS_BINS[1:] + MASS_BINS[:-1]) / 2.0

fig = plt.figure(figsize=(17, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.scatter(epss, mean_se[:, i], label="")
    plt.plot([], [], color="black", label=f"MASS: {np.log10(MASS2[i]):1.2f}")

    plt.ylim(np.mean(mean_se[:, i]) - 0.02, np.mean(mean_se[:, i]) + 0.02)
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$MAD(\sigma_{jk})$")

    plt.legend(prop={"size": 8})

plt.savefig(f"figures/eps_gaussian_mad_constrains.jpg", dpi=150)
