import sys
from typing import Iterable, Union

from constants import MASS_BINS, RADIUS
from functions import cost, se_jack
from imports import iminuit, np, plt

plt.style.use(["science", "scatter"])


def Eps(
    inds: np.ndarray,
    ep: float,
    func: str = "gaussian",
    bounds: Union[None, Iterable[float]] = None,
) -> np.ndarray:
    """Gets the best-fits using the minimiser for a particular
    epsilion value.

    Parameters
    ----------
    inds : np.ndarray
        The indices file containing the ID, Mvir, Rvir, Densities
        along with Jack-knife Densities.
    ep : float
        The epsilion value for the cost function.
    func : str, optional
        The cost function, by default "gaussian"
    bounds : Union[None, Iterable[float]], optional
        The bounds for the minimiser, by default None

    Returns
    -------
    np.ndarray
        An array of the best fit ln(cvir). Each column is a
        jack-knife with the first column being the full halo.
    """
    main = np.zeros(0)

    for halo in inds:
        M = halo[1]
        Rvir = halo[2]

        opts = np.zeros(0)

        for jack in range(9):
            density = halo[3 + (jack * 25):3 + ((jack + 1) * 25)]
            mask = np.where((RADIUS < Rvir) & (density > 0))
            density = density[mask]

            # optres = so.minimize(
            #     cost,
            #     np.log(10),
            #     args=(density, ep, M, Rvir, mask, func),
            #     bounds=bounds,
            #     method='Nelder-Mead'
            # )

            optres = iminuit.minimize(
                cost,
                np.log(10),
                args=(density, ep, M, Rvir, mask, func),
                bounds=bounds,
                method="simplex",
            )

            opts = np.append(opts, optres.x)

        if main.size == 0:
            main = opts
        else:
            main = np.vstack((main, opts))

    return main


inds = np.loadtxt("out/inds.txt")
epss = np.arange(0.01, 0.26, 0.01)

mean_se = np.array([])

func = sys.argv[1]
print(func)

for ep in epss:
    main = Eps(inds, ep, func, bounds=(np.log(1e-50), np.log(50)))
    temp = se_jack(main[:, 1:], np.mean(main[:, 1:], axis=1), 8)

    slices = np.array(
        [0, 200, 400, 600, 800, 1000, 1200, 1400, 1497, 1545, 1568])

    y = np.array([])

    for i in range(10):
        new = temp[slices[i]:slices[i + 1]]
        y = np.append(y, np.mean(new))

    if mean_se.size == 0:
        mean_se = y
    else:
        mean_se = np.vstack((mean_se, y))

    print(f"{func} {ep} done.")

np.savetxt(f"out/mean_se_{func}_constrains.out", mean_se)

##########################
plt.style.use(["science"])

MASS2 = (MASS_BINS[1:] + MASS_BINS[:-1]) / 2.0

fig = plt.figure(figsize=(17, 5))

plt.suptitle(f"{func.capitalize()} Function", y=0.93)

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.scatter(epss, mean_se[:, i], label="")
    plt.plot([], [], color="black", label=f"MASS: {np.log10(MASS2[i]):1.2f}")

    plt.ylim(np.mean(mean_se[:, i]) - 0.02, np.mean(mean_se[:, i]) + 0.02)
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$<\sigma_{jk}>$")

    plt.legend(prop={"size": 8})

plt.savefig(f"figures/eps_{func}_constrains.jpg", dpi=150)
