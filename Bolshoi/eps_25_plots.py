import sys
from typing import Iterable, Union

import scipy.optimize as so

from constants import MASS_BINS, RADIUS
from functions import cost, se_jack
from imports import iminuit, np, plt

plt.style.use(["science"])


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

slices = np.array([0, 200, 400, 600, 800, 1000, 1200, 1400, 1497, 1545, 1568])

func = sys.argv[1]
print(func)

imain = Eps(inds, 0.25, "gaussian", bounds=(np.log(1e-50), np.log(50)))
imain2 = Eps(inds, 0.25, "lorentz", bounds=(np.log(1e-50), np.log(50)))
imain3 = Eps(inds, 0.25, "abs", bounds=(np.log(1e-50), np.log(50)))

itemp = se_jack(imain[:, 1:], np.mean(imain[:, 1:], axis=1), 8)
itemp2 = se_jack(imain2[:, 1:], np.mean(imain2[:, 1:], axis=1), 8)
itemp3 = se_jack(imain3[:, 1:], np.mean(imain3[:, 1:], axis=1), 8)

x, iy, iy2, iy3, iy4, iy5, iy6 = (np.zeros(0) for _ in range(7))

iy7, iy8, iy9, z, z2, z3 = (np.zeros(0) for _ in range(6))

for i in range(slices.shape[0] - 1):
    iy = np.append(iy, np.mean(itemp[slices[i]:slices[i + 1]]))
    iy3 = np.append(iy3, np.mean(itemp2[slices[i]:slices[i + 1]]))
    iy5 = np.append(iy5, np.mean(itemp3[slices[i]:slices[i + 1]]))

    iy2 = np.append(iy2, np.std(imain[slices[i]:slices[i + 1], 0]))
    iy4 = np.append(iy4, np.std(imain2[slices[i]:slices[i + 1], 0]))
    iy6 = np.append(iy6, np.std(imain3[slices[i]:slices[i + 1], 0]))

    iy7 = np.append(iy7, np.std(itemp[slices[i]:slices[i + 1]]))
    iy8 = np.append(iy8, np.std(itemp2[slices[i]:slices[i + 1]]))
    iy9 = np.append(iy9, np.std(itemp3[slices[i]:slices[i + 1]]))

    z = np.append(z, np.mean(imain[slices[i]:slices[i + 1], 0]))
    z2 = np.append(z2, np.mean(imain2[slices[i]:slices[i + 1], 0]))
    z3 = np.append(z3, np.mean(imain3[slices[i]:slices[i + 1], 0]))

    x = np.append(x, np.mean([MASS_BINS[i], MASS_BINS[i + 1]]))

##########################
fig = plt.figure(figsize=(5, 5), dpi=150)

plt.plot(
    x,
    iy,
    "#008BF8",
    linestyle="--",
    linewidth=1.5,
    label="_Jack-knife Gaussian Iminuit",
)
plt.plot(
    x,
    iy2,
    "#DC0073",
    linestyle="--",
    linewidth=1.5,
    label="_Intrinsic Gaussian Iminuit",
)

plt.plot(x,
         iy3,
         "#008BF8",
         linestyle=":",
         linewidth=1.5,
         label="_Jack-knife Lorentz Iminuit")
plt.plot(x,
         iy4,
         "#DC0073",
         linestyle=":",
         linewidth=1.5,
         label="_Intrinsic Lorentz Iminuit")

plt.plot(
    x,
    iy5,
    "#008BF8",
    linestyle="-",
    linewidth=1.5,
    label="_Jack-knife Absolute Iminuit",
)
plt.plot(x,
         iy6,
         "#DC0073",
         linestyle="-",
         linewidth=1.5,
         label="_Intrinsic Absolute Iminuit")

plt.plot([], [], "--", color="#1E152A", linewidth=1.5, label="Gaussian")
plt.plot([], [], ":", color="#1E152A", linewidth=1.5, label="Lorentz")
plt.plot([], [], "-", color="#1E152A", linewidth=1.5, label="Absolute")

plt.plot([], [], "#008BF8", linewidth=1.5, label="Jack-knife")
plt.plot([], [], "#DC0073", linewidth=1.5, label="Intrinsic")

plt.title(f"{func.capitalize()} Loss Function, Fractional Err: 25\%")

plt.ylabel(r"$<\sigma_{\mathrm{jk}}> \mathrm{or} \ \sigma(c)$")
plt.xlabel("Mean of the bin edges")

plt.xscale("log")
plt.legend()

plt.savefig(f"figures/test_{func}1.png")
plt.clf()

##########################
fig = plt.figure(figsize=(5, 5), dpi=150)

plt.plot(
    x,
    iy7,
    "#008BF8",
    linestyle="--",
    linewidth=1.5,
    label="_Jack-knife Gaussian Iminuit",
)
plt.plot(x,
         iy8,
         "#008BF8",
         linestyle=":",
         linewidth=1.5,
         label="_Jack-knife Lorentz Iminuit")
plt.plot(
    x,
    iy9,
    "#008BF8",
    linestyle="-",
    linewidth=1.5,
    label="_Jack-knife Absolute Iminuit",
)

plt.plot([], [], "--", color="#1E152A", linewidth=1.5, label="Gaussian")
plt.plot([], [], ":", color="#1E152A", linewidth=1.5, label="Lorentz")
plt.plot([], [], "-", color="#1E152A", linewidth=1.5, label="Absolute")

plt.plot([], [], "#008BF8", linewidth=1.5, label="Jack-knife")

plt.title(f"{func.capitalize()} Loss Function, Fractional Err: 25\%")

plt.ylabel(r"$\sigma(\sigma_{jk})$")
plt.xlabel("Mean of the bin edges")

plt.xscale("log")
plt.legend()

plt.savefig(f"figures/test_{func}2.png")
plt.clf()

##########################
fig = plt.figure(figsize=(5, 5), dpi=150)

plt.plot(
    x,
    iy / z,
    "#008BF8",
    linestyle="--",
    linewidth=1.5,
    label="_Jack-knife Gaussian Iminuit",
)
plt.plot(
    x,
    iy2 / z,
    "#DC0073",
    linestyle="--",
    linewidth=1.5,
    label="_Intrinsic Gaussian Iminuit",
)

plt.plot(
    x,
    iy3 / z2,
    "#008BF8",
    linestyle=":",
    linewidth=1.5,
    label="_Jack-knife Lorentz Iminuit",
)
plt.plot(
    x,
    iy4 / z2,
    "#DC0073",
    linestyle=":",
    linewidth=1.5,
    label="_Intrinsic Lorentz Iminuit",
)

plt.plot(
    x,
    iy5 / z2,
    "#008BF8",
    linestyle="-",
    linewidth=1.5,
    label="_Jack-knife Absolute Iminuit",
)
plt.plot(
    x,
    iy6 / z2,
    "#DC0073",
    linestyle="-",
    linewidth=1.5,
    label="_Intrinsic Absolute Iminuit",
)

plt.plot([], [], "--", color="#1E152A", linewidth=1.5, label="Gaussian")
plt.plot([], [], ":", color="#1E152A", linewidth=1.5, label="Lorentz")
plt.plot([], [], "-", color="#1E152A", linewidth=1.5, label="Absolute")

plt.plot([], [], "#008BF8", linewidth=1.5, label="Jack-knife")
plt.plot([], [], "#DC0073", linewidth=1.5, label="Intrinsic")

plt.title(f"{func.capitalize()} Loss Function, Fractional Err: 25\%")

plt.ylabel(r"Fractional Dispersion")
plt.xlabel("Mean of the bin edges")

plt.xscale("log")
plt.legend()

plt.savefig(f"figures/test_{func}3.png")
plt.clf()
