from constants import RADIUS_BINS
from imports import jit, njit, np


@njit(parallel=True)
def get_points(x: float, y: float, z: float, arr_points: np.ndarray) -> np.ndarray:
    return arr_points[
        (arr_points[:, 0] < x + 10)
        & (x - 10 < arr_points[:, 0])
        & (arr_points[:, 1] < y + 10)
        & (y - 10 < arr_points[:, 1])
        & (arr_points[:, 2] < z + 10)
        & (z - 10 < arr_points[:, 2])
    ]


@jit(parallel=True, fastmath=True, forceobj=True)
def compute_R(x, y, z, arr, ind):
    points = get_points(x, y, z, arr)
    Arrays = arrays(points, x, y, z, ind)
    return np.sqrt(
        (Arrays[:, 0]) ** 2.0 + (Arrays[:, 1]) ** 2.0 + (Arrays[:, 2]) ** 2.0
    )


@jit(fastmath=True)
def Volume(factor, bn=25):
    volume = []
    for i in range(bn):
        vol = (
            factor * 4.0 / 3.0 * np.pi * (RADIUS_BINS[i + 1] ** 3 - RADIUS_BINS[i] ** 3)
        )
        volume.append(vol)
    return np.array(volume)


@njit(fastmath=True)
def rho_o(M, Rvir, Rs):
    c = Rvir / Rs
    ln_term = np.log(1.0 + c) - (c / (1.0 + c))
    rho_not = M / (4.0 * np.pi * (Rs**3.0) * ln_term)
    return rho_not


@njit()
def rho_r(Rs, M, Rvir, mask, rmin=1e-2, rmax=1e1, Nbins=25):
    dummy = np.logspace(np.log10(rmin), np.log10(rmax), Nbins + 1)
    r = (dummy[1:] + dummy[:-1]) / 2.0
    r = r[mask]
    term = r / Rs
    rho_not = rho_o(M, Rvir, Rs)
    return r, rho_not / (term * ((1.0 + term) ** 2.0))


@jit
def cinv(obs, epsilon):
    c = np.diag((epsilon * obs) ** 2)
    return np.linalg.inv(c)


@njit(parallel=True)
def chisq(obs, model, cinv, func="gaussian"):
    residual = obs - model
    cost = 0
    # residual ** 2 * cinv for every bin
    for bin in range(len(residual)):
        if func == "gaussian":
            dummy = (residual[bin] ** 2) * cinv[bin, bin]
        elif func == "lorentz":
            dummy = np.log(1 + ((residual[bin] ** 2) * cinv[bin, bin]))
        elif func == "abs":
            dummy = np.abs(residual[bin]) * np.sqrt(cinv[bin, bin])
        cost += dummy  # type: ignore
    return cost


@jit
def cost(lncvir, obs, cinv, M, Rvir, mask, func="gaussian"):  # theta is Rs, M, Rvir
    Rs = Rvir / np.exp(lncvir)
    # if lncvir < 0:
    #     return np.inf
    # Rs = Rvir / lncvir
    _, model = rho_r(Rs, M, Rvir, mask)
    Cost = chisq(obs, model, cinv, func)
    return Cost


def arrays(array: np.ndarray, X: int, Y: int, Z: int, i: int) -> np.ndarray:  # type: ignore
    array1 = array - [X, Y, Z]
    if i == 1:
        return array1
    elif i == 2:
        return np.delete(
            array1,
            np.where((array1[:, 0] > 0) & (array1[:, 1] > 0) & (array1[:, 2] > 0))[0],
            axis=0,
        )
    elif i == 3:
        return np.delete(
            array1,
            np.where((array1[:, 0] < 0) & (array1[:, 1] > 0) & (array1[:, 2] > 0))[0],
            axis=0,
        )
    elif i == 4:
        return np.delete(
            array1,
            np.where((array1[:, 0] < 0) & (array1[:, 1] < 0) & (array1[:, 2] > 0))[0],
            axis=0,
        )
    elif i == 5:
        return np.delete(
            array1,
            np.where((array1[:, 0] > 0) & (array1[:, 1] < 0) & (array1[:, 2] > 0))[0],
            axis=0,
        )
    elif i == 6:
        return np.delete(
            array1,
            np.where((array1[:, 0] > 0) & (array1[:, 1] > 0) & (array1[:, 2] < 0))[0],
            axis=0,
        )
    elif i == 7:
        return np.delete(
            array1,
            np.where((array1[:, 0] < 0) & (array1[:, 1] > 0) & (array1[:, 2] < 0))[0],
            axis=0,
        )
    elif i == 8:
        return np.delete(
            array1,
            np.where((array1[:, 0] < 0) & (array1[:, 1] < 0) & (array1[:, 2] < 0))[0],
            axis=0,
        )
    elif i == 9:
        return np.delete(
            array1,
            np.where((array1[:, 0] > 0) & (array1[:, 1] < 0) & (array1[:, 2] < 0))[0],
            axis=0,
        )


def se_jack(jacks, meanjk, num):
    if jacks.ndim == 1:
        return np.sqrt(np.sum(np.square(jacks - meanjk), axis=0) * (num - 1) / num)
    else:
        return np.sqrt(
            np.sum(np.square(jacks - meanjk[:, None]), axis=1) * (num - 1) / num
        )
