from constants import *
from imports import *


@jit(nopython=True, parallel=True)
def get_points(X, Y, Z, arr):
    arr_points_2 = arr[arr[:, 0] < X + 10]
    arr_points_2 = arr_points_2[X - 10 < arr_points_2[:, 0]]
    arr_points_2 = arr_points_2[arr_points_2[:, 1] < Y + 10]
    arr_points_2 = arr_points_2[Y - 10 < arr_points_2[:, 1]]
    arr_points_2 = arr_points_2[arr_points_2[:, 2] < Z + 10]
    arr_points_2 = arr_points_2[Z - 10 < arr_points_2[:, 2]]
    return arr_points_2


@jit(nopython=True, parallel=True, fastmath=True)
def compute_R(X, Y, Z, arr_points_2):
    R = np.sqrt(
        (X - arr_points_2[:, 0]) ** 2.0
        + (Y - arr_points_2[:, 1]) ** 2.0
        + (Z - arr_points_2[:, 2]) ** 2.0
    )
    return R


@njit(fastmath=True)
def rho_o(M, Rvir, Rs):
    c = Rvir / Rs
    ln_term = np.log(1.0 + c) - (c / (1.0 + c))
    rho_not = M / (4.0 * np.pi * (Rs**3.0) * ln_term)
    return rho_not


@njit()
def rho_r(Rs, M, Rvir, rmin=1e-2, rmax=1e1, Nbins=25):
    dummy = np.logspace(np.log10(rmin), np.log10(rmax), Nbins + 1)
    r = (dummy[1:] + dummy[:-1]) / 2.0
    mask = np.where(r < Rvir)
    r = r[mask]
    term = r / Rs
    rho_not = rho_o(M, Rvir, Rs)
    return r, rho_not / (term * ((1.0 + term) ** 2.0))

def cinv(obs):
    c = np.diag((0.25 * obs) ** 2)
    return np.linalg.inv(c)


@njit(parallel=True)
def chisq(obs, model, cinv):
    residual = obs - model
    # chis = np.dot(residual, np.dot(cinv, residual))
    chis = np.dot(residual, np.dot(cinv, np.transpose(residual)))
    chis2 = 0
    # residual ** 2 * cinv for every bin
    for bin in range(len(residual)):
        dummy = (residual[bin] ** 2) * cinv[bin, bin]
        # print(dummy)
        chis2 += dummy
    # print(f"Chis numpy: {chis}")
    # print(f"Chis 2: {chis2}\n")
    return chis


@jit()
def cost(cvir, obs, cinv, M, Rvir):  # theta is Rs, M, Rvir
    if cvir < 0:
        chis = np.inf
    else:
        Rs = Rvir / cvir
        _, model = rho_r(Rs, M, Rvir)
        chis = chisq(obs, model, cinv)
    return chis


@jit(nopython=True, parallel=True, fastmath=True)
def compute_R2(arr):
    R = np.sqrt(
        (arr[:, 0]) ** 2.0
        + (arr[:, 1]) ** 2.0
        + (arr[:, 2]) ** 2.0
    )
    return R
    
@jit(fastmath=True)
def Volume(a, bn=25):
    volume = []
    for i in range(0, bn):
        vol = a * 4.0 / 3.0 * np.pi * (RADIUS_BINS[i + 1] ** 3 - RADIUS_BINS[i] ** 3)
        volume.append(vol)
    return np.array(volume)


def arrays(arr, X, Y, Z, i):
    array1 = arr - [X, Y, Z]
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