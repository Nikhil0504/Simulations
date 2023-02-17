from typing import Tuple
from warnings import warn

from constants import PART_MASS, PERCENT, RADIUS, VOLUME
from imports import jit, njit, np


@njit(parallel=True)
def get_points(x: float, y: float, z: float,
               arr_points: np.ndarray) -> np.ndarray:
    return arr_points[(arr_points[:, 0] < x + 10) &
                      (x - 10 < arr_points[:, 0]) &
                      (arr_points[:, 1] < y + 10) &
                      (y - 10 < arr_points[:, 1]) &
                      (arr_points[:, 2] < z + 10) & (z - 10 < arr_points[:, 2])]


@jit(parallel=True, fastmath=True, forceobj=True)
def compute_R(x: float, y: float, z: float, arr: np.ndarray,
              ind: int) -> np.ndarray:
    points = get_points(x, y, z, arr)
    Arrays = arrays(points, x, y, z, ind)
    return np.sqrt((Arrays[:, 0])**2.0 + (Arrays[:, 1])**2.0 +
                   (Arrays[:, 2])**2.0)


@njit(fastmath=True)
def rho_o(M: float, Rvir: float, Rs: float):
    c = Rvir / Rs
    ln_term = np.log(1.0 + c) - (c / (1.0 + c))
    rho_not = M / (4.0 * np.pi * (Rs**3.0) * ln_term)
    return rho_not


@njit()
def rho_r(Rs: float, M: float, Rvir: float, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = RADIUS[mask]
    term = r / Rs
    rho_not = rho_o(M, Rvir, Rs)
    return r, rho_not / (term * ((1.0 + term)**2.0))


@jit
def cinv(obs, epsilon):
    warn('This method is deprecated.', DeprecationWarning, stacklevel=2)
    c = np.diag((epsilon * obs)**2)
    return np.linalg.inv(c)


@njit(fastmath=True)
def chisq(obs: np.ndarray, model: np.ndarray, epsilon: float, mask, func: str="gaussian"):
    Ndata = obs * (VOLUME[mask] / PART_MASS)
    Nmodel = model * (VOLUME[mask] / PART_MASS)
    # residual = obs - model
    residual = Ndata - Nmodel
    # residual ** 2 * cinv for every bin
    if func == "gaussian":
        # return np.sum(np.square(residual) / np.square((epsilon * obs)))
        return np.sum(np.square(residual) / (Ndata + np.square((epsilon * Ndata))))
        # return np.sum(np.square(residual) / (np.square((epsilon * Ndata))))
    elif func == "lorentz":
        # temp = np.square(residual) / np.square((epsilon * obs))
        temp = np.square(residual) / (Ndata + np.square((epsilon * Ndata)))
        return np.sum(np.log(1 + temp))
    elif func == 'abs':
        return np.sum(np.sqrt(np.square(residual) / (Ndata + np.square((epsilon * Ndata)))))
        # return np.sum(np.abs(residual) / ((epsilon * obs)))


@jit
def cost(lncvir,
         obs,
         epsilon,
         M,
         Rvir,
         mask,
         func="gaussian"):  # theta is Rs, M, Rvir
    Rs = Rvir / np.exp(lncvir)
    # if lncvir < 0:
    #     return np.inf
    # Rs = Rvir / lncvir
    _, model = rho_r(Rs, M, Rvir, mask)
    Cost = chisq(obs, model, epsilon, mask, func)
    return Cost


def arrays(array: np.ndarray, X: float, Y: float, Z: float,
           i: int) -> np.ndarray:
    array1 = array - [X, Y, Z]
    
    # Define lookup table mapping cases to conditions
    conditions = {
        2: (array1[:, 0] > 0) & (array1[:, 1] > 0) & (array1[:, 2] > 0),
        3: (array1[:, 0] < 0) & (array1[:, 1] > 0) & (array1[:, 2] > 0),
        4: (array1[:, 0] < 0) & (array1[:, 1] < 0) & (array1[:, 2] > 0),
        5: (array1[:, 0] > 0) & (array1[:, 1] < 0) & (array1[:, 2] > 0),
        6: (array1[:, 0] > 0) & (array1[:, 1] > 0) & (array1[:, 2] < 0),
        7: (array1[:, 0] < 0) & (array1[:, 1] > 0) & (array1[:, 2] < 0),
        8: (array1[:, 0] < 0) & (array1[:, 1] < 0) & (array1[:, 2] < 0),
        9: (array1[:, 0] > 0) & (array1[:, 1] < 0) & (array1[:, 2] < 0),
    }
    
    if i == 1:
        return array1
    else:
        # Keep rows where condition is False instead of np.delete
        return array1[~conditions[i]]


def se_jack(jacks, meanjk, num):
    if jacks.ndim == 1:
        return np.sqrt(
            np.sum(np.square(jacks - meanjk), axis=0) * (num - 1) / num)
    else:
        return np.sqrt(
            np.sum(np.square(jacks - meanjk[:, None]), axis=1) * (num - 1) /
            num)


def remove_outliers(array, sigma=3):
    # Removes outliers within 3 sigma
    upper_boundary = np.mean(array) + sigma * np.std(array)
    lower_boundary = np.mean(array) - sigma * np.std(array)
    mask = np.where((lower_boundary < array) & (upper_boundary > array))
    return array[mask]


@njit(fastmath=True)
def remove_outliers_2(array, th1=0.25, th2=0.75):
    # Uses IQR methods
    q1 = np.quantile(array, th1)
    q3 = np.quantile(array, th2)
    iqr = q3 - q1
    upper_boundary = q3 + 1.5 * iqr
    lower_boundary = q1 - 1.5 * iqr
    mask = np.where((lower_boundary < array) & (upper_boundary > array))
    return array[mask]
