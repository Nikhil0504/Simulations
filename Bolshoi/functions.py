from constants import RADIUS
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


@njit(fastmath=True)
def rho_o(M, Rvir, Rs):
    c = Rvir / Rs
    ln_term = np.log(1.0 + c) - (c / (1.0 + c))
    rho_not = M / (4.0 * np.pi * (Rs**3.0) * ln_term)
    return rho_not


@njit()
def rho_r(Rs, M, Rvir, mask):
    r = RADIUS[mask]
    term = r / Rs
    rho_not = rho_o(M, Rvir, Rs)
    return r, rho_not / (term * ((1.0 + term) ** 2.0))


# @jit
# def cinv(obs, epsilon):
#     c = np.diag((epsilon * obs) ** 2)
#     return np.linalg.inv(c)


@njit(parallel=True, fastmath=True)
def chisq(obs, model, epsilon, func="gaussian"):
    residual = obs - model
    # residual ** 2 * cinv for every bin
    if func == "gaussian":
        return np.sum(np.square(residual) / np.square((epsilon * obs)))
    elif func == "lorentz":
        temp = np.square(residual) / np.square((epsilon * obs))
        return np.sum(np.log(1 + temp))
    elif func == 'abs':
        return np.sum(np.abs(residual) / ((epsilon * obs)))


@jit
def cost(lncvir, obs, epsilon, M, Rvir, mask, func="gaussian"):  # theta is Rs, M, Rvir
    Rs = Rvir / np.exp(lncvir)
    # if lncvir < 0:
    #     return np.inf
    # Rs = Rvir / lncvir
    _, model = rho_r(Rs, M, Rvir, mask)
    Cost = chisq(obs, model, epsilon, func)
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
