from constants import RADIUS_BINS
from imports import np, njit, jit


@njit(parallel=True)
def get_points(x, y, z, arr_points) -> np.ndarray:
    return arr_points[
        (arr_points[:, 0] < x + 10)
        & (x - 10 < arr_points[:, 0])
        & (arr_points[:, 1] < y + 10)
        & (y - 10 < arr_points[:, 1])
        & (arr_points[:, 2] < z + 10)
        & (z - 10 < arr_points[:, 2])
    ]

@njit(parallel=True, fastmath=True)
def compute_R(x, y, z, arr, ind):
    Arrays = arrays(get_points(x, y, z, arr), x, y, z, ind)
    return np.sqrt(
        (Arrays[:, 0]) ** 2.0 + (Arrays[:, 1]) ** 2.0 + (Arrays[:, 2]) ** 2.0
    )

@jit(fastmath=True)
def Volume(factor, bn=25):
    volume = []
    for i in range(bn):
        vol = factor * 4.0 / 3.0 * np.pi * (RADIUS_BINS[i + 1] ** 3 - RADIUS_BINS[i] ** 3)
        volume.append(vol)
    return np.array(volume)

@jit()
def cinv(obs, epsilon):
    c = np.diag((epsilon * obs) ** 2)
    return np.linalg.inv(c)


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
