from imports import np


def arrays(array: np.ndarray, X: int, Y: int, Z: int, i: int) -> np.ndarray: # type: ignore
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
