from imports import *
from functions import *


class Halo:
    def __init__(self, hid: int, **kwargs):
        self.hid = hid
        self.mvir = kwargs['mvir'][hid]
        self.x = kwargs['x'][hid]
        self.y = kwargs['y'][hid]
        self.z = kwargs['z'][hid]
        self.rvir = kwargs['rvir'][hid] / 1000
        self.rs = kwargs['rs'][hid] / 1000

    @property
    def Mvir(self):
        return np.log10(self.mvir)

    @property
    def coords(self):
        return (self.x, self.y, self.z)

    @property
    def Rvir(self):
        return self.rvir

    @property
    def Rs(self):
        return self.rs

    @njit(parallel=True)
    def get_points(self, arr_points) -> np.ndarray:
        return arr_points[
            (arr_points[:, 0] < self.x + 10)
            & (self.x - 10 < arr_points[:, 0])
            & (arr_points[:, 1] < self.y + 10)
            & (self.y - 10 < arr_points[:, 1])
            & (arr_points[:, 2] < self.z + 10)
            & (self.z - 10 < arr_points[:, 2])
        ]
    
    @njit(parallel=True, fastmath=True)
    def compute_R(self, arr, ind):
        Arrays = arrays(self.get_points(arr), self.x, self.y, self.z, ind)
        return np.sqrt((Arrays[:, 0]) ** 2.0 + (Arrays[:, 1]) ** 2.0 + (Arrays[:, 2]) ** 2.0)
    

    