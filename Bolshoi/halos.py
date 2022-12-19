from constants import BIN_NO, MASS, PERCENT, RADIUS_BINS
from functions import Volume, compute_R, rho_r
from imports import np


class Halo:
    def __init__(self, hid: int, mvir, x, y, z, rvir, rs):
        self.hid = hid
        self.mvir = mvir[hid]
        self.x = x[hid]
        self.y = y[hid]
        self.z = z[hid]
        self.rvir = rvir[hid] / 1000
        self.rs = rs[hid] / 1000

    @property
    def Mvir(self):
        return self.mvir

    @property
    def coords(self):
        return (self.x, self.y, self.z)

    @property
    def Rvir(self):
        return self.rvir

    @property
    def Rs(self):
        return self.rs

    def distances(self, arr, ind):
        return compute_R(*self.coords, arr, ind)
    
    def densities(self, arr, ind, factor):
        R = self.distances(arr, ind)
        pairs, _ = np.histogram(R, bins=RADIUS_BINS)
        total_mass = np.array(pairs) * MASS * (100 / PERCENT)

        volume = Volume(factor)

        return total_mass / volume

    def NFWs(self, Rs):
        radii, rhos = rho_r(Rs, self.Mvir, self.Rvir, Nbins=BIN_NO)
        return radii, rhos