from constants import BIN_NO, MASS, PERCENT, RADIUS, RADIUS_BINS, VOLUME
from functions import cinv, compute_R, cost, rho_r
from imports import iminuit, jit, np

import scipy.optimize as so


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

    @jit(fastmath=True, parallel=True, forceobj=True, error_model="numpy")
    def densities(self, arr, ind, factor):
        R = self.distances(arr, ind)
        pairs, _ = np.histogram(R, bins=RADIUS_BINS)
        total_mass = np.array(pairs) * MASS * (100 / PERCENT)

        volume = factor * VOLUME

        return total_mass / volume

    @jit(fastmath=True, parallel=True, forceobj=True, error_model="numpy")
    def NFWs(self, Rs, mask):
        radii, rhos = rho_r(Rs, self.Mvir, self.Rvir, mask)
        return radii, rhos

    def minimise_cost(self, Den=[], eps=0.25, cost_func="gaussian"):
        den = Den

        mask = np.where((RADIUS < self.Rvir) & (den > 0))
        den = den[mask]
        c_inv = cinv(den, eps)

        # optres = so.minimize(
            # cost, [np.log(10)], args=(den, c_inv, self.Mvir, self.Rvir, mask, cost_func), method='Nelder-Mead', bounds=[(np.log(1e-50), np.log(50))]
        # )
        optres = iminuit.minimize(
            cost, [np.log(10)], args=(den, c_inv, self.Mvir, self.Rvir, mask, cost_func), method='simplex'
        )
        return optres.x
