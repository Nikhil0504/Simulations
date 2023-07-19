from dataclasses import dataclass
from typing import Union

import scipy.optimize as so
from constants import PART_MASS, RADIUS_BINS, VOLUME
from functions import cost, rho_r
from imports import iminuit, jit, np


# @dataclass(frozen=True, slots=True)
class Halo:
    # slots = ('mvir', 'rvir', 'rps')
    # mvir: float
    # rvir: float
    # rps: Union[np.ndarray, float]

    def __init__(self, mvir, rvir, rps) -> None:
        self.mvir = mvir
        self.rvir = rvir
        self.rps = rps

    @property
    def Mvir(self):
        return self.mvir

    @property
    def Rvir(self):
        return self.rvir

    @jit(fastmath=True, parallel=True, forceobj=True, error_model="numpy")
    def densities(self, factor=1):
        pairs, _ = np.histogram(self.rps, bins=RADIUS_BINS)
        total_mass = np.array(pairs) * PART_MASS

        volume = factor * VOLUME

        return total_mass / volume

    @jit(fastmath=True, parallel=True, forceobj=True, error_model="numpy")
    def NFWs(self, Rs, mask):
        radii, rhos = rho_r(Rs, self.Mvir, self.Rvir, mask)
        return radii, rhos

    def minimise_cost(self, Den=[], eps=0.1, cost_func="gaussian", lib='iminuit'):
        if Den == []:
            den = self.densities()
        else:
            den = Den

        mask = np.where(den > 0)
        den = den[mask]

        
        if lib == 'iminuit':
            optres = iminuit.minimize(cost, [843.8],
                                    args=(self.Mvir, den, mask, eps, cost_func),
                                    method='simplex',
                                    tol=1e-4,
                                    options={'stra': 2, 'maxfun': 500})
            return optres.x[0]
        
        elif lib == 'scipy':
            optres = so.basinhopping(cost, 843.8, stepsize=1,
            minimizer_kwargs={"method": "Nelder-Mead", 
                              "args": (self.Mvir, den, mask, eps, cost_func)})
            return optres.x
        
        return np.inf

