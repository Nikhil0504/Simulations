from dataclasses import dataclass
from typing import Union

import scipy.optimize as so
from constants import PART_MASS, RADIUS_BINS, VOLUME, RADIUS
from functions import cost, rho_r, cost_nfw
from imports import iminuit, jit, np


# @dataclass(frozen=True, slots=True)
class Halo:
    # slots = ('mvir', 'rvir', 'rps')
    # mvir: float
    # rvir: float
    # rps: Union[np.ndarray, float]

    def __init__(self, mvir, rvir, rps) -> None:
        self.m200m = mvir
        self.r200m = rvir
        self.rps = rps

    @property
    def M200m(self):
        return self.m200m

    @property
    def R200m(self):
        return self.r200m

    @jit(fastmath=True, parallel=True, forceobj=True, error_model="numpy")
    def densities(self, factor=1):
        pairs, _ = np.histogram(self.rps, bins=RADIUS_BINS)
        total_mass = np.array(pairs) * PART_MASS

        volume = factor * VOLUME

        return total_mass / volume

    @jit(fastmath=True, parallel=True, forceobj=True, error_model="numpy")
    def NFWs(self, Rs, mask):
        radii, rhos = rho_r(Rs, self.M200m, self.R200m, mask)
        return radii, rhos

    def minimise_cost(self, Den=[], eps=0.1, cost_func="gaussian", lib='iminuit', profile='orbiting'):
        if Den == []:
            den = self.densities()
        else:
            den = Den

        mask = (den > 0) & (RADIUS < self.R200m) & (RADIUS > 0.1)
        den = den[mask]

        
        if lib == 'iminuit':
            if profile == 'orbiting':
                optres = iminuit.minimize(cost, [843.8],
                                        args=(self.M200m, den, mask, eps, cost_func),
                                        method='simplex',
                                        tol=1e-4,
                                        options={'stra': 2, 'maxfun': 500})
                return optres.x[0]

            elif profile == 'NFW':
                optres = iminuit.minimize(cost_nfw, [np.log(5)],
                                    args=(self.M200m, den, mask, self.R200m, eps, cost_func),
                                    method='simplex',
                                    bounds=(np.log(1e-2), np.log(50)),
                                    tol=1e-4,
                                    options={'stra': 2, 'maxfun': 500})
                return optres.x[0]
            else:
                raise ValueError("Profile must be either 'orbiting' or 'NFW'")
        
        elif lib == 'scipy':
            optres = so.basinhopping(cost, 843.8, stepsize=1,
            minimizer_kwargs={"method": "Nelder-Mead", 
                              "args": (self.M200m, den, mask, eps, cost_func)})
            return optres.x
        
        return np.inf

