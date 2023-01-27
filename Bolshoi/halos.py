import scipy.optimize as so

from constants import MASS, PERCENT, RADIUS, RADIUS_BINS, VOLUME
from functions import compute_R, cost, rho_r
from imports import iminuit, jit, np


class Halo:

    def __init__(self, hid: float, mvir, x, y, z, rvir, rs):
        self.hid = int(hid)
        self.mvir = mvir[self.hid]
        self.x = x[self.hid]
        self.y = y[self.hid]
        self.z = z[self.hid]
        self.rvir = rvir[self.hid] / 1000
        self.rs = rs[self.hid] / 1000
        self.cvir = self.rvir / self.rs

    @property
    def Mvir(self):
        return int(self.mvir)

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

    def minimise_cost(self, Den=[], eps=0.25, cost_func="gaussian", lib='iminuit'):
        den = Den

        mask = np.where((RADIUS < self.Rvir) & (den > 0))
        den = den[mask]

        
        if lib == 'iminuit':
            optres = iminuit.minimize(cost, [np.log(5)],
                                    args=(den, eps, self.Mvir, self.Rvir, mask,
                                            cost_func),
                                    method='simplex',
                                    bounds=(np.log(1e-2), np.log(50)),
                                    tol=1e-4,
                                    options={'stra': 2, 'maxfun': 500})
            return optres.x
        
        elif lib == 'scipy':
            # optres = so.minimize(
            #     cost, np.log(5), 
            #     args=(den, eps, self.Mvir, self.Rvir, mask, cost_func), 
            #     method='Nelder-Mead', bounds=[(np.log(1e-2), np.log(50))],
            #     tol=1e-4,
            #     options={'maxiter': 500}
            # )

            optres = so.basinhopping(cost, np.log(5),stepsize=1,
            minimizer_kwargs={"method": "Nelder-Mead", 
                              "args": (den, eps, self.Mvir, self.Rvir, mask, cost_func),
                              "bounds": [(np.log(1e-2), np.log(50))]})

            # optres = so.shgo(cost, bounds=[(np.log(1e-5), np.log(50))],
            #                 args=(den, eps, self.Mvir, self.Rvir, mask, cost_func),
            #                 sampling_method='sobol', 
            #                 minimizer_kwargs={'method': 'Nelder-Mead'})

            return optres.x
        
        return np.inf
