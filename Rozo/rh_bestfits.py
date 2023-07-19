import tqdm
from loading import *
import scipy.optimize as so
from multiprocessing import Pool

def get_info():
    with h5.File(join(NIKHIL_PATH, 'catalogue/halo_densities_full.h5')) as hdf:
        Morb = hdf['Morb'][:]
        rho = hdf['rho_o'][:]
        R200 = hdf['R200m'][:]
        ids = hdf['HID'][:]

    return Morb, R200, rho, ids

def fit_density(ind):
    # mask where density is not zero and RADIUS > 6 * RSOFT
    mask = (rho[ind] != 0) & (RADIUS > 6 * RSOFT)
    # print(mask)

    M = Morb[ind]
    den = rho[ind][mask]

    optres = iminuit.minimize(cost, [0.038],
                              args=(M, den, mask),
                              method='simplex',
                              tol=1e-4,
                              bounds=((0, 2000)),
                              options={'stra': 2, 'maxfun': 500})
    
    return np.array([*optres.x, optres.fun])

if __name__ == '__main__':
    Morb, R200, rho, ids = get_info()

    pool = Pool(processes=8)  # create a Pool object with the default number of processes

    # apply the fit_density function to each halo index using the apply_async method
    results = [pool.apply_async(fit_density, args=(ind,)) for ind in range(len(Morb))]

    # retrieve the results from the async objects and put them into the list
    results = [r.get() for r in tqdm.tqdm(results)]
    results = np.array(results)

    # Clean up
    pool.close()
    pool.join()

    with h5.File(join(NIKHIL_PATH, 'a_bestfits.h5'), 'w') as hdf:
        hdf.create_dataset('HID', data=ids)
        hdf.create_dataset('Morb', data=Morb)
        hdf.create_dataset('R200', data=R200)
        hdf.create_dataset('rho', data=rho)
        hdf.create_dataset('a', data=results[:, 0])
        # hdf.create_dataset('a', data=results[:, 1])
        hdf.create_dataset('chi2', data=results[:, -1])
