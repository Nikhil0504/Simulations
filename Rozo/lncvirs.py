# %%
from itertools import cycle
from multiprocessing import Pool

import tqdm
from constants import *
from functions import *
from halos import Halo
from imports import *

from os.path import join

NIKHIL_PATH = '/home/nikhilgaruda/Simulations/Rozo/out'

def parallel_lncvir(args):
    key, m200, r200 = args
    with h5.File(join(SDD, 'catalogue/halo_particle_dict.h5'), 'r') as hdf_dict, \
        h5.File(join(SRC, 'data/susmita-sim_%d.h5'), 'r', driver='family', memb_size=MEMBSIZE) as part_cat:
        p_idx = hdf_dict[str(int(key))][()]
        # print(p_idx, len(p_idx))
        try:
            R = part_cat['Rp'][p_idx, 0]
        except (IndexError, TypeError):
            # print('Error with', p_idx)
            R = np.inf
        h = Halo(m200, r200, R)
        dens = h.densities()
        # you will need to change underlying cost function to use NFW instead of the paper's function.
        # cost = h.minimise_cost(Den=dens)

        # return all args and dens, cost
        full = np.array([key, m200, r200, *dens])
        return full
    

def get_info():
    with h5.File(join(SDD, 'halo_catalogue.h5'), 'r') as hdf_cat:

        # Load all halo ids with an M200m
        m200_mask = (hdf_cat['M200m'][()] > 0)
        good_m200 = hdf_cat['M200m'][()][m200_mask]
        good_r200 = hdf_cat['R200m'][()][m200_mask]
        hid = hdf_cat['OHID'][()]
        good_hid = hid[m200_mask]  # with particles
    
    return np.array([good_hid, good_m200, good_r200]).T

class ColoredBar:
    def __init__(self, total):
        self.colors = cycle(['\033[31m', '\033[32m', '\033[33m', '\033[34m', '\033[35m', '\033[36m'])
        self.bar = tqdm.tqdm(total=total, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
    def __call__(self, x):
        self.bar.update()
        self.bar.set_description(next(self.colors) + 'Processing' + '\033[0m')
        return x

if __name__ == "__main__":
    # Set up multiprocessing
    num_cpus = 20
    pool = Pool(processes=num_cpus)
    
    iterator = get_info()
    total = iterator.shape[0]

    results = []
    for result in tqdm.tqdm(pool.imap_unordered(parallel_lncvir, iterator), total=total):
        results.append(result)
    
    results = np.array(results)

    # Clean up
    pool.close()
    pool.join()

    print(results.shape)
    

    np.save(join(NIKHIL_PATH, 'catalogue/m200_densities.npy'), results)

    with h5.File(join(NIKHIL_PATH, 'catalogue/m200_densities.h5'), 'w') as hdf:
        hdf.create_dataset('HID', data=results[:, 0])
        hdf.create_dataset('M200m', data=results[:, 1])
        hdf.create_dataset('R200m', data=results[:, 2])
        # hdf.create_dataset('lnc', data=results[:, 3])
        hdf.create_dataset('densities', data=results[:, 3:])
    
    print('done')