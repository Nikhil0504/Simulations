from multiprocessing import Pool

import tqdm
from constants import *
from functions import *
from halos import Halo
from imports import *

from os.path import join

NIKHIL_PATH = '/home/nikhilgaruda/Simulations/Rozo/out'

def get_info():
    with h5.File(join(SDD, 'catalogue/halo_catalogue.h5'), 'r') as hdf:
        """
        Get halo information from hdf file. 

        Returns:
            good_halo (np.array): array of halo information
        """
        # OHID matches HID
        temp = hdf['Morb'][:]
        zero_mask = np.where((temp > 0))
        good_morb = temp[zero_mask]
        good_r200 = hdf['R200m'][:][zero_mask]
        good_x = hdf['x'][:][zero_mask]
        good_y = hdf['y'][:][zero_mask]
        good_z = hdf['z'][:][zero_mask]
        hid = hdf['OHID'][:][zero_mask]
        # combine all these good values into a single array
        good_halo = np.array([hid, good_morb, good_r200, good_x, good_y, good_z]).T
    return good_halo

def get_density(halo):
    """
    Get density of halo particles. 

    Args:
        halo (np.array): array of halo information

    Returns:
        rho (np.array): density of halo particles
    """
    with h5.File(join(SDD, 'catalogue/halo_particle_dict.h5'), 'r') as part_dict, \
        h5.File(join(SRC, 'data/susmita-sim_%d.h5'), 'r', driver='family', memb_size=MEMBSIZE) as part_cat, \
        h5.File(join(SRC, 'data/orb_class.h5'), 'r') as tags:
        p_idx = part_dict[str(int(halo[0]))][()]
        temp = tags['CLASS'][p_idx]
        orbiting = (temp == 1)
        # print(p_idx, len(p_idx))
        try:
            R = part_cat['Rp'][p_idx, 0]
            h =  Halo(halo[1], halo[2], R)
            density = h.densities()

            R_orb = R[orbiting]
            R_inf = R[~orbiting]

            h_o = Halo(halo[1], halo[2], R_orb)
            density_o = h_o.densities()

            h_i = Halo(halo[1], halo[2], R_inf)
            density_i = h_i.densities()
        except IndexError:
            # print('Error with', p_idx)
            R = np.inf
            h = Halo(halo[1], halo[2], R)
            density = h.densities()
            density_o = density
            density_i = density
    # return all the halo parameters along with the densities (orbiting and infalling)
    return np.concatenate(([halo[0]], halo[1:], density, density_o, density_i))

def main():
    good_halos = get_info()

    # limit good_halos to 10 for testing
    # good_halos = good_halos[:10]

    # Set up multiprocessing
    num_cpus = 16
    pool = Pool(processes=num_cpus)

    total = len(good_halos)
    results = []

    for result in tqdm.tqdm(pool.imap_unordered(get_density, good_halos), total=total):
        results.append(result)
    
    # close and join pool
    pool.close()    
    pool.join()
    

    results = np.array(results)

    # save results in out folder as hdf5 file
    # split result to its components
    with h5.File(join(NIKHIL_PATH, 'catalogue/halo_densities_full.h5'), 'w') as hdf:
        hdf.create_dataset('HID', data=results[:, 0])
        hdf.create_dataset('Morb', data=results[:, 1])
        hdf.create_dataset('R200m', data=results[:, 2])
        hdf.create_dataset('x', data=results[:, 3])
        hdf.create_dataset('y', data=results[:, 4])
        hdf.create_dataset('z', data=results[:, 5])
        hdf.create_dataset('rho', data=results[:, 6:6+20])
        hdf.create_dataset('rho_o', data=results[:, 6+20:6+20+20])
        hdf.create_dataset('rho_i', data=results[:, 6+20+20:6+20+20+20])
    
    # save results to npy file
    np.save(join(NIKHIL_PATH, 'catalogue/halo_densities_full.npy'), results)

if __name__ == "__main__":
    main()

