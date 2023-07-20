import tqdm
from loading import *

NIKHIL_PATH = '/home/nikhilgaruda/Simulations/Rozo/out'

with h5.File(join(NIKHIL_PATH, 'catalogue/m200_densities_new.h5'), 'r') as m200:
    ids = m200['HID'][:]
    lnc = m200['lnc'][:]
    masses = m200['M200m'][:]
    masses_2 = m200['Morb'][:]
    r200m = m200['R200m'][:]

with h5.File(join(NIKHIL_PATH, 'catalogue/halo_densities_full.h5'), 'r') as hdf:
    halo_ids = hdf['HID'][:]
    rho = hdf['rho'][:]
    rho_i = hdf['rho_i'][:]


FRAC = []

for id, logc, mass, R200m in tqdm.tqdm(zip(ids, lnc, masses, r200m)):
    mask_id = np.where(halo_ids == id)
    rho_mask = rho[mask_id][0]
    rho_i_mask = rho_i[mask_id][0]

    mask_dens = (RADIUS < R200m) & (RADIUS > 0.1)
    rho_mask = rho_mask[mask_dens] * (VOLUME[mask_dens]/PART_MASS)
    rho_i_mask = rho_i_mask[mask_dens] * (VOLUME[mask_dens]/PART_MASS)

    frac = np.sum(rho_i_mask) / np.sum(rho_mask)

    FRAC.append(frac)

# print(FRAC)

FRAC = np.array(FRAC)

# save to h5 File
with h5.File(join(NIKHIL_PATH, 'infalling_fracs.h5'), 'w') as hdf:
    hdf.create_dataset('HID', data=ids)
    hdf.create_dataset('lnc', data=lnc)
    hdf.create_dataset('M200m', data=masses)
    hdf.create_dataset('Morb', data=masses_2)
    hdf.create_dataset('frac', data=FRAC)


