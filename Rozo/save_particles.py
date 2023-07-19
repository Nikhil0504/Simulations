from loading import *
import pandas as pd

from os.path import exists, join

ds = 1
fhid = 4049605

fnamer = f'out/particle_data/hid_{fhid}_{ds}_xyz.npy'
fnamev = f'out/particle_data/hid_{fhid}_{ds}_vel.npy'
fnamet = f'out/particle_data/hid_{fhid}_{ds}_tag.npy'
fnamei = f'out/particle_data/hid_{fhid}_{ds}_idx.npy'

if all([exists(fn) for fn in [fnamei, fnamet, fnamer, fnamev]]):
    # xyz = np.load(fnamer); vel = np.load(fnamev); tag = np.load(fnamet); idx = np.load(fnamei)
    with h5.File(join(SRC, 'halo_catalogue.h5'), 'r') as hdf:
        idx_hid = np.where(hdf['OHID'][()] == fhid)[0]
        halo_id = hdf['OHID'][idx_hid][0]
        halo_r200 = hdf['R200m'][idx_hid]
        halo_rt = hdf['Rt'][idx_hid]
        halo_mass = hdf['M200m'][idx_hid]
    print(idx_hid, halo_id, halo_r200, halo_rt, halo_mass)
else:
    # Load most massive halo
    with h5.File(join(SRC, 'halo_catalogue.h5'), 'r') as hdf:
        idx_hid = np.where(hdf['OHID'][()] == fhid)[0]
        halo_id = hdf['OHID'][idx_hid][0]
        halo_mass = hdf['M200m'][idx_hid]
        halo_r200 = hdf['R200m'][idx_hid]
        halo_rt = hdf['Rt'][idx_hid]
        halo_xyz = (hdf['x'][idx_hid], hdf['y'][idx_hid], hdf['z'][idx_hid])
        halo_vel = (hdf['vx'][idx_hid], hdf['vy'][idx_hid], hdf['vz'][idx_hid])
    
    # Load particles belonging to the selected HID
    with h5.File(join(SRC, 'halo_particle_dict.h5'), 'r') as hdf:
        idx_pid = hdf[str(halo_id)][()]
    
    # Load members PIDs
    with h5.File(join(SRC, 'orbits/orbit_catalogue_%d.h5'), 'r', 
                 driver='family', memb_size=MEMBSIZE) as hdf:
        idx_pid = np.argwhere(hdf['HID'][()] == halo_id).reshape(-1)
        pid = hdf['PID'][idx_pid]
    
    # Load members tags (TRUE == orbiting)
    with h5.File(join(SRC, 'particle_classification.h5'), 'r') as hdf:
        tag = hdf['CLASS'][idx_pid]
    
    df = np.stack([pid, tag]).T
    df = pd.DataFrame(df, columns=['PID', 'TAG'])
    df.sort_values(by=['PID'], inplace=True, ignore_index=True)
    tag = df['TAG'].values
    tag = np.array(tag, dtype=bool)
    np.save(fnamet, tag)
    
    # Load all particles' PIDs
    with h5.File(join(SRC, 'particle_catalogue.h5'), 'r') as hdf:
        pid_all = hdf[f'snap99/{ds}/PID'][()]
        idx_member = np.isin(pid_all, pid)
        # Match to member's PID
        df2 = np.stack([
            hdf[f'snap99/{ds}/PID'][idx_member],
            hdf[f'snap99/{ds}/x'][idx_member],
            hdf[f'snap99/{ds}/y'][idx_member],
            hdf[f'snap99/{ds}/z'][idx_member],
            hdf[f'snap99/{ds}/vx'][idx_member],
            hdf[f'snap99/{ds}/vy'][idx_member],
            hdf[f'snap99/{ds}/vz'][idx_member],
            ]).T
    df2 = pd.DataFrame(df2, columns=['PID', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
    df2.sort_values(by=['PID'], inplace=True, ignore_index=True)
    xyz = (df2[['x', 'y', 'z']].values.T - halo_xyz).T
    vel = (df2[['vx', 'vy', 'vz']].values.T - halo_vel).T
    # Save relative positions for each particle.
    np.save(fnamer, xyz)
    np.save(fnamev, vel)
    np.save(fnamei, idx_member)