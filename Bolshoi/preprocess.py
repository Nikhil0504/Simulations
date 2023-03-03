from constants import (CACHE_PATH, HALOS_PATH, LOWER_LIMIT, POINTS_PATH,
                       UPPER_LIMIT)
from imports import np, pd


def pre_halo_cat(HP: str, UL: float, LL: float):
    """This method is used for preprocessing the halo catelogs
    downloaded from the CosmoSim website. The halo catelog is
    in the form of an ASCII file.

    We will be extracting the following parameters:
    Halo Mass: `mvir` (Msun/h)
    X, Y, Z: `x`, `y`, `z` (Mpc/h comoving)
    Halo Radius: `rvir` (kpc/h comoving)
    Scale Radius: `rs` (kpc/h comoving)

    Parameters
    ----------
    HP : str
        _description_
    UL : float
        _description_
    LL : float
        _description_
    """
    mvir = np.array([])
    x = np.array([])
    y = np.array([])
    z = np.array([])
    rvir = np.array([])
    rs = np.array([])

    with open(HP) as f:
        for line in f:
            if "#" not in line:
                l = np.fromstring(line, dtype=np.float32, sep=" ")
                data = l[10]  # halos
                check = l[5]  # sub-halos
                # only get distinct halos and M_vir between 1e11-1e15 and
                # constrain the Z to a certain point.
                if check == -1 and UL > data > LL:
                    mvir = np.append(mvir, data)
                    x = np.append(x, l[17])
                    y = np.append(y, l[18])
                    z = np.append(z, l[19])
                    rvir = np.append(rvir, l[11])
                    rs = np.append(rs, l[35])

    np.save(f"{CACHE_PATH}/halofunc_points_full", mvir)
    np.save(f"{CACHE_PATH}/rvir_points_full", rvir)
    np.save(f"{CACHE_PATH}/rs_points_full", rs)
    np.save(f"{CACHE_PATH}/x_points_full", x)
    np.save(f"{CACHE_PATH}/y_points_full", y)
    np.save(f"{CACHE_PATH}/z_points_full", z)


def pre_part_file(PP: str, chunk_size: int = 100000000):
    """This function is used for preprocessing the particle file
    from CosmoSim. This file is a CSV due to which we use pandas
    chunking function to convert it into a numpy array.

    The numpy array will be of the shape (n, 3).

    Parameters
    ----------
    PP : str
        The Particle File
    chunk : int, optional
        The chunk size for the file, by default 100000000
    """
    # Break the big csv file to chunk(s) and convert to a numpy array
    XYZ = np.ndarray(shape=(3,), dtype=np.float32)
    df = pd.read_csv(PP, chunksize=chunk_size, header=0)
    for chunk in df:
        lines = chunk.to_numpy()
        XYZ = np.vstack((XYZ, lines[:, 1:]))

    np.save(f"{CACHE_PATH}/arr_points", XYZ[1:])


def main():
    """Main function of the script."""
    print('Starting...')
    pre_halo_cat(HALOS_PATH, UPPER_LIMIT, LOWER_LIMIT)
    pre_part_file(POINTS_PATH)
    print('done!')


if __name__ == "__main__":
    main()
