from constants import (CACHE_PATH, HALOS_PATH, LOWER_LIMIT, POINTS_PATH,
                       UPPER_LIMIT)
from imports import np, pd

data_points = np.array([])
x = np.array([])
y = np.array([])
z = np.array([])
rvir = np.array([])
rs = np.array([])

with open(HALOS_PATH) as f:
    for line in f:
        if "#" not in line:
            l = np.fromstring(line, dtype=np.float32, sep=" ")
            data = l[10]  # halos
            check = l[5]  # sub-halos
            # only get distinct halos and M_vir between 1e11-1e15 and
            # constrain the Z to a certain point.
            if check == -1 and l[19] < 10.0 and UPPER_LIMIT > data > LOWER_LIMIT:
                data_points = np.append(data_points, data)
                x = np.append(x, l[17])
                y = np.append(y, l[18])
                z = np.append(z, l[19])
                rvir = np.append(rvir, l[11])
                rs = np.append(rs, l[35])

# Break the big csv file to chunk(s) and convert to a numpy array
XYZ = np.ndarray(shape=(3,), dtype=np.float32)
df = pd.read_csv(POINTS_PATH, chunksize=100000000, header=0)
for chunk in df:
    lines = chunk.to_numpy()
    XYZ = np.vstack((XYZ, lines[:, 1:]))

np.save(f"{CACHE_PATH}/halofunc_points", data_points)
np.save(f"{CACHE_PATH}/rvir_points", rvir)
np.save(f"{CACHE_PATH}/rs_points", rs)
np.save(f"{CACHE_PATH}/x_points", x)
np.save(f"{CACHE_PATH}/y_points", y)
np.save(f"{CACHE_PATH}/z_points", z)
np.save(
    f"{CACHE_PATH}/arr_points",
    XYZ[1:],
)
