from constants import *
from imports import np, os

data_points = np.array([])
r_vir = np.array([])
r_sk = np.array([])
hmass_scale = np.array([])
mean_age = np.array([])
mean_mass = np.array([])
std_age = np.array([])

with open(HALOS_PATH) as f:
    for line in f:
        if "#" not in line:
            l = np.fromstring(line, dtype=np.float64, sep=" ")
            data = l[10]  # Halo-masses
            check = l[5]  # Check for sub-halos
            if 1e15 > data > 1e11 and check == -1:
                data_points = np.append(data_points, data)
                r_vir = np.append(r_vir, l[11])  # Halo radius
                r_sk = np.append(r_sk, l[37])  # Scale radius
                hmass_scale = np.append(hmass_scale, l[63])  # Scale factor

for i in range(19):
    condition_mass = data_points[
        (MASS_BINS[i + 1] > data_points) & (data_points > MASS_BINS[i])
    ]
    condition_age = hmass_scale[
        (MASS_BINS[i + 1] > data_points) & (data_points > MASS_BINS[i])
    ]
    mean_mass = np.append(mean_mass, np.mean(condition_mass))
    mean_age = np.append(mean_age, np.mean(condition_age))
    std_age = np.append(std_age, np.std(condition_age))

if not os.path.exists(CACHE_PATH):
    os.mkdir(CACHE_PATH)

# Save the points (cache) to make it faster.
np.save(f"{CACHE_PATH}/halofunc_points", data_points)
np.save(f"{CACHE_PATH}/rvir_points", r_vir)
np.save(f"{CACHE_PATH}/rsk_points", r_sk)
np.save(f"{CACHE_PATH}/hmscale_points", hmass_scale)
np.save(f"{CACHE_PATH}/mean_mass", mean_mass)
np.save(f"{CACHE_PATH}/mean_age", mean_age)
np.save(f"{CACHE_PATH}/std_mass", std_age)
