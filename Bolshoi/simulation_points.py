# This code is for checking 4 selected halos without jackknifing them
# and saving their densities for future use.
from loading import *

# Hard coding the desired locations to get
# 4 ranges of halos
locations = np.array([[2, 31, 114]])
m_location = np.argmax(data_points)
locations = np.append(locations, m_location)

densities = []

for location in locations:
    X, Y, Z = x[location], y[location], z[location]

    arr_points_2 = get_points(X, Y, Z, arr_points)
    R = compute_R(X, Y, Z, arr_points_2)

    pairs, _ = np.histogram(R, bins=RADIUS_BINS)
    total_mass = np.array(pairs) * MASS * (100 / PERCENT)

    volume = Volume(a=1, bn=BIN_NO)
    density = total_mass / volume

    densities.append(list(density))

densities = np.array(densities)
np.save(f"{CACHE_PATH}/simulation_points", densities)
