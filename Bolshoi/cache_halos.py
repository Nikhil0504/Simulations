# %%
from loading import *

# %%
bins = MASS_BINS
r = []
for ind in range(bins.size - 1):
    m_ind = np.nonzero(
            (data_points <= bins[ind + 1]) 
            & (data_points > bins[ind])
        )[0]
    N_samples = min([len(m_ind), 200])
    print(N_samples)
    r_ind = np.random.choice(m_ind, N_samples, replace=False)
    r.extend(r_ind)

# %%
actual_ind = np.array(r)
filesave = actual_ind.reshape(-1, 1)
filesave = np.hstack(
    (
        filesave,
        data_points[actual_ind].reshape(-1, 1),
        (rvir[actual_ind] / 1000).reshape(-1, 1),
    )
)

# %%
np.savetxt("inds.txt", filesave)

# %%
indexes = np.arange(9) + 1

for index in indexes:
    obss = np.array([])
    for r in actual_ind:
        M = data_points[r]
        Rvir = rvir[r] / 1000

        X, Y, Z = x[r], y[r], z[r]
        arr_points_2 = get_points(X, Y, Z, arr_points)

        R = compute_R2(arrays(arr_points_2, X, Y, Z, index))
        pairs, _ = np.histogram(R, bins=RADIUS_BINS)
        total_mass = np.array(pairs) * MASS * (100 / PERCENT)

        if index == 1:
            volume = Volume(1)
        else:
            volume = Volume(7.0 / 8.0)

        obs = total_mass / volume

        if obss.shape[0] == 0:
            obss = np.append(obss, obs)
        else:
            obss = np.vstack((obss, obs))
    filesave = np.hstack((filesave, obss))
    np.savetxt("inds.txt", filesave)

# %%
np.savetxt("inds.txt", filesave)
