# %%
from halos import Halo
from loading import *

# %%
bins = MASS_BINS[4:]
r = []
for ind in range(bins.size - 1):
    m_ind = np.nonzero((MVIR <= bins[ind + 1]) & (MVIR > bins[ind]))[0]
    print(len(m_ind))
    r.extend(m_ind)

# %%
actual_ind = np.array(r)

filesave = actual_ind.reshape(-1, 1)
filesave = np.hstack(
    (
        filesave,
        MVIR[actual_ind].reshape(-1, 1),
        (RVIR[actual_ind] / 1000).reshape(-1, 1),
    )
)

# %%
np.savetxt("out/inds_full.txt", filesave)

# %%
indexes = np.arange(9) + 1

for index in indexes:
    obss = np.array([])
    for r in actual_ind:
        print(r)
        h = Halo(r, MVIR, X, Y, Z, RVIR, RS)

        if index == 1:
            obs = h.densities(ARR_POINTS, index, 1)
        else:
            obs = h.densities(ARR_POINTS, index, 7./8.)

        if obss.shape[0] == 0:
            obss = np.append(obss, obs)
        else:
            obss = np.vstack((obss, obs))

    filesave = np.hstack((filesave, obss))
    np.savetxt("out/inds.txt", filesave)

# # %%
# HID,Mvir, Rvir, rho, rho_jk 1-8
np.savetxt("out/inds.txt", filesave)
np.save("out/inds.npy", filesave)