from typing import Any

from loading import *

fig = plt.figure(figsize=(20, 8))
bins: Any = np.logspace(np.log10(11), np.log10(15), 50)

plt.hist(
    np.log10(data_points[1:]),
    bins=bins,
    facecolor="white",
    ec="b",
    density=True,
)

plt.title("Histogram Showing the Halo Mass Function")
plt.xlabel(r"log10 scale of $M_{vir}$")
plt.ylabel("Number Density (Normalised)")
plt.savefig("figures/nd_hm.png")
plt.clf()
