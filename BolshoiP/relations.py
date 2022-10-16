from loading import *

c = r_vir / r_sk

# Plotting the mass concentration relation
plt.hist2d(
    data_points,
    c,
    bins=[np.logspace(11, 15, 50), np.linspace(0, 30, 25)],
    norm=mpl.colors.LogNorm(),
    cmap="YlGnBu",
)
plt.title("Mass Concentration Relation")
plt.xscale("log")
plt.colorbar()
plt.ylabel("$c$")
plt.xlabel(r"$M_{\mathrm{vir}}$")
plt.savefig('figures/mass_c.png')
plt.clf()


# Plotting the mass age relation
plt.hist2d(data_points,
           hmass_scale,
           bins=[np.logspace(11, 15, 50), np.linspace(0, 1, 50)],
           norm=mpl.colors.LogNorm(),
           cmap='YlGnBu')
plt.title('Mass Age Relation')
plt.xscale('log')
plt.colorbar()
plt.ylabel(r'$a_{1/2}$')
plt.xlabel(r'$M_{\mathrm{vir}}$')
plt.savefig('figures/mass_age.png')
plt.clf()