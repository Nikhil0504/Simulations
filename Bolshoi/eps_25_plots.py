# %%
from loading import *
from halos import Halo

# %%
inds = np.loadtxt("out/inds.txt")
slices = np.array([0, 200, 400, 600, 800, 1000, 1200, 1400, 1497, 1545, 1568])[4:]
MASS_BINS = MASS_BINS[4:]

# %%
def Eps(
    inds: np.ndarray,
    ep: float,
    func: str = "gaussian",
    lib: str = "scipy"
) -> np.ndarray:
    main = np.zeros(0)

    for id, halo in enumerate(inds):
        print(f'{lib} {id}')
        h = Halo(int(halo[0]), MVIR, X, Y, Z, RVIR, RS)

        opts = np.zeros(0)

        for jack in range(9):
            density = halo[3 + (jack * 25):3 + ((jack + 1) * 25)]

            optres = h.minimise_cost(density, ep, func, lib)

            opts = np.append(opts, optres)

        if main.size == 0:
            main = opts
        else:
            main = np.vstack((main, opts))

    return main

# %%
main = Eps(inds, 0.10, 'gaussian', "scipy")
main2 = Eps(inds, 0.10, 'lorentz', "scipy")
main3 = Eps(inds, 0.10, 'abs', "scipy")

# %%
temp = se_jack(main[:, 1:], np.mean(main[:, 1:], axis=1), 8)
temp2 = se_jack(main2[:, 1:], np.mean(main2[:, 1:], axis=1), 8)
temp3 = se_jack(main3[:, 1:], np.mean(main3[:, 1:], axis=1), 8)

# %%
# imain = Eps(inds, 0.10, 'gaussian', 'scipy')
# imain2 = Eps(inds, 0.10, 'lorentz', 'scipy')
# imain3 = Eps(inds, 0.10, 'abs', 'scipy')

# itemp = se_jack(imain[:, 1:], np.mean(imain[:, 1:], axis=1), 8)
# itemp2 = se_jack(imain2[:, 1:], np.mean(imain2[:, 1:], axis=1), 8)
# itemp3 = se_jack(imain3[:, 1:], np.mean(imain3[:, 1:], axis=1), 8)

# %%
X, Y, y2, y3, y4, y5, y6 = (np.zeros(0) for _ in range(7))

for i in range(slices.shape[0] - 1):
    Y = np.append(Y, np.median(temp[slices[i]: slices[i+1]]))
    y3 = np.append(y3, np.median(temp2[slices[i]: slices[i+1]]))
    y5 = np.append(y5, np.median(temp3[slices[i]: slices[i+1]]))

    y2 = np.append(y2, np.std(main[slices[i]: slices[i + 1], 0]))
    y4 = np.append(y4, np.std(main2[slices[i]: slices[i + 1], 0]))
    y6 = np.append(y6, np.std(main3[slices[i]: slices[i + 1], 0]))

# %%
X, iy, iy2, iy3, iy4, iy5, iy6 = (np.zeros(0) for _ in range(7))

for i in range(slices.shape[0] - 1):
    iy = np.append(iy, np.mean(temp[slices[i]: slices[i+1]]))
    iy3 = np.append(iy3, np.mean(temp2[slices[i]: slices[i+1]]))
    iy5 = np.append(iy5, np.mean(temp3[slices[i]: slices[i+1]]))

    iy2 = np.append(iy2, np.std(main[slices[i]: slices[i + 1], 0]))
    iy4 = np.append(iy4, np.std(main2[slices[i]: slices[i + 1], 0]))
    iy6 = np.append(iy6, np.std(main3[slices[i]: slices[i + 1], 0]))

    X = np.append(X, np.mean([MASS_BINS[i], MASS_BINS[i + 1]]))

# %%
plt.style.use(['science'])

fig = plt.figure(figsize=(5, 5), dpi=150)

plt.plot(X, Y, '#008BF8', linestyle='--', linewidth=1.5, label='_Jack-knife Gaussian')
plt.plot(X, y2, '#DC0073', linestyle='--', linewidth=1.5, label='_Intrinsic Gaussian')

plt.plot(X, y3, '#008BF8', linestyle='dotted', linewidth=1.5, label='_Jack-knife Lorentz')
plt.plot(X, y4, '#DC0073', linestyle='dotted', linewidth=1.5, label='_Intrinsic Lorentz')

# plt.plot(X, iy, '#008BF8', linestyle='--', linewidth=1.5, label='_Jack-knife Gaussian Mean')
# plt.plot(X, iy2, '#DC0073', linestyle='--', linewidth=1.5, label='_Intrinsic Gaussian Mean')

# plt.plot(X, iy3, '#008BF8', linestyle='dotted', linewidth=1.5, label='_Jack-knife Lorentz Mean')
# plt.plot(X, iy4, '#DC0073', linestyle='dotted', linewidth=1.5, label='_Intrinsic Lorentz Mean')

plt.plot(X, y5, '#008BF8', linestyle='-', linewidth=1.5, label='_Jack-knife Absolute')
plt.plot(X, y6, '#DC0073', linestyle='-', linewidth=1.5, label='_Intrinsic Absolute')

# plt.plot(X, iy5, '#008BF8', linestyle='-', linewidth=1.5, label='_Jack-knife Absolute Mean')
# plt.plot(X, iy6, '#DC0073', linestyle='-', linewidth=1.5, label='_Intrinsic Absolute Mean')

plt.plot([], [], '--', color='#1E152A', linewidth=1.5, label='Gaussian')
plt.plot([], [], ':', color='#1E152A', linewidth=1.5, label='Lorentz')
plt.plot([], [], '-', color='#1E152A', linewidth=1.5, label='Absolute')
plt.plot([], [], '#008BF8', linewidth=1.5, label='Jack-knife')
plt.plot([], [], '#DC0073', linewidth=1.5, label='Intrinsic')

# plt.ylabel(r'$<\sigma_{\mathrm{jk}}> \mathrm{or} \ \sigma(c)$')
plt.ylabel(r'$Med(\sigma_{\mathrm{jk}}) \ \mathrm{or} \ \sigma(c)$')
plt.xlabel(r'$M_{\rm{vir}}$')
plt.xscale('log')
plt.legend()
# plt.show()
plt.savefig('1.png')
plt.clf()

# # %%
# plt.figure(figsize=(20,16), dpi=150)
# # plt.tight_layout()
# for i in range(slices.shape[0] - 1):
#     print(i)
#     plt.subplot(5, 5, i+1)
#     # plt.title(f'{ep * 100} \%')
#     new = itemp2[slices[i]: slices[i + 1]]
#     new2 = temp2[slices[i]: slices[i + 1]]


#     tempd = y3/iy3

#     plt.hist(new, facecolor=(0.86, 0, 0.45, 0.7), hatch='//', histtype='stepfilled', edgecolor=(0.12, 0.08, 0.16, 0.8), bins='fd', zorder=10)
#     plt.hist(new2, facecolor=(0.44, 0.68, 0.43, 0.65), hatch='.', histtype='stepfilled', edgecolor=(0.12, 0.08, 0.16, 0.8), bins='fd')

#     if i == 0:
#         plt.hist([], facecolor=(0.86, 0, 0.45, 0.7), hatch='//', histtype='stepfilled', edgecolor=(0.12, 0.08, 0.16, 0.8), label='Iminuit')
#         plt.hist([], facecolor=(0.44, 0.68, 0.43, 0.65), hatch='..', histtype='stepfilled', edgecolor=(0.12, 0.08, 0.16, 0.8), label='Scipy')
    
#     plt.plot([], color='black', label=f'\% Disagreement: {tempd[i]:1.3f}')

#     plt.legend(prop={'size': 8})
# plt.savefig('2.png')
# plt.clf()


# %%
# plt.style.use(['science', 'scatter'])

# fig = plt.figure(figsize=(5, 5), dpi=300)

# plt.plot(X, Y/iy, label='Gaussian')
# plt.plot(X, y3/iy3, label='Lorentz')
# plt.plot(X, y5/iy5, label='Absolute')

# plt.xlabel('Mass of bin edges')
# plt.ylabel(r'$Med(\sigma_{\mathrm{jk, scipy}}) - <\sigma_{\mathrm{jk, iminuit}}>$')

# plt.xscale('log')

# # plt.ylim((0.95, 1.06))

# plt.legend()
# # plt.show()
# plt.savefig('3.png')
# plt.clf()
