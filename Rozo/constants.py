import numpy as np
from os.path import abspath

# MAIN CONSTANTS
BIN_NO = 20
UPPER_LIMIT = 1e15
LOWER_LIMIT = 1e12

RSOFT = 0.015  # Softening length in Mpc/h
BOXSIZE = 1_000  # Mpc / h


RADIUS_BINS = np.logspace(np.log10(RSOFT), np.log10(5), BIN_NO+1, base=10)
MASS_BINS = np.logspace(12, 15, 11)

PART_MASS = 7.754657e+10
PIVOT_MASS = 1e14

RADIUS = (RADIUS_BINS[1:] + RADIUS_BINS[:-1]) / 2.0
VOLUME = 4. / 3. * np.pi * np.diff(RADIUS_BINS**3)

COSMO = {
    "flat": True,
    "H0": 70,
    "Om0": 0.3,
    "Ob0": 0.0469,
    "sigma8": 0.8355,
    "ns": 1,
}

RHOCRIT = 2.77536627e+11  # h^2 M_sun / Mpc^3
RHOM = RHOCRIT * COSMO["Om0"]  # h^2 M_sun / Mpc^3

MEMBSIZE = int(10 * 1000**3)  # 10.0 GB file

# PATHS
SRC = abspath('/spiff/edgarmsc/simulations/Banerjee/')
SDD = SRC
# NIKHIL_PATH = abspath('/spiff/nikhilgaruda/cache/Rozo/')
NIKHIL_PATH = abspath('/home/nikhilgaruda/Simulations/Rozo/out')