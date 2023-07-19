from typing import Tuple

from constants import PART_MASS, PIVOT_MASS, RADIUS, VOLUME
from imports import jit, njit, np
from scipy import integrate


### NFW Profile ###
@njit(fastmath=True)
def rho_o(M: float, Rvir: float, Rs: float):
    c = Rvir / Rs
    ln_term = np.log(1.0 + c) - (c / (1.0 + c))
    rho_not = M / (4.0 * np.pi * (Rs**3.0) * ln_term)
    return rho_not


@njit()
def rho_r(Rs: float, M: float, Rvir: float, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = RADIUS[mask]
    term = r / Rs
    rho_not = rho_o(M, Rvir, Rs)
    return r, rho_not / (term * ((1.0 + term)**2.0))

### New Density Profile ###
def rho_orb(Morb, r_h = None, a = None, alpha_inf = None, mask = np.array([])):
    """New Density Profile based on Rozo et al. (in prep)

    rho_orb(r|M_orb) = A * (r/(epsilon * r_h))^-alpha(r) * exp(-1/2 * (r/r_h)^2)))
    alpha(r) = alpha_inf * (r/(epsilon * r_h)) / (1 + (r/(epsilon * r_h)))
    
    Best Fit Parameters:
    r_h = (843.8 / 1000) * (Morb / PIVOT_MASS)**(0.221)
    epsilon = 0.038
    alpha_inf = 2.021 * (Morb / PIVOT_MASS)**(-0.053)
    """
    if mask.size == 0: 
        r = RADIUS
    else:
        r = RADIUS[mask]

    # change Kpc to Mpc
    if r_h == None:
        r_h = (843.8 / 1000) * (Morb / PIVOT_MASS)**(0.221)

    if a == None:
        a = 0.038
    else:
        a = a

    alphar = alpha_r(Morb, r, r_h, a, alpha_inf=alpha_inf)

    # M_orb = int_0^inf rho_orb(r|M_orb) * r^2 * dr
    A = normalisation_const(Morb, r_h, a)

    rho = A * (r / (a * r_h))**(-alphar) * np.exp(-1/2 * (r / r_h)**2)
    return rho


def alpha_r(Morb, r, r_h, epsilon, alpha_inf=None):
    if alpha_inf == None:
        alpha_inf = 2.021 * (Morb / PIVOT_MASS)**(-0.053)
    else:
        alpha_inf = alpha_inf
    return alpha_inf * (r / (epsilon * r_h)) / (1 + (r / (epsilon * r_h)))


def normalisation_const(Morb, r_h, epsilon):
    # A = Morb / 4 * pi * int_0^inf (r/(epsilon * r_h))^-alpha(r) * exp(-1/2 * (r/r_h)^2))) * r^2 * dr
    integrand = lambda r: (r / (epsilon * r_h))**(-alpha_r(Morb, r, r_h, epsilon)) * np.exp(-1/2 * (r / r_h)**2) * r**2

    integral = integrate.quad(integrand, 0, np.inf)[0]
    # print(f'Integral: {integral}')
    return Morb / (4 * np.pi * integral)


### Chi-Squared and Cost Functions ###
@njit(fastmath=True)
def chisq(obs: np.ndarray, model: np.ndarray, epsilon: float, mask, func: str="gaussian"):
    """
    Compute the chi-squared statistic for a given 
    observed and model distribution.
    
    Parameters
    ----------
    obs : numpy.ndarray, shape (N,)
        Observed distribution.
    model : numpy.ndarray, shape (N,)
        Model distribution.
    epsilon : float
        Uncertainty in observed distribution.
    mask : numpy.ndarray, shape (N,), boolean
        Boolean mask indicating which indices in `obs` and `model` 
        to use in the chi-squared calculation.
    func : str, optional
        The type of function to use in the chi-squared calculation. 
        Can be "gaussian", "lorentz", or "abs".
        Defaults to "gaussian".
    
    Returns
    -------
    float
        The computed chi-squared statistic.
    """
    # Scale the counts by the volume and particle mass
    Ndata = obs * (VOLUME[mask] / PART_MASS)
    Nmodel = model * (VOLUME[mask] / PART_MASS)

    # Compute the residual
    residual = Ndata - Nmodel

    # Compute the residual squared divided by the error squared
    denom = Ndata + np.square((epsilon * Ndata))

    if func == "gaussian":
        chisq = np.sum(np.square(residual) / denom)
    elif func == "lorentz":
        temp = np.square(residual) / denom
        chisq = np.sum(np.log(1 + temp))
    elif func == 'abs':
        chisq = np.sum(np.sqrt(np.square(residual) / denom))
    else:
        chisq = np.inf
        # raise(NotImplementedError(f"Function {func} not implemented"))


    return chisq


@jit(forceobj=True, fastmath=True)
def cost(params, M, obs, mask, epsilon=0.1, func="gaussian"):
    # rval, a = params
    a = params
    # r_h = (rval / 1000) * (M / PIVOT_MASS)**(0.221)
    r_h = None
    # alpha_inf = params * (M / PIVOT_MASS)**(-0.053)
    alpha_inf = None
    model = rho_orb(M, mask=mask, r_h=r_h, a=a, alpha_inf=alpha_inf)
    Cost = chisq(obs, model, epsilon, mask, func)
    return Cost

@jit(forceobj=True, fastmath=True)
def cost_nfw(lncvir, M, obs, mask, Rvir, epsilon=0.1, func="gaussian"):
    Rs = Rvir / np.exp(lncvir)
    _, model = rho_r(Rs, M, Rvir, mask)
    Cost = chisq(obs, model, epsilon, mask, func)
    return Cost

### Jackknife Resampling ###
def se_jack(jacks: np.ndarray, meanjk: np.ndarray, num: int) -> np.ndarray:
    """
    Calculate the standard error of the mean using jackknife resampling.

    Parameters
    ----------
    jacks : np.ndarray
        A NumPy array of jackknife samples, where each row represents a jackknife sample.
    meanjk : np.ndarray
        The mean of the jackknife samples.
    num : int
        The number of samples in the original dataset.

    Returns
    -------
    np.ndarray
        A NumPy array of the same shape as `meanjk`, 
        containing the standard error of the mean for each element.

    Notes
    -----
    The standard error is calculated using the formula:

    - If `jacks` is a one-dimensional array:
        sqrt(sum((jacks - meanjk) ** 2) * (num - 1) / num)

    - If `jacks` is a two-dimensional array:
        sqrt(sum((jacks[i, :] - meanjk[i]) ** 2) * (num - 1) / num)

    """
    if jacks.ndim == 1:
        return np.sqrt(
            np.sum(np.square(jacks - meanjk), axis=0) * (num - 1) / num)
    else:
        meanjk_expand = meanjk[:, None]  # Calculate once to avoid redundancy
        return np.sqrt(
            np.sum(np.square(jacks - meanjk_expand), axis=1) * (num - 1) / num)


### Outlier Removal ###
def remove_outliers(array, sigma=3):
    # Removes outliers within 3 sigma
    upper_boundary = np.mean(array) + sigma * np.std(array)
    lower_boundary = np.mean(array) - sigma * np.std(array)
    mask = np.where((lower_boundary < array) & (upper_boundary > array))
    return array[mask]


@njit(fastmath=True)
def remove_outliers_2(array, th1=0.25, th2=0.75):
    # Uses IQR methods
    q1 = np.quantile(array, th1)
    q3 = np.quantile(array, th2)
    iqr = q3 - q1
    upper_boundary = q3 + 1.5 * iqr
    lower_boundary = q1 - 1.5 * iqr
    mask = np.where((lower_boundary < array) & (upper_boundary > array))
    return array[mask]
