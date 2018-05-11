from __future__ import division
from collections import namedtuple
from math import factorial
import numpy as np
from scipy.ndimage import rotate


FilterSet = namedtuple("FilterSet", ["p1", "p2", "n1", "n2"])


def motion_filters(size, res, csigx=0.35, cordx=4, gsigy=0.05, k=60, theta=0):
    """Create a set of Adelson-Bergen motion filters.

    Parameters
    ----------
    size : 3-tuple
        Size of the filter along the x, y, t dimensions.
    res : 3-tuple
        Resolution of the filter along the x, y, t dimensions.
    csigx : float
        Size (sigma) of the Cauchy function envelope (on x dimension).
    cordx : float
        Order of the Cauchy functions.
    gsigy : float
        Size (sigma) of the Gaussian function envelope (on y dimension).
    k : float
        Time-scale of the temporal impulse response functions.
    theta : float
        Preferred motion direction (in degrees). Default is rightward.

    Returns
    -------
    filters : named tuple

    """
    # Spatial filters on the x axis: odd and even cauchy functions
    xx = filter_grid(size[0], res[0], center=True)
    fx_e, fx_o = cauchy(xx, csigx, cordx)

    # Spatial filter on the y axis: gaussian envelope
    yy = filter_grid(size[1], res[1], center=True)
    fy = np.exp((-yy ** 2) / (2 * gsigy ** 2))

    # Temporal filters on the t axis: odd and even difference of Poissons
    tt = filter_grid(size[2], res[2], center=False)
    ft_e = temporal_impulse_response(tt, 5, k)
    ft_o = temporal_impulse_response(tt, 3, k)

    # Convert 1D filters to 3D arrays
    qxe, qy, qte = np.meshgrid(fx_e, fy, ft_e, indexing="ij")
    qxo, qy, qto = np.meshgrid(fx_o, fy, ft_o, indexing="ij")

    # Combine the spatial and temporal filters
    p1 = qy * (qto * qxe + qte * qxo)
    p2 = qy * (qto * qxo - qte * qxe)
    n1 = qy * (qto * qxe - qte * qxo)
    n2 = qy * (qto * qxo + qte * qxe)
    filters = [p1, p2, n1, n2]

    # Apply a rotation in the x, y plane
    if theta:
        filters = [rotate(f, theta, reshape=False) for f in filters]

    # Normalize the filter energy
    energy = np.sum(np.square(p1))
    p2 *= np.sqrt(energy / np.sum(np.square(p2)))
    n1 *= np.sqrt(energy / np.sum(np.square(n1)))
    n2 *= np.sqrt(energy / np.sum(np.square(n2)))

    # Pad on the temporal axis to make causal filters
    pad_width = [(0, 0), (0, 0), (size[-1] - 1, 0)]
    filters = [np.pad(f, pad_width, "constant") for f in filters]

    return FilterSet(*filters)


def filter_bank(thetas, *args, **kwargs):
    """Return a list of filters selective for each direction in thetas.

    Parameters
    ----------
    thetas : list
        List of direction preferences, in degrees.

    Other arguments and keyword arguments are passed to motion_filters.

    Returns
    -------
    filters : list
        List of FilterSet objects corresponding to thetas.

    """
    filters = []
    for theta in thetas:
        filters.append(motion_filters(*args, theta=theta, **kwargs))
    return filters


def cauchy(x, s, n=4, x0=0):
    """Return even and odd cauchy functions.

    Parameters
    ----------
    x : array
        Grid to evaluate the function on.
    s : float
        Size of the envelope (sigma).
    n : int
        Order of the cauchy function.

    Returns
    -------
    even_func, odd_func : arrays with same size as x

    """
    q = np.arctan2(x - x0, s)
    qn = np.power(np.cos(q), n)
    return qn * np.cos(n * q), qn * np.sin(n * q)


def gaussian_envelope(y, s):
    """Return a gaussian function with specified width.

    Parameters
    ----------
    y : array
        Grid to evaluate the function on.
    s : float
        Size of the envelope (sigma).

    returns
    -------
    envelope : array with same size as y

    """
    return np.exp((-y ** 2) / (2 * s ** 2))


def temporal_impulse_response(t, n, k):
    """Return a difference of Poissons function.

    Parameters
    ----------
    t : array
        Grid to evaluate the function on.
    n : int
        Impulse response parameter.
    k : float
        Time-scale factor for the impulse response.

    Returns
    -------
    t_func : array with same size t

    """
    return ((k * t) ** n
            * np.exp(-k * t)
            * (1 / factorial(n) - (k * t) ** 2 / factorial(n + 2)))


def filter_grid(size, dx, center=False):
    """Define a grid in image space based on size and sampling."""
    x = np.arange(0, size * dx, dx)
    if center:
        x = (x + dx / 2) - (size / 2) * dx
    assert x.size == size
    return x


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def apply_motion_energy_filters(stim, filter_array):
    """Compute motion energy through convolution.

    Parameters
    ----------
    stim : 3D array
        Movie of stimulus intensities.
    filter_array : list of FilterSet objects
        Motion energy filters.

    Returns
    -------
    energy_array : list of 3D arrays
        Movies of opponent motion energy for each orientation in `filter_array`
        with same shape as `stim`.

    """
    from numpy.fft import fftn, ifftn

    if isinstance(filter_array, list):
        single_filter = False
    else:
        single_filter = True
        filter_array = [filter_array]

    # Select one filter to get shape information for padding
    filter_shape = filter_array[0][0].shape
    nfft = np.add(stim.shape, filter_shape) - 1

    # FFT the stimulus
    stim_fft = fftn(stim, nfft)

    # Loop over each set of filters with different orientations
    energy_array = []
    for direction_filters in filter_array:

        # Loop over the filters in this set and convolve with the stimulus
        direction_energy = []
        for f in direction_filters:
            res = crop_convolved(ifftn(stim_fft * fftn(f, nfft)), stim.shape)
            direction_energy.append(res.real)

        # Compute opponent motion energy
        p1, p2, n1, n2 = direction_energy
        pref, null = p1 ** 2 + p2 ** 2, n1 ** 2 + n2 ** 2
        energy_array.append(pref - null)

    if single_filter:
        energy_array, = energy_array

    return energy_array


def crop_convolved(a, shape):
    """Crop an array to shape after circular convolution."""
    start = np.subtract(a.shape, shape) // 2
    end = np.add(start, shape)
    indices = tuple(slice(s, e) for s, e in zip(start, end))
    return a[indices]
