import numpy as np
from scipy import stats, integrate
from scipy.special import gamma, beta
from scipy.ndimage import maximum_filter, rotate
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML


def dot_movie(radius, density, size, speed, coherence, ppd, framerate,
              duration, moments, seed=None):
    """Make a 3D array with a patch from the random dot stimulus.

    Parameters
    ----------
    radius : float
        Size of the dot aperture in degrees.
    density : float
        Dots per degree squared per second.
    size : int
        Width of each dot in pixels.
    speed :float
        Speed of the coherent motion in degrees per second.
    coherence : float
        Proportion of dots that move coherently on each frame.
    ppd : float
        Pixels per degree.
    framerate : float
        Sampling rate for the temporal dimension.
    duration : float
        Length of the movie in seconds.
    moments : four-tuple
        Mean, std dev, skewness, and kurtosis of angular displacements,
        in degrees.
    seed : int or None
        Seed for the random number generator to get reproducible movies.

    Returns
    -------
    movie : 3d array
        Spatiotemporal array with the stimulus movie.

    """
    rs = np.random.RandomState(seed)

    # Initialize the spatial grid
    n_pix = int(2 * radius * ppd)
    grid = np.linspace(-radius, radius, n_pix)
    xx, yy = np.meshgrid(grid, grid)

    # Define thetas to sample dot displacements from
    thetas = np.linspace(-np.pi, np.pi, 1000)

    # Initialize the dots
    n_dots = int(np.round(density * np.pi * radius ** 2 / 75))
    xy = rs.uniform(-radius, radius, (2, n_dots))

    frames = []
    for _ in range(int(duration / framerate)):

        # Create a blank image for this frame
        img = np.zeros_like(xx)

        # Determine which dots will move coherently
        signal = rs.rand(n_dots) < coherence

        # Sample the angles of coherent dot motion
        mean, std, skew, kurtosis = moments
        angle = pearsrnd(thetas, np.deg2rad(mean), np.deg2rad(std),
                         skew, kurtosis, n_dots)

        # Convert to x y displacements
        norm = speed * framerate
        dxdy = np.array([norm * np.cos(angle), norm * np.sin(angle)])

        # Update dot positions
        xy = np.where(signal,
                      xy + dxdy,
                      rs.uniform(-radius, radius, (2, n_dots)))

        # Wrap-around for dots that have moved outside the aperture
        oob = np.abs(xy) > radius
        xy[oob] = -xy[oob] + dxdy[oob]

        # Draw dots in the screen image
        i, j = np.round(xy * ppd + radius * ppd - size / 2).astype(np.int)
        img[i, j] = 1
        img = maximum_filter(img, size)

        frames.append(img)

    movie = np.stack(frames, axis=-1)
    return movie


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def grating_movie(radius, cpd, speed, ppd, framerate, duration, angle):
    """Make a 3D array showing a drifting grating.

    Parameters
    ----------
    radius : float
        Radius of the aperture, in degrees.
    cpd : float
        Spatial frequency of the grating (cycles per degree).
    speed : float
        Drift speed, in degrees per second.
    ppd : float
        Spatial resolution of the display, in pixels per degree.
    framerate : float
        Temporal resolution of the movie, in frames per second.
    duration : float
        Duration of the movie, in seconds.
    angle : float
        Orientation of the grating, in degrees.

    Returns
    -------
    movie : 3D numpy array
        Spatiotemporal array with the simulus movie.

    """
    dx = 1 / ppd
    npix = 2 * radius * ppd
    n_frames = duration // framerate
    xx = np.linspace(-np.pi, np.pi, npix + 1)[:-1] * cpd * dx * npix
    grating, _ = np.meshgrid(np.sin(xx), xx, indexing="ij")

    step = speed * dx / dx
    offsets = np.arange(0, n_frames * step, step).round().astype(int)
    movie = np.stack([
        rotate(np.roll(grating, i, 0), angle, reshape=False) for i in offsets
    ], axis=-1)
    movie = (movie + 1) / 2

    return movie


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def play_movie(movie, framerate=1 / 60, aperture=False, size=5, as_html5=True,
               **kwargs):
    """Turn a 3D movie array into a matplotlib animation or HTML movie.

    Parameters
    ----------
    movie : 3D numpy array
        Array with time on the final axis.
    framerate : float
        Temporal resolution of the movie, in frames per second.
    aperture : bool
        If True, show only a central circular aperture.
    size : float
        Size of the underlying matplotlib figure, in inches.
    as_html : bool
        If True, return an HTML5 video; otherwise return the underying
        matplotlib animation object (e.g. to save to .gif).

    Returns
    -------
    anim : HTML object or FuncAnimation object
        Animation, format depends on `as_html`.

    """
    # Initialize the figure and an empty array for the frames
    f, ax = plt.subplots(figsize=(size, size))
    f.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()

    kwargs.setdefault("vmin", 0)
    kwargs.setdefault("vmax", 1)
    kwargs.setdefault("cmap", "gray")
    array = ax.imshow(np.zeros(movie.shape[:-1]), **kwargs)

    # Define the part of the image to show
    if aperture:
        aperture = circular_aperture(movie)
    else:
        aperture = np.ones(movie.shape[:2], np.bool)

    # Define animation functions
    def init_movie():
        return array,

    def animate_movie(i):
        frame = movie[..., i].astype(np.float)
        frame[~aperture] = np.nan
        array.set_data(np.rot90(frame))
        return array,

    # Produce the animation
    anim = animation.FuncAnimation(f,
                                   frames=movie.shape[-1],
                                   interval=framerate * 1000,
                                   blit=True,
                                   func=animate_movie,
                                   init_func=init_movie)

    plt.close(f)

    if as_html5:
        return HTML(anim.to_html5_video())
    return anim


def circular_aperture(a):
    """Define circular aperture for an array with square spatial dims."""
    n = a.shape[0]
    xx = np.linspace(-n / 2, n / 2, n)
    xx, yy = np.meshgrid(xx, xx)
    return np.sqrt(xx ** 2 + yy ** 2) < (n / 2)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def pearspdf(x, mu, sigma, skew, kurt):
    """Pearson distribution probability density function.

      Returns the probability distribution denisty of the pearsons distribution
      with mean `mu`, standard deviation `sigma`, skewness `skew` and
      kurtosis `kurt`, evaluated at the values in x.

      Some combinations of moments are not valid for any random variable, and
      in particular, the kurtosis must be greater than the square of the
      skewness plus 1.  The kurtosis of the normal distribution is defined to
      be 3.

      The seven distribution types in the Pearson system correspond to the
      following distributions:

         Type 0: Normal distribution
         Type 1: Four-parameter beta
         Type 2: Symmetric four-parameter beta
         Type 3: Three-parameter gamma
         Type 4: Not related to any standard distribution.
                 Density proportional to:
                    (1+((x-a / b)^2)^(-c) * exp(-d*arctan((x-a / b)).
         Type 5: Inverse gamma location-scale
         Type 6: F location-scale
         Type 7: Student's t location-scale


      References:
         [1] Johnson, N.L., S. Kotz, and N. Balakrishnan (1994) Continuous
             Univariate Distributions, Volume 1,  Wiley-Interscience.
         [2] Devroye, L. (1986) Non-Uniform Random Variate Generation,
             Springer-Verlag.

    Translated from the MATLAB pearspdf function.

    Original Author Information:

    Pierce Brady
    Smart Systems Integration Group - SSIG
    Cork Institute of Technology, Ireland.

    """
    eps = np.finfo(np.float).eps
    realmin = np.finfo(np.double).tiny

    x_orig = x.copy()
    x = (x - mu) / sigma

    beta1 = skew ** 2
    beta2 = kurt

    if sigma <= 0:
        raise ValueError("Standard deviation must be positive")
    if beta2 <= (beta1 + 1):
        raise ValueError("Skewness must be greater than kurtosis plus 1")

    c0 = (4 * beta2 - 3 * beta1)
    c1 = skew * (beta2 + 3)
    c2 = (2 * beta2 - 3 * beta1 - 6)

    if c1 == 0:
        if beta2 == 3:
            pearstype = 0
        else:
            if beta2 < 3:
                pearstype = 2
            elif beta2 > 3:
                pearstype = 7

            a1 = -np.sqrt(abs(c0 / c2))
            a2 = -a1

    elif c2 == 0:
        pearstype = 3
        a1 = -c0 / c1
    else:
        kappa = c1 ** 2 / (4*c0 * c2)
        if kappa < 0:
            pearstype = 1
        elif kappa < (1 - eps):
            pearstype = 4
        elif kappa <= (1 + eps):
            pearstype = 5
        else:
            pearstype = 6

        # Solve the quadratic for general roots a1 and a2
        # and sort by their real parts
        tmp = -(c1 + np.sign(c1) * np.sqrt(c1 ** 2 - 4 * c0 * c2)) / 2
        a1 = tmp / c2
        a2 = c0 / tmp

    denom = 10 * beta2 - 12 * beta1 - 18
    if abs(denom) > np.sqrt(realmin):
        c0 = c0 / denom
        c1 = c1 / denom
        c2 = c2 / denom
    else:
        pearstype = 1

    # -- Normal: standard support (-Inf,Inf)
    if pearstype == 0:

        m1 = 0
        m2 = 1
        p = stats.norm.pdf(x, m1, m2)

    # -- Four-parameter beta: standard support (a1,a2)

    elif pearstype == 1:

        if abs(denom) > np.sqrt(realmin):
            m1 = (c1 + a1) / (c2 * (a2 - a1))
            m2 = -(c1 + a2) / (c2 * (a2 - a1))
        else:
            # c1 and c2 -> Inf, but c / c2 has finite limit
            m1 = c1 / (c2 * (a2 - a1))
            m2 = -c1 / (c2 * (a2 - a1))

        # Transform to 0-1 interval
        x = (x - a1) / (a2 - a1)

        p = stats.beta.pdf(x, m1 + 1, m2 + 1)

    elif pearstype == 2:

        # symmetric four-parameter beta: standard support (-a1,a1)
        m = (c1 + a1) / (c2 * 2 * abs(a1))
        x = (x - a1) / (2 * abs(a1))
        p = stats.beta.pdf(x, m + 1, m + 1)

    # -- three-parameter gamma: standard support (a1,Inf) or (-Inf,a1)

    elif pearstype == 3:

        m = (c0 / c1 - c1) / c1
        x = (x - a1) / c1

        p = stats.gamma.pdf(x, m + 1)

    # -- Pearson IV is not a transformation of a standard distribution:

    elif pearstype == 4:

        # density  proportional to
        # (1+((x-lambda / a)^2)^(-m) * exp(-nu*arctan((x-lambda / a)),
        # standard support (-Inf,Inf)

        m = 1 / (2 * c2)
        nu = 2 * c1 * (1 - m) / np.sqrt((4 * c0 * c2 - c1 ** 2))
        b = 2 * (m - 1)
        a = np.sqrt(b ** 2 * (b-1) / (b ** 2 + nu ** 2))
        lam = a * nu / b
        p = _pearson4pdf(x, m, nu, a, lam) / sigma

    # -- Inverse gamma loc-scale: standard support (-C1,Inf) or (-Inf,-C1)

    elif pearstype == 5:

        C1 = c1 / (2 * c2)

        x = -((c1 - C1) / c2) / (x + C1)
        p = stats.gamma.pdf(x, 1 / c2 - 1)

    # --  F location-scale: standard support (a2,Inf) or (-Inf,a1)

    elif pearstype == 6:

        m1 = (a1 + c1) / (c2 * (a2 - a1))
        m2 = -(a2 + c1) / (c2 * (a2 - a1))

        if a2 < 0:

            nu1 = 2 * (m2 + 1)
            nu2 = -2 * (m1 + m2 + 1)
            x = (x - a2) / (a2 - a1) * (nu2 / nu1)

            p = stats.f.pdf(x, nu1, nu2)

        else:

            nu1 = 2 * (m1 + 1)
            nu2 = -2 * (m1 + m2 + 1)
            x = (x - a1) / (a1 - a2) * (nu2 / nu1)

            p = stats.f.pdf(x, nu1, nu2)

    # -- t location-scale: standard support (-Inf,Inf)

    elif pearstype == 7:

        nu = 1 / c2 - 1
        x = x / np.sqrt(c0 / (1 - c2))
        p = stats.t.pdf(x, nu)

    # Normalize the PDF
    p /= integrate.trapz(p, x_orig)

    return p


def _pearson4pdf(x, m, nu, a, lam):

    x = (x - lam) / a
    p1 = (np.abs(gamma(m + (nu / 2) * 1j) / gamma(m)) ** 2
          / (a * beta(m - 5, 5 * np.ones_like(m))))
    p2 = (1 + x ** 2) ** -m * np.exp(-nu * np.arctan(x))
    assert not np.isnan(p1).any()
    assert not np.isnan(p2).any()
    return p1 * p2


def pearsrnd(x, mu, sigma, skew, kurt, size=1, random_state=None):
    """Random samples from the Pearson distribution on support mesh x."""
    if random_state is None:
        rs = np.random.RandomState()

    p = pearspdf(x, mu, sigma, skew, kurt)
    p /= p.sum()
    return rs.choice(x, replace=True, p=p, size=size)
