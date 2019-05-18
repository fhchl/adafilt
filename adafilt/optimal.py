"""Functions for the optimal filtering problem."""

import numpy as np
from scipy.signal import csd
from adafilt.utils import atleast_2d, atleast_3d


def static_filter(p, g, n, squeeze=True):
    """Compute the optimal cancellation filter from primary and secondary paths.

    Parameters
    ----------
    p : array_like, shape (N1[, L])
        Primary path impulse response.
    g : array_like, shape (N2[, L[, M]])
        Secondary path impulse response.
    n : int
        Output filter length.
    squeeze: bool, optional
        Squeeze output dimensions.

    Returns
    -------
    numpy.ndarray, shape (n,[, M])
        Optimal filter in frequency domain.

    """
    p = atleast_2d(p)
    g = atleast_3d(g)

    P = np.fft.fft(p, n=n, axis=0)
    G = np.fft.fft(g, n=n, axis=0)

    M = G.shape[2]

    W = np.zeros((n, M), dtype=complex)
    for i in range(n):
        W[i] = - np.linalg.lstsq(G[i], P[i], rcond=None)[0]

    return W if not squeeze else W.squeeze()


def wiener_filter(x, d, n, g=None, constrained=False):
    """Compute optimal wiener filter for single channel control.

    From Elliot, Signal Processing for Optimal Control, Eq. 3.3.26

    Parameters
    ----------
    x : array_like
        Reference signal.
    d : array_like
        Disturbance signal.
    n : int
        Output filter length.
    g : None or array_like, optional
        Secondary path impulse response.
    constrained : bool, optional
        If True, constrain filter to be causal.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Optimal wiener filter in freqency domain.

    """
    if g is None:
        g = [1]

    G = np.fft.fft(g, n=n)

    _, Sxd = csd(x, d, nperseg=n, return_onesided=False)
    _, Sxx = csd(x, x, nperseg=n, return_onesided=False)

    if not constrained:
        return - Sxd / Sxx / G

    c = np.ones(n)
    c[n // 2:] = 0
    # half at DC and Nyquist
    c[0] = 0.5
    if n % 2 == 0:
        c[n // 2] = 0.5

    # minimum phase and allpass components of G
    Gmin = np.exp(np.fft.fft(c * np.fft.ifft(2 * np.log(np.abs(G)), n=n), n=n))
    Gall = G / Gmin

    # spectral factor
    F = np.exp(np.fft.fft(c * np.fft.ifft(np.log(Sxx), n=n), n=n))

    h = np.ones(n)
    h[n // 2:] = 0
    return - np.fft.fft(h * np.fft.ifft(Sxd / F.conj() / Gall), n=n) / (F * Gmin)


def multi_channel_wiener_filter(x, d, n, g=None, constrained=False, rho=0, beta=0):
    """Compute multichannel optimal wiener filter.

    From Elliot, Signal Processing for Optimal Control, Eq. 5.3.31

    Parameters
    ----------
    x : array_like, shape (N1[, K])
        K reference signals.
    d : array_like, shape (N2[, L])
        L disturbance signals.
    n : int
        Output filter length.
    g : None or array_like, shape (N3[, L[, M]]), optional
        Secondary path impulse response.
    constrained : bool, optional
        If True, constrain filter to be causal.
    rho: float
        Regularize through control effort weighting.
    beta: float
        Regularize through reference noise.

    Returns
    -------
    numpy.ndarray, shape (n,[, M])
        Optimal wiener filter in freqency domain.

    """
    if g is None:
        g = [1]

    x = atleast_2d(x)
    d = atleast_2d(d)
    g = atleast_3d(g)

    Nin = x.shape[1]
    _, Nmic, Nout = g.shape

    G = np.fft.fft(g, n=n, axis=0)

    Sxx = np.zeros((n, Nin, Nin), dtype=complex)
    for i in range(Nin):
        for j in range(Nin):
            _, S = csd(x[:, i], x[:, j], nperseg=n, return_onesided=False)
            Sxx[:, i, j] = S

    Sxd = np.zeros((n, Nmic, Nin), dtype=complex)
    for i in range(Nmic):
        for j in range(Nin):
            _, S = csd(x[:, j], d[:, i], nperseg=n, return_onesided=False)
            Sxd[:, i, j] = S

    if not constrained:
        return - np.linalg.pinv(G) @ Sxd @ np.linalg.pinv(Sxx + beta * np.identity(Nin))

    def hermtrans(x):
        return x.conj().transpose([0, 2, 1])

    def spectral_factor(x):
        c = np.ones((n, 1, 1))
        c[n // 2:] = 0
        # half at DC and Nyquist
        c[0] = 0.5
        if n % 2 == 0:
            c[n // 2] = 0.5
        return np.exp(np.fft.fft(c * np.fft.ifft(np.log(x), axis=0), axis=0))

    # spectral factor of G.H @ G is Gmin.H
    print(np.log(hermtrans(G) @ G + rho * np.identity(Nout)))
    Gmin = hermtrans(spectral_factor(hermtrans(G) @ G + rho * np.identity(Nout)))

    from response import Response
    fig = Response.from_freq(1, Gmin.T).plot()
    Gmin = spectral_factor(hermtrans(G) @ G + rho * np.identity(Nout))
    print(Gmin)
    Response.from_freq(1, Gmin.T).plot(use_fig=fig, show=True, flim=(1e-2, 0.5))
    Gall = G @ np.linalg.pinv(Gmin)

    # spectral factor if Sxx
    invF = np.linalg.pinv(spectral_factor(Sxx + beta * np.identity(Nin)))

    h = np.ones((n, 1, 1))
    h[n // 2:] = 0
    GhSFinvh = hermtrans(Gall) @ Sxd @ hermtrans(invF)
    GhSFinvh = np.fft.fft(h * np.fft.ifft(GhSFinvh, axis=0), axis=0)  # remove causal part
    return - np.linalg.pinv(Gmin) @ GhSFinvh @ invF
