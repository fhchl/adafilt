import numpy as np
import warnings


def lfilter(b, a, x, zi=None):
    """Wrap olafilt as scipy.signal.lfilter."""
    warnings.warn("lfilter is deprecated.", DeprecationWarning)
    return olafilt(b, x, zi=zi)


def olafilt(b, x, zi=None, squeeze=True):
    """Filter a multi dimensional array with an FIR filter matrix.

    Filter a data sequence, `x`, using a FIR filter given in `b`.
    Filtering uses the overlap-add method converting both `x` and `b`
    into frequency domain first.  The FFT size is determined as the
    next higher power of 2 of twice the length of `b`.

    Parameters
    ----------
    b : array_like, shape ([[Nin,] Nout,] M)
        The impulse response of the filter matrix.
    x : array_like, shape ([Nin,] N)
        Signal to be filtered.
    zi : array_like, shape ([[Nout,] K), optional
        Initial condition of the filter, but in reality just the
        runout of the previous computation.  If `zi` is None or not
        given, then zero initial state is assumed.
    squeeze : bool, optional
        If `True`, squeeze dimensions from output arrays.

    Returns
    -------
    y : numpy.ndarray, shape ([Nout,] N)
        The output of the digital filter.
    zf : numpy.ndarray, shape ([Nout,] M - 1), optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.

    Notes
    -----
    Based on olfilt from `https://github.com/jthiem/overlapadd`
    """
    # bring into broadcasting shape
    b = np.array(b, copy=False, ndmin=3)
    x = np.array(x, copy=False, ndmin=2)

    _, Nout, L_I = b.shape

    # find power of 2 larger that 2*L_I (from abarnert on Stackoverflow)
    L_F = 2 << (L_I - 1).bit_length()  # FFT Size
    L_S = L_F - L_I + 1  # length of segments
    L_sig = x.shape[-1]
    offsets = range(0, L_sig, L_S)

    # handle complex or real input
    if np.iscomplexobj(b) or np.iscomplexobj(x):
        fft_func = np.fft.fft
        ifft_func = np.fft.ifft
        res = np.zeros((Nout, L_sig + L_F), dtype=np.complex128)
    else:
        fft_func = np.fft.rfft
        ifft_func = np.fft.irfft
        res = np.zeros((Nout, L_sig + L_F))

    B = fft_func(b, n=L_F)

    # overlap and add
    for n in offsets:
        Xseg = fft_func(x[..., n : n + L_S], n=L_F)
        res[..., n : n + L_F] += ifft_func(np.einsum("ik,ijk->jk", Xseg, B))

    if zi is not None:
        zi = np.array(zi, copy=True, ndmin=2)
        res[..., : zi.shape[-1]] = res[..., : zi.shape[-1]] + zi
        y = res[..., :L_sig]
        zf = res[..., L_sig : L_sig + L_I - 1]
        return (y.squeeze(), zf.squeeze()) if squeeze else (y, zf)
    else:
        y = res[..., :L_sig]
        return y.squeeze() if squeeze else y


def wgn(x, snr, unit=None):
    """Create white Gaussian noise with relative noise level SNR.

    Parameters
    ----------
    x : ndarray
        Signal.
    SNR : float
        Relative magnitude of noise, i.e. SNR = E(x)/E(n).
    unit : None or str, optional
        If `dB`, SNR is specified in dB, i.e. SNR = 10*log(E(x)/E(n)).

    Returns
    -------
    n: numpy.ndarray
        Noise.

    Examples
    --------
    Add noise with 0dB SNR to a sinusoidal signal:

    >>> t = np.linspace(0, 1, 1000000, endpoint=False)
    >>> x = np.sin(2*np.pi*10*t)
    >>> snr = 2
    >>> snrdB = 10*np.log10(snr)
    >>> n = wgn(x, snrdB, "dB")
    >>> xn = x + n
    >>> energy_x = np.linalg.norm(x)**2
    >>> energy_n = np.linalg.norm(n)**2
    >>> np.allclose(snr * energy_n, energy_x)
    True

    """
    if unit == "dB":
        snr = 10 ** (snr / 10)

    if np.iscomplexobj(x):
        n = np.random.standard_normal(x.shape) + 1j * np.random.standard_normal(x.shape)
    else:
        n = np.random.standard_normal(x.shape)

    n *= 1 / np.sqrt(snr) * np.linalg.norm(x) / np.linalg.norm(n)

    return n


def check_lengths(length, blocklength, h_pri, h_sec):
    primax = np.argmax(np.abs(h_pri))
    secmax = np.argmax(np.abs(h_sec))

    assert blocklength <= secmax, f"{blocklength} <= {secmax}"
    assert blocklength <= primax - secmax, f"{blocklength} <= {primax - secmax}"
    assert (
        length > primax - secmax - blocklength
    ), f"{length} > {primax} - {secmax} - {blocklength}"
