import numpy as np


def lfilter(b, a, x, zi=None):
    """Wrap olafilt as scipy.signal.lfilter."""
    return olafilt(b, x, zi=zi)


def olafilt(b, x, zi=None):
    """Filter a one-dimensional array with an FIR filter.

    Filter a data sequence, `x`, using a FIR filter given in `b`.
    Filtering uses the overlap-add method converting both `x` and `b`
    into frequency domain first.  The FFT size is determined as the
    next higher power of 2 of twice the length of `b`.

    Parameters
    ----------
    b : one-dimensional numpy array
        The impulse response of the filter
    x : one-dimensional numpy array
        Signal to be filtered
    zi : one-dimensional numpy array, optional
        Initial condition of the filter, but in reality just the
        runout of the previous computation.  If `zi` is None or not
        given, then zero initial state is assumed.

    Returns
    -------
    y : array
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.

    Source: https://github.com/jthiem/overlapadd
    """
    b = np.asarray(b)
    x = np.asarray(x)

    L_I = b.shape[0]
    # Find power of 2 larger that 2*L_I (from abarnert on Stackoverflow)
    L_F = 2 << (L_I - 1).bit_length()
    L_S = L_F - L_I + 1
    L_sig = x.shape[0]
    offsets = range(0, L_sig, L_S)

    # handle complex or real input
    if np.iscomplexobj(b) or np.iscomplexobj(x):
        fft_func = np.fft.fft
        ifft_func = np.fft.ifft
        res = np.zeros(L_sig + L_F, dtype=np.complex128)
    else:
        fft_func = np.fft.rfft
        ifft_func = np.fft.irfft
        res = np.zeros(L_sig + L_F)

    FDir = fft_func(b, n=L_F)

    # overlap and add
    for n in offsets:
        res[n : n + L_F] += ifft_func(fft_func(x[n : n + L_S], n=L_F) * FDir)

    if zi is not None:
        res[: zi.shape[0]] = res[: zi.shape[0]] + zi
        return res[:L_sig], res[L_sig:]
    else:
        return res[:L_sig]


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
