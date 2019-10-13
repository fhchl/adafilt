import numpy as np
import warnings


def lfilter(b, a, x, zi=None):
    """Wrap olafilt as scipy.signal.lfilter."""
    warnings.warn("lfilter is deprecated.", DeprecationWarning)
    return olafilt(b, x, zi=zi)


def atleast_2d(a):
    """Similar to numpy.atleast_2d, but always appends new axis."""
    a = np.asanyarray(a)
    if a.ndim == 0:
        result = a.reshape(1, 1)
    elif a.ndim == 1:
        result = a[:, np.newaxis]
    else:
        result = a
    return result


def atleast_3d(a):
    """Similar to numpy.atleast_3d, but always appends new axis."""
    a = np.asanyarray(a)
    if a.ndim == 0:
        result = a.reshape(1, 1, 1)
    elif a.ndim == 1:
        result = a[:, np.newaxis, np.newaxis]
    elif a.ndim == 2:
        result = a[..., np.newaxis]
    else:
        result = a
    return result


def atleast_4d(a):
    """Similar to numpy.atleast_3d, but always appends new axis."""
    a = np.asanyarray(a)
    if a.ndim == 0:
        result = a.reshape(1, 1, 1, 1)
    elif a.ndim == 1:
        result = a[:, np.newaxis, np.newaxis, np.newaxis]
    elif a.ndim == 2:
        result = a[..., np.newaxis, np.newaxis]
    elif a.ndim == 3:
        result = a[..., np.newaxis]
    else:
        result = a
    return result


def olafilt(b, x, zi=None, sum_inputs=True):
    """Filter a multi dimensional array with an FIR filter matrix.

    Filter a data sequence, `x`, using a FIR filter given in `b`.
    Filtering uses the overlap-add method converting both `x` and `b`
    into frequency domain first.  The FFT size is determined as the
    next higher power of 2 of twice the length of `b`.

    Parameters
    ----------
    b : array_like, shape (m[, L[, M]] )
        The impulse response of the filter matrix with `L` outputs and `M` inputs.
    x : array_like, shape (n[, K])
        `K` signals to be filtered.
    zi : array_like, shape (m - 1[, L[, M[, K]]]), optional
        Initial condition of the filter, but in reality just the
        runout of the previous computation.  If `zi` is None (default), then zero
        initial state is assumed.
    sum_inputs : bool, optional
        If `True`, sum the result over all inputs, assuming `M == K`. If `False`, return
        all possible combinations of filtering `x` with the filters `b` without summing.


    Returns
    -------
    y : numpy.ndarray
        The output of the digital filter. Has shape (n[, L]), if `sum_inputs==True`,
        else shape (n[, L[, M[, K]]]).
    zf : numpy.ndarray
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter state. Has shape (m - 1[, L]), if `sum_inputs=True`, else shape
        (m - 1[, L[, M[, K]]]).

    Notes
    -----
    Based on olafilt from `https://github.com/jthiem/overlapadd`

    """
    # bring into broadcasting shape
    b = np.asarray(b)
    x = np.asarray(x)

    L_I = b.shape[0]
    L_sig = x.shape[0]

    # find power of 2 larger that 2*L_I (from abarnert on Stackoverflow)
    L_F = int(2 << (L_I - 1).bit_length())  # FFT Size
    L_S = L_F - L_I + 1  # length of segments
    offsets = range(0, L_sig, L_S)

    # compute output shape
    if b.ndim == 1:
        bsig = "n"
    elif b.ndim == 2:
        bsig = "nl"
    elif b.ndim == 3:
        bsig = "nlm"
    else:
        raise ValueError("b must not have more than 3 dimensions")

    if x.ndim == 1:
        xsig = "n"
    elif x.ndim == 2:
        xsig = "nk"
    else:
        raise ValueError("x must not have more than 2 dimensions")

    outsig = "n" + bsig[1:] + xsig[1:]
    outshape = [L_sig + L_F] + list(b.shape[1:]) + list(x.shape[1:])

    if sum_inputs:
        outsig = outsig.replace("m", "")
        outsig = outsig.replace("k", "")
        if "m" in bsig:
            outshape.pop(2)
        if "k" in xsig:
            outshape.pop(-1)

    # handle complex or real input
    if np.iscomplexobj(b) or np.iscomplexobj(x):
        fft_func = np.fft.fft
        ifft_func = np.fft.ifft
        res = np.zeros(outshape, dtype=np.complex128)
    else:
        fft_func = np.fft.rfft
        ifft_func = np.fft.irfft
        res = np.zeros(outshape)

    B = fft_func(b, n=L_F, axis=0)

    # overlap and add
    for n in offsets:
        Xseg = fft_func(x[n : n + L_S], n=L_F, axis=0)
        res[n : n + L_F] += ifft_func(
            np.einsum(f"{bsig},{xsig}->{outsig}", B, Xseg), axis=0
        )

    if zi is not None:
        res[: zi.shape[0]] = res[: zi.shape[0]] + zi
        return res[:L_sig], res[L_sig : L_sig + L_I - 1]

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
