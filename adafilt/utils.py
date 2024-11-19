import numpy as np


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


def einsum_outshape(subscripts, *operants):
    """Compute the shape of output from `numpy.einsum`.

    Does not support ellipses.

    """
    if "." in subscripts:
        raise ValueError(f"Ellipses are not supported: {subscripts}")

    insubs, outsubs = subscripts.replace(",", "").split("->")
    if outsubs == "":
        return ()
    insubs = np.array(list(insubs))
    innumber = np.concatenate([op.shape for op in operants])
    outshape = []
    for o in outsubs:
        (indices,) = np.where(insubs == o)
        try:
            outshape.append(innumber[indices].max())
        except ValueError:
            raise ValueError(f"Invalid subscripts: {subscripts}")
    return tuple(outshape)


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


def fifo_extend(a, b):
    """Right-extend a with b and popleft same number of elements."""
    n = len(b)
    a[:-n] = a[n:]
    a[-n:] = b


def fifo_append_left(a, b):
    """Left-extend a with b and popleft same number of elements."""
    a[1:] = a[:-1]
    a[:1] = b
