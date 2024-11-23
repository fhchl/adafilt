"""Adaptive filtering classes."""
from abc import ABC, abstractmethod

import numpy as np

from adafilt.utils import (
    atleast_2d,
    atleast_4d,
    einsum_outshape,
    fifo_append_left,
    fifo_extend,
)


def olafilt(b, x, subscripts=None, zi=None):
    """Efficiently filter a long signal with a FIR filter using overlap-add.

    Filter a data sequence, `x`, using a FIR filter given in `b`.
    Filtering uses the overlap-add method converting both `x` and `b`
    into frequency domain first.  The FFT size is determined as the
    next higher power of 2 of twice the length of `b`.

    Multi-channel fitering is support via `numpy.einsum` notation.

    Parameters
    ----------
    b : array_like, shape (m[, ...])
        Filter matrix with `m` taps.
    x : array_like, shape (n[, ...])
        Input signal.
    subscripts : str or None, optional
        String that defines the matrix operations in the multichannel case using the
        notation from `numpy.einsum`. Subscripts for `b` and `x` and output must start
        with the same letter, e.g. `nlmk,nk->nl`.
    zi : int or array_like, shape (m - 1[, ...]), optional
        Initial condition of the filter, but in reality just the runout of the previous
        computation.  If `zi` is None (default), then zero initial state is assumed.
        Zero initial state can be explicitly passes with `0`. Shape after first
        dimention must be compatible with output defined via `subscripts`.

    Returns
    -------
    y : numpy.ndarray
        The output of the digital filter. The precise output shape is defined by
        `subscripts`, but always `y.shape[0] == n`.
    zf : numpy.ndarray
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter state. The precise output shape is defined by `subscripts`, but
        always `zf.shape[0] == m - 1`.

    Notes
    -----
    Based on olafilt from `https://github.com/jthiem/overlapadd`

    """
    b = np.asarray(b)
    x = np.asarray(x)

    if (b.ndim > 1 or x.ndim > 1) and subscripts is None:
        raise ValueError("Supply `subscripts` argument for multi-channel filtering.")

    M = b.shape[0]
    Nx = x.shape[0]

    # TODO: use more optimal choice of FFT size
    #       (https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method#Efficiency_considerations)
    # find power of 2 larger that 2*L_I (from abarnert on Stackoverflow)
    N = int(2 << (M - 1).bit_length())  # FFT Size
    step_size = N - M + 1  # length of segments /
    offsets = range(0, Nx, step_size)

    outshape = (Nx + N,)
    if subscripts is not None:
        outshape += einsum_outshape(subscripts, b, x)[1:]

    # handle complex or real input
    if np.iscomplexobj(b) or np.iscomplexobj(x):
        fft_func = np.fft.fft
        ifft_func = np.fft.ifft
        C = np.zeros((N,) + outshape[1:], dtype=np.complex128)
        res = np.zeros(outshape, dtype=np.complex128)
    else:
        fft_func = np.fft.rfft
        ifft_func = np.fft.irfft
        C = np.zeros((N // 2 + 1,) + outshape[1:], dtype=np.complex128)
        res = np.zeros(outshape)

    B = fft_func(b, n=N, axis=0)

    # overlap and add
    for n in offsets:
        Xseg = fft_func(x[n : n + step_size], n=N, axis=0)

        if subscripts is None:
            # fast 1D case
            C[:] = B * Xseg
        else:
            C[:] = np.einsum(subscripts, B, Xseg)

        res[n : n + N] += ifft_func(C, axis=0)

    if zi is not None:
        res[: M - 1] += zi
        return res[:Nx], res[Nx : Nx + M - 1]

    return res[:Nx]


class FIRFilter:
    """An overlap-add Filter."""

    def __init__(self, w, subscripts=None, zi=None):
        """Create overlap-add filter object.

        Parameters
        ----------
        w : array_like
            Filter taps.
        subscripts: str or none, optional
            Defines multi-channel case with `numpy.einsum` notation. See
            `olafilt` for details.
        zi : None or array_like, optional
            Initial filter state.

        """
        self.w = np.asarray(w)
        self._subscripts = subscripts
        self._zi = zi if zi is not None else 0

    def __call__(self, x):
        """See `FIRFilter.filt`."""
        return self.filt(x)

    def filt(self, x):
        """Filter signal x.

        Parameters
        ----------
        x : array_like
            Input signal

        Returns
        -------
        y : numpy.ndarray
            Output signal with `y.shape[0] == x.shape[0]`.

        """
        y, self._zi = olafilt(self.w, x, subscripts=self._subscripts, zi=self._zi)
        return y


class Delay:
    """A simple delay."""

    def __init__(
        self, n_delay, n_sig=None, zi=None, blocklength=None, dtype=np.float64
    ):
        """Create simple delay.

        Parameters
        ----------
        n_delay : int
            Delay by `n_delay` samples.
        n_sig : int
            Number of channels.
        zi : numpy.ndarray, shape (n_delay, [n_sig]) or None, optional
            Initial filter condition.

        """
        if n_sig is not None:
            self._zi = np.zeros((n_delay, n_sig), dtype=dtype)
        else:
            self._zi = np.zeros(n_delay, dtype=dtype)

        if zi is not None:
            self._zi[:] = zi

        self._n_delay = n_delay
        self._blocklength = blocklength

    def __call__(self, x):
        """See `Delay.filt`."""
        return self.filt(x)

    def filt(self, x, out=None):
        """Filter signal.

        Parameters
        ----------
        x : numpy.ndarray, shape (N,) or (N, M)
            Signal with samples along first dimension
        out : numpy.ndarray, shape (N,) or (N, M), optional
            Preallocated output array

        Returns
        -------
        numpy.ndarray
            The filtered signal of shape (N, ) or (N, M)

        """
        x = np.asarray(x)

        if out is None:
            out = np.empty(x.shape, dtype=x.dtype)

        n = x.shape[0]
        if n <= self._n_delay:
            out[:] = self._zi[:n]
            self._zi[:-n:] = self._zi[n:]
            self._zi[-n:] = x
        else:
            out[: self._n_delay] = self._zi
            out[self._n_delay :] = x[: -self._n_delay]
            self._zi[:] = x[-self._n_delay :]

        return out


class AdaptiveFilter(ABC):
    """Base class for adaptive filters."""

    blocklength: int
    length: int
    w: np.ndarray

    def reset(self):
        """Reset filter object."""
        raise NotImplementedError

    @abstractmethod
    def filt(self, x):
        """Filter a block."""

    @abstractmethod
    def adapt(self, x, e):
        """Adapt to a block."""

    def __call__(self, x, d, sec_path_coeff=None, sec_path_est=None):
        """Compute filter output, error, and coefficients.

        Parameters
        ----------
        x : (N,) array_like
            Reference signal. `N` must be divisable by filter blocklength.
        d : (N,) array_like
            Desired signal. `N` must be divisable by filter blocklength.
        sec_path_coeff : array_like or None, optional
            Coefficients of the secondary path filter model.
        sec_path_est : None, optional
            Estimate of secondary path filter model.

        Returns
        -------
        y : (N,) numpy.ndarray
            Filter output.
        u : (N,) numpy.ndarray
            Filter output at error signal.
        e : (N,) numpy.ndarray
            Error signal.
        w : (N, length)
            Filter coefficients at each block.

        """
        x = np.atleast_1d(x)
        d = np.atleast_1d(d)
        assert x.ndim == 1
        assert x.shape[0] % self.blocklength == 0
        assert x.shape == d.shape

        if sec_path_est is not None:
            fx = olafilt(sec_path_est, x)  # filtered reference signal
        else:
            fx = x

        x = x.reshape((-1, self.blocklength))
        d = d.reshape((-1, self.blocklength))
        n_blocks = x.shape[0]
        w = np.zeros((n_blocks, self.length))  # filter history
        e = np.zeros((n_blocks, self.blocklength))  # error signal
        y = np.zeros((n_blocks, self.blocklength))  # filtered output signal
        u = np.zeros((n_blocks, self.blocklength))  # control signal at error mic
        fx = fx.reshape((-1, self.blocklength))

        if sec_path_coeff is not None:
            zi = 0

        for n in range(n_blocks):
            w[n] = self.w
            y[n] = self.filt(x[n])

            # control signal at error sensor
            if sec_path_coeff is not None:
                u[n], zi = olafilt(sec_path_coeff, y[n], zi=zi)
            else:
                u[n] = y[n]

            # error signal
            e[n] = d[n] - u[n]

            self.adapt(fx[n], e[n])

        return y.reshape(-1), u.reshape(-1), e.reshape(-1), w


class LMSFilter(AdaptiveFilter):
    """A sample-wise Least-Mean-Square adaptive filter."""

    def __init__(
        self,
        length,
        stepsize=0.1,
        leakage=1,
        initial_coeff=None,
        normalized=True,
        epsilon_power=1e-5,
    ):
        """Create sample-wise Least-Mean-Square adaptive filter object.

        Parameters
        ----------
        length : int
            Length of filter coefficient vector.
        stepsize : float, optional
            Adaptation step size
        leakage : float, optional
            Leakage factor.
        initial_coeff : (length,) array_like or None, optional
            Initial filter coefficient vector. If `None` defaults to zeros.
        normalized : bool, optional
            If `True` take normalize step size with signal power.
        epsilon_power : float, optional
            Add this to power normalization factor to avoid instability at very small
            signal levels.

        Notes
        -----
        The normalised LMS algorithm is optimal in terms of convergence speed, or
        tracking capabilities, but will not necessarily be optimal in terms of final
        mean square error (Hansen, p. 419)

        """
        assert 0 < leakage and leakage <= 1
        self.blocklength = 1
        self.length = length
        self.stepsize = stepsize
        self.leakage = leakage
        self.epsilon_power = epsilon_power
        self.normalized = normalized

        self.reset()

        if initial_coeff is not None:
            self.w[:] = initial_coeff

    def reset(self):
        self.w = np.zeros(self.length)
        self._xfiltbuff = np.zeros(self.length)
        self._xbuff = np.zeros(self.length)

    def filt(self, x):
        """Filtering step.

        Parameters
        ----------
        x : complex
            Reference signal.

        Returns
        -------
        y : complex
            Filter output.

        """
        fifo_append_left(self._xfiltbuff, x)
        y = self.w.conj().dot(self._xfiltbuff)
        return y

    def adapt(self, x, e):
        """Adaptation step.

        Parameters
        ----------
        x : complex
            Reference signal.
        e : complex
            Error signal, e.g. `d - y`.

        """
        fifo_append_left(self._xbuff, x)
        xvec = np.asarray(self._xbuff)

        if self.normalized:
            stepsize = self.stepsize / (np.dot(xvec, xvec) + self.epsilon_power)
        else:
            stepsize = self.stepsize

        self.w = self.leakage * self.w + stepsize * xvec * np.conj(e)


class FastBlockLMSFilter(AdaptiveFilter):
    """A fast, block-wise LMS adaptive filter based on overlap-save sectioning."""

    def __init__(
        self,
        length=32,
        blocklength=32,
        stepsize=0.1,
        leakage=1,
        power_averaging=0.5,
        initial_coeff=None,
        epsilon_power=1e-5,
        constrained=True,
        normalized=True,
    ):
        """Create fast, block-wise normalized LMS adaptive filter object.

        After Haykins Table 8.1.

        Parameters
        ----------
        length : int, optional
            Length of filter coefficient vector.
        blocklength : int, optional
            Number of samples in one block.
        stepsize : float, optional
            Adaptation step size.
        leakage : float or (length,) array_like, optional
            Leakage factor.
        power_averaging : float, optional
            Averaging factor for signal power.
        initial_coeff : (length,) array_like or None, optional
            Initial filter coefficient vector. If `None` defaults to zeros.
        initial_power : float, optional
            initial signal power.
        epsilon_power : float, optional
            Add this to power normalization factor to avoid instability at very small
            signal levels.
        constrained : bool, optional
            Causally constrain the filter.
        normalized : bool, optional
            If `True`, normalizes step size with signal power. This improves convergence
            speed at the cost of bias in the filter.

        Notes
        -----
        The normalised LMS algorithm is optimal in terms of convergence speed, or
        tracking capabilities, but will not necessarily be optimal in terms of final
        mean square error (Hansen, p. 419)

        """
        assert length >= blocklength, "Filter must be at least as long as block"

        self.blocklength = blocklength
        self.stepsize = stepsize
        self.power_averaging = power_averaging
        self.constrained = constrained
        self.normalized = normalized
        self.epsilon_power = epsilon_power

        if length is None:
            length = blocklength
        self.length = length

        if isinstance(leakage, (int, float)):
            self.leakage = leakage
        else:
            leakage = np.asarray(leakage)
            self.leakage = np.zeros(2 * length)
            self.leakage[:length] = leakage  # nyquist bin is zero!
            self.leakage[length + 1 :] = leakage[:0:-1]  # mirror around nyquist bin

        self.reset()

        if initial_coeff is not None:
            assert len(initial_coeff) == length
            self.W[:] = np.fft.rfft(initial_coeff, n=2 * length)

    @property
    def w(self):
        w = np.fft.irfft(self.W)
        if self.constrained:
            w = w[: self.length]
        return w

    def reset(self):
        self._P = 0
        self.W = np.zeros(self.length + 1, dtype=complex)
        self._xfiltbuff = np.zeros(2 * self.length)
        self._xbuff = np.zeros(2 * self.length)
        self._ebuff = np.zeros(self.length)

    def filt(self, x):
        """Filtering step.

        Parameters
        ----------
        x : (blocklength,) array_like
            Reference signal.

        Returns
        -------
        y : (blocklength,) numpy.ndarray
            Filter output.

        """
        assert len(x) == self.blocklength
        fifo_extend(self._xfiltbuff, x)

        # NOTE: X is computed twice per adaptation cycle if filt and adapt are fed with
        # the same signal. Needed for FxLMS.
        X = np.fft.rfft(self._xfiltbuff)
        y = np.fft.irfft(self.W * X)[-self.blocklength :]
        return y

    def adapt(self, x, e, window=None):
        """Adaptation step.

        If `self.locked == True` perform no adaptation, but fill buffers and estimate
        power.

        Parameters
        ----------
        x : (blocklength,) array_like
            Reference signal.
        e : (blocklength,) array_like
            Error signal.

        """
        assert len(x) == self.blocklength
        assert len(e) == self.blocklength

        fifo_extend(self._xbuff, x)
        fifo_extend(self._ebuff, e)

        X = np.fft.rfft(self._xbuff)
        E = np.fft.rfft(np.concatenate((np.zeros(self.length), self._ebuff)))

        if self.normalized:
            # signal power estimation
            self._P = (
                self.power_averaging * self._P
                + (1 - self.power_averaging) * np.abs(X) ** 2
            )
            # normalization factor
            D = 1 / (self._P + self.epsilon_power)
        else:
            D = 1

        update = D * X.conj() * E

        if self.constrained:
            # note that constraining the frequency dependent step size to be causal
            # can be problematc, see Elliot p. 153
            U = np.fft.irfft(update)
            U[self.length :] = 0  # make it causal
            update = np.fft.rfft(U)

        # filter weight adaptation
        self.W = self.leakage * self.W + self.stepsize * update


class MultiChannelBlockLMS(AdaptiveFilter):
    """A multi-channel block-wise LMS adaptive filter."""

    def __init__(
        self,
        Nout=1,
        Nin=1,
        Nsens=1,
        length=32,
        blocklength=32,
        stepsize=0.1,
        leakage=1,
        initial_coeff=None,
        constrained=True,
        normalized="sum_errors",
        power_averaging=0.5,
        epsilon_power=1e-5,
    ):
        """Create multi-channel block-wise LMS adaptive filter object."""
        assert length >= blocklength, "`length` must larger or equal `blocklength`"
        assert length % blocklength == 0, "`length` must be multiple of `blocklength`"

        self.Nin = Nin
        self.Nout = Nout
        self.Nsens = Nsens
        self.blocklength = blocklength
        self.stepsize = stepsize
        self.constrained = constrained
        self.normalized = normalized
        self.leakage = leakage
        self.power_averaging = power_averaging
        self.epsilon_power = epsilon_power

        self._num_saved_blocks = length // blocklength

        if length is None:
            length = blocklength
        self.length = length

        self.reset(filt=True)

        if initial_coeff is not None:
            assert initial_coeff.shape[0] == length
            self.W[:] = np.fft.rfft(initial_coeff, axis=0, n=2 * length)

    def reset(self, filt=False):
        self._P = 0
        self.W = np.zeros(
            ((2 * self.length) // 2 + 1, self.Nout, self.Nin), dtype=complex
        )
        self._xbuff = np.zeros(
            (
                2 * self.length,
                self.Nsens,
                self.Nout,
                self.Nin,
            )
        )
        self._xfiltbuff = np.zeros((2 * self.length, self.Nin))
        self._ebuff = np.zeros((self.length, self.Nsens))

        if filt:
            self._zifilt = 0

    @property
    def w(self):
        w = np.fft.irfft(self.W, axis=0)
        if self.constrained:
            w = w[: self.length]
        return w

    def filt_time_fast(self, x):
        """Filter reference signal in time domain.

        This is slightly different to `MultiChannelBlockLMS.filt` and
        `MultiChannelBlockLMS.filt`: the convolution of the last block is computed
        with the old filter. Might be faster for some filter dimensions.

        Parameters
        ----------
        x : (blocklength, Nin) array_like
            Reference signal.

        Returns
        -------
        y : (blocklength, Nout) numpy.ndarray
            Filter output.

        """
        x = atleast_2d(x)
        assert x.shape[0] == self.blocklength
        assert x.shape[1] == self.Nin

        # NOTE: filtering could also be done in FD. When is each one better?
        # NOTE: give olafilt the FFT of w?
        y, self._zifilt = olafilt(self.w, x, "nmk,nk->nm", zi=self._zifilt)

        return y

    def filt_time(self, x):
        """Filter reference signal in time domain.

        Parameters
        ----------
        x : (blocklength, Nin) array_like
            Reference signal.

        Returns
        -------
        y : (blocklength, Nout) numpy.ndarray
            Filter output.

        """
        x = atleast_2d(x)
        assert x.shape[0] == self.blocklength
        assert x.shape[1] == self.Nin
        fifo_extend(self._xfiltbuff, x)

        # NOTE: filtering could also be done in FD. When is each one better?
        # NOTE: give olafilt the FFT of w?
        y, _ = olafilt(self.w, self._xfiltbuff, "nmk,nk->nm", zi=self._zifilt)

        return y[-self.blocklength :]

    def filt(self, x):
        """Filter reference signal in frequency domain.

        Parameters
        ----------
        x : (blocklength, Nin) array_like
            Reference signal.

        Returns
        -------
        y : (blocklength, Nout) numpy.ndarray
            Filter output.

        """
        x = atleast_2d(x)
        assert x.shape[0] == self.blocklength
        assert x.shape[1] == self.Nin
        fifo_extend(self._xfiltbuff, x)

        X = np.fft.rfft(self._xfiltbuff, axis=0)
        y = np.fft.irfft(np.einsum("nmk,nk->nm", self.W, X), axis=0)
        return y[-self.blocklength :]

    def adapt(self, x, e):
        """Adaptation step.

        If `self.locked == True` perform no adaptation, but fill buffers and estimate
        power.

        Parameters
        ----------
        x : (blocklength, Nsens, Nout, Nin) array_like
            Reference signal.
        e : (blocklength, Nsens) array_like
            Error signal.

        """

        x = atleast_4d(x)
        e = atleast_2d(e)

        assert x.shape == (self.blocklength, self.Nsens, self.Nout, self.Nin)
        assert e.shape == (self.blocklength, self.Nsens)

        fifo_extend(self._xbuff, x)
        fifo_extend(self._ebuff, e)

        X = np.fft.rfft(self._xbuff, axis=0)
        E = np.fft.rfft(
            np.concatenate((np.zeros((self.length, self.Nsens)), self._ebuff)),
            axis=0,
        )

        if self.normalized:
            if self.normalized == "elementwise":
                power = np.abs(X) ** 2
            elif self.normalized == "sum_errors":
                power = np.sum(np.abs(X) ** 2, axis=1, keepdims=True)
            else:
                raise ValueError(f'Unknown normalization "{self.normalized}".')

            self._P = (
                self.power_averaging * self._P + (1 - self.power_averaging) * power
            )
            D = 1 / (self._P + self.epsilon_power)  # normalization factor
        else:
            D = 1

        update = np.einsum("nlmk,nl->nmk", D * X.conj(), E)

        if self.constrained:  # make it causal
            ut = np.fft.irfft(update, axis=0)  # FIXME: pass n to all irffts
            ut[self.length :] = 0
            update = np.fft.rfft(ut, axis=0)

        self.W = self.leakage * self.W + self.stepsize * update  # update filter


class RLSFilter(LMSFilter):
    """A recursive Least-Squares filter.

    After the fading-memory filter of Simons, p. 210
    """

    def __init__(
        self,
        length,
        alpha=1,
        initial_covariance=10000,
        initial_coeff=None,
        meas_noise_var=0,
        proc_noise_var=0,
    ):
        """Create recursive Least-Squares filter object.

        Parameters
        ----------
        length : int > 0
            Number of filter taps.
        alpha : float >= 1, optional
            Forgetting factor.
        initial_covariance : float > 0 or np.ndarray [shape=(length)], optional
            Initial covariance of the filter coefficients `w`. If float, initialize as
            diagonal matrix.
        initial_coeff : np.ndarray [shape=(length)], optional
            Initial filter coefficients. If `None`, initialize with zeros.
        meas_noise_var : float >= 0, optinal
            Measurement noise variance.
        proc_noise_var : float >= 0, optinal
            Process noise variance

        """
        assert initial_covariance > 0
        assert meas_noise_var >= 0
        assert proc_noise_var >= 0
        assert alpha >= 1
        self.length = length
        self.alpha = alpha
        self.initial_covariance = initial_covariance
        self.meas_noise_var = meas_noise_var
        self.proc_noise_var = proc_noise_var
        self.blocklength = 1
        self.reset()

        if initial_coeff is not None:
            self.w[:] = initial_coeff

    def reset(self):
        self.w = np.zeros(self.length)
        self._xfiltbuff = np.zeros(self.length)
        self._xbuff = np.zeros(self.length)
        if isinstance(self.initial_covariance, (int, float)):
            self.P = np.eye(self.length) * self.initial_covariance
        elif self.initial_covariance.size == self.length:
            self.P = np.diag(self.initial_covariance)
        else:
            raise ValueError("Invalid value for initial_covariance.")

    def adapt(self, x, e):
        """Adaptation step.

        Parameters
        ----------
        x : complex
            Reference signal.
        e : complex
            Error signal, i.e. difference of desired signal and filter output (`d - y`).

        """
        eps = 1e-8

        fifo_append_left(self._xbuff, x)
        u = self._xbuff[:, None]

        self.P = self.alpha**2 * self.P + self.proc_noise_var * np.eye(self.length)
        Pu = self.P @ u
        k = Pu / (u.conj().T @ Pu + self.meas_noise_var + eps)
        self.w += k[:, 0] * e.conj()
        self.P -= k @ Pu.conj().T
