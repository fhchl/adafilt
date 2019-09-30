"""Adaptive filtering classes."""

import numpy as np
from collections import deque

from adafilt.utils import olafilt, atleast_2d


class SimpleFilter:
    """A overlap-and-add Filter."""

    def __init__(self, w, sum_inputs=True, squeeze=True):
        """Create overlap-add filter object.

        Parameters
        ----------
        w : array_like
            Filter taps.

        """
        w = np.asarray(w)
        self.w = w
        self.sum_inputs = sum_inputs
        self.squeeze = squeeze

        if sum_inputs:
            self.zi = np.zeros((w.shape[0] - 1, 1))
        else:
            self.zi = np.zeros((w.shape[0] - 1, 1, 1, 1))

    def __call__(self, x):
        """Filter signal x.

        Parameters
        ----------
        x : array_like
            Input signal

        Returns
        -------
        y : numpy.ndarray
            Output signal. Has same length as `x`.

        """
        y, self.zi = olafilt(
            self.w, x, zi=self.zi, sum_inputs=self.sum_inputs, squeeze=self.squeeze
        )
        return y


class AdaptiveFilter:
    """Base class for adaptive filters."""

    def __init__(self):
        raise NotImplementedError

    def reset(self):
        """Reset filter object."""
        raise NotImplementedError

    def filt(self, x):
        """Filter a block."""
        raise NotImplementedError

    def adapt(self, x, e):
        """Adapt to a block."""
        raise NotImplementedError

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
            zi = np.zeros(len(sec_path_coeff) - 1)

        for n in range(n_blocks):
            w[n] = self.w
            y[n] = self.filt(x[n])

            # control signal at error sensor
            if sec_path_coeff is not None:
                u[n], zi = olafilt(sec_path_coeff, y[n], zi=zi)
            else:
                u[n] = y[n]

            # error signal
            e[n] = d[n] + u[n]

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
        minimum_power=1e-5,
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
        minimum_power : float, optional
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
        self.minimum_power = minimum_power
        self.normalized = normalized
        self.locked = False
        self.w = np.zeros(length)
        if initial_coeff is not None:
            self.w[:] = initial_coeff
        self.xadaptbuff = deque(np.zeros(length), maxlen=length)
        self.xfiltbuff = deque(np.zeros(length), maxlen=length)

    def filt(self, x):
        """Filtering step.

        Parameters
        ----------
        x : float
            Reference signal.

        Returns
        -------
        y : float
            Filter output.
        """
        x = float(x)
        self.xfiltbuff.appendleft(x)
        y = np.dot(self.w, self.xfiltbuff)
        return y

    def adapt(self, x, e):
        """Adaptation step.

        Parameters
        ----------
        x : float
            Reference signal.
        e : float
            Error signal.
        """
        x = float(x)
        e = float(e)

        self.xadaptbuff.appendleft(x)

        if self.locked:
            return

        xvec = np.array(self.xadaptbuff)

        if self.normalized:
            stepsize = self.stepsize / (np.dot(xvec, xvec) + self.minimum_power)
        else:
            stepsize = self.stepsize

        self.w = self.leakage * self.w - stepsize * xvec * np.conj(e)

    def reset(self):
        self.w = np.zeros(self.length)
        self.xfiltbuff = deque(np.zeros(self.length), maxlen=self.length)
        self.xadaptbuff = deque(np.zeros(self.length), maxlen=self.length)


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
        initial_power=0,
        minimum_power=1e-5,
        constrained=True,
        normalized=True,
    ):
        """Create fast, block-wise normalized LMS adaptive filter object.

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
        minimum_power : float, optional
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

        TODO:
        - Gain or power constraints on filters
          (rafaelyComputationallyEfficientFrequencydomain2000)
        - Unbiased normalized algorithm
          (elliottFrequencydomainAdaptationCausal2000)
        """
        assert length >= blocklength, "Filter must be at least as long as block"

        self.blocklength = blocklength
        self.stepsize = stepsize
        self.power_averaging = power_averaging
        self.constrained = constrained
        self.normalized = normalized
        self.locked = False
        self.minimum_power = minimum_power
        self.initial_power = initial_power

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

        # attributes that reset with reset()
        self.P = initial_power
        self.W = np.zeros(2 * length, dtype=complex)
        self.xfiltbuff = deque(np.zeros(2 * length), maxlen=2 * length)
        self.xadaptbuff = deque(np.zeros(2 * length), maxlen=2 * length)
        self.eadaptbuff = deque(np.zeros(length), maxlen=length)

        if initial_coeff is not None:
            w = np.concatenate((initial_coeff, np.zeros(length)))
            self.W[:] = np.fft.fft(w)

    @property
    def w(self):
        return np.real(np.fft.ifft(self.W)[: self.length])

    def reset(self):
        self.P = self.inital_power
        self.W = np.zeros(2 * self.length, dtype=complex)
        self.xfiltbuff = deque(np.zeros(2 * self.length), maxlen=2 * self.length)
        self.xadaptbuff = deque(np.zeros(2 * self.length), maxlen=2 * self.length)
        self.eadaptbuff = deque(np.zeros(self.length), maxlen=self.length)

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
        self.xfiltbuff.extend(x)

        # NOTE: X is computed twice per adaptation cycle if filt and adapt are fed with
        # the same signal. Needed for FxLMS.
        X = np.fft.fft(self.xfiltbuff)

        y = np.real(np.fft.ifft(X * self.W)[-self.blocklength :])
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

        self.xadaptbuff.extend(x)
        self.eadaptbuff.extend(e)

        # reference signal in frequency domain
        X = np.fft.fft(self.xadaptbuff)

        if self.normalized:
            # signal power estimation
            self.P = (
                self.power_averaging * self.P
                + (1 - self.power_averaging) * np.abs(X) ** 2
            )
            # normalization factor
            D = 1 / (self.P + self.minimum_power)
        else:
            D = 1

        if self.locked:
            return

        # error signal in frequency domain
        E = np.fft.fft(np.concatenate((np.zeros(self.length), self.eadaptbuff)))

        update = D * X.conj() * E

        if self.constrained:
            # note that constraining the frequency dependent step size to be causal
            # can be problematc, see Elliot p. 153
            U = np.fft.ifft(update)
            U[self.length :] = 0  # make it causal
            update = np.fft.fft(U)

        # filter weight adaptation
        self.W = self.leakage * self.W - self.stepsize * update

        if window is not None:
            w = np.fft.ifft(self.W)
            w[: self.length] *= window
            self.W = np.fft.fft(w)


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
        power_averaging=0.5,
        initial_coeff=None,
        constrained=True,
    ):
        """Create multi-channel block-wise LMS adaptive filter object."""
        assert length >= blocklength, "Filter must be at least as long as block"
        assert length % blocklength == 0

        self.Nin, self.Nout, self.Nsens = Nin, Nout, Nsens
        self.blocklength = blocklength
        self.stepsize = stepsize
        self.num_saved_blocks = length // blocklength
        self.power_averaging = power_averaging
        self.constrained = constrained
        self.locked = False

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

        # attributes that reset with reset()
        self.P = initial_power
        self.W = np.zeros((nout, nref, 2 * length), dtype=complex)  # shape (nout, 2 * length)
        self.xfiltbuff = deque(np.zeros(2 * length), maxlen=2 * length)  # shape (nout, 2 * length)
        self.xadaptbuff = deque(np.zeros(2 * length), maxlen=2 * length)
        self.eadaptbuff = deque(np.zeros(length), maxlen=length)

        if initial_coeff is not None:
            w = np.concatenate((initial_coeff, np.zeros(length)))
            self.W[:] = np.fft.fft(w)

    def filt(self, x):
        """Filtering step.

        Parameters
        ----------
        x : (blocklength, nref) array_like
            Reference signal.

        Returns
        -------
        y : (blocklength, nour) numpy.ndarray
            Filter output.
        """
        assert x.shape[0] == self.blocklength
        assert x.shape[1] == self.nref

        self.xfiltbuff.extend(x)

        # NOTE: X is computed twice per adaptation cycle if filt and adapt are fed with
        # the same signal. Needed for FxLMS.
        X = np.fft.fft(self.xfiltbuff, axis=0)  # shape ()

        y = np.real(np.fft.ifft(self.W @ X)[-self.blocklength :])
        return y
