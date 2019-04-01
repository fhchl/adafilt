"""
TODO: use olafilt From https://github.com/jthiem/overlapadd instead lfilter.
"""
from collections import deque
from itertools import cycle

import numpy as np
from scipy.signal import lfilter


class FakeInterface:
    """A fake audio interface."""

    def __init__(
        self, buffsize, signal, h_pri=[1], h_sec=[1], noise=lambda x: x, y_init=None
    ):
        self.buffsize = buffsize
        self.orig_signal = signal
        self.signal = cycle(signal.reshape(-1, buffsize))
        self.noise = noise
        self.h_pri = h_pri
        self.h_sec = h_sec
        self.zi_pri = np.zeros(len(h_pri) - 1)
        self.zi_sec = np.zeros(len(h_sec) - 1)

    def rec(self):
        return self.playrec(np.zeros(self.buffsize))

    def playrec(self, y, send_signal=True):
        y = np.atleast_1d(y)

        if send_signal:
            x = np.atleast_1d(next(self.signal))  # reference signal
        else:
            x = np.zeros(self.buffsize)

        d, self.zi_pri = lfilter(
            self.h_pri, 1, x, zi=self.zi_pri
        )  # primary path signal at error mic
        u, self.zi_sec = lfilter(
            self.h_sec, 1, y, zi=self.zi_sec
        )  # secondary path signal at error mic
        d = self.noise(d)

        # NOTE: should this be plus?
        e = d - u  # error signal

        return x, e, u, d

    def reset(self):
        """Reset filter to initial condition."""
        self.signal = cycle(self.orig_signal.reshape(-1, self.buffsize))
        self.zi_pri = np.zeros(len(self.h_pri) - 1)
        self.zi_sec = np.zeros(len(self.h_sec) - 1)


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
            fx = lfilter(sec_path_est, 1, x)  # filtered reference signal
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
                u[n], zi = lfilter(sec_path_coeff, 1, y[n], zi=zi)
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
        leakage=0,
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
        """
        assert 0 <= leakage and leakage < 1 / stepsize
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

        self.w = (1 - stepsize * self.leakage) * self.w + stepsize * xvec * np.conj(e)

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
        leakage=0,
        power_averaging=0.5,
        initial_coeff=None,
        initial_power=0,
        minimum_power=1e-5,
        constrained=True,
    ):
        """Create fast, block-wise LMS adaptive filter object.

        Parameters
        ----------
        length : int, optional
            Length of filter coefficient vector.
        blocklength : int, optional
            Number of samples in one block.
        stepsize : float, optional
            Adaptation step size.
        leakage : float, optional
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
            Description

        Deleted Parameters
        ------------------
        normalized : bool, optional
            If `True` take normalize step size with signal power.
        """
        self.blocklength = blocklength
        self.stepsize = stepsize
        self.power_averaging = power_averaging
        self.constrained = constrained
        self.leakage = 0
        self.locked = False

        self.minimum_power = minimum_power
        self.initial_power = initial_power
        if initial_coeff:
            self.W[:] = np.fft.fft(initial_coeff)

        if length is None:
            length = blocklength
        self.length = length

        # attributes that reset with reset()
        self.P = initial_power
        self.W = np.zeros(2 * length, dtype=complex)
        self.xfiltbuff = deque(np.zeros(2 * length), maxlen=2 * length)
        self.xadaptbuff = deque(np.zeros(2 * length), maxlen=2 * length)
        self.eadaptbuff = deque(np.zeros(length), maxlen=length)

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

        y = np.real(np.fft.ifft(X * self.W)[self.length :])
        y = y[-self.blocklength :]  # only output the newest block
        return y

    def adapt(self, x, e):
        """Adaptation step.

        If `self.locked == True` perform no adaptation, but fill buffers.

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

        if self.locked:
            return

        X = np.fft.fft(self.xadaptbuff)

        # signal power estimation
        self.P = (
            self.power_averaging * self.P + (1 - self.power_averaging) * np.abs(X) ** 2
        )
        D = 1 / (self.P + self.minimum_power)

        # tap weight adaptation
        E = np.fft.fft(np.concatenate((np.zeros(self.length), self.eadaptbuff)))
        self.W *= 1 - self.stepsize * self.leakage
        if self.constrained:
            Phi = np.fft.ifft(D * X.conj() * E)[: self.length]
            self.W += self.stepsize * np.fft.fft(
                np.concatenate((Phi, np.zeros(self.length)))
            )
        else:
            self.W += self.stepsize * D * X.conj() * E
