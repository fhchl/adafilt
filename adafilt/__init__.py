"""Adaptive filtering classes."""

import numpy as np
from adafilt.utils import olafilt, atleast_2d, atleast_4d, fifo_extend, fifo_append_left


class SimpleFilter:
    """A overlap-and-add Filter."""

    def __init__(self, w, subscripts=None, zi=None):
        """Create overlap-add filter object.

        Parameters
        ----------
        w : array_like
            Filter taps.
        subscripts: str or none, optional
            Defines multi-channel case with `numpy.einsum` notation. See
            `adafilt.utils.olafilt` for details.
        zi : None or array_like, optional
            Initial filter state.

        """
        w = np.asarray(w)
        self.w = w
        self.subscripts = subscripts
        self.zi = zi if zi is not None else 0

    def __call__(self, x):
        """See `SimpleFilter.filt`."""
        # TODO: take out?
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
        y, self.zi = olafilt(self.w, x, subscripts=self.subscripts, zi=self.zi)
        return y


class Delay:
    """A simple delay."""

    def __init__(self, nsamples, zi=None):
        """Create simple delay.

        Parameters
        ----------
        nsamples : int
            Delay by `nsamples` samples.
        zi : array_like or None, optional
            Initial filter condition.

        """
        self.zi = zi
        self.nsamples = nsamples

    def __call__(self, x):
        """See `Delay.filt`."""
        return self.filt(x)

    def filt(self, x):
        """Filter signal.

        Parameters
        ----------
        x : array_like, shape (N,) or (N, M)
            Signal with samples along first dimension

        Returns
        -------
        numpy.ndarray
            The filtered signal of shape (N, ) or (N, M)

        """
        x_orig_shape = x.shape
        x = atleast_2d(x)
        nout, nsig = x.shape

        if self.zi is None:
            # first filtering: fill with zeros
            self.zi = np.zeros((self.nsamples, nsig))

        zx = np.concatenate((self.zi, x), axis=0)
        out = zx[:nout]
        self.zi = zx[nout:]
        return out.reshape(x_orig_shape)


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

        self.reset()

        if initial_coeff is not None:
            self.w[:] = initial_coeff

    def reset(self):
        self.w = np.zeros(self.length)
        self.xfiltbuff = np.zeros(self.length)
        self.xbuff = np.zeros(self.length)

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
        fifo_append_left(self.xfiltbuff, x)
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

        fifo_append_left(self.xbuff, x)

        if self.locked:
            return

        xvec = np.array(self.xbuff)

        if self.normalized:
            stepsize = self.stepsize / (np.dot(xvec, xvec) + self.minimum_power)
        else:
            stepsize = self.stepsize

        self.w = self.leakage * self.w - stepsize * xvec * np.conj(e)


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
        initial_power=1e-2,
        minimum_power=1e-5,
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
        - use rfft if appropriate. Saves 20% time.
        - Gain or power constraints on filters
          (rafaelyComputationallyEfficientFrequencydomain2000)
        - Unbiased normalized algorithm
          (elliottFrequencydomainAdaptationCausal2000)
        - Faster?: Soo, J-S., and Khee K. Pang. "Multidelay block frequency domain
        adaptive filter." IEEE Transactions on Acoustics, Speech, and Signal Processing
        38.2 (1990): 373-376.

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

        self.reset()

        if initial_coeff is not None:
            assert len(initial_coeff) == length
            self.W[:] = np.fft.rfft(initial_coeff, n=2 * length)

        self.P = initial_power

    @property
    def w(self):
        w = np.fft.irfft(self.W)
        if self.constrained:
            w = w[: self.length]
        return w

    def reset(self):
        self.P = 0
        self.W = np.zeros((2 * self.length) // 2 + 1, dtype=complex)
        self.xfiltbuff = np.zeros(2 * self.length)
        self.xbuff = np.zeros(2 * self.length)
        self.ebuff = np.zeros(self.length)

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
        fifo_extend(self.xfiltbuff, x)

        # NOTE: X is computed twice per adaptation cycle if filt and adapt are fed with
        # the same signal. Needed for FxLMS.
        X = np.fft.rfft(self.xfiltbuff)
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

        fifo_extend(self.xbuff, x)
        fifo_extend(self.ebuff, e)

        X = np.fft.rfft(self.xbuff)
        E = np.fft.rfft(np.concatenate((np.zeros(self.length), self.ebuff)))

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

        update = D * X.conj() * E

        if self.constrained:
            # note that constraining the frequency dependent step size to be causal
            # can be problematc, see Elliot p. 153
            U = np.fft.irfft(update)
            U[self.length :] = 0  # make it causal
            update = np.fft.rfft(U)

        # filter weight adaptation
        self.W = self.leakage * self.W - self.stepsize * update


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
    ):
        """Create multi-channel block-wise LMS adaptive filter object."""
        assert length >= blocklength, "Filter must be at least as long as block"
        assert length % blocklength == 0

        self.Nin = Nin
        self.Nout = Nout
        self.Nsens = Nsens
        self.blocklength = blocklength
        self.stepsize = stepsize
        self.num_saved_blocks = length // blocklength
        self.constrained = constrained
        self.locked = False
        self.normalized = False
        self.leakage = leakage

        if length is None:
            length = blocklength
        self.length = length

        self.reset()

        if initial_coeff is not None:
            assert initial_coeff.shape[0] == length
            self.W[:] = np.fft.rfft(initial_coeff, axis=0, n=2 * length)

        self.zifilt = 0

    def reset(self, filt=False):
        self.W = np.zeros(
            ((2 * self.length) // 2 + 1, self.Nout, self.Nin), dtype=complex
        )
        self.xbuff = np.zeros(
            (
                2 * self.num_saved_blocks * self.blocklength,
                self.Nsens,
                self.Nout,
                self.Nin,
            )
        )
        self.ebuff = np.zeros((self.num_saved_blocks * self.blocklength, self.Nsens))

        if filt:
            self.zifilt = 0

    @property
    def w(self):
        w = np.fft.irfft(self.W, axis=0)
        if self.constrained:
            w = w[: self.length]
        return w

    def filt(self, x):
        """Filter reference signal.

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
        y, self.zifilt = olafilt(self.w, x, "nmk,nk->nm", zi=self.zifilt)
        return y

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

        fifo_extend(self.xbuff, x)
        fifo_extend(self.ebuff, e)

        X = np.fft.rfft(self.xbuff, axis=0)
        E = np.fft.rfft(
            np.concatenate(
                (np.zeros((self.length, self.Nsens)), self.ebuff)
            ),
            axis=0,
        )

        if self.normalized:
            # TODO: implement
            D = 1
        else:
            D = 1

        if self.locked:
            return

        update = D * np.einsum("nlmk,nl->nmk", X.conj(), E)

        if self.constrained:
            # make it causal
            ut = np.fft.irfft(update, axis=0)
            ut[self.length :] = 0
            update = np.fft.rfft(ut, axis=0)

        # update filter
        self.W = self.leakage * self.W - self.stepsize * update
