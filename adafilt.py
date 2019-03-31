"""
TODO: use olafilt From https://github.com/jthiem/overlapadd instead lfilter.
TODO: alternative to slow ring buffers?
"""

import numpy as np
from scipy.signal import lfilter
from collections import deque


class FakeInterface:
    """A fake audio interace."""

    def __init__(
        self, buffsize, signal, h_pri=[1], h_sec=[1], noise=lambda x: x, y_init=None
    ):
        self.buffsize = buffsize
        self.orig_signal = signal
        self.signal = iter(signal.reshape(-1, buffsize))
        self.noise = noise
        self.h_pri = h_pri
        self.h_sec = h_sec
        self.zi_pri = np.zeros(len(h_pri) - 1)
        self.zi_sec = np.zeros(len(h_sec) - 1)

    def rec(self):
        return self.playrec(np.zeros(self.buffsize))

    def playrec(self, y, send_signal=True):
        # TODO: make mute signal kwarg of this func
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
        """Reset simulation to inital condition."""
        self.signal = iter(self.orig_signal.reshape(-1, self.buffsize))
        self.zi_pri = np.zeros(len(self.h_pri) - 1)
        self.zi_sec = np.zeros(len(self.h_sec) - 1)


def blockwise_input_form(arr, blocksize):
    arr = np.concatenate((np.zeros(blocksize - 1), arr))
    out = np.lib.stride_tricks.as_strided(
        arr,
        shape=(arr.shape[0] - blocksize + 1, blocksize),
        strides=(arr.strides[0], arr.strides[0]),
        writeable=False,
    )
    return np.fliplr(out)


class AdaptiveFilter:
    def __init__(self):
        raise NotImplementedError

    def adapt(self, x, e):
        raise NotImplementedError

    def filt(self, x):
        raise NotImplementedError

    def reset(self):
        self.w = np.zeros(self.filtsize)
        self.xbuff = deque(np.zeros(self.filtsize), maxlen=self.filtsize)
        self.zifilt = np.zeros(self.filtsize - 1)

    def run(self, x, d):
        """
        Parameters
        ----------

        x : array (N,)
            Reference signal
        d : array (M,)
            Desired signal, M>=N

        """
        x = np.asarray(x)
        d = np.asarray(d)

        N = x.shape[0]
        w = np.zeros((N, self.filtsize))  # filter history
        e = np.zeros(N)  # error signal
        y = np.zeros(N)  # filtered output signal

        for n in range(N):
            y[n] = self.filt(x[n])
            e[n] = d[n] - y[n]
            w[n] = self.adapt(x[n], e[n])

        return y, e, w

    def run_filtered_reference(self, x, d, sec_path_coeff, sec_path_coeff_est):
        """
        Parameters
        ----------

        x : array (N,)
            Reference signal
        d : array (M,)
            Desired signal, M>=N

        """
        # filtered reference signal
        fx = np.convolve(x, sec_path_coeff_est)

        N = x.shape[0]
        w = np.zeros((N + 1, self.filtsize))  # filter history
        e = np.zeros(N)  # error signal
        y = np.zeros(N)  # filtered output signal
        u = np.zeros(N)  # control signal at error mic

        x = blockwise_input_form(x, self.filtsize)
        fx = blockwise_input_form(fx, self.filtsize)

        for n in range(N):
            y[n] = self.filt(x[n])
            yblocks = blockwise_input_form(
                y, len(sec_path_coeff)
            )  # TODO: use simple indexing
            u[n] = np.dot(sec_path_coeff, yblocks[n])
            e[n] = d[n] - u[n]
            w[n + 1] = self.adapt(fx[n], e[n])

        return y, u, e, w

    def adafilt(self, x, e, fx=None):
        """Adaptively filter x according to e and fx.

        Parameters
        ----------
        x : (N,) array_like
            Reference signal.
        e : (N,) array_like
            Error signal.
        fx : (N,) array_like or None, optional
            Filtered reference signal.

        Returns
        -------
        y: (N,) ndarray
            Filter output.
        """
        x = np.asarray(x)
        e = np.asarray(e)
        assert x.ndim == 1
        assert x.shape == e.shape
        if fx is not None:
            fx = np.asarray(fx)
            assert x.shape == fx.shape

        assert len(x) % self.blocksize == 0

        nblocks = int(len(x) / self.blocksize)

        y = np.zeros(len(x))
        M = self.blocksize
        for n in range(nblocks):
            slce = slice(n * M, (n + 1) * M)

            if fx is not None:
                self.adapt(fx[slce], e[slce])
            else:
                self.adapt(x[slce], e[slce])

            # filter
            y[n * M : (n + 1) * M] = self.filt(x[slce])

        return y


class LMSFilter(AdaptiveFilter):
    def __init__(self, filtsize, mu=0.1, leak=0, w_init=None, normalized=True):
        assert 0 <= leak and leak < 1 / mu
        self.blocksize = 1
        self.filtsize = filtsize
        self.mu = mu
        self.leak = leak
        self.normalized = normalized
        self.lock = False
        self.w = np.zeros(filtsize)
        if w_init is not None:
            self.w[:] = w_init

        self.zifilt = np.zeros(filtsize - 1)
        self.xbuff = deque(np.zeros(filtsize), maxlen=filtsize)
        self.ebuff = deque(np.zeros(filtsize), maxlen=filtsize)
        self.fxbuff = deque(np.zeros(filtsize), maxlen=filtsize)

    def filt2(self, x):
        """Filter x.

        Parameters
        ----------
        x : (N,) array_like
            Signal.

        Returns
        -------
        y : (N, array_like)
            Filtered signal.
        """
        x = np.atleast_1d(x)
        y, self.zifilt = lfilter(self.w, 1, x, zi=self.zifilt)
        return y.squeeze()  # return single number if input was single number

    def filt(self, x):
        assert isinstance(x, float)
        self.xbuff.appendleft(x)
        y = np.dot(self.w, self.xbuff)
        return y

    def adapt(self, x, e):
        """Adapt filter coefficients.

        Parameters
        ----------
        x : (blocksize,) array_like
            Reference signal.
        e : (blocksize,) array_like
            Error signal
        """
        x = np.atleast_1d(x)
        e = np.atleast_1d(e)

        assert len(x) == self.blocksize
        assert len(e) == self.blocksize

        if self.lock:
            return

        N = len(x)
        for n in range(N):
            self.xbuff.appendleft(x[n])
            xvec = np.array(self.xbuff)

            if self.normalized:
                mu = self.mu / (np.dot(xvec, xvec) + 1e-5)
            else:
                mu = self.mu

            self.w = (1 - mu * self.leak) * self.w + mu * xvec * np.conj(e[n])


class FastBlockLMSFilter(AdaptiveFilter):
    """Fast Block LMS filter based on overlap-save sectioning."""

    def __init__(
        self,
        blocksize,
        filtsize=None,
        mu=0.1,
        forget=0.1,
        leak=0,
        w_init=None,
        eps=1e-5,
        constrained=True,
    ):
        self.blocksize = blocksize
        self.mu = mu
        self.forget = forget
        self.constrained = constrained
        self.P = 0
        self.leak = 0
        self.eps = eps
        if filtsize is None:
            filtsize = blocksize
        self.W = np.zeros(2 * filtsize, dtype=complex)
        if w_init:
            self.W[:] = np.fft.fft(w_init)
        self.filtsize = filtsize
        self.last_x = np.zeros(filtsize)
        self.xbuff = deque(np.zeros(filtsize), maxlen=filtsize)
        self.fxbuff = deque(np.zeros(filtsize), maxlen=filtsize)

    def reset(self):
        self.W = np.zeros(2 * self.filtsize, dtype=complex)
        super().reset()

    def filt(self, x):
        # TODO: only compute once per filt adapt cycle
        self.X = np.fft.fft(np.concatenate((self.last_x, x)))
        self.last_x = x
        y = np.real(np.fft.ifft(self.X * self.W)[self.filtsize :])
        return y

    def adapt(self, x, e):
        assert len(x) == self.filtsize
        assert len(e) == self.filtsize

        # TODO: only compute once per filt adapt cycle
        self.X = np.fft.fft(np.concatenate((self.last_x, x)))

        # signal power estimation
        self.P = self.forget * self.P + (1 - self.forget) * np.abs(self.X) ** 2
        D = 1 / (self.P + self.eps)

        # tap weight adaptation
        E = np.fft.fft(np.concatenate((np.zeros(self.filtsize), e)))
        self.W *= 1 - self.mu * self.leak
        if self.constrained:
            Phi = np.fft.ifft(D * self.X.conj() * E)[: self.filtsize]
            self.W += self.mu * np.fft.fft(
                np.concatenate((Phi, np.zeros(self.filtsize)))
            )
        else:
            self.W += self.mu * D * self.X.conj() * E

        return self.W

    def run(self, x, d):
        """
        Parameters
        ----------

        x : array (N,)
            Reference signal
        d : array (M,)
            Desired signal, M>=N

        """
        x = np.atleast_1d(x)
        d = np.atleast_1d(d)
        assert x.ndim == 1
        assert x.shape[0] % self.blocksize == 0
        assert x.shape == d.shape

        x = x.reshape((-1, self.blocksize))
        d = d.reshape((-1, self.blocksize))

        n_blocks = x.shape[0]
        w = np.zeros((n_blocks, 2 * self.blocksize))  # filter history
        e = np.zeros((n_blocks, self.blocksize))  # error signal
        y = np.zeros((n_blocks, self.blocksize))  # filtered output signal

        for n in range(n_blocks):
            y[n] = self.filt(x[n])
            e[n] = d[n] - y[n]
            W = self.adapt(x[n], e[n])
            w[n] = np.real(np.fft.ifft(W))

        return y.reshape(-1), e.reshape(-1), w

    def run_filtered_reference(self, x, d, sec_path_coeff, sec_path_coeff_est):
        """
        Parameters
        ----------

        x : array (N,)
            Reference signal
        d : array (M,)
            Desired signal, M>=N

        """
        x = np.atleast_1d(x)
        d = np.atleast_1d(d)
        assert x.ndim == 1
        assert x.shape[0] % self.blocksize == 0
        assert x.shape == d.shape

        fx = lfilter(sec_path_coeff_est, 1, x)  # filtered reference signal

        x = x.reshape((-1, self.blocksize))
        d = d.reshape((-1, self.blocksize))
        fx = fx.reshape((-1, self.blocksize))
        n_blocks = x.shape[0]

        w = np.zeros((n_blocks, 2 * self.blocksize))  # filter history
        e = np.zeros((n_blocks, self.blocksize))  # error signal
        y = np.zeros((n_blocks, self.blocksize))  # filtered output signal
        u = np.zeros((n_blocks, self.blocksize))  # control signal at error mic

        zi = np.zeros(len(sec_path_coeff) - 1)
        for n in range(n_blocks):
            y[n] = self.filt(x[n])
            u[n], zi[:] = lfilter(sec_path_coeff, 1, y[n], zi=zi)
            e[n] = d[n] - u[n]
            W = self.adapt(fx[n], e[n])
            w[n] = np.real(np.fft.ifft(W))

        return y.flatten(), u.flatten(), e.flatten(), w
