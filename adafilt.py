"""
TODO: use olafilt From https://github.com/jthiem/overlapadd instead lfilter.
"""

import numpy as np
from scipy.signal import lfilter
from collections import deque


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

    def run(self, x, d):
        """
        Parameters
        ----------

        x : array (N,)
            Reference signal
        d : array (M,)
            Desired signal, M>=N

        """
        x = blockwise_input_form(x, self.filtsize)
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


class LMSFilter(AdaptiveFilter):
    def __init__(self, filtsize, mu=0.1, leak=0, w_init=None, normalized=True):
        assert 0 <= leak and leak < 1 / mu

        self.filtsize = filtsize
        self.mu = mu
        self.leak = leak
        self.normalized = normalized
        self.lock = False
        self.w = np.zeros(filtsize)
        if w_init is not None:
            self.w[:] = w_init

    def filt(self, x):
        x = np.asarray(x)
        assert x.ndim == 1
        assert x.size == self.filtsize

        y = np.dot(self.w, x)
        return y

    def adapt(self, x, e):
        x = np.asarray(x)
        assert x.ndim == 1
        assert x.size == self.filtsize

        if self.lock:
            return self.w

        if self.normalized:
            mu = self.mu / (np.dot(x, x) + 1e-5)
        else:
            mu = self.mu
        self.w = (1 - mu * self.leak) * self.w + mu * x * np.conj(e)
        return self.w


class BlockAdaptiveFilter(AdaptiveFilter):
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


class FastBlockLMSFilter(BlockAdaptiveFilter):
    """Fast Block LMS filter based on overlap-save sectioning."""

    def __init__(
        self,
        blocksize,
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
        self.last_x = np.zeros(blocksize)
        self.eps = eps
        self.W = np.zeros(2 * blocksize, dtype=complex)
        if w_init:
            self.W[:] = np.fft.fft(w_init)

    def filt(self, x):
        # TODO: only compute once per filt adapt cycle
        self.X = np.fft.fft(np.concatenate((self.last_x, x)))
        self.last_x = x
        y = np.real(np.fft.ifft(self.X * self.W)[self.blocksize :])
        return y

    def filt(self, x):
        # TODO: only compute once per filt adapt cycle
        self.X = np.fft.fft(np.concatenate((self.last_x, x)))
        self.last_x = x
        y = np.real(np.fft.ifft(self.X * self.W)[self.blocksize :])
        return y

    def adapt(self, x, e):
        assert len(x) == self.blocksize
        assert len(e) == self.blocksize

        # TODO: only compute once per filt adapt cycle
        self.X = np.fft.fft(np.concatenate((self.last_x, x)))

        # signal power estimation
        self.P = self.forget * self.P + (1 - self.forget) * np.abs(self.X) ** 2
        D = 1 / (self.P + self.eps)

        # tap weight adaptation
        E = np.fft.fft(np.concatenate((np.zeros(self.blocksize), e)))
        self.W *= 1 - self.mu * self.leak
        if self.constrained:
            Phi = np.fft.ifft(D * self.X.conj() * E)[: self.blocksize]
            self.W += self.mu * np.fft.fft(
                np.concatenate((Phi, np.zeros(self.blocksize)))
            )
        else:
            self.W += self.mu * D * self.X.conj() * E

        return self.W
