import numpy as np


def blockwise_input_form(arr, blocksize):
    arr = np.concatenate((np.zeros(blocksize - 1), arr))
    out = np.lib.stride_tricks.as_strided(
        arr,
        shape=(arr.shape[0] - blocksize + 1, blocksize),
        strides=(arr.strides[0], arr.strides[0]),
        writeable=False,
    )
    return np.fliplr(out)


def extend_to_length(x, n):
    return np.concatenate((x, np.zeros(n - len(x))))


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
        x = blockwise_input_form(x, self.L)
        N = x.shape[0]
        w = np.zeros((N, self.L))  # filter history
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
        w = np.zeros((N, self.L))  # filter history
        e = np.zeros(N)  # error signal
        y = np.zeros(N)  # filtered output signal
        u = np.zeros(N)  # control signal at error mic

        x = blockwise_input_form(x, self.L)
        fx = blockwise_input_form(fx, self.L)

        for n in range(N):
            y[n] = self.filt(x[n])
            yblocks = blockwise_input_form(
                y, len(sec_path_coeff)
            )  # TODO: use simple indexing
            u[n] = np.dot(sec_path_coeff, yblocks[n])
            e[n] = d[n] - u[n]
            w[n] = self.adapt(fx[n], e[n])

        return y, u, e, w


class LMSFilter(AdaptiveFilter):
    def __init__(self, order, mu=0.1, leak=0, w_init=None, normalized=True):
        assert 0 <= leak and leak < 1 / mu

        self.L = order
        self.mu = mu
        self.leak = leak
        self.w = np.zeros(order)
        self.normalized = normalized
        if w_init is not None:
            self.w[:] = w_init

    def filt(self, x):
        y = np.dot(self.w, x)
        return y

    def adapt(self, x, e):
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

        N = x.shape[0]
        w = np.zeros((N, 2 * self.blocksize))  # filter history
        e = np.zeros((N, self.blocksize))  # error signal
        y = np.zeros((N, self.blocksize))  # filtered output signal

        for n in range(N):
            y[n] = self.filt(x[n])
            e[n] = d[n] - y[n]
            W = self.adapt(x[n], e[n])
            w[n] = np.real(np.fft.ifft(W))

        return y.flatten(), e.flatten(), w


class FastBlockLMSFilter(BlockAdaptiveFilter):
    """Fast Block LMS filter based on overlap-save sectioning."""

    def __init__(self, blocksize, mu=0.1, forget=0.1, leak=0, eps=1e-5):
        self.W = np.zeros(2 * blocksize, dtype=complex)
        self.mu = mu
        self.last_x = np.zeros(blocksize)
        self.forget = forget
        self.P = 0
        self.eps = eps
        self.blocksize = blocksize
        self.leak = 0

    def filt(self, x):
        assert len(x) == self.blocksize

        self.X = np.fft.fft(np.concatenate((self.last_x, x)))
        self.last_x = x
        y = np.real(np.fft.ifft(np.diag(self.X) @ self.W)[self.blocksize :])
        return y

    def adapt(self, x, e):
        assert len(x) == self.blocksize
        assert len(e) == self.blocksize

        # signal power estimation
        self.P = self.forget * self.P + (1 - self.forget) * np.abs(self.X) ** 2
        D = 1 / (self.P + self.eps)

        # tap weight adaptation
        E = np.fft.fft(np.concatenate((np.zeros(self.blocksize), e)))
        Phi = np.fft.ifft(D * self.X.conj() * E)[: self.blocksize]
        self.W = (1 - self.mu * self.leak) * self.W + self.mu * np.fft.fft(
            np.concatenate((Phi, np.zeros(self.blocksize)))
        )
        return self.W
