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

    def adapt(self, x, d):
        raise NotImplementedError

    def predict(self, x):
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
            y[n] = self.predict(x[n])
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
        #fx = extend_to_length(fx, len(d))

        N = x.shape[0]
        w = np.zeros((N, self.L))  # filter history
        e = np.zeros(N)  # error signal
        y = np.zeros(N)  # filtered output signal
        u = np.zeros(N)  # control signal at error mic

        x = blockwise_input_form(x, self.L)
        fx = blockwise_input_form(fx, self.L)

        for n in range(N):
            y[n] = self.predict(x[n])
            yblocks = blockwise_input_form(y, len(sec_path_coeff))  # TODO: use simple indexing
            u[n] = np.dot(sec_path_coeff, yblocks[n])
            e[n] = d[n] - u[n]
            w[n] = self.adapt(fx[n], e[n])

        return y, u, e, w


class LMSFilter(AdaptiveFilter):
    def __init__(self, order, mu=0.1, leak=0, w=None):
        assert 0 <= leak and leak < 1 / mu

        self.L = order
        self.mu = mu
        self.leak = leak
        self.w = np.zeros(order)
        if w is not None:
            self.w[:] = w

    def predict(self, x):
        y = np.dot(self.w, x)
        return y

    def adapt(self, x, e):
        self.w = (1 - self.mu * self.leak) * self.w + self.mu * x * np.conj(e)
        return self.w


class NLMSFilter(LMSFilter):
    def adapt(self, x, e):
        mu_norm = self.mu / (np.dot(x, x) + 1e-5)
        self.w = (1 - mu_norm * self.leak) * self.w + mu_norm * x * np.conj(e)
        return self.w
