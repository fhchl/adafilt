from itertools import cycle
import numpy as np
from adafilt.utils import lfilter


class FakeInterface:
    """A fake audio interface."""

    def __init__(
        self, blocklength, signal, h_pri=[1], h_sec=[1], noise=None, y_init=None
    ):
        self.blocklength = blocklength
        self.orig_signal = signal
        self.signal = cycle(signal.reshape(-1, blocklength))
        if noise is None:
            noise = np.zeros(blocklength)
        self.noise = cycle(noise.reshape(-1, blocklength))
        self.h_pri = h_pri
        self.h_sec = h_sec
        self.zi_pri = np.zeros(len(h_pri) - 1)
        self.zi_sec = np.zeros(len(h_sec) - 1)

    def rec(self):
        return self.playrec(np.zeros(self.blocklength))

    def playrec(self, y, send_signal=True):
        y = np.atleast_1d(y)

        if send_signal:
            x = np.atleast_1d(next(self.signal))  # reference signal
        else:
            x = np.zeros(self.blocklength)

        d, self.zi_pri = lfilter(
            self.h_pri, 1, x, zi=self.zi_pri
        )  # primary path signal at error mic
        u, self.zi_sec = lfilter(
            self.h_sec, 1, y, zi=self.zi_sec
        )  # secondary path signal at error mic
        d += next(self.noise)

        # NOTE: should this be plus?
        e = d - u  # error signal

        return x, e, u, d

    def reset(self):
        """Reset filter to initial condition."""
        self.signal = cycle(self.orig_signal.reshape(-1, self.blocklength))
        self.zi_pri = np.zeros(len(self.h_pri) - 1)
        self.zi_sec = np.zeros(len(self.h_sec) - 1)
