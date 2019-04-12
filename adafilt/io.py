"""Summary
"""
from itertools import cycle
import numpy as np
from adafilt.utils import lfilter


class FakeInterface:
    """A fake signal interface."""

    def __init__(self, blocklength, signal, h_pri=[1], h_sec=[1], noise=None):
        """Create a fake block based signal interface.

        Parameters
        ----------
        blocklength : int
            Number of samples in a block.
        signal : array_like
            The disturbance signal.
        h_pri : array_like, optional
            Primary path impulse response.
        h_sec : list, optional
            Secondary path impulse response.
        noise : None, optional
            Description
        """
        self.blocklength = blocklength
        self.orig_signal = signal
        self.signal = cycle(signal.reshape(-1, blocklength))
        if noise is None:
            noise = np.zeros(blocklength)
        self.orig_noise = noise
        self.noise = cycle(noise.reshape(-1, blocklength))
        self.h_pri = h_pri
        self.h_sec = h_sec
        self._zi_pri = np.zeros(len(h_pri) - 1)
        self._zi_sec = np.zeros(len(h_sec) - 1)

    def rec(self):
        """Record one block of the disturbance after the primary path.

        Returns
        -------
        e : (blocklength,) ndarray
            Recorded (error) signal.
        """
        return self.playrec(np.zeros(self.blocklength))

    def playrec(self, y, send_signal=True):
        """Simulatenously play through secondary path while recording the result.

        Parameters
        ----------
        y : (blocklength,) ndarray
            Control signal.
        send_signal : bool, optional
            If `False`, turn of the disturbance. Can be used to 'measure' the secondary
            path.

        Returns
        -------
        e : (blocklength,) ndarray
            Recorded (error) signal.
        """
        y = np.atleast_1d(y)

        if send_signal:
            x = np.atleast_1d(next(self.signal))  # reference signal
        else:
            x = np.zeros(self.blocklength)

        d, self._zi_pri = lfilter(
            self.h_pri, 1, x, zi=self._zi_pri
        )  # primary path signal at error mic
        u, self._zi_sec = lfilter(
            self.h_sec, 1, y, zi=self._zi_sec
        )  # secondary path signal at error mic
        d += next(self.noise)

        # NOTE: should this be plus?
        e = d - u  # error signal

        return x, e, u, d

    def reset(self):
        """Reset interface to initial condition."""
        self.signal = cycle(self.orig_signal.reshape(-1, self.blocklength))
        self.noise = cycle(self.orig_noise.reshape(-1, self.blocklength))
        self._zi_pri = np.zeros(len(self.h_pri) - 1)
        self._zi_sec = np.zeros(len(self.h_sec) - 1)
