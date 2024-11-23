"""Input and output classes."""

from itertools import cycle

import numpy as np

from adafilt import olafilt


class FakeInterface:
    """A fake signal interface."""

    def __init__(
        self, blocklength, signal=None, h_pri=[1], h_sec=[1], noise=None, cycle_h=False
    ):
        """Create a fake block based signal interface.

        Parameters
        ----------
        blocklength : int
            Number of samples in a block.
        signal : array_like or None, optional
            The disturbance signal of shape (n,[ K]).
        h_pri : array_like, optional
            Primary path impulse responses of shape (m, [L, [K]])
        h_sec : list, optional
            Secondary path impulse response of shape (l, [L, [M]]).
        noise : int, array_like or None, optional
            Noise at microphones in dB. If array_like of length (n,[ L])
        cycle : bool, optional
            Cycle through h_pri and h_sec with each iteration to simulate time
            chainging paths. Then h_pri and h_sec need to have an additional first block
            axis.

        """
        h_pri = np.asarray(h_pri)
        h_sec = np.asarray(h_sec)

        if signal is None:
            signal = np.zeros((blocklength, *h_pri.shape[1:2]))
        else:
            signal = np.asarray(signal)

        self.blocklength = blocklength
        self._orig_signal = signal
        self._orig_noise = noise
        self.nblocks = signal.shape[0] // blocklength

        # check correct h shapes
        if not cycle_h:
            # cycle through one element
            h_pri = h_pri[None]
            h_sec = h_sec[None]
        assert (h_pri.ndim > 2) == (h_sec.ndim > 2)

        # check correct signal shape
        assert signal.ndim in [1, 2]
        assert (signal.ndim == 2 and h_pri.ndim == 4) or (
            signal.ndim == 1 and h_pri.ndim in [2, 3]
        ), (
            "Incompatible signal (n,[ K]) and h_pri (m, [L, [K]]) shapes:"
            + f"{signal.shape}, {h_pri.shape}"
        )

        # check correct noise shape
        if noise is not None:
            noise = np.asarray(noise)
            assert noise.ndim in [1, 2]
            assert (noise.ndim == 2 and h_pri.ndim == 4) or (
                noise.ndim == 1 and h_pri.ndim in [2, 3]
            ), "Incompatible noise and h_pri shapes"
        else:
            self.noise = None

        self._orig_h_pri = h_pri
        self._orig_h_sec = h_sec

        self.reset()

    def rec(self):
        """Record one block of the disturbance after the primary path.

        Returns
        -------
        e : (blocklength,) ndarray
            Recorded (error) signal.

        """
        return self.playrec(np.zeros((self.blocklength, *self._orig_h_pri.shape[1:2])))

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
        x, e, u, d : (blocklength,) ndarray
            Reference signal, error signal, control signal at error microphone, primary
            at error microphone.

        """
        if y is None:
            y = np.zeros((self.blocklength, *self._orig_h_sec.shape[3:4]))
        else:
            y = np.atleast_1d(y)

        if send_signal:
            x = np.atleast_1d(next(self.signal))  # reference signal
        else:
            x = np.zeros((self.blocklength, *self._orig_signal.shape[1:2]))

        subscripts_sec = (
            "nlm"[: self._orig_h_sec.ndim - 1] + "," + "nm"[: y.ndim] + "->n"
        )
        subscripts_pri = (
            "nlk"[: self._orig_h_pri.ndim - 1]
            + ","
            + "nk"[: self._orig_signal.ndim]
            + "->n"
        )
        if self._orig_h_sec.ndim - 1 > 1:
            subscripts_sec += "l"
            subscripts_pri += "l"

        # primary path signal at error mic
        d, self._zi_pri = olafilt(next(self.h_pri), x, subscripts_pri, zi=self._zi_pri)
        # secondary path signal at error mic
        u, self._zi_sec = olafilt(next(self.h_sec), y, subscripts_sec, zi=self._zi_sec)

        e = d + u  # error signal

        if self.noise is not None:
            e += next(self.noise)

        return x, e, u, d

    def reset(self):
        """Reset interface to initial condition."""
        self.signal = cycle(
            self._orig_signal.reshape(
                self.nblocks, self.blocklength, *self._orig_signal.shape[1:2]
            )
        )

        self.h_pri = cycle(self._orig_h_pri)
        self.h_sec = cycle(self._orig_h_sec)

        self._zi_pri = 0
        self._zi_sec = 0

        if self._orig_noise is not None:
            self.noise = cycle(
                self._orig_noise.reshape(
                    self.nblocks, self.blocklength, *self._orig_noise.shape[1:2]
                )
            )
