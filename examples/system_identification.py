"""A filtered-reference Least-Mean-Square (FxLMS) filter."""

import numpy as np
import matplotlib.pyplot as plt

from adafilt import FastBlockLMSFilter
from adafilt.io import FakeInterface

length = 512  # number of adaptive FIR filter taps
blocklength = 128  # length of I/O buffer and blocksize of filter
n_buffers = 150  # size of simulation

# plant
h = np.zeros(512)
h[255] = 1

# white noise signal
signal = np.random.normal(0, 1, size=n_buffers * blocklength)

# the adaptive filter
filt = FastBlockLMSFilter(length, blocklength, power_averaging=0.9)

# simulates an audio interface with primary and secondary paths and 40 dB SNR noise
# at the error sensor
sim = FakeInterface(blocklength, h_sec=h)

# aggregate signals during simulation
elog = []
felog = []
wslog = []
ylog = []

for i in range(n_buffers):

    # identification noise
    u = np.random.normal(0, 1, blocklength)

    # record error signal e
    _, d, _, _ = sim.playrec(u)

    # filter prediction
    y = filt.filt(u)

    # error signal
    e = y - d

    # weight adaptation
    filt.adapt(u, e)

    # relative filter error
    fe = np.sum((h - filt.w)**2) / np.sum((h)**2)

    elog.append(e)
    felog.append(fe)
    ylog.append(u)
    wslog.append(filt.w)


fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(14, 8), constrained_layout=True)

ax[0, 0].set_title("Signals")
ax[0, 0].plot(np.concatenate(ylog), label="y", alpha=0.8)
ax[0, 0].plot(np.concatenate(elog), label="e", alpha=0.7)
ax[0, 0].set_xlabel("Sample")
ax[0, 0].legend()

ax[0, 1].set_title("Filter weights")
ax[0, 1].plot(wslog)
ax[0, 1].set_xlabel("Block")


ax[1, 0].set_title("Error")
ax[1, 0].plot(10 * np.log10(felog))
ax[1, 0].set_xlabel("Sample")
ax[1, 0].set_ylabel("Error [dB]")


ax[1, 1].set_title("Final filter")
ax[1, 1].plot(filt.w, label="estimate")
ax[1, 1].plot(h, label="truth")
ax[1, 1].set_xlabel("Tap")
ax[1, 1].legend()

plt.show()
