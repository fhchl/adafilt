"""A modified filtered-reference Least-Mean-Square filter."""

import numpy as np
import matplotlib.pyplot as plt

from adafilt import FastBlockLMSFilter, SimpleFilter
from adafilt.io import FakeInterface
from adafilt.utils import olafilt, wgn

length = 512  # number of adaptive FIR filter taps
blocklength = 128  # length of I/O buffer and blocksize of filter
n_buffers = 150  # size of simulation

# primary and secondary paths
h_pri = np.zeros(1024)
h_pri[-1] = 1
h_sec = np.zeros(512)
h_sec[-1] = 1

# white noise signal
signal = np.random.normal(0, 1, size=n_buffers * blocklength)

# the adaptive filter
filt = FastBlockLMSFilter(length, blocklength)

# the dummy adaptive filter
filt_dummy = FastBlockLMSFilter(
    length, blocklength, stepsize=0.1, leakage=0.99999, power_averaging=0.9
)

# secondary path estimate has to account for block size
plant_model_fx = SimpleFilter(np.concatenate((np.zeros(blocklength), h_sec)))
plant_model_fy = SimpleFilter(np.concatenate((np.zeros(blocklength), h_sec)))

# simulates an audio interface with primary and secondary paths and 40 dB SNR noise
# at the error sensor
sim = FakeInterface(
    blocklength,
    signal,
    h_pri=h_pri,
    h_sec=h_sec,
    noise=wgn(olafilt(h_pri, signal), 20, "dB"),
)


# aggregate signals during simulation
xlog = []
elog = []
wslog = []
ylog = []

y = np.zeros(blocklength)  # control signal is zero for first block
for i in range(n_buffers):

    # record reference signal x and error signal e while playing back y
    x, e, _, _ = sim.playrec(y)

    y = filt.filt(x)

    fx = plant_model_fx(x)
    fy = plant_model_fy(y)

    y_dummy = filt_dummy.filt(fx)
    p_dummy = e - fy
    e_dummy = y_dummy + p_dummy
    filt_dummy.adapt(fx, e_dummy)

    # copy filter coefficients
    filt.W = filt_dummy.W

    xlog.append(x)
    elog.append(e)
    ylog.append(y.copy())
    wslog.append(filt.w)


fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(14, 8), constrained_layout=True)

ax[0, 0].set_title("Signals")
ax[0, 0].plot(np.concatenate(xlog), label="x", alpha=1)
ax[0, 0].plot(np.concatenate(ylog), label="y", alpha=0.8)
ax[0, 0].plot(np.concatenate(elog), label="e", alpha=0.7)
ax[0, 0].set_xlabel("Sample")
ax[0, 0].legend()

ax[0, 1].set_title("Filter weights")
ax[0, 1].plot(wslog)
ax[0, 1].set_xlabel("Block")


ax[1, 0].set_title("Error Energy")
ax[1, 0].plot(10 * np.log10(np.array(np.concatenate(elog)) ** 2))
ax[1, 0].set_xlabel("Sample")
ax[1, 0].set_ylabel("Error [dB]")


ax[1, 1].set_title("Final filter")
ax[1, 1].plot(filt.w)
ax[1, 1].set_xlabel("Tap")

plt.show()
