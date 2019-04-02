import numpy as np
import matplotlib.pyplot as plt

from adafilt import FastBlockLMSFilter
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
filt = FastBlockLMSFilter(
    length, blocklength, stepsize=0.1, leakage=0.99999, power_averaging=0.9
)

# simulates an audio interface with primary and secondary paths and 40 dB SNR noise
sim = FakeInterface(
    blocklength, signal, h_pri=h_pri, h_sec=h_sec, noise=wgn(signal, 40, "dB")
)

# secondary path estimate has to account for block size
h_sec_estimate = np.concatenate((np.zeros(blocklength), h_sec))

# aggregate signals during simulation
xlog = []
elog = []
wslog = []
ylog = []

zi = np.zeros(len(h_sec_estimate) - 1)  # initialize overlap-add filter with zeros
y = np.zeros(blocklength)  # control signal is zero for first block
for i in range(n_buffers):

    # simulate the audio interface that records reference signal x and error signal e
    x, e, _, _ = sim.playrec(y)

    # filter the reference signal
    fx, zi = olafilt(h_sec_estimate, x, zi=zi)

    # adapt filter
    filt.adapt(fx, e)

    # filter
    y = filt.filt(x)

    xlog.append(x)
    elog.append(e)
    ylog.append(y.copy())
    wslog.append(filt.w)


fig, ax = plt.subplots(nrows=3, figsize=(8, 8 / 1.3), constrained_layout=True)

ax[0].set_title("Signals")
ax[0].plot(np.concatenate(xlog), label="reference x")
ax[0].plot(np.concatenate(ylog), label="filter output y")
ax[0].plot(np.concatenate(elog), label="error signal e")
ax[0].set_xlabel("Sample")
ax[0].legend()

ax[1].set_title("Filter coefficients")
ax[1].plot(np.stack((wslog)))
ax[1].set_xlabel("Block")

ax[2].set_title("Error energy")
ax[2].plot(10 * np.log10(np.concatenate(elog) ** 2))
ax[2].set_xlabel("Sample")

plt.show()
