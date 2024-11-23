"""A filtered-reference Least-Mean-Square (FxLMS) filter."""

import matplotlib.pyplot as plt
import numpy as np
from adafilt import FastBlockLMSFilter, FIRFilter
from adafilt.io import FakeInterface
from adafilt.utils import wgn


def moving_rms(x, N):
    return np.sqrt(np.convolve(x**2, np.ones((N,)) / N, mode="valid"))


length = 64  # number of adaptive FIR filter taps
blocklength = 4  # length of I/O buffer and blocksize of filter
n_buffers = 10000  # size of simulation
estimation_phase = 2000

# primary and secondary paths
h_pri = np.zeros(64)
h_pri[60] = 1
h_sec = np.zeros(64)
h_sec[20] = 1

# simulates an audio interface with primary and secondary paths and 40 dB SNR noise
# at the error sensor
signal = np.random.normal(0, 1, size=n_buffers * blocklength)
sim = FakeInterface(
    blocklength, signal, h_pri=h_pri, h_sec=h_sec, noise=wgn(signal, 20, "dB")
)

# the adaptive filter
filt = FastBlockLMSFilter(
    length, blocklength, stepsize=0.01, leakage=0.99999, power_averaging=0.9
)
filt.locked = True

# secondary path estimate has to account for block size
plant_model = FIRFilter(np.zeros(blocklength + length))

# adaptive plant model
adaptive_plant_model = FastBlockLMSFilter(
    length, blocklength, stepsize=0.01, leakage=0.99999
)

# aggregate signals during simulation
elog = []
e_plog = []
wlog = []
ylog = []
glog = []
ulog = []
dlog = []
fxlog = []

y = np.zeros(blocklength)  # control signal is zero for first block
for i in range(n_buffers):
    # identification noise
    if i < estimation_phase:
        v = np.random.normal(0, 1, blocklength)
    else:
        v = np.random.normal(0, 0.01, blocklength)
        adaptive_plant_model.stepsize = 0.0001

    # record reference signal x and error signal e while playing back y
    x, e, u, d = sim.playrec(-y + v)
    # adaptive plant model prediction
    y_p = adaptive_plant_model.filt(v)
    # plant estimation error
    e_p = e - y_p
    # adapt plant model
    adaptive_plant_model.adapt(v, e_p)
    # copy plant model
    plant_model.w[blocklength:] = adaptive_plant_model.w
    # filter the reference signal
    fx = plant_model(x)

    if i >= estimation_phase:
        # adapt filter
        filt.adapt(fx, e)
        # filter
        y = filt.filt(x)

    ulog.append(u)
    dlog.append(d)
    elog.append(e)
    fxlog.append(fx)
    e_plog.append(e_p)
    ylog.append(y)
    wlog.append(filt.w.copy())
    glog.append(adaptive_plant_model.w.copy())


fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(14, 8), constrained_layout=True)

ax = ax.flatten()

ax[0].set_title("Signals")
ax[0].plot(np.concatenate(ylog), label="y", alpha=0.8)
ax[0].plot(np.concatenate(elog), label="Signal at error mic: e", alpha=0.7)
ax[0].set_xlabel("Sample")
ax[0].legend()

ax[1].set_title("Filter weights")
ax[1].plot(glog, "--")
ax[1].plot(wlog, "-")
ax[1].set_xlabel("Block")

ax[2].set_title("Error Signals")
# ax[2].plot(10 * np.log10(moving_rms(np.concatenate(elog), 512) ** 2), label="e")
# ax[2].plot(10 * np.log10(moving_rms(np.concatenate(e_plog), 512) ** 2), label="e_p")
ax[2].plot(moving_rms(np.concatenate(elog), 512) ** 2, label="e")
ax[2].plot(moving_rms(np.concatenate(e_plog), 512) ** 2, label="e_p")
ax[2].set_xlabel("Sample")
ax[2].set_ylabel("Error [dB]")
ax[2].legend()

# the optimal filter
wopt = -np.fft.irfft(np.fft.rfft(h_pri) / np.fft.rfft(np.roll(h_sec, blocklength)))

ax[3].set_title("Final filter")
ax[3].plot(-filt.w, "x", label="control filter")
ax[3].plot(wopt, "+", label="optimal filter")
ax[3].plot(plant_model.w[blocklength:], "o", label="plant model")
ax[3].plot(h_sec, label="plant")
ax[3].set_xlabel("Tap")
ax[3].legend()

ax[4].set_title("Filtered reference and primary disturbance")
ax[4].plot(np.concatenate(dlog), label="d", alpha=0.7)
ax[4].plot(np.concatenate(fxlog), label="fx", alpha=0.8)
ax[4].set_xlabel("Sample")
ax[4].legend()

pri_path_error = np.sum((wopt + wlog) ** 2, axis=1) / np.sum((wopt) ** 2)
sec_path_error = np.sum((h_sec - glog) ** 2, axis=1) / np.sum((h_sec) ** 2)
ax[5].set_title("Filtered reference and primary disturbance")
ax[5].plot(
    10 * np.log10(pri_path_error), label="primary path estimation error", alpha=0.7
)
ax[5].plot(
    10 * np.log10(sec_path_error), label="secondary path estimation error", alpha=0.7
)
ax[5].set_xlabel("Sample")
ax[5].legend()

plt.show()
