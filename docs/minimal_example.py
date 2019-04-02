import numpy as np

from adafilt import FastBlockLMSFilter
from adafilt.io import FakeInterface
from adafilt.utils import olafilt, wgn

length = 8  # number of adaptive FIR filter taps
blocklength = 2  # length of I/O buffer and blocksize of filter
n_buffers = 150  # size of simulation

# primary and secondary paths
h_pri = [0, 0, 0, 0, 0, 0, 0, 0.5]
h_sec = [0, 0, 0, 1, 0, 0, 0, 0]

# white noise signal
signal = np.random.normal(0, 1, size=n_buffers * blocklength)

# the adaptive filter
filt = FastBlockLMSFilter(length, blocklength, stepsize=0.1, leakage=0.9999)

# simulates an audio interface with primary and secondary paths and 40 dB SNR noise
sim = FakeInterface(
    blocklength, signal, h_pri=h_pri, h_sec=h_sec, noise=wgn(signal, 40, "dB")
)

# secondary path estimate has to account for block size
h_sec_estimate = np.concatenate((np.zeros(blocklength), h_sec))

zi = np.zeros(len(h_sec_estimate) - 1)  # initialize overlap-add filter with zeros
y = np.zeros(blocklength)               # control signal is zero for first block
for i in range(n_buffers):

    # simulate the audio interface that records reference signal x and error signal e
    x, e, _, _ = sim.playrec(y)

    # filter the reference signal
    fx, zi = olafilt(h_sec_estimate, x, zi=zi)

    # adapt filter
    filt.adapt(fx, e)

    # filter
    y = filt.filt(x)
