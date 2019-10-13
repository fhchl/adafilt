"""Multichannel block FxLMS filtering. Imparative style."""

import numpy as np
import matplotlib.pyplot as plt
from adafilt.utils import olafilt
from adafilt.optimal import multi_channel_wiener_filter
from collections import deque

n = 16
Nin = 1  # references
Nout = 2  # control sources
Nsens = 4  # error sensors
nblocks = 500
blocklength = 4
length = 4 * blocklength

# primary path
h_pri = np.zeros((n, Nsens, Nin))
h_pri[2] = 1

# secondary path
h_sec = np.zeros((n, Nsens, Nout))
h_sec[0] = 1  # np.eye(Nsens)

x = np.random.normal(size=(nblocks * blocklength, Nin))  # reference signal
d = olafilt(h_pri, x)  # disturbance

# optimal wiener filter
Wiener = multi_channel_wiener_filter(x, d, length, h_sec)

plt.figure()
plt.title("Wiener solution")
plt.plot(np.real(np.fft.ifft(Wiener, axis=0)).reshape(length, -1))

# prepare filter
num_saved_blocks = length // blocklength
W = np.zeros((2 * length, Nout, Nin))
stepsize = 0.001

G = np.fft.fft(h_sec, axis=0, n=2 * length)

# prepare loop
zi_pri = np.zeros((n - 1, Nsens))
zi_sec = np.zeros((n - 1, Nsens))

xlog = []
elog = []
wslog = []
ylog = []
fxlog = []
dlog = []
ulog = []

xbuff = deque(
    np.zeros((2 * num_saved_blocks, blocklength, Nin)), maxlen=2 * num_saved_blocks
)
ebuff = deque(np.zeros((num_saved_blocks, blocklength, Nsens)), maxlen=num_saved_blocks)

for i in range(nblocks):
    x = np.random.normal(size=(blocklength, Nin))
    xbuff.append(x)

    X = np.fft.fft(np.concatenate(xbuff), axis=0)

    # filter
    y = np.real(np.fft.ifft(W @ X[..., None], axis=0))[-blocklength:, :, 0]

    # acoustic paths and summing at error mic
    d, zi_pri = olafilt(h_pri, x, zi=zi_pri, sum_inputs=True)
    u, zi_sec = olafilt(h_sec, y, zi=zi_sec, sum_inputs=True)
    e = d + u
    ebuff.append(e)

    # adapt
    E = np.fft.fft(
        np.concatenate((np.zeros((length, Nsens)), np.concatenate(ebuff))), axis=0
    )
    update = G.conj().transpose([0, 2, 1]) @ E[..., None] @ X.conj()[:, None, :]

    if True:
        # make it causal
        ut = np.real(np.fft.ifft(update, axis=0))
        ut[length:] = 0
        update = np.fft.fft(ut, axis=0)

    W = W - stepsize * update

    xlog.append(x)
    elog.append(e)
    ylog.append(y)
    dlog.append(d)
    ulog.append(u)
    wslog.append(np.real(np.fft.ifft(W, axis=0))[:length])


plt.figure()
plt.plot(np.concatenate(xlog), label="x")
plt.plot(np.concatenate(ulog), label="u")
plt.plot(np.concatenate(dlog), label="d")
plt.legend()

plt.plot()
plt.figure()
plt.title("Final filter")
plt.plot(wslog[-1].reshape(length, -1))
plt.xlabel("Tap")

plt.figure()
plt.title("Error Energy")
plt.plot(10 * np.log10(np.array(np.concatenate(elog)) ** 2))
plt.xlabel("Sample")
plt.ylabel("Error [dB]")

plt.figure()
plt.title("output")
plt.plot(np.concatenate(ylog))
plt.xlabel("Sample")

plt.show()
