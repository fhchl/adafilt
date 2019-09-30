"""A filtered-reference Least-Mean-Square (FxLMS) filter."""

import numpy as np
import matplotlib.pyplot as plt

from adafilt import LMSFilter, SimpleFilter

length = 16  # number of adaptive FIR filter taps
n_buffers = 400  # size of simulation

# plant
plant = SimpleFilter([0, 0, 0, 0, 0, 0, 0, 0, 1])
model = SimpleFilter([0, 1])

# the adaptive filter
iw = np.zeros(length)
iw[0] = 1
filt = LMSFilter(length, stepsize=0.5, initial_coeff=iw)

# aggregate signals during simulation
elog = []
dlog = []
wslog = []
dhatlog = []
rlog = []

dhat = 0
for i in range(n_buffers):

    # signal
    x = np.random.normal(0, 1)

    d = plant(x)
    r = model(x)

    dhat = filt.filt(r)

    e = d + dhat

    filt.adapt(r, e)

    rlog.append(r)
    elog.append(e)
    wslog.append(filt.w)
    dlog.append(d)
    dhatlog.append(dhat)


fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(14, 8), constrained_layout=True)

ax[0, 0].set_title("Signals")
ax[0, 0].plot(dhatlog, label="dhat", alpha=0.8)
ax[0, 0].plot(dlog, label="d", alpha=0.7)
ax[0, 0].plot(rlog, label="r", alpha=0.7)
ax[0, 0].set_xlabel("Sample")
ax[0, 0].legend()

ax[0, 1].set_title("Filter weights")
ax[0, 1].plot(wslog)
ax[0, 1].set_xlabel("Block")

ax[1, 0].set_title("Error")
ax[1, 0].plot(elog)
ax[1, 0].set_xlabel("Sample")
ax[1, 0].set_ylabel("Error")

ax[1, 1].set_title("Final filter")
ax[1, 1].plot(filt.w, label="estimate")
ax[1, 1].set_xlabel("Tap")
ax[1, 1].legend()

plt.show()
