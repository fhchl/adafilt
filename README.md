adafilt
========

Adaptive filters for Python.

```python
"""A filtered-reference Least-Mean-Square (FxLMS) filter."""

import numpy as np
import matplotlib.pyplot as plt

from adafilt import FastBlockLMSFilter, SimpleFilter
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

# secondary path estimate has to account for block size
plant_model = SimpleFilter(np.concatenate((np.zeros(blocklength), h_sec)))

# simulates an audio interface with primary and secondary paths and 40 dB SNR noise
# at the error sensor
sim = FakeInterface(
    blocklength,
    signal,
    h_pri=h_pri,
    h_sec=h_sec,
    noise=wgn(olafilt(h_pri, signal), 40, "dB"),
)

elog = []
y = np.zeros(blocklength)  # control signal is zero for first block
for i in range(n_buffers):

    # record reference signal x and error signal e while playing back y
    x, e, _, _ = sim.playrec(y)

    # filter the reference signal
    fx = plant_model(x)

    # adapt filter
    filt.adapt(fx, e)

    # filter
    y = filt.filt(x)

    elog.append(e)

plt.plot(np.concatenate(elog), label="e", alpha=0.7)
plt.xlabel("Sample")
plt.ylabel("Error Signal")
plt.show()
```

Find the full example [here](https://github.com/fhchl/adafilt/blob/master/examples/fxLMS.py).
