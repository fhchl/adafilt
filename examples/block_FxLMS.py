import numpy as np
import matplotlib.pyplot as plt
from adafilt.utils import olafilt


n = 32

# primary path
h_pri = np.zeros((n))
h_pri[1] = 0.5

# secondary path
h_sec = np.zeros((n))
h_sec[0] = 1

# prepare filter
blocklength = 4
length = 4
n_buffers = 10000
W = np.zeros((2 * length))
stepsize = 0.1

# prepare loop
zi_pri = np.zeros((n - 1))
zi_sec = np.zeros((n - 1))

xlog = []
elog = []
wslog = []
ylog = []
fxlog = []
dlog = []
ulog = []

xold = np.zeros((blocklength))
for i in range(n_buffers):
    x = np.random.normal(size=(blocklength))
    xconcat = np.concatenate((xold, x))
    xold = x.copy()

    X = np.fft.fft(xconcat)

    # filter output
    y = np.real(np.fft.ifft(W * X))[-length :]

    # acoustic paths and summing at error mic
    d, zi_pri = olafilt(h_pri, x, zi=zi_pri)
    # u, zi_sec = olafilt(h_sec, y, zi=zi_sec)
    e = d - y

    # adapt
    E = np.fft.fft(np.concatenate((np.zeros(e.shape), e)))
    update = stepsize * E * X.conj()

    # if True:
    #     # make it causal
    #     U = np.fft.ifft(update)
    #     U[length :] = 0
    #     update = np.fft.fft(U)

    W = W + stepsize * update

    xlog.append(x)
    elog.append(e)
    ylog.append(y)
    dlog.append(d)
    ulog.append(y)
    wslog.append(np.real(np.fft.ifft(W))[: length])


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
