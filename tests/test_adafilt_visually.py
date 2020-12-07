# -*- coding: utf-8 -*-
import numpy as np
import numpy.testing as npt
from scipy.signal import lfilter
import matplotlib.pyplot as plt

from adafilt.io import FakeInterface
from adafilt.optimal import wiener_filter
from adafilt import (
    kalman_filter_predict, kalman_filter_update
)

class TestKalmanFilter():

    def test_converges(self):
        # linear dynamic system equations
        A = np.eye(2) # dont move
        B = np.zeros((2, 2))
        Q = np.eye(2) * 0.1

        # measurement equations
        H = np.eye(2)  # directly observe
        R = np.eye(2) * 100

        # initial condition
        x = np.array([[2, 2]]).T
        P = np.eye(2) * 100  # no knowledge

        x_true = np.array([[1, 3]]).T
        x_trues = []
        measurements = []
        x_estimates = []
        P_estimates = []
        for _S in range(100):
            x_true = A @ x_true + np.random.multivariate_normal([0, 0], Q)[:, None]  # advance system
            z = H @ x_true + np.random.multivariate_normal([0, 0], R)[:, None]  # measurement

            x, P = kalman_filter_update(x, z, H, P, R)
            x, P = kalman_filter_predict(x, np.zeros((2, 1)), A, B, P, Q)

            measurements.append(z)
            x_trues.append(x_true)
            x_estimates.append(x)
            P_estimates.append(P)

        measurements = np.stack(measurements)
        x_trues = np.stack(x_trues)
        x_estimates = np.stack(x_estimates)
        P_estimates = np.stack(P_estimates)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(x_trues[:, 0], label='truth')
        plt.plot(measurements[:, 0], label='measurement')
        plt.plot(x_estimates[:, 0], label='estimate')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(x_trues[:, 1], label='truth')
        plt.plot(measurements[:, 1], label='measurement')
        plt.plot(x_estimates[:, 1], label='estimate')
        plt.legend()
        plt.show()


import adafilt

class TestStatic:

    def test_compare_least_squares_and_wiener(self):
        # This test shows what often, least_squares is better, but it is MUCH less robust, i.e. sometimes, the error with LS is huge.
        # compare these with RLS?
        N = 10000
        M = 16
        Mest = 32
        x = np.random.normal(size=N)
        e = np.random.normal(scale=0.1, size=N)
        h = np.random.normal(size=M)
        y = np.convolve(x, h, mode='full')[:N] + e
        ns = range(2 * Mest, N, Mest)

        ewiener = []
        for n in ns:
            hwiener = np.fft.ifft(- adafilt.optimal.wiener_filter(x[:n], y[:n], Mest))
            ewiener.append(np.sum((h - hwiener[:M])**2)+ np.sum(hwiener[M:]**2))

        els = []
        for n in ns:
            hls = adafilt.static.least_squares(x[:n], y[:n], Mest, chop=True)
            els.append(np.sum((h - hls[:M])**2) + np.sum(hls[M:]**2))


        filt = adafilt.RLSFilter(length=Mest)
        erls = []
        for xs, yd in zip(x, y):
            yest = filt.filt(xs)
            e = yd - yest
            filt.adapt(xs, e)
            erls.append(np.sum((h - filt.w[:M])**2) + np.sum(filt.w[M:]**2))

        plt.figure()
        plt.suptitle('Comparing least squares and wiener filter using Welch')
        plt.subplot(211)
        plt.plot(ns, 10*np.log10(ewiener), label='wiener')
        plt.plot(ns, 10*np.log10(els), label='ls with chopping')
        plt.plot(10*np.log10(erls), label='ls with chopping')
        plt.ylabel('Error [dB]')
        plt.xlabel('Samples used for estimation')
        plt.title('Convergence')
        plt.legend()

        plt.subplot(212)
        plt.plot(h, label='truth')
        plt.plot(hwiener, label='wiener')
        plt.plot(hls, label='ls with chopping')
        plt.xlabel('n')
        plt.title('Truth and estimates')
        plt.show()