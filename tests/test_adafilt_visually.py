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



