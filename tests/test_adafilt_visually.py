# -*- coding: utf-8 -*-
import numpy as np
import numpy.testing as npt
from scipy.signal import lfilter
import matplotlib.pyplot as plt

from adafilt.io import FakeInterface
from adafilt.optimal import wiener_filter
from adafilt import KalmanFilter


class TestKalmanFilter():

    def test_converges(self):
        # linear dynamic system equations
        A = np.eye(1) # dont move
        Q = np.eye(1) * 0

        # measurement equations
        H = np.eye(1)  # directly observe
        R = np.eye(1) * 0.1

        # initial condition
        x = np.array([[1]])
        P = np.eye(1) * 10  # no knowledge

        filt = KalmanFilter(x=x, P=P, A=A, H=H, Q=Q, R=R)
        x_true = np.array([[2]]).T
        x_trues = []
        measurements = []
        x_estimates = [x]
        P_estimates = [P]
        for _S in range(300):
            x_true = A @ x_true + np.random.multivariate_normal([0], Q)[:, None]  # advance system
            y = H @ x_true + np.random.multivariate_normal([0], R)[:, None]  # measurement

            filt.predict()
            filt.update(y)

            measurements.append(y)
            x_trues.append(x_true)
            x_estimates.append(filt.x)
            P_estimates.append(filt.P)

        measurements = np.stack(measurements)
        x_trues = np.stack(x_trues)
        x_estimates = np.stack(x_estimates)
        P_estimates = np.stack(P_estimates)
        plt.figure()
        plt.plot(x_trues[:, 0], label='truth')
        plt.plot(measurements[:, 0], label='measurement')
        plt.plot(x_estimates[:, 0], label='estimate')
        plt.plot(x_estimates[:, 0] + np.sqrt(P_estimates[:, 0]), 'g--', label='1 std')
        plt.plot(x_estimates[:, 0] - np.sqrt(P_estimates[:, 0]), 'g--', )
        plt.legend()
        plt.show()








