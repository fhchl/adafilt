# -*- coding: utf-8 -*-
import numpy as np
import numpy.testing as npt
from scipy.signal import lfilter

from adafilt.utils import olafilt
from adafilt.io import FakeInterface


class TestOlafilt:
    def test_behaves_like_scipy(self):
        m = 123
        n = 1024
        h = np.random.random(m)
        x = np.random.random(n)
        zi = np.random.random(m - 1)

        yola, zfola = olafilt(h, x, zi=zi)
        y, zf = lfilter(h, 1, x, zi=zi)

        npt.assert_almost_equal(yola, y)
        npt.assert_almost_equal(zfola, zf)

        assert np.array_equal(zi.shape, zfola.shape)
        assert np.array_equal(zi.shape, zf.shape)

    def test_many_to_many(self):
        m = 123
        n = 1024
        K, L = (5, 7)
        x = np.random.random((n, K))
        h = np.random.random((m, L, K))
        zi = np.random.random((m - 1, L, K))
        zj = np.sum(zi, axis=-1)

        yola, zfola = olafilt(h, x, zi=zj, squeeze=False)

        y = np.zeros((n, L))
        zf = np.zeros((m - 1, L))
        for o in range(L):
            for i in range(K):
                yt, zft = lfilter(h[:, o, i], 1, x[:, i], zi=zi[:, o, i])
                y[:, o] += yt
                zf[:, o] += zft

        npt.assert_almost_equal(y, yola)
        npt.assert_almost_equal(zf, zfola)

    def test_multiple_outputs(self):
        m = 123
        n = 1024
        L = 7
        x = np.random.random(n)
        h = np.random.random((m, L))
        zi = np.random.random((m - 1, L))

        yola, zfola = olafilt(h, x, zi=zi)
        for o in range(L):
            y, zf = lfilter(h[:, o], 1, x, zi=zi[:, o])
            npt.assert_almost_equal(y, yola[:, o])
            npt.assert_almost_equal(zf, zfola[:, o])

    def test_multiple_inputs(self):
        m = 123
        n = 1024
        L, K = (1, 3)
        x = np.random.random((n, K))
        h = np.random.random((m, L, K))
        zi = np.random.random((m - 1, L))

        yola, zfola = olafilt(h, x, zi=zi)
        y = 0
        zf = 0
        for i in range(K):
            yt, zft = lfilter(h[:, 0, i], 1, x[:, i], zi=zi[:, 0] / K)
            y += yt
            zf += zft

        npt.assert_almost_equal(zf, zfola)
        npt.assert_almost_equal(y, yola)

    def test_does_not_modify_inputs(self):
        m = 123
        n = 1024
        K, L = (4, 8)
        x = np.random.random((n, K))
        h = np.random.random((m, L, K))
        zi = np.random.random((m - 1, L))

        x0 = x.copy()
        zi0 = zi.copy()
        h0 = h.copy()

        yola, zfola = olafilt(h, x, zi=zi)

        np.array_equal(h, h0)
        np.array_equal(x, x0)
        np.array_equal(zi, zi0)

    def test_does_not_sum(self):
        m = 123
        n = 1024
        L, M, K = (4, 3, 2)
        x = np.random.random((n, K))
        h = np.random.random((m, L, M))
        zi = np.random.random((m - 1, L, M, K))

        yola, zfola = olafilt(h, x, zi=zi, squeeze=False, sum_inputs=False)

        y = np.zeros((n, L))
        zf = np.zeros((m - 1, L))
        for l in range(L):
            for m in range(M):
                for k in range(K):
                    y, zf = lfilter(h[:, l, m], 1, x[:, k], zi=zi[:, l, m, k])
                    npt.assert_almost_equal(y, yola[:, l, m, k])
                    npt.assert_almost_equal(zf, zfola[:, l, m, k])

    def test_works_with_any_input_shape(self):
        def test_shape(L, M, K):
            m = 123
            n = 1024
            x = np.random.random((n, K))
            h = np.random.random((m, L, M))
            zi = np.random.random((m - 1, L, M, K))

            yola, zfola = olafilt(h, x, zi=zi, squeeze=False, sum_inputs=False)

            y = np.zeros((n, L))
            zf = np.zeros((m - 1, L))
            for l in range(L):
                for m in range(M):
                    for k in range(K):
                        y, zf = lfilter(h[:, l, m], 1, x[:, k], zi=zi[:, l, m, k])
                        npt.assert_almost_equal(y, yola[:, l, m, k])
                        npt.assert_almost_equal(zf, zfola[:, l, m, k])

        Ls = range(1, 4)
        Ms = range(1, 4)
        Ks = range(1, 4)

        [test_shape(L, M, K) for L in Ls for M in Ms for K in Ks]


class TestIO:
    def test_FakeInterface_output(self):

        h_pri = np.random.normal(0, 1, 5)
        h_sec = np.random.normal(0, 1, 24)
        buffsize = 16
        buffers = 100
        signal = np.random.normal(0, 1, size=buffers * buffsize)
        sim = FakeInterface(buffsize, signal, h_pri=h_pri, h_sec=h_sec)

        ys = []
        xs = []
        es = []
        us = []
        ds = []

        for i in range(buffers):
            y = np.random.normal(0, 1, buffsize)
            x, e, u, d = sim.playrec(y, send_signal=True)
            xs.append(x)
            es.append(e)
            ds.append(d)
            us.append(u)
            ys.append(y)

        y = np.concatenate(ys)
        x = np.concatenate(xs)
        e = np.concatenate(es)
        u = np.concatenate(us)
        d = np.concatenate(ds)

        assert np.all(x == signal)
        npt.assert_almost_equal(d, olafilt(h_pri, x))
        npt.assert_almost_equal(u, olafilt(h_sec, y))
        npt.assert_almost_equal(e, u + d)

    def test_FakeInterface_filtering(self):
        h_pri = [0, 0, 0, 0, 1, 0, 1]  # primary path impulse response
        h_sec = [0, 0, 1, 0.5]
        buffsize = 16
        buffers = 100
        signal = np.random.normal(0, 1, size=buffers * buffsize)
        sim = FakeInterface(buffsize, signal, h_pri=h_pri, h_sec=h_sec)
        sim = FakeInterface(buffsize, signal, h_pri=h_pri, h_sec=h_sec)

        # measure primary path
        xs = []
        es = []
        for i in range(buffers):
            x, e, _, _ = sim.rec()
            xs.append(x)
            es.append(e)

        xs = np.concatenate(xs)
        es = np.concatenate(es)

        h_pri_meas = np.fft.irfft(np.fft.rfft(es) / np.fft.rfft(xs))[:10]

        # measure secondary path
        ys = []
        us = []
        for i in range(buffers):
            y = np.random.normal(0, 1, size=buffsize)
            _, e, _, _ = sim.playrec(y, send_signal=False)
            ys.append(y)
            us.append(e)

        ys = np.concatenate(ys)
        us = np.concatenate(us)

        h_sec_meas = np.fft.irfft(np.fft.rfft(us) / np.fft.rfft(ys))[:10]

        npt.assert_almost_equal(h_pri_meas[:len(h_pri)], h_pri, decimal=2)
        npt.assert_almost_equal(h_sec_meas[:len(h_sec)], h_sec, decimal=2)
