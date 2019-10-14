# -*- coding: utf-8 -*-
import numpy as np
import numpy.testing as npt
from scipy.signal import lfilter

from adafilt.utils import olafilt
from adafilt.io import FakeInterface
from adafilt.optimal import wiener_filter
from adafilt import MultiChannelBlockLMS, Delay, SimpleFilter


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

    def test_multiple_outputs(self):
        m = 123
        n = 1024
        L = 7
        x = np.random.random(n)
        h = np.random.random((m, L))
        zi = np.random.random((m - 1, L))

        yola, zfola = olafilt(h, x, zi=zi)
        for l in range(L):
            y, zf = lfilter(h[:, l], 1, x, zi=zi[:, l])
            npt.assert_almost_equal(y, yola[:, l])
            npt.assert_almost_equal(zf, zfola[:, l])

    def test_nozi_multiple_inputs(self):
        m = 123
        n = 1024
        L, M = (1, 3)
        x = np.random.random((n,))
        h = np.random.random((m, L, M))

        yola = olafilt(h, x, zi=None)
        y = 0
        for m in range(M):
            yt = lfilter(h[:, 0, m], 1, x, zi=None)
            y += yt

        npt.assert_almost_equal(y, yola[:, 0])

    def test_multiple_inputs(self):
        m = 123
        n = 1024
        L, M = (1, 3)
        x = np.random.random((n,))
        h = np.random.random((m, L, M))
        ziall = np.random.random((m - 1, L, M))
        zi = ziall.sum(axis=-1)

        yola, zfola = olafilt(h, x, zi=zi)
        y = 0
        zf = 0
        for m in range(M):
            yt, zft = lfilter(h[:, 0, m], 1, x, zi=ziall[:, 0, m])
            y += yt
            zf += zft

        npt.assert_almost_equal(zf, zfola[:, 0])
        npt.assert_almost_equal(y, yola[:, 0])

    def test_nozi_many_to_many(self):
        m = 123
        n = 1024
        L, M = (10, 5)

        x = np.random.random((n,))
        h = np.random.random((m, L, M))

        yola = olafilt(h, x, zi=None, sum_inputs=True)

        y = np.zeros((n, L))
        for l in range(L):
            for m in range(M):
                yt = lfilter(h[:, l, m], 1, x, zi=None)
                y[:, l] += yt

        npt.assert_almost_equal(y, yola)

    def test_many_to_many(self):
        m = 123
        n = 1024
        L, M = (10, 5)

        x = np.random.random((n,))
        h = np.random.random((m, L, M))
        ziall = np.random.random((m - 1, L, M))
        zi = ziall.sum(axis=-1)
        yola, zfola = olafilt(h, x, zi=zi, sum_inputs=True)

        y = np.zeros((n, L))
        zf = np.zeros((m - 1, L))
        for l in range(L):
            for m in range(M):
                yt, zft = lfilter(h[:, l, m], 1, x, zi=ziall[:, l, m])
                y[:, l] += yt
                zf[:, l] += zft

        npt.assert_almost_equal(y, yola)
        npt.assert_almost_equal(zf, zfola)

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

        yola, zfola = olafilt(h, x, zi=zi, sum_inputs=False)

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

            yola, zfola = olafilt(h, x, zi=zi, sum_inputs=False)

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

    def test_works_with_any_input_shape_summed_no_zi(self):
        def test_shape(L, M, K):
            m = 123
            n = 1024
            x = np.random.random((n, K))
            h = np.random.random((m, L, M))

            yola = olafilt(h, x, zi=None, sum_inputs=True)

            for l in range(L):
                y = 0
                for m in range(M):
                    for k in range(K):
                        y += lfilter(h[:, l, m], 1, x[:, k], zi=None)
                npt.assert_almost_equal(y, yola[:, l], err_msg=f'{L, M, K}')

        Ls = range(1, 4)
        Ms = range(1, 4)
        Ks = range(1, 4)

        [test_shape(L, M, K) for L in Ls for M in Ms for K in Ks]

    def test_works_with_any_input_shape_summed(self):
        def test_shape(L, M, K):
            m = 123
            n = 1024
            x = np.random.random((n, K))
            h = np.random.random((m, L, M))
            ziall = np.random.random((m - 1, L, M, K))
            zi = ziall.sum(axis=(-1, -2))

            yola, zfola = olafilt(h, x, zi=zi, sum_inputs=True)

            for l in range(L):
                y = 0
                zf = 0
                for m in range(M):
                    for k in range(K):
                        yt, zft = lfilter(h[:, l, m], 1, x[:, k], zi=ziall[:, l, m, k])
                        y += yt
                        zf += zft
                npt.assert_almost_equal(y, yola[:, l], err_msg=f'{L, M, K}')
                npt.assert_almost_equal(zf, zfola[:, l], err_msg=f'{L, M, K}')

        Ls = range(1, 4)
        Ms = range(1, 4)
        Ks = range(1, 4)

        [test_shape(L, M, K) for L in Ls for M in Ms for K in Ks]



class TestSimpleFilter:
    def test_output_shape_1x1(self):
        h = [1, 0, 0]
        filt = SimpleFilter(h, sum_inputs=True)
        x = np.random.normal(size=100)
        y = filt.filt(x)
        assert x.shape == (100,)
        npt.assert_almost_equal(y, x)


class TestFakeInterface:
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

    def test_FakeInterface_output_multichannel(self):
        L, M, K = 10, 5, 3
        npri = 1024
        nsec = 512
        h_pri = np.random.normal(size=(npri, L, K))
        h_sec = np.random.normal(size=(nsec, L, M))

        buffsize = 16
        buffers = 100
        signal = np.random.normal(size=(buffers * buffsize, K))
        noise = np.random.normal(size=(buffers * buffsize, L))
        sim = FakeInterface(buffsize, signal, noise=noise, h_pri=h_pri, h_sec=h_sec)

        ys = []
        xs = []
        es = []
        us = []
        ds = []

        for i in range(buffers):
            y = np.random.normal(size=buffsize)
            x, e, u, d = sim.playrec(y, send_signal=True)
            ys.append(y)
            xs.append(x)
            es.append(e)
            ds.append(d)
            us.append(u)

        y = np.concatenate(ys)
        x = np.concatenate(xs)
        e = np.concatenate(es)
        u = np.concatenate(us)
        d = np.concatenate(ds)

        assert np.all(x == signal)
        npt.assert_almost_equal(d, olafilt(h_pri, x))
        npt.assert_almost_equal(u, olafilt(h_sec, y))
        npt.assert_almost_equal(e, u + d + noise)

    def test_FakeInterface_filtering(self):
        h_pri = [0, 0, 0, 0, 1, 0, 1]  # primary path impulse response
        h_sec = [0, 0, 1, 0.5]
        buffsize = 16
        buffers = 500
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

        npt.assert_almost_equal(h_pri_meas[: len(h_pri)], h_pri, decimal=2)
        npt.assert_almost_equal(h_sec_meas[: len(h_sec)], h_sec, decimal=2)


class TestOptimal:
    def test_wiener_filter_unconstrained_causal(self):
        h = [1, -1, 0.5]
        x = np.random.random(2 ** 16)
        y = olafilt(h, x)

        h_est = -np.real(np.fft.ifft(wiener_filter(x, y, 32, constrained=False)))
        npt.assert_almost_equal(h, h_est[: len(h)], decimal=2)

    def test_wiener_filter_constrained_causal(self):
        h = [1, -1, 0.5]
        x = np.random.random(2 ** 16)
        y = olafilt(h, x)

        h_est = -np.real(np.fft.ifft(wiener_filter(x, y, 32, constrained=True)))
        npt.assert_almost_equal(h, h_est[: len(h)], decimal=2)

    def test_wiener_filter_constrained_noncausal(self):
        h = [1, 0, 1]
        x = np.random.random(2 ** 16)
        y = olafilt(h, x)

        h_est = -np.real(
            np.fft.ifft(wiener_filter(x, y, 256, g=[0, 1], constrained=True))
        )
        npt.assert_almost_equal([0, 1, 0], h_est[: len(h)], decimal=2)


class TestDelay:
    def test_delays_properly(self):
        ndelay = 10
        delay = Delay(ndelay)
        nsamp = 64
        x = np.arange(1, nsamp + 1)

        y = delay.filt(x)
        assert y.shape == x.shape
        npt.assert_array_equal(y, np.concatenate((np.zeros(ndelay), x[:-ndelay])))

        x2 = np.arange(nsamp + 1, nsamp * 2 + 1)
        y2 = delay.filt(x2)

        x = np.concatenate((x, x2))
        y = np.concatenate((y, y2))

        npt.assert_array_equal(y[ndelay:], x[:-ndelay])


class TestMultiChannelBlockLMS:
    def test_filt(self):
        for (L, M, K) in [
            (L, M, K) for L in range(1, 4) for M in range(1, 4) for K in range(1, 4)
        ]:
            print((L, M, K))
            length = 16
            blocks = 16
            blocklength = 16
            w = np.random.normal(size=(length, L, M))
            xs = np.random.normal(size=(blocks, blocklength, K))
            x = np.concatenate(xs)

            filt = MultiChannelBlockLMS(
                length=length,
                blocklength=blocklength,
                initial_coeff=w,
                Nin=K,
                Nout=M,
                Nsens=L,
            )

            # w set correctly
            npt.assert_almost_equal(w, filt.w)

            # single block works
            npt.assert_almost_equal(olafilt(w, xs[0]), filt.filt(xs[0]))

            filt.zifilt = 0  # reset

            # many nlocks work
            y = []
            for xb in xs:
                xb_copy = xb.copy()
                y.append(filt.filt(xb))
                assert np.array_equal(xb_copy, xb)
            y = np.concatenate(y)

            npt.assert_almost_equal(olafilt(w, x), y)
            yref = np.zeros(y.shape)
            for l in range(L):
                for m in range(M):
                    for k in range(K):
                        yref[:, l] += lfilter(w[:, l, m], 1, x[:, k])

            npt.assert_almost_equal(y, yref, err_msg=f"shape: {(L, M, K)}")
