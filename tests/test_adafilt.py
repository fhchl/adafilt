# -*- coding: utf-8 -*-
import numpy as np
import numpy.testing as npt
from scipy.signal import lfilter

from adafilt.utils import olafilt


class TestOlafilt:
    def test_behaves_like_scipy(self):
        M = 123
        N = 1024
        h = np.random.random(M)
        x = np.random.random(N)
        zi = np.random.random(M - 1)

        yola, zfola = olafilt(h, x, zi=zi)
        y, zf = lfilter(h, 1, x, zi=zi)

        npt.assert_almost_equal(yola, y)
        npt.assert_almost_equal(zfola, zf)

        assert np.array_equal(zi.shape, zfola.shape)
        assert np.array_equal(zi.shape, zf.shape)

    def test_many_to_many(self):
        M = 123
        N = 1024
        Nin, Nout = (5, 7)
        x = np.random.random((Nin, N))
        h = np.random.random((Nin, Nout, M))
        zi = np.random.random((Nin, Nout, M - 1))
        zj = np.sum(zi, axis=0)

        yola, zfola = olafilt(h, x, zi=zj, squeeze=False)

        y = np.zeros((Nout, N))
        zf = np.zeros((Nout, M - 1))
        for o in range(Nout):
            for i in range(Nin):
                yt, zft = lfilter(h[i, o], 1, x[i], zi=zi[i, o])
                y[o] += yt
                zf[o] += zft

        npt.assert_almost_equal(y, yola)
        npt.assert_almost_equal(zf, zfola)

    def test_multiple_outputs(self):
        M = 123
        N = 1024
        Nout = 7
        x = np.random.random(N)
        h = np.random.random((Nout, M))
        zi = np.random.random((Nout, M - 1))

        yola, zfola = olafilt(h, x, zi=zi)
        for o in range(Nout):
            y, zf = lfilter(h[o], 1, x, zi=zi[o])
            npt.assert_almost_equal(y, yola[o])
            npt.assert_almost_equal(zf, zfola[o])

    def test_multiple_inputs(self):
        M = 123
        N = 1024
        Nin = 2
        Nout = 1
        x = np.random.random((Nin, N))
        h = np.random.random((Nin, Nout, M))
        zi = np.random.random((Nout, M - 1))
        # zi = np.zeros((Nout, M - 1))

        yola, zfola = olafilt(h, x, zi=zi)
        y = 0
        zf = 0
        for i in range(Nin):
            yt, zft = lfilter(h[i, 0], 1, x[i], zi=zi[0] / Nin)
            y += yt
            zf += zft

        npt.assert_almost_equal(zf, zfola)
        npt.assert_almost_equal(y, yola)

    def test_does_not_modify_inputs(self):
        M = 123
        N = 1024
        Nin = 5
        Nout = 8

        x = np.random.random((Nin, N))
        h = np.random.random((Nin, Nout, M))
        zi = np.random.random((Nout, M - 1))

        x0 = x.copy()
        zi0 = zi.copy()
        h0 = h.copy()

        yola, zfola = olafilt(h, x, zi=zi)

        np.array_equal(h, h0)
        np.array_equal(x, x0)
        np.array_equal(zi, zi0)


if __name__ == '__main__':
    TestOlafilt().test_behaves_like_scipy()
    TestOlafilt().test_many_to_many()
    TestOlafilt().test_multiple_inputs()
    TestOlafilt().test_multiple_outputs()
    TestOlafilt().test_behaves_like_scipy()
