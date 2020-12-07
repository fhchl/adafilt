import numpy as np
from scipy.linalg import solve_toeplitz, lstsq, toeplitz, convolution_matrix

import tqdm

def least_squares(x, y, M, chop=False, **lstsq_kwargs):
    assert len(y) == len(x), 'input must have same length'
    y = y[M-1:]  # first m-1 elements can not be used in estimation
    N = len(y)
    assert N > 0, 'len(x) must be larger or equal than len(y) + M - 1: not enough data'

    X = toeplitz(x[M-1:], np.flip(x[:M]))  # convolution matrix shape N x M
    if not chop:
        # Least squares needs O(M^2 N) operations. Faster Toeplitz least squares solver?
        # `scipy.linalg.solve_toeplitz` does not support rectangular matrices.
        h = lstsq(X, y, check_finite=False, **lstsq_kwargs)[0]  #
        return h
    assert X.shape[0] == len(y)

    # estimate h from mean over sub-problems
    Nseg = N // M
    Xseg = X[:Nseg * M].reshape(Nseg, M, M)
    yseg = y[:Nseg * M].reshape(Nseg, M)
    h = 0
    for Xs, ys in tqdm.tqdm(zip(Xseg, yseg)):
        # Toeplitz solve needs O(M^2) operations, so this is O(M^2 Nseg)
        h += solve_toeplitz((Xs[:, 0], Xs[0, :]), ys, check_finite=False) / Nseg

    return h