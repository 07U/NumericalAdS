"""
    Module for the Spectral Methods differentiation matrix.
"""

"""
    Computes the spectral methods differentiation matrix and Chabysev grid.

    Input arguments
        N - Size of the grid. There will be N + 1 grid points.

    Return
        An array which includes:
            D - The differentiation matrix of size (N+1)X(N+1)
            x = Chebyshev grid of size N+1
"""
def cheb(N):
    from numpy import array

    if(N == 0):
        D = array([0])
        x = array([1])
        return [D, x]

    from numpy import arange
    from numpy import newaxis
    from numpy import r_
    from numpy import ones
    from numpy import eye
    from numpy import tile
    from numpy import diag
    from numpy import cos
    from numpy import pi

    ind = arange(N + 1)
    x = cos(pi * ind / N)
    if(N % 2 == 0):                              # Numerical accuracy treatment.
        x[N / 2] = 0
    temp = r_[([[2]], ones((N - 1, 1)))]
    temp = r_[(temp, [[2]])]
    temp *= (- 2) * (ind[:, newaxis] % 2 - 0.5)
    X = tile(x, (N + 1, 1))
    dX = X.T - X
    D = (temp * (1 / temp).T) / (dX + eye(N + 1))        # off-diagonal entries.
    D = D - diag(D.T.sum(axis = 0))                      # diagonal entries.
    if(N % 2 == 0):                              # Numerical accuracy treatment.
        D[N / 2, N / 2] = 0
    return [D, x]