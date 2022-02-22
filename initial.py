"""
    Module for the initial conditions.
"""

from cheb import cheb
from numpy import exp

"""
    Size of the grid. There will be N + 1 grid points.
"""
N = 32

"""
    Spectral methods variables.
"""
ch = cheb(N)
D = ch[0]
rho = ch[1]

"""
    The initial horizon position.
"""
rh = 1

"""
    The initial guess for lambda.
"""
lam0 = -0.15675982326818474

"""
    The initial condition a_1.
"""
a_1 = - 1 / 2

"""
    Beta parameters.
"""
beta_0 = 5
u_0 = 0.25
omega = 0.15

"""
    Computes initial beta, as function of rho.
"""
def beta0():
    return beta_0 * exp(-(((1 - rho) / (2 * rh) - u_0) / omega)**2)

"""
    Computes the derivative of initial beta according to rho, as function of rho.
"""
def dbeta0():
    return beta0() * ((1 - rho) / (2 * rh) - u_0) / (rh * omega**2)