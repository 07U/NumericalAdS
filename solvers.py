"""
    Module for the spatial solution of Einstein's equations.
"""

import initial as i
from numpy import diag
from numpy import dot
from numpy.linalg import solve
from numpy import arange
from numpy import polyfit
from numpy import polyval
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

"""
    Values which are used by all functions.
"""
D2 = dot(i.D, i.D)
rhoTilde = i.rho - 1
rhoTilde2 = rhoTilde**2
rhoTilde3 = rhoTilde**3
rhoTilde4 = rhoTilde**4
rhoTilde5 = rhoTilde**5
rhoTilde6 = rhoTilde**6
rhoTilde7 = rhoTilde**7
rhoTilde14 = rhoTilde**14
twoRh = 2 * i.rh
twoRh2 = twoRh**2
twoRh3 = twoRh**3
twoRh4 = twoRh**4
twoRh5 = twoRh**5
twoRh6 = twoRh**6
twoRh7 = twoRh**7
twoRh9 = twoRh**9
twoRh11 = twoRh**11
twoRh12 = twoRh**12
twoRh13 = twoRh**13

rhoTilde3a_1 = rhoTilde3 * i.a_1
rhoTilde9a_1 = rhoTilde**9 * i.a_1
twoRh3a_1 = twoRh3 * i.a_1
twoRh6a_1 = twoRh6 * i.a_1
twoRhrhoTilde = twoRh * rhoTilde
twoRhrhoTilde2 = twoRh * rhoTilde2
twoRh3rhoTilde = twoRh3 * rhoTilde
twoRh5rhoTilde2 = twoRh5 * rhoTilde2
twoRh6rhoTilde = twoRh6 * rhoTilde
twoRh6rhoTilde8 = twoRh6 * rhoTilde**8
twoRh7rhoTilde6 = twoRh7 * rhoTilde6
twoRh12rhoTilde2 = twoRh12 * rhoTilde2

twoRh2rhoTilde3a_1 = twoRh2 * rhoTilde3a_1
twoRh3rhoTilde3a_1 = twoRh3 * rhoTilde3a_1
twoRh12rhoTildea_1 = twoRh12 * rhoTilde * i.a_1

twoRh3_rhoTilde3a_1 = twoRh3 + rhoTilde3a_1

"""
    The L1 and L2 operators in sigma's equation.
"""
sigmaL1L2 = dot(diag(48 * twoRhrhoTilde), i.D) \
            + dot(diag(4 * twoRhrhoTilde2), D2)

"""
    Some constants which are being used in sigmap's equation.
"""
sigmapConst1 = 2 * twoRh2 * rhoTilde2 * twoRh3_rhoTilde3a_1
sigmapConst2 = twoRh3rhoTilde * (twoRh3 - 2 * rhoTilde3a_1)

"""
    A constant which is being used in betap's equation.
"""
betapConst1 = twoRhrhoTilde * twoRh3_rhoTilde3a_1

"""
    Some constants which are being used in alpha's equation.
"""
alphaConst1 = 14 * (twoRh12 * rhoTilde - 2 * twoRh9 * rhoTilde4 * i.a_1)
alphaConst2 = 2 * (twoRh6 * rhoTilde7 - 2 * twoRh3a_1 * rhoTilde**10)
alphaConst3 = twoRh2rhoTilde3a_1 - twoRh5 / 2
alphaConst4 = 12 * twoRh6 + 18 * twoRh3rhoTilde3a_1
alphaConst5 = (5 /4) * twoRh3 + 3 * rhoTilde3a_1

"""
    Solves the sigma equation of motion.

    ((120 (2 rh)^6
     + (rho - 1)^6 (3 beta(rho) + (rho - 1) (d beta/d rho)(rho))^2)
            / (2 rh)^5)
                sigma(rho)
    + 48 (2 rh) (rho - 1) (d sigma/d rho)(rho)
    + 4 (2 rh) (rho - 1)^2 (d^2 sigma/d rho^2)(rho)

    =

    ((rho - 1) lambda - 2 rh)
        (3 beta(rho) + (rho - 1) (d beta/d rho)(rho))^2

    Input arguments
        lam - The lambda value which will be used for computation.
        beta - The beta values which will be used for computation.
        dbeta - The partial derivative acording to rho of beta which will be used for computation.

    Return
        sigma - The solution of the equation of motion on sigma.
"""
def sigmaSolve(lam, beta, dbeta):

    sharedTerm = (3 * beta + rhoTilde * dbeta)**2

    L0 = diag((120 * twoRh6 + rhoTilde6 * sharedTerm) / twoRh5)
    f = (rhoTilde * lam - twoRh) * sharedTerm

    return solve(L0 + sigmaL1L2, f)

"""
    Solves the sigmap equation of motion with division.

    ((2 rh)^5 (3 (rho - 1) lambda  - 2 (2 rh))
     - (rho - 1)^6 (8 sigma(rho) + (rho - 1) (d sigma/d rho)(rho)))
        2 (2 rh)
            sigmap(rho)
    - 2 (2 rh) (rho - 1)
        ((2 rh)^5 (2 rh - (rho - 1) lambda) + (rho - 1)^6 sigma(rho))
            (d sigmap/d rho)(rho)

    =

    2 (2 rh)^2 (rho - 1)^2 ((2 rh)^3 + (rho - 1)^3 a_1) lambda
        (7 sigma(rho) + (rho - 1) (d sigma/d rho)(rho))
    - (2 rh)^4 lambda^2
        (4 (2 rh)^3 a_1 + 5 (rho - 1)^3 sigma(rho)
        + (rho - 1)^4 (d sigma/d rho)(rho))
    - (rho - 1)
        (3 sigma(rho)
            (3 (2 rh)^6 - 4 (2 rh)^3 (rho - 1)^3 a_1 + (rho - 1)^6 sigma(rho))
        + 2 (2 rh)^3 (rho - 1) (0.5 (2 rh)^3 - (rho - 1)^3 a_1)
            (d sigma/d rho)(rho))

    Input arguments
        lam - The lambda value which will be used for computation.
        beta - The beta values which will be used for computation.
        dbeta - The partial derivative acording to rho of beta which will be used for computation.
        sigma - The sigma values which were calculated using lam, beta and dbeta.
        dsigma - The partial derivative according to rho of initial sigma.

    Return
        sigmap - The solution of the equation of motion on sigmap.
"""
def sigmapDivSolve(lam, sigma, dsigma):

    rhoTildelam = rhoTilde * lam
    rhoTilde6sigma = rhoTilde6 * sigma
    rhoTildedsigma = rhoTilde * dsigma

    L0 = diag(2 * twoRh \
                  * (twoRh5 * (2 * twoRh - 3 * rhoTildelam) \
                    + rhoTilde6 * (8 * sigma + rhoTildedsigma)))
    L1 = dot(diag(2 * twoRhrhoTilde \
                      * (twoRh5 * (twoRh - rhoTildelam) + rhoTilde6sigma)) \
            , i.D)
    f = twoRh4 * lam**2 \
            * (4 * twoRh3a_1 + rhoTilde3 * (5 * sigma + rhoTildedsigma)) \
        - sigmapConst1 * lam * (7 * sigma + rhoTildedsigma) \
        + rhoTilde \
            * (3 * sigma \
                  * (3 * twoRh6 - 4 * twoRh3rhoTilde3a_1 + rhoTilde6sigma) \
              + sigmapConst2 * dsigma)
    return solve(L0 + L1, f)

"""
    Solves the betap equation of motion with division.

    ((2 rh)^5 (2 rh - 2 (rho - 1) lambda)
     + (rho - 1)^6 (7 sigma(rho) + (rho - 1) (d sigma/d rho)(rho)))
        betap(rho)
    + (rho - 1) ((2 rh)^5 (2 rh - (rho - 1) lambda) + (rho - 1)^6 sigma(rho))
        (d betap/d rho)(rho)

    =

    2 rh (3 beta(rho) + (rho - 1) (d beta/d rho)(rho))
        ((2 rh)^2 (rho - 1)^3 a_1
         + 2 rh (rho - 1) ((2 rh)^3 + (rho - 1)^3 a_1) lambda
         - 0.5 (2 rh)^3 ((rho - 1)^2 lambda^2 + (2 rh)^2)
         + (rho - 1)^5 sigmap(rho))

    Input arguments
        lam - The lambda value which will be used for computation.
        beta - The beta values which will be used for computation.
        dbeta - The partial derivative acording to rho of beta which will be used for computation.
        sigma - The sigma values which were calculated using lam, beta and dbeta.
        dsigma - The partial derivative according to rho of initial sigma.
        sigmap - The sigmap values which were calculated using lam, beta and dbeta.

    Return
        betap - The solution of the equation of motion on betap.
"""
def betapDivSolve(lam, beta, dbeta, sigma, dsigma, sigmap):

    rhoTildelam = rhoTilde * lam

    L0 = diag(twoRh5 * (twoRh - 2 * rhoTildelam) \
              + rhoTilde6 * (7 * sigma + rhoTilde * dsigma))
    L1 = dot(diag(rhoTilde \
                      * (twoRh5 * (twoRh - rhoTildelam) + rhoTilde6 * sigma)) \
            , i.D)
    f = twoRh * (3 * beta + rhoTilde * dbeta) \
            * (twoRh2rhoTilde3a_1 + betapConst1 * lam \
              - (twoRh3 / 2) * (rhoTildelam**2 + twoRh2) + rhoTilde5 * sigmap)
    return solve(L0 + L1, f)

"""
    Solves the alpha equation of motion.

    24 (2 rh) ((2 rh)^5 (2 rh - (rho - 1) lambda) + (rho - 1)^6 sigma(rho))^2
        alpha(rho)
    + 16 (2 rh) (rho - 1)
        ((2 rh)^5 (2 rh - (rho - 1) lambda) + (rho - 1)^6 sigma(rho))^2
            (d alpha/d rho)(rho)
    + 2 (2 rh) (rho - 1)^2
        ((2 rh)^5 (2 rh - (rho - 1) lambda) + (rho - 1)^6 sigma(rho))^2
            (d^2 alpha/d rho^2)(rho)

    =

    2 (2 rh)^12 (rho - 1) (7 sigma(rho) - 6 a_1 lambda^3)
    - 28 (2 rh)^9 (rho - 1)^4 a_1 sigma(rho)
    + 2 (2 rh)^6 (rho - 1)^7 sigma(rho)^2
    - 4 (2 rh)^3 (rho - 1)^10 a_1 sigma(rho)^2
    - 3 (rho - 1) beta(rho) betap(rho)
        ((2 rh)^5 (2 rh - (rho - 1) lambda) + (rho - 1)^6 sigma(rho))^2
    - 20 (2 rh)^7 (rho - 1)^6 sigma(rho) sigmap(rho)
    - (2 rh)^12 (rho - 1)^2 betap(rho) dbeta(rho)
    - 2 (2 rh)^6 (rho - 1)^8 betap(rho) sigma(rho) dbeta(rho)
    - (rho - 1)^14 betap(rho) sigma(rho)^2 dbeta(rho)
    - 4 (2 rh)^2
        ((2 rh)^5
            ((rho - 1)^2 dsigma(rho)
                ((2 rh)^2 (rho - 1)^3 a_1 + (rho - 1)^5 sigmap(rho)
                - 0.5 (2 rh)^5)
            - (2 rh)^6 sigmap(rho))
        + (rho - 1)^2 lambda
            (3 (rho - 1)^9 a_1 sigma(rho)^2
            + 0.5 (2 rh)^3 sigma(rho)
                (12 (2 rh)^6 + 18 (2 rh)^3 (rho - 1)^3 a_1
                - (rho - 1)^7 betap(rho) dbeta(rho))
            + (2 rh)^6 (rho - 1)
                (((2 rh)^3 + (rho - 1)^3 a_1) dsigma(rho)
                - 0.5 (2 rh)^3 betap(rho) dbeta(rho)))
        - 2 (2 rh)^5 lambda^2
            (2.5 (2 rh)^6 a_1
            + (rho - 1)^3
                ((1.25 (2 rh)^3 + 3 (rho - 1)^3 a_1) sigma(rho)
                + 0.125 (2 rh)^3 (rho - 1)
                    (2 dsigma(rho) - betap(rho) dbeta(rho)))))

    Input arguments
        lam - The lambda value which will be used for computation.
        beta - The beta values which will be used for computation.
        dbeta - The partial derivative acording to rho of beta which will be used for computation.
        sigma - The sigma values which were calculated using lam, beta and dbeta.
        dsigma - The partial derivative according to rho of initial sigma.
        sigmap - The sigmap values which were calculated using lam, beta and dbeta.
        betap - The betap values which were calculated using lam, beta and dbeta.

    Return
        alpha - The solution of the equation of motion on alpha.
"""
def alphaSolve(lam, beta, dbeta, sigma, dsigma, sigmap, betap):

    sharedTerm = (twoRh5 * (twoRh - rhoTilde * lam) + rhoTilde6 * sigma)**2

    L0 = diag(24 * twoRh * sharedTerm)
    L1 = dot(diag(16 * twoRhrhoTilde * sharedTerm), i.D)
    L2 = dot(diag(2 * twoRhrhoTilde2 * sharedTerm), D2)

    sigma2 = sigma**2
    betapDbeta = betap * dbeta

    f = sigma * (alphaConst1 - 2 * twoRh6rhoTilde8 * betapDbeta) \
               - 12 * twoRh12rhoTildea_1 * lam**3 \
               + alphaConst2 * sigma2 \
               - 3 * rhoTilde * beta * betap * sharedTerm \
               - 20 * twoRh7rhoTilde6 * sigma * sigmap \
               - betapDbeta * (twoRh12rhoTilde2 + rhoTilde14 * sigma2) \
               - 4 * twoRh2 \
                   * (twoRh5rhoTilde2 * dsigma \
                         * (alphaConst3 + rhoTilde5 * sigmap) \
                     - twoRh11 * sigmap \
                     + rhoTilde2 * lam \
                         * (3 * rhoTilde9a_1 * sigma2
                           + (twoRh3 / 2) * sigma \
                               * (alphaConst4 - rhoTilde7 * betapDbeta) \
                           + twoRh6rhoTilde \
                               * (twoRh3_rhoTilde3a_1 * dsigma \
                               - (twoRh3 / 2) * betapDbeta)) \
                     - 2 * twoRh5 * lam**2 \
                         * ((5 / 2) * twoRh6a_1 \
                           + rhoTilde3 \
                               * (alphaConst5 * sigma \
                                 + (twoRh3rhoTilde / 8) \
                                     *(2 * dsigma - betapDbeta))))
    return solve(L0 + L1 + L2, f)

"""
    Plots a given function values on a Chebyshev grid and an interpolation line.

    Input arguments
        val - The function values on the Chebyshev grid.
"""
def plotInGrid(val):

    rrho = arange(-1, 1, 0.01)
    vval = polyval(polyfit(i.rho, val, i.N), rrho)
    plot(i.rho, val, '.', rrho, vval)
    show()