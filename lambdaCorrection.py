"""
    Module which handle the correction of lambda, using the boundary condition on the horizon.
"""

import initial as i
import solvers as s
from numpy import arange
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

"""
    Some constants.
"""
maxNumOfIterations = 10**4
epsilon = 10**(-14)

rha_1 = i.rh * i.a_1
rh3over2 = i.rh**3 / 2

"""
    Computes the value of Sigmap (big sigma!) at the horizon.

    Input arguments
        lam - The lambda value which was used for computation.
        sigmapH - The value of sigmap at the horizon.

    Return
        The value of Sigmap at the horizon.
"""
def sigmapHorizonValue(lam, sigmapH):
    return sigmapH + rha_1 * (i.rh - lam) + rh3over2 * (lam + i.rh)**2

"""
    Computes the value of Sigmap (big sigma!) at the horizon.

    Input arguments
        lam - The lambda value which will be used for computation.
        beta - The beta values which will be used for computation.
        dbeta - The partial derivative acording to rho of beta which will be used for computation.

    Return
        The value of Sigmap at the horizon with the values of sigma, the partial derivative according
        to rho of sigma and sigmap, which were computed in the proccess.
"""
def getSigmapAtHorizon(lam, beta, dbeta):
    sigma = s.sigmaSolve(lam, beta, dbeta)
    dsigma = s.dot(i.D, sigma)
    sigmap = s.sigmapDivSolve(lam, sigma, dsigma)
    return [sigmapHorizonValue(lam, sigmap[-1]), sigma, dsigma, sigmap]

"""
    Plots the values of Sigmap (big sigma!) at the horizon as function of lambda for a given beta.

    Input arguments
        lower - The smallest value of lambda in the plot.
        higher - The biggest value of lambda in the plot.
        space - The space between grid points which will be used to plot the graph.
        beta - The beta values which will be used for computation.
        dbeta - The partial derivative acording to rho of beta which will be used for computation.
"""
def plotSigmapAsFunctionOfLambda(lower, higher, space, beta, dbeta):
    if lower > higher:
        lower, higher = higher, lower
    lamVec = arange(lower, higher, space)
    valVec = lamVec.copy()
    for k in range(0, lamVec.size):
        [valVec[k], sigma, dsigma, sigmap] = getSigmapAtHorizon(lamVec[k], beta, dbeta)
    plot(lamVec, valVec)
    show()

"""
    Finds the value of lambda which satisfy the boundary condition on Sigmap at the horizon using
    Steffensen's method.
    Because this function needs some arguments that have been computed before it was called,
    and because another computation of them is expensive, this function gets them as input
    arguments.

    Input arguments
        lam - The initial lambda value.
        horizon - The value of Sigmap at the horizon.
        beta - The beta values which will be used for computation.
        dbeta - The partial derivative acording to rho of beta which will be used for computation.
        sigma - The initial sigma values which was calculated using lam, beta and dbeta.
        dsigma - The partial derivative according to rho of initial sigma.
        sigmap - The initial sigmap values which was calculated using lam, beta and dbeta.

    Return
        The values of lambda, Sigmap at the horizon, sigma, dsigma, and sigmap.
        Another return value is a flag which states if a root was found.
        If this flag is true, the values above relate to the value of lambda which is the root.
        If this flag is false, the values above relate to the value of lambda which gave the minimal
        Sigma value at the horizon.
"""
def steffensen(lam, horizon, beta, dbeta, sigma, dsigma, sigmap):

    lamMin = lam
    hMin = horizon
    sigmaMin = sigma.copy()
    dsigmaMin = dsigma.copy()
    sigmapMin = sigmap.copy()

    for k in range(0, maxNumOfIterations):
        [temp, sigma, dsigma, sigmap] = getSigmapAtHorizon(lam + horizon, beta, dbeta)
        if temp == horizon:
            break
        lam -= (horizon**2 / (temp - horizon))
        [horizon, sigma, dsigma, sigmap] = getSigmapAtHorizon(lam, beta, dbeta)
        absH = abs(horizon)
        if absH < abs(hMin):
            lamMin = lam
            hMin = horizon
            sigmaMin = sigma.copy()
            dsigmaMin = dsigma.copy()
            sigmapMin = sigmap.copy()
            if absH < epsilon:
                return [lam, horizon, sigma, dsigma, sigmap, True]
    print("Steffensen method failed. Returning closest result.")
    return [lamMin, hMin, sigmaMin, dsigmaMin, sigmapMin, False]