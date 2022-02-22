"""
    Module for the time evolution methods and ODEs.
"""

import initial as i
import solvers as s
from numpy import dot

"""
    Some constants.
"""
dt = 10**(-3)

rh2 = i.rh**2
rh4 = i.rh**4

lambdaDotConst1 = rh4 + s.twoRh * i.a_1
lambdaDotConst2 = 2 * (i.rh**3 - i.a_1)
lambdaDotConst3 = 1 / (6 * rh4)

betaDotConst1 = s.twoRh * s.rhoTilde3a_1 - 8 * rh4
betaDotConst2 = s.rhoTilde * (s.twoRh3 + s.rhoTilde3a_1)
betaDotConst3 = 2 * rh2 * s.rhoTilde2
betaDotConst4 = s.twoRh4 * s.rhoTilde[1:]

"""
    Computes the value of (d lambda/d t).

    Input arguments
        lam - The lambda value which was used for computation.
        betapH - The value of betap at the horizon.
        alphaH - The value of alpha at the horizon.

    Return
        The value of (d lambda/d t).
"""
def lambdaDotValue(lam, betapH, alphaH):
    return lambdaDotConst3 \
               * (3 * s.twoRh * alphaH + betapH**2 \
                 + 3 * rh2 \
                     * (lambdaDotConst1 + lam * (lambdaDotConst2 + rh2 * lam)))

"""
    Computes the value of (d lambda/d t).

    Input arguments
        lam - The lambda value which will be used for computation.
        beta - The beta values which will be used for computation.
        dbeta - The partial derivative acording to rho of beta which will be used for computation.

    Return
        The value of (d lambda/d t).
"""
def lambdaDot(lam, beta, dbeta):
    sigma = s.sigmaSolve(lam, beta, dbeta)
    dsigma = dot(i.D, sigma)
    sigmap = s.sigmapDivSolve(lam, sigma, dsigma)
    betap = s.betapDivSolve(lam, beta, dbeta, sigma, dsigma, sigmap)
    alpha = s.alphaSolve(lam, beta, dbeta, sigma, dsigma, sigmap, betap)
    return lambdaDotValue(lam, betap[-1], alpha[-1])

"""
    Computes the values of the partial derivative (d beta/d t).
    Because this function needs some arguments that have been computed before it was called,
    and because another computation of them is expensive, this function gets them as input
    arguments.

    Input arguments
        lam - The lambda value which was used for computation.
        beta - The beta values which will be used for computation.
        dbeta - The partial derivative acording to rho of beta which will be used for computation.
        betap - The betap values which were calculated using lam, beta and dbeta.
        alpha - The alpha values which were calculated using lam, beta and dbeta.

    Return
        The values of the partial derivative (d beta/d t).
"""
def betaDotValue(lam, beta, dbeta, betap, alpha):
    betaDot = beta.copy()
    betaDot[1:] = ((3 * beta + s.rhoTilde * dbeta) \
                      * (s.rhoTilde5 * alpha \
                        + s.twoRh \
                            * (betaDotConst1 \
                              + lam * (betaDotConst2 - betaDotConst3 * lam) \
                              + 2 * betaDotConst3 \
                                  * lambdaDotValue(lam, betap[-1], alpha[-1]))) \
                    - s.twoRh5 * betap)[1:] \
                        / betaDotConst4
    dbetap = dot(i.D, betap)
    betaDot[0] = 3 * beta[0] * lam \
                 - s.twoRh * (2 * dbeta[0] + dbetap[0])
    return betaDot

"""
    Computes the values of the partial derivative (d beta/d t).

    Input arguments
        lam - The lambda value which will be used for computation.
        beta - The beta values which will be used for computation.
        dbeta - The partial derivative acording to rho of beta which will be used for computation.

    Return
        The values of the partial derivative (d beta/d t).
"""
def betaDot(lam, beta, dbeta):
    sigma = s.sigmaSolve(lam, beta, dbeta)
    dsigma = dot(i.D, sigma)
    sigmap = s.sigmapDivSolve(lam, sigma, dsigma)
    betap = s.betapDivSolve(lam, beta, dbeta, sigma, dsigma, sigmap)
    alpha = s.alphaSolve(lam, beta, dbeta, sigma, dsigma, sigmap, betap)
    return betaDotValue(lam, beta, dbeta, betap, alpha)

"""
    Computes the next step values of lambda and beta using fourth-order Runge-Kutta method.
    Because this function needs some arguments that have been computed before it was called,
    and because another computation of them is expensive, this function gets them as input
    arguments.

    Input arguments
        lam - The lambda value which was used for computation.
        beta - The beta values which will be used for computation.
        dbeta - The partial derivative acording to rho of beta which will be used for computation.
        betap - The betap values which were calculated using lam, beta and dbeta.
        alpha - The alpha values which were calculated using lam, beta and dbeta.

    Return
        The next step values of lambda and beta.
"""
def RungeKutte4(lam, beta, dbeta, betap, alpha):
    l1 = dt * lambdaDotValue(lam, betap[-1], alpha[-1])
    b1 = dt * betaDotValue(lam, beta, dbeta, betap, alpha)
    tempLam = lam + l1 / 2
    tempBeta = beta + b1 / 2
    tempDbeta = dot(i.D, tempBeta)
    l2 = dt * lambdaDot(tempLam, tempBeta, tempDbeta)
    b2 = dt * betaDot(tempLam, tempBeta, tempDbeta)
    tempLam = lam + l2 / 2
    tempBeta = beta + b2 / 2
    tempDbeta = dot(i.D, tempBeta)
    l3 = dt * lambdaDot(tempLam, tempBeta, tempDbeta)
    b3 = dt * betaDot(tempLam, tempBeta, tempDbeta)
    tempLam = lam + l3
    tempBeta = beta + b3
    tempDbeta = dot(i.D, tempBeta)
    l4 = dt * lambdaDot(tempLam, tempBeta, tempDbeta)
    b4 = dt * betaDot(tempLam, tempBeta, tempDbeta)
    lamOut = lam + (l1 + 2 * (l2 + l3) + l4) / 6
    betaOut = beta + (b1 + 2 * (b2 + b3) + b4) / 6
    dbetaOut = dot(i.D, betaOut)
    return [lamOut, betaOut, dbetaOut]

"""
    Computes the next step values of lambda and beta using four-step Adams-Bashforth method
    which is initialized with fourth-order Runge-Kutta method.
    Because this function needs some arguments that have been computed before it was called,
    and because another computation of them is expensive, this function gets them as input
    arguments.

    WARNING:
        This function uses static variables. This means that it can be used only once. To use it again,
        it is needed to create another Python console session and import the module again.

    Input arguments
        lam - The lambda value which was used for computation.
        beta - The beta values which will be used for computation.
        dbeta - The partial derivative acording to rho of beta which will be used for computation.
        betap - The betap values which were calculated using lam, beta and dbeta.
        alpha - The alpha values which were calculated using lam, beta and dbeta.

    Return
        The next step values of lambda and beta.
"""
def AdamsBashforth4(lam, beta, dbeta, betap, alpha):
    AdamsBashforth4.lambdaDot[-3] = AdamsBashforth4.lambdaDot[-2]
    AdamsBashforth4.lambdaDot[-2] = AdamsBashforth4.lambdaDot[-1]
    AdamsBashforth4.lambdaDot[-1] = AdamsBashforth4.lambdaDot[0]
    AdamsBashforth4.lambdaDot[0] = lambdaDotValue(lam, betap[-1], alpha[-1])
    AdamsBashforth4.betaDot[-3] = AdamsBashforth4.betaDot[-2]
    AdamsBashforth4.betaDot[-2] = AdamsBashforth4.betaDot[-1]
    AdamsBashforth4.betaDot[-1] = AdamsBashforth4.betaDot[0]
    AdamsBashforth4.betaDot[0] = betaDotValue(lam, beta, dbeta, betap, alpha)
    if(AdamsBashforth4.counter < 3):
        [lamOut, betaOut, dbetaOut] = RungeKutte4(lam, beta, dbeta, betap, alpha)
        AdamsBashforth4.counter += 1
    else:
        lamOut = lam \
                 + (dt / 24) \
                     * (55 * AdamsBashforth4.lambdaDot[0] \
                       - 59 * AdamsBashforth4.lambdaDot[-1] \
                       + 37 * AdamsBashforth4.lambdaDot[-2] \
                       - 9 * AdamsBashforth4.lambdaDot[-3])
        betaOut = beta \
                  + (dt / 24) \
                      * (55 * AdamsBashforth4.betaDot[0] \
                        - 59 * AdamsBashforth4.betaDot[-1] \
                        + 37 * AdamsBashforth4.betaDot[-2] \
                        - 9 * AdamsBashforth4.betaDot[-3])
        dbetaOut = dot(i.D, betaOut)
    return [lamOut, betaOut, dbetaOut]
# Static variables. See warning in the function description.
AdamsBashforth4.counter = 0
AdamsBashforth4.lambdaDot = [0, 0, 0, 0]
AdamsBashforth4.betaDot = [0, 0, 0, 0]

"""
    Computes the next step values of lambda and beta using a Predictor-Corrector method, in
    which the four-step Adams-Bashforth is used as a predictor and the three-step Adams-Moulton
    method is used as a corrector. The proccess is initialized with fourth-order Runge-Kutta
    method.
    Because this function needs some arguments that have been computed before it was called,
    and because another computation of them is expensive, this function gets them as input
    arguments.

    WARNING:
        This function uses static variables. This means that it can be used only once. To use it again,
        it is needed to create another Python console session and import the module again.

    Input arguments
        lam - The lambda value which was used for computation.
        beta - The beta values which will be used for computation.
        dbeta - The partial derivative acording to rho of beta which will be used for computation.
        betap - The betap values which were calculated using lam, beta and dbeta.
        alpha - The alpha values which were calculated using lam, beta and dbeta.

    Return
        The next step values of lambda and beta.
"""
def AdamsMoulton3(lam, beta, dbeta, betap, alpha):
    AdamsMoulton3.lambdaDot[-3] = AdamsMoulton3.lambdaDot[-2]
    AdamsMoulton3.lambdaDot[-2] = AdamsMoulton3.lambdaDot[-1]
    AdamsMoulton3.lambdaDot[-1] = AdamsMoulton3.lambdaDot[0]
    AdamsMoulton3.lambdaDot[0] = lambdaDotValue(lam, betap[-1], alpha[-1])
    AdamsMoulton3.betaDot[-3] = AdamsMoulton3.betaDot[-2]
    AdamsMoulton3.betaDot[-2] = AdamsMoulton3.betaDot[-1]
    AdamsMoulton3.betaDot[-1] = AdamsMoulton3.betaDot[0]
    AdamsMoulton3.betaDot[0] = betaDotValue(lam, beta, dbeta, betap, alpha)
    if(AdamsMoulton3.counter < 3):
        [lamOut, betaOut, dbetaOut] = RungeKutte4(lam, beta, dbeta, betap, alpha)
        AdamsMoulton3.counter += 1
    else:
        tempLam = lam \
                  + (dt / 24) \
                      * (55 * AdamsMoulton3.lambdaDot[0] \
                        - 59 * AdamsMoulton3.lambdaDot[-1] \
                        + 37 * AdamsMoulton3.lambdaDot[-2] \
                        - 9 * AdamsMoulton3.lambdaDot[-3])
        tampBeta = beta \
                   + (dt / 24) \
                       * (55 * AdamsMoulton3.betaDot[0] \
                         - 59 * AdamsMoulton3.betaDot[-1] \
                         + 37 * AdamsMoulton3.betaDot[-2] \
                         - 9 * AdamsMoulton3.betaDot[-3])
        tempDbeta = dot(i.D, tampBeta)
        lambdaDot_plus1 = lambdaDot(tempLam, tampBeta, tempDbeta)
        betaDot_plus1 = betaDot(tempLam, tampBeta, tempDbeta)
        lamOut = lam \
                 + (dt / 24) \
                     * (9 * lambdaDot_plus1 \
                       + 19 * AdamsMoulton3.lambdaDot[0] \
                       - 5 * AdamsMoulton3.lambdaDot[-1] \
                       + AdamsMoulton3.lambdaDot[-2])
        betaOut = beta \
                  + (dt / 24) \
                      * (9 * betaDot_plus1 \
                        + 19 * AdamsMoulton3.betaDot[0] \
                        - 5 * AdamsMoulton3.betaDot[-1] \
                        + AdamsMoulton3.betaDot[-2])
        dbetaOut = dot(i.D, betaOut)
    return [lamOut, betaOut, dbetaOut]
# Static variables. See warning in the function description.
AdamsMoulton3.counter = 0
AdamsMoulton3.lambdaDot = [0, 0, 0, 0]
AdamsMoulton3.betaDot = [0, 0, 0, 0]