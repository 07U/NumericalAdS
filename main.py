"""
    Module for the main program.
"""

import initial as i
import solvers as s
import lambdaCorrection as l
import odeIntegration as o
from numpy import zeros
from numpy import linspace
from numpy import floor
#from numpy import meshgrid
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.pyplot import figure
from matplotlib.pyplot import plot
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import show
#from matplotlib import cm

"""
    Some constants.
"""
# If Sigmap value at horizon is bigger than this, a lambda correction proccess will be initialized.
correctionEpsilon = 10**(-10)

# If the target time of the simulation is big, we may want to not plot for each time step.
#reducer = 250

"""
    Gets initial conditions and calculate sigma, sigmap, betap and alpha values in a given time.
    In comment are some expressions which can be used for generating somr plots.

    Input arguments
        targetTime - The simulation will run until this time.
        lam - The initial value (or guess) of lambda.
        beta - The initial beta values.
        dbeta - The derivative according to rho of initial beta.

    Return
        The values of all functions at the given target time.
        It can return any of the vectors/matrices which used for ploting.
"""
def main(targetTime, lam, beta, dbeta):
    iterations = floor(targetTime / o.dt)
    plotVec = zeros((2,iterations))     # For generating a 2D plot.
    #plotMat = zeros((i.rho.size, iterations / reducer))    # For generating a 3D plot.
    #fig = figure()     # For generating a 3D plot.
    #ax = Axes3D(fig)   # For generating a 3D plot.
    it = 0
    time = linspace(0, targetTime, iterations)
    #plotTime = linspace(0, targetTime, iterations / reducer)   # For generating a 3D plot.
    out_lam = lam
    out_beta = beta
    out_dbeta = dbeta
    out_sigma = 0
    out_dsigma = 0
    out_sigmap = 0
    out_betap = 0
    out_alpha = 0
    print('Starting run with targetTime = ', targetTime, end = '')
    print(', dt = ', o.dt, end = '')
    print(', epsilon = ', l.epsilon, end = '')
    print(' and correctionEpsilon = ', correctionEpsilon)
    for t in time:
        [horizon, sigma, dsigma, sigmap] = l.getSigmapAtHorizon(lam, beta, dbeta)
        if abs(horizon) > correctionEpsilon:
            [lam, horizon, sigma, dsigma, sigmap, succeeded] = \
                                                  l.steffensen(lam, horizon, beta, dbeta, sigma, dsigma, sigmap)
            if(not succeeded):
                print('Can\'t find lambda. Sigmap(rh,', t, end = '')
                print(') = ', horizon, end = '')
                print('. Returning intermediate values.')
                break
        plotVec[0,it] = t   # For generating a 2D plot.
        plotVec[1,it] = beta[0] / i.a_1     # For generating a 2D plot.
        #if it % reducer == 0:      # For generating a 3D plot.
        #    plotMat[:,it / reducer] = beta.copy()      # For generating a 3D plot.
        betap = s.betapDivSolve(lam, beta, dbeta, sigma, dsigma, sigmap)
        alpha = s.alphaSolve(lam, beta, dbeta, sigma, dsigma, sigmap, betap)
        out_lam = lam
        out_beta = beta
        out_dbeta = dbeta
        out_sigma = sigma
        out_dsigma = dsigma
        out_sigmap = sigmap
        out_betap = betap
        out_alpha = alpha
        #[lam, beta, dbeta] = o.RungeKutte4(lam, beta, dbeta, betap, alpha)
        #[lam, beta, dbeta] = o.AdamsBashforth4(lam, beta, dbeta, betap, alpha)
        [lam, beta, dbeta] = o.AdamsMoulton3(lam, beta, dbeta, betap, alpha)
        it += 1
    plot(plotVec[0,:], plotVec[1,:])
    #plotTime, rho = meshgrid(plotTime, i.rho)      # For generating a 3D plot.
    #ax.plot_wireframe(plotTime, rho, plotMat, rstride=1, cstride=1)    # For generating a 3D plot.
    #fig.colorbar(surf, shrink=0.5, aspect=5)   # For generating a 3D plot.
    show()
    return [out_lam, out_beta, out_dbeta, out_sigma, out_dsigma, out_sigmap, out_betap, out_alpha, plotVec]