The Python code and final document of my 2015 Physics Bachelor project in the Technion - Israel Institute of Technology.
The goal of the project was to reproduce Fig. 4 of https://doi.org/10.1007/JHEP07(2014)086 (also available in https://arxiv.org/abs/1309.1439 or https://ar5iv.org/html/1309.1439).



Instructions for running the program
************************************

1. Get Python 3.x.
Python can be downloaded freely. One simple way to get it (without installation required) is to download it from
https://winpython.github.io/
Note 1: The program will only work on 3.x Python version. It was tested on 64-bit Python 3.3.5.0.
Note 2: If WinPython isn't being used, additional libraries should be installed, like numpy, matplotlib and/or others which the former depends on.
		The interpreter will prompt about missing libraries. Please use Google to see how to install them...

2. Put the code where the console/interpreter executable is.

3. Run the interpreter and type
			from main import *
From that moment you are familiar with the main function and all other functions and constants from the other modules.
To use it you can type something like (assumes that the user uses the code as it was provided)
			[lam1, beta1, dbeta1, sigma1, dsigma1, sigmap1, betap1, alpha1, plotVec] = main(30, i.lam0, i.beta0(), i.dbeta0())
This will initiate the program, using the initial lambda and beta functions which are defined in initial.py module. The simulation target time is 30 time units.

4. Creating the graph of the first quasi-normal mode from the PDF.
This part assumes that the user uses the code as it was provided. Type the following lines

from main import *
[lam1, beta1, dbeta1, sigma1, dsigma1, sigmap1, betap1, alpha1, plotVec] = main(30, i.lam0, i.beta0(), i.dbeta0())
t = plotVec[0,:]
ba = plotVec[1,:]
from numpy import sin
import matplotlib.pyplot as plt
plt.plot(t, ba * i.exp(t * 2.66385), 'b-', lw=3, label='y = e^(2.66385 * t) * b_3(t) / a_1(t)')
plt.plot(t, 10 * sin(t * 1.84942 + 2.53), 'r--', linewidth=3, label='y = 10 * sin(t * 1.84942 + 2.53)')
legend = plt.legend(loc = 'upper right')
show()

and the graph should appear.
