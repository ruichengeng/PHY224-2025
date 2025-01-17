# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:15:01 2025

@author: ruich
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x=np.array([0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.80])
y=np.array([-3.46, -4.78, -6.32, -6.20, -8.60, -11.57, -12.27, -15.70, -16.71])

plt.plot(x,y, marker='o', label='Data')

def quad_quiz(x_val, A, B, C):
    return A*x_val**2+B*x_val+C

popt, pcov = curve_fit(quad_quiz, x, y)
#perr = np.sqrt(np.diag(pcov))

plt.plot(x, quad_quiz(x, popt[0], popt[1], popt[2]), marker = 's', label = "Quadratic Curve Fit")
plt.xlabel("x values")
plt.ylabel("y values")
plt.legend()
plt.show()

print ("A value:", popt[0])
print ("B value:", popt[1])
print ("C value:", popt[2])