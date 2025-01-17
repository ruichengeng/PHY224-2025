# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:14:21 2025

@author: ruich
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Imports the data from file named co2_annmean_mlo for the annual average of daily measurement of atmospheric co2 from 1960 to 2023 from the top of Mauna Loa in Hawaii.
year, mean, unc = np.loadtxt("co2_annmean_mlo.csv", delimiter = ',', skiprows=44, unpack=True) #Skips 44 rows so that the first data entry is from 1959 aka the first row containing data

#plt.plot(year, mean)

#################################
plt.plot(year, mean, label='Data')
#yerr_up= mean+unc
#yerr_down= mean-unc
#yerr=[-1*unc, unc]
unc2=unc*50
plt.errorbar(year, mean, yerr=unc, capsize=1, fmt = 'none', ecolor = "blue")

def quad_quiz(x_val, A, B, C):
    return A*x_val**2+B*x_val+C

# def exponential_model(x_val, A, B, C):
#     return A*(x_val-1959)**B+C

def exponential_model(x_val, A, B, C):
    return A*np.exp(B*(x_val-1959))+C

popt, pcov = curve_fit(quad_quiz, year, mean)
popt2, pcov2 = curve_fit(exponential_model, year, mean, p0=(1, 1e-6, 200))

#plt.plot(year, quad_quiz(year, popt[0], popt[1], popt[2]), label = "Quadratic Curve Fit")

#plt.plot(year, exponential_model(year, popt2[0], popt2[1], popt2[2]), label = "Exponential Curve Fit")
plt.xlabel("x values")
plt.ylabel("y values")
plt.legend()
plt.show()

print ("A value:", popt2[0])
print ("B value:", popt2[1])
print ("C value:", popt2[2])

print (quad_quiz(2060, popt[0], popt[1], popt[2]))
print (exponential_model(2060, popt2[0], popt2[1], popt2[2]))

#perr = np.sqrt(np.diag(pcov))

#plt.plot(year, quad_quiz(year, popt[0], popt[1], popt[2]), label = "Quadratic Curve Fit")
#plt.xlabel("x values")
#plt.ylabel("y values")
#plt.legend()
#plt.show()

#print ("A value:", popt[0])
#print ("B value:", popt[1])
#print ("C value:", popt[2])

#print (quad_quiz(2060, popt[0], popt[1], popt[2]))