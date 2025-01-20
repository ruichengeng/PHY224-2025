# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:14:21 2025

Created by Rui (Richard) Chen Geng Li
Student number: 1003048454

Code created for PHY224 Fitting Exercise 1
Due Sunday January 20th, 2025
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Imports the data from file named co2_annmean_mlo for the annual average of daily measurement of atmospheric co2 from 1960 to 2023 from the top of Mauna Loa in Hawaii.
#Below: We skips 44 rows so that the first data entry is from 1959 aka the first row containing data
#We also separated the data category based on separation character ',' 
#Then we unpacked these into different arrays of data according to the type (i.e. years separated from co2 level separated from the uncertainty)
year, mean, unc = np.loadtxt("co2_annmean_mlo.csv", delimiter = ',', skiprows=44, unpack=True)


#Here we define the different types of fitting models, the independent variable is x_val
#We will use curve_fit function later to estimate/fit the values of the coefficients A, B, and C where they are applicable

#Linear fitting model using the same style as the formula f(x)=ax+b
def linear_model(x_val, A, B):
    return A*x_val+B

#Power fitting models:

#Quadratic fitting model using the same style as f(x)=ax^2+bx+c
def quadratic_model(x_val, A, B, C):
    return A*x_val**2+B*x_val+C


#Note for both power_model and exponential_model we encountered the runtime error:
#RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800.
#To fix this, we introduced the term -1959 to x_val to fit the model aka x_val[0]=1959-1959 = 0 for the computer to be able to handle the simulation.

#Power fitting model using the same style as f(x)=ax^b+c
def power_model(x_val, A, B, C):
    return A*(x_val-1959)**B+C
    
#Exponential fitting model using the same style as f(x)=ae^(bx)+c
def exponential_model(x_val, A, B, C):
    return A*np.exp(B*(x_val-1959))+C





#################################
plt.plot(year, mean, label='Data')
#yerr_up= mean+unc
#yerr_down= mean-unc
#yerr=[-1*unc, unc]
unc2=unc*50
plt.errorbar(year, mean, yerr=unc, capsize=1, fmt = 'none', ecolor = "blue")

lin_year=year[-20:]
print(lin_year)

popt, pcov = curve_fit(power_model, year, mean)
popt2, pcov2 = curve_fit(exponential_model, year, mean, p0=(1, 1e-6, 200))

plt.plot(year, power_model(year, popt[0], popt[1], popt[2]), label = "Quadratic Curve Fit")

plt.plot(year, exponential_model(year, popt2[0], popt2[1], popt2[2]), label = "Exponential Curve Fit")
plt.xlabel("x values")
plt.ylabel("y values")
plt.legend()
plt.show()

print ("A value:", popt2[0])
print ("B value:", popt2[1])
print ("C value:", popt2[2])

print (power_model(2060, popt[0], popt[1], popt[2]))
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