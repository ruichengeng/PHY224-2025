# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:23:56 2025

Created by Rui (Richard) Chen Geng Li
Student number: 1003048454

Code created for PHY224 Circuit Exercise 2
Prof. Sergio De La Barrera
Ta: Weigeng Peng
Due February 16th, 2025
"""

#Necessary modules
import numpy as np #Useful helper functions that can convert our data into formats acceptable to the curve fitting components.
from scipy.optimize import curve_fit #The main component of this fitting exercise
import matplotlib.pyplot as plt #Handles the plotting aspect of this lab


source, volt, amp, volt_unc, amp_unc = np.loadtxt("Circuit_Lab_Ex2_Data.csv", delimiter = ',', skiprows=1, unpack=True)

#amp=amp*0.001

def power_model(x_val, a, b):#x_val is the voltage
    return a*(x_val**b)#Returns the current

def linear_log_model(x_val, a, b):
    return b*np.log(x_val)+np.log(a) #Returns the log of the current

def ideal_model(x_val, a):
    return a*x_val**(3.0/5.0)

def abc_model(x_val, a, b, c, d):
    return a*x_val**b+c*x_val**d

# popt, pcov = curve_fit(power_model, volt, amp, p0=(8.0, 0.56), sigma=amp_unc, absolute_sigma=True, maxfev=10000)

# plt.plot(volt, amp, color = "blue")
# plt.plot(volt, power_model(volt, popt[0], popt[1]), color = "red")



popt, pcov = curve_fit(abc_model, volt, amp, sigma=amp_unc, absolute_sigma=True, maxfev=10000)

plt.plot(volt, amp, color = "blue")
plt.plot(volt, abc_model(volt, popt[0], popt[1], popt[2], popt[3]), color = "red")



# popt, pcov = curve_fit(ideal_model, volt, amp, p0=8.5, sigma=amp_unc, absolute_sigma=True)

# plt.plot(volt, amp, color = "blue")
# plt.plot(volt, ideal_model(volt, popt[0]), color = "red")

# popt, pcov = curve_fit(linear_log_model, volt, np.log(amp))#, sigma=amp_unc, absolute_sigma=True)

# plt.plot(volt, amp, color = "blue")
# plt.plot(np.log(volt), linear_log_model(volt, popt[0], popt[1]), color = "red")
# plt.xcale('log')
# plt.yscale('log')