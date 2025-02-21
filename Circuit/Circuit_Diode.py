# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 06:06:02 2025

Turki Almansoori
Rui (Richard) Chen Geng Li
Student number: 1003048454

Code created for PHY224 Fitting Exercise 3
Prof. Sergio De La Barrera
Ta: Weigeng Peng
Due February 23rd, 2025
"""

#Necessary modules
import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 


#Loading the dataset
source, volt, amp, volt_unc, amp_unc = np.loadtxt("circuit_diode_data.csv", delimiter = ',', skiprows=1, unpack=True)


#Prediction Models to be fitted
#x_val is the voltage, and the returned value is the current.
def power_model(x_val, a, b):
    return a*(x_val**b)

def linear_log_model(x_val, a, b):
    return b*np.log(x_val)+np.log(a) #Returns the log of the current


pow_popt, pow_pcov = curve_fit(power_model, volt, amp, sigma = amp_unc, absolute_sigma = True)
log_popt, log_pcov = curve_fit(linear_log_model, volt, np.log(amp), sigma = amp_unc /(amp*np.log(10)), absolute_sigma = True)


plt.plot(volt, amp)