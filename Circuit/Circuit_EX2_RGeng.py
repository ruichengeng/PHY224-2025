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


source, volt, amp = np.loadtxt("Circuit_Lab_Ex2_Data.csv", delimiter = ',', skiprows=1, unpack=True)


def power_model(x_val, a, b):#x_val is the voltage
    return a*(x_val**b)#Returns the current

def linear_log_model(x_val, a, b):
    return b*np.log(x_val)+np.log(a) #Returns the log of the current

popt, pcov = curve_fit(power_model, volt, amp)

plt.plot(volt, amp, color = "blue")
plt.plot(volt, power_model(volt, popt[0], popt[1]), color = "red")