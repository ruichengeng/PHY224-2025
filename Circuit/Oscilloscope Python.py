# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:12:12 2025

@author: ruich
"""

#Necessary modules
import numpy as np #Useful helper functions that can convert our data into formats acceptable to the curve fitting components.
from scipy.optimize import curve_fit #The main component of this fitting exercise
import matplotlib.pyplot as plt #Handles the plotting aspect of this lab

time, ch1_volt, ch2_volt, math_volt = np.loadtxt("scope_5.csv", delimiter = ',', skiprows=2, unpack=True)

plt.plot(time, ch1_volt, color="red")
plt.plot(time, ch2_volt, color = "orange")
plt.plot(time, math_volt, color = "blue")
plt.show()

pt2_up_time, pt2_up_volt, pt2_down_time, pt2_down_volt = np.loadtxt("circuit_pt2_data.csv", delimiter = ',', skiprows=2, unpack=True)




def volt_model_up(x_val, R, C):
    return pt2_up_volt[0]*np.exp((x_val)/(R*C))

def volt_model_down(x_val, R, C):
    return pt2_down_volt[0]*np.exp((-1*x_val)/(R*C))

up_popt, up_pcov = curve_fit(volt_model_up, pt2_up_time, pt2_up_volt, absolute_sigma = True)
down_popt, down_pcov = curve_fit(volt_model_down, pt2_down_time, pt2_down_volt, absolute_sigma = True)



plt.plot(pt2_up_time, pt2_up_volt, color = "orange")
plt.plot(pt2_down_time, pt2_down_volt, color = "cyan")
plt.plot(pt2_up_time, volt_model_up(pt2_up_time, up_popt[0], up_popt[1]), color = "red")
plt.plot(pt2_down_time, volt_model_down(pt2_down_time, down_popt[0], down_popt[1]), color = "blue")

plt.show()
