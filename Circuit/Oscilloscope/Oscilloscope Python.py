# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:12:12 2025

@author: ruich
"""

#Necessary modules
import numpy as np #Useful helper functions that can convert our data into formats acceptable to the curve fitting components.
from scipy.optimize import curve_fit #The main component of this fitting exercise
import matplotlib.pyplot as plt #Handles the plotting aspect of this lab

time, ch1_volt, ch2_volt, math_volt, volt_unc = np.loadtxt("scope_4_2.csv", delimiter = ',', skiprows=2, unpack=True)

plt.plot(time, ch1_volt, color="red")
plt.plot(time, ch2_volt, color = "orange")
plt.plot(time, math_volt, color = "blue")
plt.show()

pt2_up_time, pt2_up_volt, pt2_down_time, pt2_down_volt, time_unc1, volt_unc1, time_unc2, volt_unc2 = np.loadtxt("circuit_pt2_data.csv", delimiter = ',', skiprows=2, unpack=True)




# def volt_model_up(x_val, R, C):
#     return pt2_up_volt[0]*np.exp((x_val-pt2_up_time[0])/(R*C))

# def volt_model_down(x_val, R, C):
#     return pt2_down_volt[0]*np.exp((-1*x_val)/(R*C))

# up_popt, up_pcov = curve_fit(volt_model_up, pt2_up_time, pt2_up_volt, sigma=volt_unc1, absolute_sigma = True)
# down_popt, down_pcov = curve_fit(volt_model_down, pt2_down_time, pt2_down_volt, sigma=volt_unc2, absolute_sigma = True)



# plt.plot(pt2_up_time, pt2_up_volt, color = "orange")
# plt.plot(pt2_down_time, pt2_down_volt, color = "cyan")
# plt.plot(pt2_up_time, volt_model_up(pt2_up_time, up_popt[0], up_popt[1]), color = "red")
# plt.plot(pt2_down_time, volt_model_down(pt2_down_time, down_popt[0], down_popt[1]), color = "blue")

# plt.show()


data_up_volt = np.array([])
data_down_volt = np.array([])
data_up_time = np.array([])
data_down_time = np.array([])
data_up_unc = np.array([])
data_down_unc = np.array([])

for i in range(len(time)):
    if time[i] <0.0:
        data_up_time = np.append(data_up_time, time[i])
        data_up_volt = np.append(data_up_volt, ch2_volt[i])
        data_up_unc = np.append(data_up_unc, volt_unc[i])
    elif time[i]>0.0:
        data_down_time = np.append(data_down_time, time[i])
        data_down_volt = np.append(data_down_volt, ch2_volt[i])
        data_down_unc = np.append(data_down_unc, volt_unc[i])

def volt_model_up(x_val, R, C):
    return data_up_volt[0]*np.exp(-1*(x_val-data_up_time[0])/(R*C))

def volt_model_down(x_val, R, C):
    return data_down_volt[0]*np.exp((-1*x_val)/(R*C))

up_popt, up_pcov = curve_fit(volt_model_up, data_up_time, data_up_volt, p0=(470, 2.2e-8), sigma=data_up_unc, absolute_sigma = True)
down_popt, down_pcov = curve_fit(volt_model_down, data_down_time, data_down_volt,p0=(470, 2.2e-8), sigma=data_down_unc, absolute_sigma = True)



plt.plot(data_up_time, data_up_volt, color = "orange")
plt.plot(data_down_time, data_down_volt, color = "cyan")
plt.plot(data_up_time, volt_model_up(data_up_time, up_popt[0], up_popt[1]), color = "red")
plt.plot(data_down_time, volt_model_down(data_down_time, down_popt[0], down_popt[1]), color = "blue")

plt.show()
