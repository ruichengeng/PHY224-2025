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

forward_volt = np.array([])
forward_amp = np.array([])
forward_volt_unc = np.array([])
forward_amp_unc = np.array([])

for f in range(len(volt)):
    if volt[f] >=0.1:
        forward_volt = np.append(forward_volt, volt[f])
        forward_amp = np.append(forward_amp, amp[f])
        forward_volt_unc = np.append(forward_volt_unc, volt_unc[f])
        forward_amp_unc = np.append(forward_amp_unc, amp_unc[f])


#Prediction Models to be fitted
#x_val is the voltage, and the returned value is the current.
def power_model(x_val, a, b):
    return a*(x_val**b)

def linear_log_model(x_val, a, b):
    return b*np.log(x_val)+np.log(a) #Returns the log of the current

def exponential_model(x_val, a, b, c, d):
    return a*(b**(c*x_val+d))

def shockley_model(x_val, a, b, c):
    return a*(np.exp(b*x_val)-c)

# exp_popt, exp_pcov = curve_fit(exponential_model, forward_volt, forward_amp, p0 = (0.6353310698, 1031.75089, 3.723623702, -2.268601667), sigma=forward_amp_unc, absolute_sigma = True, maxfev = 10000)
exp_popt, exp_pcov = curve_fit(exponential_model, volt, amp, p0 = (0.6353310698, 1031.75089, 3.723623702, -2.268601667), sigma=amp_unc, absolute_sigma = True, maxfev = 10000)

shock_popt, shock_pcov = curve_fit(shockley_model, volt, amp, p0 = (0.6353310698, 25.0, 0.1), sigma=amp_unc, absolute_sigma = True, maxfev = 10000)



# pow_popt, pow_pcov = curve_fit(power_model, volt, amp, sigma = amp_unc, absolute_sigma = True)
log_amp_unc = forward_amp_unc /(forward_amp*np.log(10))
log_volt_unc = forward_volt_unc/(forward_volt*np.log(10))
log_popt, log_pcov = curve_fit(linear_log_model, forward_volt, np.log(forward_amp), sigma = log_amp_unc, absolute_sigma = True)


# plt.errorbar(forward_volt, forward_amp, xerr=log_volt_unc, yerr=log_amp_unc, fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)

#plt.plot(forward_volt, exponential_model(forward_volt, log_popt[0], log_popt[1]), label="Positive Log Model Fitting", color="blue", linewidth = 1)
#plt.plot(forward_volt, np.exp(linear_log_model(forward_volt, log_popt[0], log_popt[1])), label="Positive Log Model Fitting", color="blue", linewidth = 1)

plt.errorbar(volt, amp, xerr=volt_unc, yerr=amp_unc, fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)
exp_volt=np.arange(-1.5, 0.9, 0.05)
plt.plot(exp_volt, exponential_model(exp_volt, exp_popt[0], exp_popt[1], exp_popt[2], exp_popt[3]), label="Positive Exp Model Fitting", color="blue", linewidth = 1)
plt.xticks(np.arange(-1.5,1.0, step=0.25))





plt.plot(exp_volt, shockley_model(exp_volt, shock_popt[0], shock_popt[1], shock_popt[2]), label="Shockley Model Fitting", color="green", linewidth = 1)



# plt.xscale('log')
# plt.yscale('log')
# plt.xticks(np.arange(0.0,1.0, step=0.1))
plt.legend()
plt.show()
