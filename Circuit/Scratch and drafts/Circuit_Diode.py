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
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 


#Loading the dataset
source, volt, amp, volt_unc, amp_unc = np.loadtxt("circuit_diode_data.csv", delimiter = ',', skiprows=2, unpack=True)



def exponential_model(x_val, a, b, c, d):
    return a*(b**(c*x_val+d))

def shockley_model(x_val, a):
    return -1*amp[0]*(np.exp(x_val*a)-1.0)



# exp_popt, exp_pcov = curve_fit(exponential_model, forward_volt, forward_amp, p0 = (0.6353310698, 1031.75089, 3.723623702, -2.268601667), sigma=forward_amp_unc, absolute_sigma = True, maxfev = 10000)
exp_popt, exp_pcov = curve_fit(exponential_model, volt, amp, p0 = (0.6353310698, 1031.75089, 3.723623702, -2.268601667), sigma=amp_unc, absolute_sigma = True, maxfev = 10000)

shock_popt, shock_pcov = curve_fit(shockley_model, volt, amp, p0=(20.473), sigma=amp_unc, absolute_sigma = True, maxfev = 100000)

#Parameters uncertainties
exp_pcov = np.diag(exp_pcov)
shock_pcov = np.diag(shock_pcov)

print("Exponential a = ", exp_popt[0], r' \pm ', np.sqrt(exp_pcov[0]))
print("Exponential b = ", exp_popt[1], r' \pm ', np.sqrt(exp_pcov[1]))
print("Exponential c = ", exp_popt[2], r' \pm ', np.sqrt(exp_pcov[2]))
print("Exponential d = ", exp_popt[3], r' \pm ', np.sqrt(exp_pcov[3]))

print("Shockley a = ", shock_popt[0], r' \pm ', np.sqrt(shock_pcov[0]))


plt.errorbar(volt, amp, xerr=volt_unc, yerr=amp_unc, fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)
plt.plot(volt, exponential_model(volt, exp_popt[0], exp_popt[1], exp_popt[2], exp_popt[3]), label="Positive Exp Model Fitting", color="blue", linewidth = 1)

plt.plot(volt, shockley_model(volt, shock_popt[0]), label="Shockley Model Fitting", color="green", linewidth = 1)
plt.legend()
plt.show()


exp_model_data = exponential_model(volt, exp_popt[0], exp_popt[1], exp_popt[2], exp_popt[3])
shockley_model_data = shockley_model(volt, shock_popt[0])


chi2_exp=np.sum( (amp - exp_model_data)**2 / amp_unc**2 )
reduced_chi2_exp = chi2_exp/(volt.size - exp_popt.size)


chi2_shock=np.sum( (amp - shockley_model_data)**2 / amp_unc**2 )
reduced_chi2_shock = chi2_shock/(volt.size - shock_popt.size)


print("Reduced Chi-Squared Values:")
print("Exponential Model: ", reduced_chi2_exp)
print("Shockley Model: ", reduced_chi2_shock)


#Thermal Voltage Calculation
thermal_volt_lower = 1.0/(shock_popt[0])
thermal_volt_higher = 1.0/(2.0*shock_popt[0])

print("Given that the ideality factor η varies between 1 and 2, then our thermal voltage varies between: ", thermal_volt_lower*1000.0, "mV and ", thermal_volt_higher*1000.0, "mV")
print("The leakage current is approximately: ", -1.0*amp[0], "mA or approximately: ", -1000.0*amp[0], "μA")