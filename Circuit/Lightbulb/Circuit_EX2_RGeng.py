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


#Reading of the data file
source, volt, amp, volt_unc, amp_unc = np.loadtxt("Circuit_Lab_Ex2_Data.csv", delimiter = ',', skiprows=1, unpack=True)

amp=amp*0.001
amp_unc = amp_unc*0.001

#Data sorting
pos_source = np.array([])
pos_volt = np.array([])
pos_amp = np.array([])
pos_volt_unc = np.array([])
pos_amp_unc = np.array([])

neg_source = np.array([])
neg_volt = np.array([])
neg_amp = np.array([])
neg_volt_unc = np.array([])
neg_amp_unc = np.array([])

#Data sorting gets rid of the 0V entry, as it gave curve fitting runtime error for all models.
#We are separating the values based on positive versus negative voltages, and these will be fitted separately.
for s in range(len(source)):
    if source[s] <0.0:
        neg_source = np.append(neg_source, source[s])
        neg_volt = np.append(neg_volt, volt[s])
        neg_amp = np.append(neg_amp, amp[s])
        neg_volt_unc = np.append(neg_volt_unc, volt_unc[s])
        neg_amp_unc = np.append(neg_amp_unc, amp_unc[s])
    elif source[s] >0.0:
        pos_source = np.append(pos_source, source[s])
        pos_volt = np.append(pos_volt, volt[s])
        pos_amp = np.append(pos_amp, amp[s])
        pos_volt_unc = np.append(pos_volt_unc, volt_unc[s])
        pos_amp_unc = np.append(pos_amp_unc, amp_unc[s])


#Prediction Models to be fitted
#x_val is the voltage, and the returned value is the current.
def power_model(x_val, a, b):
    return a*(x_val**b)

def linear_log_model(x_val, a, b):
    return b*np.log(x_val)+np.log(a) #Returns the log of the current

def ideal_model(x_val, a):
    return a*x_val**(3.0/5.0)


#Calculation of the new uncertainty for the logarithmic model, via uncertainty propagation.
pos_log_volt_unc = pos_volt_unc /(pos_volt*np.log(10))
pos_log_amp_unc = pos_amp_unc /(pos_amp*np.log(10))
neg_log_volt_unc = -1*neg_volt_unc /(neg_volt*np.log(10))
neg_log_amp_unc = -1*neg_amp_unc /(neg_amp*np.log(10))


#Curve Fitting for each of the models.
#Note: for the negative fitting, we will be imputting a positive value so that we don't encounter runtime error.
ppow_popt, ppow_pcov = curve_fit(power_model, pos_volt, pos_amp, p0=(8.5, 0.6), sigma=pos_amp_unc, absolute_sigma=True)
npow_popt, npow_pcov = curve_fit(power_model, -1*neg_volt, -1*neg_amp, p0=(8.5, 0.6), sigma=neg_amp_unc, absolute_sigma=True)

plog_popt, plog_pcov = curve_fit(linear_log_model, pos_volt, np.log(pos_amp), sigma=pos_log_amp_unc, absolute_sigma=True)
nlog_popt, nlog_pcov = curve_fit(linear_log_model, -1*neg_volt, np.log(-1*neg_amp), sigma=neg_log_amp_unc, absolute_sigma=True)

pideal_popt, pideal_pcov = curve_fit(ideal_model, pos_volt, pos_amp, sigma=pos_amp_unc, absolute_sigma=True)
nideal_popt, nideal_pcov = curve_fit(ideal_model, -1*neg_volt, -1*neg_amp, sigma=neg_amp_unc, absolute_sigma=True)


#Model Uncertainties
#Power Model Unc
model_unc = np.sqrt((pos_volt**ppow_popt[1]*np.sqrt(ppow_pcov[0][0]))**2 +(ppow_popt[0]*pos_volt**ppow_popt[1]*np.log(pos_volt)*np.sqrt(ppow_pcov[1][1]))**2)

#Predicted Model Data
ppow_model_data = power_model(pos_volt, ppow_popt[0], ppow_popt[1])
npow_model_data = -1*power_model(-1*neg_volt, npow_popt[0], npow_popt[1])

ppow_chi2=np.sum( (pos_amp - ppow_model_data)**2 / model_unc**2 )
#npow_chi2=np.sum( (neg_amp - npow_model_data)**2 / neg_amp_unc**2 )
# pow_reduced_chi2 = (ppow_chi2+npow_chi2)/((pos_amp.size+neg_amp.size) - 2)
# print("Reduced chi2 power = ", pow_reduced_chi2)
ppow_reduced_chi2 = ppow_chi2/(len(pos_amp) - len(ppow_popt))
#npow_reduced_chi2 = npow_chi2/(neg_amp.size - len(npow_popt))


#Plotting the 
plt.errorbar(pos_volt, pos_amp, xerr=pos_volt_unc, yerr=pos_amp_unc, fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)
plt.errorbar(neg_volt, neg_amp, xerr=neg_volt_unc, yerr=neg_amp_unc, fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)

plt.plot(pos_volt, power_model(pos_volt, ppow_popt[0], ppow_popt[1]), label="Power Model Fitting", color="green", linewidth = 1)
plt.plot(neg_volt, -1*power_model(-1*neg_volt, npow_popt[0], npow_popt[1]), label="Power Model Fitting", color="green", linewidth = 1)

plt.plot(pos_volt, ideal_model(pos_volt, pideal_popt[0]), label="Ideal Model Fitting", color="blue", linewidth = 1)
plt.plot(neg_volt, -1*ideal_model(-1*neg_volt, nideal_popt[0]), label="Ideal Model Fitting", color="blue", linewidth = 1)

plt.legend()
plt.show()
#plt.subplot(2,1,2)



# plt.errorbar(np.log(pos_volt), np.log(pos_amp), xerr=np.abs(np.log(pos_volt_unc)), yerr=np.abs(np.log(pos_amp_unc)), fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)
# plt.errorbar(np.log(-1*neg_volt), np.log(-1*neg_amp), xerr=np.abs(np.log(neg_volt_unc)), yerr=np.abs(np.log(neg_amp_unc)), fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)
plt.errorbar(pos_volt, pos_amp, xerr=pos_log_volt_unc, yerr=pos_log_amp_unc, fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)
plt.errorbar(-1*neg_volt,-1*neg_amp, xerr=neg_log_volt_unc, yerr=neg_log_amp_unc, fmt='o', ecolor="red", label="Measured Data", marker = ".", color = "red", markersize = 5)

plt.plot(pos_volt, np.exp(linear_log_model(pos_volt, plog_popt[0], plog_popt[1])), label="Positive Log Model Fitting", color="blue", linewidth = 1)
plt.plot(-1*neg_volt, np.exp(linear_log_model(-1*neg_volt, nlog_popt[0], nlog_popt[1])), label="Negative Log Model Fitting", color="blue", linewidth = 1)


plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()






















# Function to compute reduced chi-squared
def reduced_chi_squared(observed, expected, uncertainty, num_params):
    chi2 = np.sum((observed - expected)**2 / uncertainty**2)
    dof = len(observed) - num_params  # Degrees of freedom
    return chi2, chi2 / dof

# --- Power Model Uncertainty ---
model_unc_power = np.sqrt(
    (pos_volt**ppow_popt[1] * np.sqrt(ppow_pcov[0][0]))**2 +
    (ppow_popt[0] * pos_volt**ppow_popt[1] * np.log(pos_volt) * np.sqrt(ppow_pcov[1][1]))**2
)

# --- Log Model Uncertainty (Propagation) ---
log_amp_uncertainty = pos_amp_unc / (pos_amp * np.log(10))

# --- Ideal Model Uncertainty ---
model_unc_ideal = np.abs( (3/5) * pideal_popt[0] * pos_volt**(-2/5) * pos_volt_unc )

# --- Compute Model Predictions ---
ppow_model_data = power_model(pos_volt, *ppow_popt)
plog_model_data = np.exp(linear_log_model(pos_volt, *plog_popt))
pideal_model_data = ideal_model(pos_volt, *pideal_popt)

# --- Compute Chi-Squared for Each Model ---
ppow_chi2, ppow_red_chi2 = reduced_chi_squared(pos_amp, ppow_model_data, model_unc_power, len(ppow_popt))
plog_chi2, plog_red_chi2 = reduced_chi_squared(pos_amp, plog_model_data, log_amp_uncertainty, len(plog_popt))
pideal_chi2, pideal_red_chi2 = reduced_chi_squared(pos_amp, pideal_model_data, model_unc_ideal, len(pideal_popt))

# --- Print Results ---
print(f"Power Model: χ² = {ppow_chi2:.2f}, Reduced χ² = {ppow_red_chi2:.2f}")
print(f"Log Model: χ² = {plog_chi2:.2f}, Reduced χ² = {plog_red_chi2:.2f}")
print(f"Ideal Model: χ² = {pideal_chi2:.2f}, Reduced χ² = {pideal_red_chi2:.2f}")




# popt, pcov = curve_fit(power_model, volt, amp, p0=(8.0, 0.56), sigma=amp_unc, absolute_sigma=True, maxfev=10000)

# plt.plot(volt, amp, color = "blue")
# plt.plot(volt, power_model(volt, popt[0], popt[1]), color = "red")




# popt, pcov = curve_fit(ideal_model, volt, amp, p0=8.5, sigma=amp_unc, absolute_sigma=True)

# plt.plot(volt, amp, color = "blue")
# plt.plot(volt, ideal_model(volt, popt[0]), color = "red")

# popt, pcov = curve_fit(linear_log_model, volt, np.log(amp))#, sigma=amp_unc, absolute_sigma=True)

# plt.plot(volt, amp, color = "blue")
# plt.plot(np.log(volt), linear_log_model(volt, popt[0], popt[1]), color = "red")
# plt.xcale('log')
# plt.yscale('log')