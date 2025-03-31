# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:08:54 2025

Turki Almansoori
Rui (Richard) Chen Geng Li

Code created for PHY224 Free Choice Lab: Electron Mass Charge Ratio
Prof. Sergio De La Barrera
Due April 4th, 2025
"""

#Necessary modules
import numpy as np 
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 

#Global variables
n = 130 #Number of coil turns
R= 15.5 #Distance middle to middle of the coil's thickness
k_char = (1.0/np.sqrt(2.0))*((4.0/5.0)**(3.0/2.0))*scipy.constants.mu_0*n/R #Characteristic of coil dimensions

#32 outer, 31 mid, 29.8 inner Radius
#17.5 outer, 15.5 mid, 13.2 inner separation distance

#Data reading
#Constant Current data
cc_current, cc_voltage, cc_ps_volt, cc_diameter, cc_current_unc, cc_voltage_unc, cc_ps_volt_unc, cc_diameter_unc = np.loadtxt("Constant_Current_data (w unc).csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)
cc_current = np.abs(cc_current)

#Eliminating the last point to correct for very small voltage
cc_current = cc_current[:-1]
cc_voltage = cc_voltage[:-1]
cc_ps_volt = cc_ps_volt[:-1]
cc_diameter = cc_diameter[:-1]
cc_current_unc = cc_current_unc[:-1]
cc_voltage_unc = cc_voltage_unc[:-1]
cc_ps_volt_unc = cc_ps_volt_unc[:-1]
cc_diameter_unc = cc_diameter_unc[:-1]

#Constant Voltage data
cv_current, cv_voltage, cv_ps_volt, cv_diameter, cv_current_unc, cv_voltage_unc, cv_ps_volt_unc, cv_diameter_unc = np.loadtxt("Constant_Voltage_data (w unc).csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)

cv_current = np.abs(cv_current)

#Local variable for the fit
#deltaV = 149.996 #Used for constant voltage fitting, average of largest and lowest measured values.
deltaV = (np.max(cv_voltage)+np.min(cv_voltage))/2.0 #Used for constant voltage fitting, average of largest and lowest measured values.
I = (np.max(cc_current)+np.min(cc_current))/2.0 #Used for constant current fitting, average of largest and lowest measured values.

#Bc values based on constant voltage data
Bc = k_char*cv_current*np.sqrt(2)
Bc_unc = k_char*cv_current_unc*np.sqrt(2)

#Corrections made to Bc
#Temporary rho value until measurement is done
rho = np.abs(5.0-(cv_diameter/2.0))
Bc *= 1.0-((rho**4)/((R**4)*((0.6583+0.29*(rho**2)/(R**2))**2)))

#Magnetic Field Bc Prediction Model
def magnetic_fit_model(x_val, a, b):
    return a*(1.0/x_val) - b #Where b is B_e

#Curve fitting for the magnetic field to obtain B_e
b_popt, b_pcov = curve_fit(magnetic_fit_model, cv_diameter/2.0, Bc, sigma = Bc_unc, absolute_sigma = True)

#Calculation of I_0
I0 = b_popt[1]/k_char
I0_unc = np.sqrt(b_pcov[1][1])/k_char

#Constant Current Prediction Model
def const_Current_model(x_val, a):
    return a*np.sqrt(x_val) #Returns r

#Constant Voltage Prediction Model
def const_Voltage_model(x_val, a):
    return a/(x_val+I0/np.sqrt(2))

#Curve fitting for the constant voltage
cv_popt, cv_pcov = curve_fit(const_Voltage_model, cv_current, cv_diameter/2.0, sigma = cv_diameter_unc/2.0, absolute_sigma = True)

#Curve fitting for the constant current
cc_popt, cc_pcov = curve_fit(const_Current_model, cc_voltage, cc_diameter/2.0, sigma = cc_diameter_unc/2.0, absolute_sigma = True)


#Propagating model uncertainties
#Magnetic fit model uncertainties
b_a_unc = np.sqrt(b_pcov[0][0]) #Uncertainty for fitted parameter a
b_b_unc = np.sqrt(b_pcov[1][1]) #Uncertainty for fitted parameter b
b_unc_pt1 = (b_popt[0]/(cv_diameter/2.0))*np.sqrt((b_a_unc/b_popt[0])**2 + (cv_diameter_unc/cv_diameter)**2) #For a/x unc
b_unc_model = np.sqrt((b_unc_pt1)**2 + (b_b_unc)**2) #For the addition of a/x and b

#Constant current uncertainties
cc_a_unc = np.sqrt(cc_pcov[0][0]) #Uncertainty for fitted parameter a
cc_unc_pt1 = 0.5*np.sqrt(cc_voltage)*cc_voltage_unc/cc_voltage #Uncertainty for sqrt(x)
cc_unc_model = cc_popt[0]*np.sqrt(cc_voltage)*np.sqrt((cc_a_unc/cc_popt[0])**2 + (cc_unc_pt1/np.sqrt(cc_voltage))**2)

#Constant voltage uncertainties
cv_a_unc = np.sqrt(cv_pcov[0][0]) #Uncertainty for fitted parameter a
cv_I0_unc = I0_unc/np.sqrt(2) #Uncertainty for I0/sqrt(2)
cv_unc_pt1 = np.sqrt((cv_current_unc)**2 + (cv_I0_unc)**2) #Uncertainty for x+I0/sqrt(2)
cv_unc_model = (cv_popt[0]/(cv_current + I0/np.sqrt(2)))*np.sqrt((cv_a_unc/cv_popt[0])**2 + (cv_unc_pt1/(cv_current + I0/np.sqrt(2)))**2) #Uncertainty for a/(x+I0/sqrt(2))


#Plotting magnetic fit model


#Plotting constant current
plt.figure(figsize = (8, 12))
#Prediction plot
plt.subplot(2, 1, 1)
plt.errorbar(cc_voltage, (cc_diameter/2.0), xerr = cc_voltage_unc, yerr = cc_diameter_unc/2.0, color = "red", fmt = 'o', label = "Measured Data")
plt.plot(cc_voltage, const_Current_model(cc_voltage, *cc_popt), color = "green", label = "Model Prediction")
plt.title("Constant Current Prediction Model")
plt.xlabel("Voltage(V)")
plt.ylabel("Radius (cm)")
plt.legend()

#Residual calculation
cc_prediction = const_Current_model(cc_voltage, *cc_popt)
cc_residual = cc_diameter/2.0 - cc_prediction

#Residual plot
plt.subplot(2, 1, 2)
plt.plot(cc_voltage, np.zeros(cc_voltage.size), color = "blue", label = "Zero residual reference line")
plt.errorbar(cc_voltage, cc_residual, xerr = cc_voltage_unc, yerr = np.sqrt(cc_diameter_unc**2 + np.sqrt(cc_pcov[0][0])**2), color = "red", fmt = 'o', label = "Residual between measured and predicted data")
plt.title("Residual of the constant current model")
plt.xlabel("Voltage(V)")
plt.ylabel("Error: Radius (cm)")

plt.legend()
plt.show()

#Plotting constant voltage
plt.figure(figsize = (8, 12))
#Prediction plot
plt.subplot(2, 1, 1)
plt.errorbar(cv_current, (cv_diameter/2.0), xerr = cv_current_unc, yerr = cv_diameter_unc/2.0, color = "red", fmt = 'o', label = "Measured Data")
plt.plot(cv_current, const_Voltage_model(cv_current, *cv_popt), color = "green", label = "Model Prediction")
plt.title("Constant Voltage Prediction Model")
plt.xlabel("Current (A)")
plt.ylabel("Radius (cm)")
plt.legend()

#Residual calculation
cv_prediction = const_Voltage_model(cv_current, *cv_popt)
cv_residual = cv_diameter/2.0 - cv_prediction

#Residual plot
plt.subplot(2, 1, 2)
plt.plot(cv_current, np.zeros(cv_voltage.size), color = "blue", label = "Zero residual reference line")
plt.errorbar(cv_current, cv_residual, xerr = cv_current_unc, yerr = np.sqrt(cv_diameter_unc**2 + np.sqrt(cv_pcov[0][0])**2), color = "red", fmt = 'o', label = "Residual between measured and predicted data")
plt.title("Residual of the constant voltage model")
plt.xlabel("Current (A)")
plt.ylabel("Error: Radius (cm)")

plt.legend()
plt.show()


#Reduced Chi Square Calculation
#Constant current
cc_chi2 = np.sum((cc_residual**2)/((cc_diameter_unc/2)**2 + cc_unc_model**2))
cc_chi2_r = cc_chi2/(cc_voltage.size - cc_popt.size)
print("Constant Current Reduced Chi2 is: ", cc_chi2_r)

#Constant voltage
cv_chi2 = np.sum((cv_residual**2)/((cv_diameter_unc/2)**2 + cv_unc_model**2))
cv_chi2_r = cv_chi2/(cv_voltage.size - cv_popt.size)
print("Constant Voltage Reduced Chi2 is: ", cv_chi2_r)



#Printing estimated charge to mass ratio
#Via constant current
cc_a_inv = 1.0/cc_popt[0]
cc_a_inv /= (k_char*(I+ I0/np.sqrt(2)))
print("Via constant current, the charge to mass ratio is: ", cc_a_inv**2, " C/kg ±")

#Via constant voltage
cv_a_inv = 1.0/cv_popt[0]
cv_a_inv*=(np.sqrt(deltaV)/k_char)
print("Via constant voltage, the charge to mass ratio is: ", cv_a_inv**2, " C/kg ±")
