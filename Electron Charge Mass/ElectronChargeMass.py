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
cc_current, cc_voltage, cc_ps_volt, cc_diameter = np.loadtxt("Constant_Current_data.csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)
cc_current = np.abs(cc_current)

#Eliminating the last point to correct for very small voltage
cc_current = cc_current[:-1]
cc_voltage = cc_voltage[:-1]
cc_ps_volt = cc_ps_volt[:-1]
cc_diameter = cc_diameter[:-1]

#Constant Voltage data
cv_current, cv_voltage, cv_ps_volt, cv_diameter = np.loadtxt("Constant_Voltage_data.csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)

cv_current = np.abs(cv_current)

#Local variable for the fit
#deltaV = 149.996 #Used for constant voltage fitting, average of largest and lowest measured values.
deltaV = (np.max(cv_voltage)+np.min(cv_voltage))/2.0 #Used for constant voltage fitting, average of largest and lowest measured values.
I = (np.max(cc_current)+np.min(cc_current))/2.0 #Used for constant current fitting, average of largest and lowest measured values.

#B_c values based on constant voltage data
Bc = k_char*cv_current*np.sqrt(2)

#Magnetic Field Bc Prediction Model
def magnetic_fit_model(x_val, a, b):
    return a*(1.0/x_val) - b #Where b is B_e

#Curve fitting for the magnetic field to obtain B_e
b_popt, b_pcov = curve_fit(magnetic_fit_model, cv_diameter/2.0, Bc)

#Calculation of I_0
I0 = b_popt[1]/k_char

#Constant Current Prediction Model
def const_Current_model(x_val, a):
    return a*np.sqrt(x_val) #Returns r

#Constant Voltage Prediction Model
def const_Voltage_model(x_val, a):
    return a/(x_val+1.0/np.sqrt(2)*I0)

#Curve fitting for the constant voltage
cv_popt, cv_pcov = curve_fit(const_Voltage_model, cv_current, cv_diameter/2.0)

#Curve fitting for the constant current
cc_popt, cc_pcov = curve_fit(const_Current_model, cc_voltage, cc_diameter/2.0)


#Temp
plt.errorbar(cc_voltage, (cc_diameter/2.0), color = "red", fmt = 'o')
plt.plot(cc_voltage, const_Current_model(cc_voltage, *cc_popt), color = "green")
plt.title("Constant Current")
plt.show()
plt.errorbar(cv_current, (cv_diameter/2.0), color = "red", fmt = 'o')
plt.plot(cv_current, const_Voltage_model(cv_current, *cv_popt), color = "green")
plt.title("Constant Voltage")


#Printing estimated charge to mass ratio
#Via constant current
cc_a_inv = 1.0/cc_popt[0]
cc_a_inv /= (k_char*(I+ I0/np.sqrt(2)))
print("Via constant current, the charge to mass ratio is: ", cc_a_inv**2, " C/kg")

#Via constant voltage
cv_a_inv = 1.0/cv_popt[0]
cv_a_inv*=(np.sqrt(deltaV)/k_char)
print("Via constant voltage, the charge to mass ratio is: ", cv_a_inv**2, " C/kg")
