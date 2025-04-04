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
R= .155 #SI Unit in meters. Distance middle to middle of the coil's thickness
R_unc = 0.002
k_char = (1.0/np.sqrt(2.0))*((4.0/5.0)**(3.0/2.0))*scipy.constants.mu_0*n/R #Characteristic of coil dimensions
k_char_unc = k_char*R_unc/(R**2)

#Data reading
#Constant Current data
cc_current, cc_voltage, cc_ps_volt, cc_diameter, cc_current_unc, cc_voltage_unc, cc_ps_volt_unc, cc_diameter_unc = np.loadtxt("Constant_Current_data (w unc).csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)
cc_current = np.abs(cc_current)
cc_diameter = cc_diameter*0.01 #Convertion to meters
cc_diameter_unc = cc_diameter_unc*0.01

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
cv_diameter *= 0.01
cv_diameter_unc *= 0.01

#Local variable for the fit
deltaV = (np.max(cv_voltage)+np.min(cv_voltage))/2.0 #Used for constant voltage fitting, average of largest and lowest measured values.
deltaV_unc = deltaV-np.min(cv_voltage)
I = (np.max(cc_current)+np.min(cc_current))/2.0 #Used for constant current fitting, average of largest and lowest measured values.
I_unc = I-np.min(cc_current)

#Bc values based on constant voltage data
Bc = k_char*cv_current*np.sqrt(2)
Bc_unc = k_char*cv_current*np.sqrt(2)*np.sqrt((k_char_unc/k_char)**2 + (cv_current_unc/cv_current)**2)

#Corrections made to Bc
rho = np.zeros(cv_diameter.size)
rho_unc = np.zeros(cv_diameter_unc.size)
rg = 5.5 * 0.01 #cm converted to m
rg_unc = 0.2 * 0.01
for r in range(len(cv_diameter)):
    if (cv_diameter[r]/2.0)<(rg/2.0):
        rho[r] = rg-cv_diameter[r]/2.0
        rho_unc[r]=np.sqrt(rg_unc**2+(0.5*cv_diameter_unc[r])**2)
    elif (cv_diameter[r]/2.0)>=(rg/2.0):
        rho[r]=cv_diameter[r]/2.0
        rho_unc[r]=0.5*cv_diameter_unc[r]
        
# rho = rg-cv_diameter/2.0

#Temporary rho value until measurement is done
# rho = np.abs(5.0-(cv_diameter/2.0))
# Bc *= 1.0-((rho**4)/((R**4)*((0.6583+0.29*(rho**2)/(R**2))**2)))

for p in range(len(rho)):
    if rho[p]>0.2*R and rho[p]<0.5*R:
        Bc[p] *= 1.0-((rho[p]**4)/((R**4)*((0.6583+0.29*(rho[p]**2)/(R**2))**2)))
        #Propagating the uncertainty in the Bc correction
        temp_Bc_pt1 = (rho[p]**2)/(R**2)#rho^2/R^2
        temp_Bc_pt1_unc = temp_Bc_pt1*np.sqrt(((2.0*rho[p]*rho_unc[p])/(rho[p]**2))**2 + ((2.0*R*R_unc)/(R**2))**2)
        temp_Bc_pt2 = 0.29*temp_Bc_pt1#For 0.29rho^2/R^2
        temp_Bc_pt2_unc = 0.29*temp_Bc_pt1_unc
        temp_Bc_pt3 = ((0.6583+0.29*(rho[p]**2)/(R**2))**2)#For ((0.6583+0.29*(rho[p]**2)/(R**2))**2)
        temp_Bc_pt3_unc = 2.0*(0.6583+0.29*(rho[p]**2)/(R**2))*temp_Bc_pt2_unc
        temp_Bc_pt4 = (R**4)*temp_Bc_pt3 #For the entire denominator of the fraction
        temp_Bc_pt4_unc = temp_Bc_pt4*np.sqrt(((4.0*(R**3)*R_unc)/(R**4))**2 + (temp_Bc_pt3_unc/temp_Bc_pt3)**2)
        temp_Bc_pt5 = (rho[p]**4)/temp_Bc_pt4 #The entire fraction
        temp_Bc_pt5_unc = temp_Bc_pt5*np.sqrt(((4.0*(rho[p]**3)*rho_unc[p])/rho[p])**2 + (temp_Bc_pt4_unc/temp_Bc_pt4)**2)
        temp_correction_unc = temp_Bc_pt5_unc * -1.0 #We have a minus sign in front
        Bc_unc[p] = Bc[p]*np.sqrt((Bc_unc[p]/Bc[p])**2 + (temp_correction_unc/(1.0-temp_Bc_pt5))**2)#Need to add the stuff


#Magnetic Field Bc Prediction Model
def magnetic_fit_model(x_val, a, b):
    return a*(1.0/x_val) - b #Where b is B_e

#Curve fitting for the magnetic field to obtain B_e
b_popt, b_pcov = curve_fit(magnetic_fit_model, cv_diameter/2.0, Bc, sigma = Bc_unc, absolute_sigma = True)

#Calculation of I_0
I0 = b_popt[1]/k_char
I0_unc = I0*np.sqrt((np.sqrt(b_pcov[1][1])/b_popt[1])**2 + (k_char_unc/k_char)**2)

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
plt.figure(figsize = (8, 12))
#Prediction plot
plt.subplot(2, 1, 1)
plt.errorbar(100.0*cv_diameter/2.0, 1000.0*Bc, xerr=100.0*cv_diameter_unc/2.0, yerr = 1000.0*b_unc_model, color = "red", fmt = 'o', label = "Calculation based on measurement data")
plt.plot(100.0*cv_diameter/2.0, 1000.0*magnetic_fit_model(cv_diameter/2.0, *b_popt), color = "green", label = "Magnetic Fit Model Prediction")
plt.title("Magnetic Fit Prediction")
plt.xlabel("Radius (cm)")
plt.ylabel(r'Magnetic Field ($mT$)')
plt.legend()

#Residual calculation
b_prediction = magnetic_fit_model(cv_diameter/2.0, *b_popt)
b_residual = Bc - b_prediction

#Residual plot
plt.subplot(2, 1, 2)
plt.plot(100.0*cv_diameter/2.0, np.zeros(cv_voltage.size), color = "blue", label = "Zero residual reference line")
plt.errorbar(100.0*cv_diameter/2.0, 1000.0*b_residual, xerr = 100.0*cv_diameter_unc/2.0, yerr = 1000.0*b_unc_model, color = "red", fmt = 'o', label = "Residual between measured and predicted data")
plt.title("Residual of the magnetic fit model")
plt.xlabel("Radius (cm)")
plt.ylabel(r'Error: Magnetic Field ($mT$)')

plt.legend()
plt.show()

#Plotting constant current
plt.figure(figsize = (8, 12))
#Prediction plot
plt.subplot(2, 1, 1)
plt.errorbar(cc_voltage, 100.0*(cc_diameter/2.0), xerr = cc_voltage_unc, yerr = 100.0*cc_diameter_unc/2.0, color = "red", fmt = 'o', label = "Measured Data")
plt.plot(cc_voltage, 100.0*const_Current_model(cc_voltage, *cc_popt), color = "green", label = "Model Prediction")
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
plt.errorbar(cc_voltage, 100.0*cc_residual, xerr = cc_voltage_unc, yerr = 100.0*np.sqrt(cc_diameter_unc**2 + np.sqrt(cc_pcov[0][0])**2), color = "red", fmt = 'o', label = "Residual between measured and predicted data")
plt.title("Residual of the constant current model")
plt.xlabel("Voltage(V)")
plt.ylabel("Error: Radius (cm)")

plt.legend()
plt.show()

#Plotting constant voltage
plt.figure(figsize = (8, 12))
#Prediction plot
plt.subplot(2, 1, 1)
plt.errorbar(cv_current, 100.0*(cv_diameter/2.0), xerr = cv_current_unc, yerr = 100.0*cv_diameter_unc/2.0, color = "red", fmt = 'o', label = "Measured Data")
plt.plot(cv_current, 100.0*const_Voltage_model(cv_current, *cv_popt), color = "green", label = "Model Prediction")
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
plt.errorbar(cv_current, 100.0*cv_residual, xerr = cv_current_unc, yerr = 100.0*np.sqrt(cv_diameter_unc**2 + np.sqrt(cv_pcov[0][0])**2), color = "red", fmt = 'o', label = "Residual between measured and predicted data")
plt.title("Residual of the constant voltage model")
plt.xlabel("Current (A)")
plt.ylabel("Error: Radius (cm)")

plt.legend()
plt.show()


#Reduced Chi Square Calculation
#Magnetic fit
#b_chi2 = np.sum((b_residual**2)/((b_unc_model)**2))
b_chi2 = np.sum((b_residual**2)/((cv_diameter_unc/2.0)**2 + (b_unc_model)**2))
b_chi2_r = b_chi2/(cv_diameter.size - b_popt.size)
print("Magnetic Fit Reduced Chi2 is: ", b_chi2_r)

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
cc_a_inv_unc = np.abs(-1.0 * np.sqrt(cc_pcov[0][0])/(cc_popt[0]**2))
cc_a_inv_pt2 = cc_a_inv/k_char
cc_a_inv_pt2_unc = cc_a_inv_pt2*np.sqrt((cc_a_inv_unc/cc_a_inv)**2+(k_char_unc/k_char)**2)
cc_a_inv_pt3 = cc_a_inv_pt2 / (I+ I0/np.sqrt(2))
cc_a_inv_pt3_unc = cc_a_inv_pt3*np.sqrt((cc_a_inv_pt2_unc/cc_a_inv_pt2)**2+((np.sqrt(I_unc**2+(I0_unc/np.sqrt(2))**2))/(I+ I0/np.sqrt(2)))**2)
cc_ratio = cc_a_inv_pt3**2
cc_ratio_unc = 2*(cc_a_inv_pt3)*cc_a_inv_pt3_unc

#Conversion for scientific notations
cc_ratio *= 1e-11
cc_ratio_unc *= 1e-11
cc_ratio = round(cc_ratio, 1)
cc_ratio_unc = round(cc_ratio_unc, 1)
print("Via constant current, the charge to mass ratio is: ", cc_ratio, "x10^11 C/kg ±",  cc_ratio_unc, "x10^11 C/kg")

#Via constant voltage
cv_a_inv = 1.0/cv_popt[0]
cv_a_inv_unc = np.abs(-1.0*np.sqrt(cv_pcov[0][0])/(cv_popt[0]**2))
cv_a_inv_deltaV_k_char = (np.sqrt(deltaV)/k_char)
cv_a_inv_deltaV_k_char_unc = cv_a_inv_deltaV_k_char*np.sqrt(((0.5*deltaV_unc/(deltaV**0.5))/(np.sqrt(deltaV)))**2 + (k_char_unc/k_char)**2)
cv_a_inv_pt2 = cv_a_inv * cv_a_inv_deltaV_k_char
cv_a_inv_pt2_unc = cv_a_inv_pt2 * np.sqrt((cv_a_inv_unc/cv_a_inv)**2 + ((cv_a_inv_deltaV_k_char_unc)/(cv_a_inv_deltaV_k_char))**2)
cv_ratio = cv_a_inv_pt2**2
cv_ratio_unc = 2.0*cv_a_inv_pt2_unc*cv_a_inv_pt2

#Conversion for scientific notations
cv_ratio *= 1e-11
cv_ratio_unc *= 1e-11
cv_ratio = round(cv_ratio, 1)
cv_ratio_unc = round(cv_ratio_unc, 1)
print("Via constant voltage, the charge to mass ratio is: ", cv_ratio, "x10^11 C/kg ±", cv_ratio_unc, "x10^11 C/kg")
