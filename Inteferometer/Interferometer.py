# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:12:00 2025

Turki Almansoori
Rui (Richard) Chen Geng Li

Code created for PHY224 Free Choice Lab: Interferometer
Prof. Sergio De La Barrera
Due March 23rd, 2025
"""

#Necessary modules
import numpy as np 
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 

#Theoretical Values of certain variables
wavelength_theo = 532e-9 #in unit of nano meters
thermal_coefficient_theo = 23e-6 #Coefficient of thermal expansion of aluminium per degrees celsius


#Data imports
w_reading, w_dN, w_reading_unc, w_dN_unc = np.loadtxt("Final_Wavelength_data.csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)

#Data import of the unfixed data
w_reading_u, w_dN_u, w_reading_unc_u, w_dN_unc_u = np.loadtxt("Final_Wavelength_data - Unfixed.csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)

#Prediction Model
#Wavelength
def deltaN_model(x_val, a):
    return x_val*(2.0/a)

#Obtaining the total change in fringe counts at the specific measuring point
w_dN_total = np.zeros(w_dN.size)
w_dN_total[0]=w_dN[0]
for n in range(1, len(w_dN)):
    w_dN_total[n]=w_dN[n]+w_dN_total[n-1]
    
#Calculating the fringe count uncertainty via propagation due to the addition above.
for w in range(1, len(w_dN_unc)):
    w_dN_unc[w]=np.sqrt((w_dN_unc[w-1]**2) + (w_dN_unc[w]**2))

#Obtaining the total change in fringe counts like above, but for the unfixed dataset
w_dN_total_u = np.zeros(w_dN_u.size)
w_dN_total_u[0]=w_dN_u[0]
for n in range(1, len(w_dN_u)):
    w_dN_total_u[n]=w_dN_u[n]+w_dN_total_u[n-1]
    
#Calculating the fringe count uncertainty via propagation due to the addition above.
for w in range(1, len(w_dN_unc_u)):
    w_dN_unc_u[w]=np.sqrt((w_dN_unc_u[w-1]**2) + (w_dN_unc_u[w]**2))

#Curve_fit
w_popt, w_pcov = curve_fit(deltaN_model, w_reading, w_dN_total, p0=(0.57), sigma=w_dN_unc, absolute_sigma = True)

#Curve_fit the unfixed data
w_u_popt, w_u_pcov = curve_fit(deltaN_model, w_reading_u, w_dN_total_u, p0=(0.57), sigma=w_dN_unc_u, absolute_sigma = True)

#Propagating Model Uncertainty
wave_model_unc_pt1 = 2.0*w_reading_unc
wave_model_unc = deltaN_model(w_reading, *w_popt)*np.sqrt((np.sqrt(w_pcov[0][0])/w_popt[0])**2 + (wave_model_unc_pt1/(2.0*w_reading))**2)
w_unc_total = np.sqrt(w_dN_unc**2 + wave_model_unc**2)

#Plotting
plt.figure(figsize = (12, 4))
#Unfixed dataset
plt.subplot(1, 2, 1)
plt.errorbar(w_reading_u, w_dN_total_u, xerr=w_reading_unc_u, yerr=w_dN_unc_u, fmt = "o", color = "red", label = "Measured Data", markersize = 0.5)
plt.plot(w_reading_u, deltaN_model(w_reading_u, *w_u_popt), color = "blue", label="Prediction Data")
plt.plot(w_reading_u, w_reading_u*2.0/(wavelength_theo*1e6), color = "green", label="Theoretical Prediction")
plt.xlabel("Change in unit of micrometer (µm)")
plt.ylabel("Change in unit of fringe count")
plt.title("Unfixed Data Wavelength Prediction")
plt.legend()
#Fixed dataset
plt.subplot(1, 2, 2)
plt.errorbar(w_reading, w_dN_total, xerr=w_reading_unc, yerr=w_dN_unc, fmt = "o", color = "red", label = "Measured Data", markersize = 0.5)
plt.plot(w_reading, deltaN_model(w_reading, *w_popt), color = "blue", label="Prediction Data")
plt.plot(w_reading, w_reading*2.0/(wavelength_theo*1e6), color = "green", label="Theoretical Prediction")
plt.xlabel("Change in unit of micrometer (µm)")
plt.ylabel("Change in unit of fringe count")
plt.title("Fixed Data Wavelength Prediction")
plt.legend()
plt.show()

#Residuals
w_residual = w_dN_total - deltaN_model(w_reading, *w_popt)
w_zero_line = np.zeros(w_residual.size)
plt.plot(w_reading, w_zero_line, color = "blue", label = "Reference zero residual line")
plt.errorbar(w_reading, w_residual, xerr=w_reading_unc, yerr=w_unc_total, color = "red", fmt = "o", label = "Residual between measurement and prediction", markersize = 0.5)
plt.xlabel("Change in unit of micrometer (µm)")
plt.ylabel("Error in the fringe count")
plt.title("Wavelength Residual between the measurement data and the prediction data")
plt.legend()
plt.show()

#Printing the predicted wavelength
print("Predicted Wavelength = ", w_popt[0]*1000.0, "nm ± ", np.sqrt(w_pcov[0][0])*1000.0, "nm")

#Reduced Chi Square
reduced_chi2 = np.sum((w_residual)**2/w_unc_total**2) /(w_residual.size - w_popt.size)
print ("The Reduced Chi Square Value For Wavelength Prediction is: ", reduced_chi2)

###################### Index of Refraction ################################

#Variables
thickness = 7.70e-3 #mm converted to m
thickness_unc = 0.03e-3

#Reading Data
ir_reading, ir_min, ir_dN, ir_min_unc, ir_dN_unc = np.loadtxt("Index_of_Refraction_2.csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)

#Obtaining the total change in fringe counts at the specific measuring point
ir_dN_total = np.zeros(ir_dN.size)
ir_dN_total[0]=ir_dN[0]
for n in range(1, len(ir_dN)):
    ir_dN_total[n]=ir_dN[n]+ir_dN_total[n-1]


#Converting arc minutes to degrees, so that we can add them.
ir_reading += ir_min/60.0
ir_reading_unc = ir_min_unc/60.0

#Processing the data as our measurement was in descending degree measurements
#Then apply an offset and then convert to radians
ir_reading*=-1.0
ir_reading-=ir_reading[0]
ir_reading = np.radians(ir_reading)
ir_reading[0]=1e-9
ir_reading_unc = np.radians(ir_reading_unc)

#Propagating the uncertainties to due offsetting.
for u in range(1, len(ir_dN_unc)):
    ir_reading_unc[u]=np.sqrt((ir_reading_unc[0]**2) + (ir_reading_unc[u]**2))

ir_dN_unc[0]=0.1
ir_dN_unc_old = np.array(ir_dN_unc)
#dN Uncertainty propagation for the fringe count due to addition
for u in range(1, len(ir_dN_unc)):
    ir_dN_unc[u]=np.sqrt((ir_dN_unc[u-1]**2) + (ir_dN_unc[u]**2))

#Prediction Model
#Index of Refraction
def index_refraction_model(x_val, a):
    return (thickness/(w_popt[0]*1e-6))*((x_val)**2)*(1.0-(1.0/a))

#Curve_fit
ir_popt, ir_pcov = curve_fit(index_refraction_model, ir_reading, ir_dN_total, p0=(1.68), sigma = ir_dN_unc, absolute_sigma = True)

#Plotting the main plot
plt.errorbar(ir_reading, ir_dN_total, xerr=ir_reading_unc, yerr=ir_dN_unc, fmt = "o", color = "red", label = "Index of Refraction Measurement", markersize = 0.5)
plt.plot(ir_reading, index_refraction_model(ir_reading, *ir_popt), color = "blue", label = "Prediction")
plt.fill_between(ir_reading, (thickness/(0.532*1e-6))*((ir_reading)**2)*(1.0-(1.0/1.4)), (thickness/(0.532*1e-6))*((ir_reading)**2)*(1.0-(1.0/1.7)), color='green', alpha=0.35, interpolate=True, label = "Theoretical Prediction Range (Upper: n=1.7, lower: n=1.4)")
plt.xlabel("Change in rotation in unit of radians (rad)")
plt.ylabel("Change in unit of fringe count")
plt.title("Index of Refraction Prediction")
plt.legend(fontsize=8)
plt.show()

#Uncertainty Propagation
ir_f_pt1 = (thickness/(w_popt[0]*1e-6))
ir_unc_pt1 = ir_f_pt1*np.sqrt((thickness_unc/thickness)**2 + (np.sqrt(w_pcov[0][0])/w_popt[0])**2)
ir_f_pt2 = ((ir_reading)**2)
ir_unc_pt2 = ir_f_pt2*2.0*ir_reading_unc/ir_reading
ir_unc_pt3 = ir_f_pt1*ir_f_pt2*np.sqrt((ir_unc_pt1/ir_f_pt1)**2 + (ir_unc_pt2/ir_f_pt2)**2)
ir_f_pt4 = (1.0-(1.0/ir_popt[0]))
ir_unc_pt4 = np.sqrt(w_pcov[0][0])/(w_popt[0]**2)
ir_model_unc = index_refraction_model(ir_reading, *ir_popt)*np.sqrt((ir_unc_pt3/(ir_f_pt1*ir_f_pt2))**2 + (ir_unc_pt4/ir_f_pt4)**2)
ir_total_unc = np.sqrt((ir_model_unc)**2 + (ir_dN_unc)**2)

#Residual plot
ir_residual = ir_dN_total - index_refraction_model(ir_reading, *ir_popt)
ir_zero_line = np.zeros(ir_residual.size)
plt.plot(ir_reading, ir_zero_line, color = "blue", label = "Reference zero residual line")
plt.errorbar(ir_reading, ir_residual, xerr=ir_reading_unc, yerr=ir_total_unc, color = "red", fmt = "o", label = "Residual between measurement and prediction", markersize = 0.5)
plt.xlabel("Change in rotation in unit of radians (rad)")
plt.ylabel("Error in the fringe count")
plt.title("Index of Refraction Residual between the measurement data and the prediction data")
plt.legend(fontsize = 8, loc = 'upper left')
plt.show()

#Printing the prediction of index of refraction
print("Predicted Index of Refraction: n = ", ir_popt[0], " ± ", np.sqrt(ir_pcov[0][0]))

#Reduced Chi Squared Value
ir_reduced_chi2 = np.sum((ir_residual)**2/(ir_total_unc**2)) /(ir_residual.size - ir_popt.size)
print ("The Reduced Chi Square Value For Index of Refraction Prediction is: ", ir_reduced_chi2)


###################### Thermal Expansion of Aluminium ################################

#Reading Data
t_temp, t_dN, t_current, t_temp_unc, t_dN_unc = np.loadtxt("Thermal_Expansion_Data_Combined.csv", 
                                                  delimiter = ',', 
                                                  skiprows=1, unpack=True)

#Variables
L0=0.09012 #Aluminium rod length at base temperature
L0_unc= 0.00002

#Prediction Model
def thermal_expansion_model(x_val, a):
    return (2.0*L0/(w_popt[0]*1e-6))*a*(x_val-t_temp[0])


#Total fringe counts at specific measurement point
t_dN_total = np.zeros(t_dN.size)
t_dN_total[0]=t_dN[0]
for n in range(1, len(t_dN)):
    t_dN_total[n]=t_dN[n]+t_dN_total[n-1]
    
#Calculating the fringe count uncertainty via propagation due to the addition above.
for t in range(1, len(t_dN_unc)):
    t_dN_unc[t]=np.sqrt((t_dN_unc[t-1]**2) + (t_dN_unc[t]**2))

#Curve_fit
t_popt, t_pcov = curve_fit(thermal_expansion_model, t_temp, t_dN_total, p0=(29.33790993115546746e-06), sigma = t_dN_unc, absolute_sigma=True)

#Main plot for the data and prediction comparison
plt.errorbar(t_temp, t_dN_total, xerr = t_temp_unc, yerr= t_dN_unc, fmt="o", color = "red", label = "Measurement Data", markersize = 0.5)
plt.plot(t_temp, thermal_expansion_model(t_temp, *t_popt), color = "blue", label = "Prediction")
plt.plot(t_temp, (2.0*L0/(wavelength_theo))*thermal_coefficient_theo*(t_temp-t_temp[0]), color = "green", label = "Theoretical Prediction")
plt.xlabel("Temperature in degrees Celsius (C)")
plt.ylabel("Change in unit of fringe counts")
plt.title("Thermal Coefficient of Aluminium Prediction")
plt.legend()
plt.show()

#Model Uncertainty Propagation
t_f_pt1 = (2.0*L0/(w_popt[0]*1e-6))
t_unc_pt1 = t_f_pt1*np.sqrt((L0_unc/L0)**2 + (np.sqrt(w_pcov[0][0])/w_popt[0])**2)
t_f_pt2 = (t_temp-t_temp[0])
t_f_pt2[0]=1e-9
t_unc_pt2 = np.sqrt(t_temp_unc**2 + t_temp_unc[0]**2)
t_model_unc = thermal_expansion_model(t_temp, *t_popt)*np.sqrt((t_unc_pt1/t_f_pt1)**2 + (t_unc_pt2/t_f_pt2)**2 + (np.sqrt(t_pcov[0][0])/t_popt[0])**2)
t_model_unc[0]=1e-9
t_total_unc = np.sqrt(t_model_unc**2 + t_dN_unc**2)

#Residual plot
t_residual = t_dN_total - thermal_expansion_model(t_temp, *t_popt)
t_zero_line = np.zeros(t_residual.size)
plt.plot(t_temp, t_zero_line, color = "blue", label = "Reference zero residual line")
plt.errorbar(t_temp, t_residual, xerr=t_unc_pt2, yerr = t_total_unc, color = "red", fmt = "o", label = "Residual between measurement and prediction", markersize = 0.5)
plt.xlabel("Temperature in degrees Celsius (C)")
plt.ylabel("Error in the fringe count")
plt.title("Thermal Coefficient Residual between the measurement data and the prediction data")
plt.legend(loc = 'lower left')
plt.show()

#Printing the prediction of thermal expansion coefficient
print("Predicted thermal expansion coefficient for aluminium: ", t_popt[0], "/C", " ± ", np.sqrt(t_pcov[0][0]), "/C")

#Reduced Chi Squared Value
t_reduced_chi2 = np.sum((t_residual)**2/(t_model_unc**2)) /(t_residual.size - t_popt.size)
print ("The Reduced Chi Square Value for Thermal Expansion Coefficient is: ", t_reduced_chi2)



